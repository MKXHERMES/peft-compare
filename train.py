# ~/peft-compare/train.py

import os
import sys
import time
import yaml
import torch
import wandb
from dataclasses import dataclass
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Config Loading ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def validate_config(cfg: dict):
    method = cfg["method"]
    assert method in ("lora", "qlora", "dora"), f"Unknown method: {method}"

    # load_in_4bit is now per-method (v5) instead of one blanket flag -- on the rented
    # 24GB card, LoRA/DoRA use a full-precision base while QLoRA keeps 4-bit NF4. Check
    # all three keys exist so a typo here fails loudly instead of silently defaulting.
    bit_map = cfg["model"]["load_in_4bit"]
    for m in ("lora", "qlora", "dora"):
        assert m in bit_map, f"model.load_in_4bit is missing the '{m}' key"

    eff_batch = cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    print(f"[config] method={method}, model={cfg['model']['name']}, "
          f"load_in_4bit={bit_map[method]}, effective_batch={eff_batch}")


# ── Model Loading ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    method = cfg["method"]
    model_name = cfg["model"]["name"]
    use_4bit = cfg["model"]["load_in_4bit"][method]

    print(f"[model] Loading {model_name} (4bit={use_4bit}) ...")

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=(method == "qlora"),  # double quant only for QLoRA
        )
    else:
        bnb_cfg = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    if use_4bit:
        # k-bit-specific prep: casts norms to fp32, enables input grads for checkpointing,
        # etc. Required for the QLoRA path.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
        )
    elif cfg["training"]["gradient_checkpointing"]:
        # Full-precision path (LoRA/DoRA): prepare_model_for_kbit_training is 4-bit-only,
        # so replicate the one piece of it that still applies here. With a frozen base
        # model, gradient checkpointing needs SOME tensor in the graph to require grad for
        # backward to flow through the checkpointed blocks into the LoRA adapters --
        # enable_input_require_grads() is what provides that. Skipping this with a frozen
        # base + checkpointing silently produces zero gradients rather than an error, so
        # don't remove it even though it looks like a no-op.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=cfg["peft"]["r"],
        lora_alpha=cfg["peft"]["lora_alpha"],
        target_modules=cfg["peft"]["target_modules"],
        lora_dropout=cfg["peft"]["lora_dropout"],
        bias=cfg["peft"]["bias"],
        use_dora=(method == "dora"),     # DoRA toggle
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ── GPU Logging ─────────────────────────────────────────────────────────────────

def log_gpu_stats(step: int, prefix: str = "train"):
    if not torch.cuda.is_available():
        return
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    wandb.log({
        f"{prefix}/gpu_mem_allocated_gb": mem_alloc,
        f"{prefix}/gpu_mem_reserved_gb": mem_reserved,
        "step": step,
    })
    print(f"[gpu] step={step} allocated={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB")


# ── Training ────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config("config.yaml")
    validate_config(cfg)

    method = cfg["method"]
    t_cfg = cfg["training"]
    l_cfg = cfg["logging"]

    # Derived from `method` alone (v5) -- output_dir and run_name used to be separate
    # manual fields in config.yaml, which meant 3 things had to be kept in sync by hand
    # across the three runs instead of 1. Forgetting to bump output_dir was a real risk
    # (silently overwrites a previous method's checkpoint).
    output_dir = os.path.join("runs", method)
    run_name = f"{method}-full"

    # W&B init — must happen before TrainingArguments to avoid duplicate runs
    wandb.init(
        project=l_cfg["wandb_project"],
        group=l_cfg["wandb_group"],
        name=run_name,
        config=cfg,
    )

    # Log initial GPU state
    log_gpu_stats(step=0, prefix="init")

    # Dataset
    print("[data] Loading dataset from disk...")
    dataset = load_from_disk(cfg["data"]["path"])
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    # Model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg["batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        max_steps=t_cfg["max_steps"],
        warmup_steps=t_cfg["warmup_steps"],
        bf16=t_cfg["bf16"],
        fp16=t_cfg["fp16"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        load_best_model_at_end=False,
        report_to="wandb",
        dataloader_num_workers=t_cfg["dataloader_num_workers"],
        save_only_model=True,             # Adapter weights only — saves disk per run
        optim="adamw_torch_fused",        # was paged_adamw_8bit. That optimizer pages
                                           # 8-bit state to CPU to save VRAM -- a real win
                                           # on the 4GB card, but the LoRA adapter is ~3M
                                           # params regardless of base precision, so its
                                           # optimizer state was always tiny (~tens of MB).
                                           # On 24GB there's nothing left for paging to buy
                                           # us; the fused kernel is the faster choice now.
        dataloader_pin_memory=t_cfg.get("dataloader_pin_memory", True),
        dataset_text_field=cfg["data"]["text_field"],
        max_length=t_cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Train
    print(f"[train] Starting {method} run -> {output_dir} (W&B name: {run_name})")
    log_gpu_stats(step=0, prefix="pre_train")
    start = time.time()

    trainer.train()

    elapsed = time.time() - start
    samples_per_sec = len(train_ds) / elapsed

    wandb.log({
        "total_train_time_s": elapsed,
        "samples_per_second": samples_per_sec,
    })
    log_gpu_stats(step=t_cfg["max_steps"], prefix="post_train")

    print(f"[train] Done. {elapsed:.0f}s | {samples_per_sec:.2f} samples/sec")

    # Save final adapter
    final_path = os.path.join(output_dir, "checkpoint-final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[save] Adapter saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()