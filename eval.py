# ~/peft-compare/eval.py
# Run after all three training runs are complete.
# Usage: python eval.py --methods lora qlora dora

import argparse
import csv
import gc
import os
import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer as rs


# v5 update: load_in_4bit is NO LONGER hardcoded True for every method here. Under the
# original 4GB-card config, all three checkpoints really were trained against a 4-bit
# base, so hardcoding True was correct and intentionally left no flag to drift out of
# sync. On the rented 24GB card, config.yaml's model.load_in_4bit is per-method
# (lora=false, qlora=true, dora=false) -- LoRA/DoRA now train against a full-precision
# base. If eval.py still forced 4-bit for every method, it would silently attach the
# LoRA/DoRA adapters to the WRONG base representation at eval time and produce garbage
# (or just badly degraded) summaries with no error raised -- this is exactly the kind of
# drift the original comment was trying to prevent, just inverted. Fix: read
# model.load_in_4bit straight out of config.yaml in main() (the same file train.py reads)
# and thread it through as use_4bit, so train.py and eval.py can never disagree about a
# given method's base precision.
METHODS = {
    "lora":  {"checkpoint": "runs/lora/checkpoint-final"},
    "qlora": {"checkpoint": "runs/qlora/checkpoint-final"},
    "dora":  {"checkpoint": "runs/dora/checkpoint-final"},
}


def load_model_for_eval(base_model_name: str, checkpoint_path: str, method: str, use_4bit: bool):
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=(method == "qlora"),  # matches train.py exactly
        )
    else:
        bnb_cfg = None
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation (decoder-only)
    return model, tokenizer


def build_eval_inputs(tokenizer, document_texts: list, max_input_tokens: int) -> dict:
    # Reserves a fixed token budget for the "### Document:" / "### Summary:" template up
    # front and truncates only the document text to fit what's left, then assembles the
    # sequence manually. This guarantees the model always sees the "### Summary:" cue
    # regardless of document length -- tokenizing the full assembled prompt string and
    # letting truncation land wherever it happens to fall could (and at the old
    # max_seq_length=256, did) truncate the marker away entirely for long documents.
    header = "### Document:\n"
    marker = "\n### Summary:\n"
    header_ids = tokenizer(header, add_special_tokens=False)["input_ids"]
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    reserved = len(bos_ids) + len(header_ids) + len(marker_ids)
    doc_budget = max(max_input_tokens - reserved, 1)

    sequences = []
    for doc in document_texts:
        doc_ids = tokenizer(doc, add_special_tokens=False, truncation=True, max_length=doc_budget)["input_ids"]
        sequences.append(bos_ids + header_ids + doc_ids + marker_ids)

    # Manual left-pad to longest in batch (tokenizer.padding_side="left" set in load_model_for_eval)
    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = [], []
    for s in sequences:
        pad_len = max_len - len(s)
        input_ids.append([pad_id] * pad_len + s)
        attention_mask.append([0] * pad_len + [1] * len(s))
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)}


def generate_summaries_batch(model, tokenizer, document_texts: list, max_input_tokens: int, max_new_tokens: int) -> list:
    inputs = build_eval_inputs(tokenizer, document_texts, max_input_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # Greedy — deterministic for fair comparison
            pad_token_id=tokenizer.eos_token_id,
        )
    # Left-padding means every row's real input ends at the same column, so a single
    # slice index recovers each row's newly generated tokens.
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[input_len:], skip_special_tokens=True).strip()
        for row in outputs
    ]


def evaluate_method(method: str, base_model_name: str, test_ds, use_4bit: bool, max_input_tokens: int,
                     max_new_tokens: int, batch_size: int, n: int = 200) -> dict:
    info = METHODS[method]
    print(f"\n[eval] Evaluating {method} from {info['checkpoint']} (4bit={use_4bit}) ...")

    model, tokenizer = load_model_for_eval(base_model_name, info["checkpoint"], method, use_4bit)

    # Log VRAM after model load to confirm it's all on GPU
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  [vram] after load: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    n_eff = min(n, len(test_ds))
    subset = test_ds.select(range(n_eff))
    done, last_print = 0, 0

    for batch_start in range(0, n_eff, batch_size):
        batch = subset.select(range(batch_start, min(batch_start + batch_size, n_eff)))

        doc_parts, ref_summaries = [], []
        for ex in batch:
            full_text = ex["text"]
            doc_parts.append(full_text.split("### Summary:")[0].replace("### Document:\n", "").strip())
            ref_summaries.append(full_text.split("### Summary:")[1].strip())

        pred_summaries = generate_summaries_batch(model, tokenizer, doc_parts, max_input_tokens, max_new_tokens)

        for ref, pred in zip(ref_summaries, pred_summaries):
            scores.append(scorer.score(ref, pred)["rougeL"].fmeasure)

        # Release cached blocks after every batch -- guards against fragmentation
        # building up across many sequential generate() calls of varying length.
        torch.cuda.empty_cache()

        done += len(batch)
        if done - last_print >= 20 or done == n_eff:
            cur_alloc = torch.cuda.memory_allocated() / 1e9
            cur_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [{done}/{n_eff}] running ROUGE-L avg: {sum(scores)/len(scores):.4f} "
                  f"| vram alloc={cur_alloc:.2f}GB reserved={cur_reserved:.2f}GB")
            last_print = done

    avg_rouge_l = sum(scores) / len(scores)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"[eval] {method}: ROUGE-L={avg_rouge_l:.4f}, peak VRAM={peak_vram:.2f}GB")

    # Reset peak memory stats for next method
    torch.cuda.reset_peak_memory_stats()

    # Free memory before loading the next method's model. A plain `del` + empty_cache() is
    # NOT reliably enough here: accelerate's device_map="auto" dispatch attaches hooks to
    # the model's submodules that create Python reference cycles, so refcounting alone
    # won't destroy the model on `del`, and empty_cache() can only release memory that's
    # already unreferenced. Forcing the cyclic GC first is what actually frees it.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    freed_alloc = torch.cuda.memory_allocated() / 1e9
    freed_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  [vram] after cleanup: allocated={freed_alloc:.2f}GB, reserved={freed_reserved:.2f}GB")

    return {
        "method": method,
        "rouge_l": round(avg_rouge_l, 4),
        "peak_vram_gb": round(peak_vram, 3),
        "n_eval_samples": n_eff,
    }, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["lora", "qlora", "dora"])
    parser.add_argument("--base_model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--n", type=int, default=200, help="number of test samples to evaluate per method")
    parser.add_argument("--data_path", default="data/billsum_processed")
    parser.add_argument("--output", default="results/metrics.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="generation batch size. Default raised to 8 for the rented 24GB "
                              "card (was 1 on the 4GB card). Watch the [vram] log after each "
                              "model loads -- if it's not close to the 4GB-card baseline, you "
                              "have headroom to go higher; if it's uncomfortably close to 24GB, "
                              "drop this back down.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                         help="max tokens generated per summary. Keep at 128 for numbers you intend to "
                              "publish -- BillSum reference summaries run well past 64 tokens, so a lower "
                              "cap deflates ROUGE-L for all three methods equally. Fine to drop to 64 for "
                              "a quick 'does it run' smoke test only.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    max_input_tokens = train_cfg["training"]["max_seq_length"]
    bit_map = train_cfg["model"]["load_in_4bit"]
    print(f"[config] max_input_tokens={max_input_tokens}, load_in_4bit={bit_map} (read from {args.config})")

    dataset = load_from_disk(args.data_path)
    test_ds = dataset["test"]

    os.makedirs("results", exist_ok=True)
    results = []
    per_method_scores = {}

    for method in args.methods:
        if not os.path.exists(METHODS[method]["checkpoint"]):
            print(f"[skip] {method} checkpoint not found at {METHODS[method]['checkpoint']}")
            continue
        result, scores = evaluate_method(method, args.base_model, test_ds, use_4bit=bit_map[method],
                                           max_input_tokens=max_input_tokens, max_new_tokens=args.max_new_tokens,
                                           batch_size=args.batch_size, n=args.n)
        results.append(result)
        per_method_scores[method] = scores

    # Write CSV
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[done] Results saved to {args.output}")
        print("\n=== FINAL RESULTS ===")
        for r in results:
            print(f"  {r['method']:6s} | ROUGE-L={r['rouge_l']:.4f} | VRAM={r['peak_vram_gb']:.2f}GB")

    # Per-document scores, for paired significance testing across methods.
    # Wide format: row i is the same document for every method, since each
    # evaluate_method() call walks the identical test_ds with the same n_eff --
    # column alignment is guaranteed without needing to store document IDs.
    if per_method_scores:
        scores_path = os.path.join("results", "per_example_scores.csv")
        methods_present = list(per_method_scores.keys())
        n_docs = len(next(iter(per_method_scores.values())))
        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_idx"] + methods_present)
            for i in range(n_docs):
                writer.writerow([i] + [per_method_scores[m][i] for m in methods_present])
        print(f"[done] Per-example scores saved to {scores_path}")


if __name__ == "__main__":
    main()