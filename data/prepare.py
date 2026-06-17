#!/usr/bin/env python3
"""
EUR-Lex-sum dataset preparation for fine-tuning comparison.
Downloads, preprocesses, and saves to disk for reproducibility.
"""

import os
from datasets import load_dataset

# Create data directory if needed
os.makedirs('data', exist_ok=True)

print("Downloading BillSum (US Congressional bills summarization)...")
ds = load_dataset('FiscalNote/billsum', split='train')
print(f"Full dataset size: {len(ds)} samples")

# Select 2000 samples for comfortable RAM/VRAM fit
print("Selecting 2000 samples for training...")
ds = ds.select(range(min(2000, len(ds))))

def format_prompt(example):
    """Format examples as: Document -> Summary prompt-response pairs."""
    return {
        'text': f"### Document:\n{example['text'][:1024]}\n### Summary:\n{example['summary']}"
    }

print("Formatting prompts...")
ds = ds.map(format_prompt, remove_columns=[col for col in ds.column_names if col != 'text'])

# Train/test split with fixed seed for reproducibility
print("Creating train/test split (90/10)...")
ds_split = ds.train_test_split(test_size=0.1, seed=42)

print(f"Train samples: {len(ds_split['train'])}")
print(f"Test samples: {len(ds_split['test'])}")

# Save to disk — never re-download during experiments
print("Saving to disk at data/billsum_processed...")
ds_split.save_to_disk('data/billsum_processed')

print("✓ Data pipeline complete!")
print(f"Dataset saved to: {os.path.abspath('data/billsum_processed')}")