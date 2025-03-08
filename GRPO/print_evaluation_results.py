#!/usr/bin/env python
# coding: utf-8

################################################################################
# This code computes the number of correct answers from a HuggingFace dataset.
# It compares the columns "Extracted Answer" to "Ground Truth" and can
# optionally limit evaluation to the first n_examples.
# Usage:
# python3 print_evaluations_results.py --hub_dataset_id YOUR_DATASET_ID [--n_examples N]
################################################################################

import argparse
from datasets import load_dataset
from math_verify import verify

parser = argparse.ArgumentParser(
    description="Load a dataset from the Hugging Face Hub and count how many times the 'Extracted Answer' equals the 'Ground Truth'."
)
parser.add_argument(
    "--hub_dataset_id",
    type=str,
    required=True,
    help="The dataset ID on the Hugging Face Hub (e.g., 'your-username/your-dataset')."
)
parser.add_argument(
    "--n_examples",
    type=int,
    default=None,
    help="Optional number of examples to evaluate (if not provided, all examples will be used)."
)
args = parser.parse_args()

# Load the dataset from the Hugging Face Hub; assuming it has a 'train' split.
dataset = load_dataset(args.hub_dataset_id, split="train")

# If n_examples is provided, only consider the first n_examples.
if args.n_examples is not None:
    dataset = dataset.select(range(min(args.n_examples, len(dataset))))

# Initialize counter and get total number of records
match_count = 0
total_records = len(dataset)

# Iterate over the dataset and count matches between "Extracted Answer" and "Ground Truth"
for record in dataset:
    if verify(record.get("Extracted Answer"), record.get("Ground Truth")):
        match_count += 1

print(f"Total records evaluated: {total_records}")
print(f"Correct: {match_count}")
print(f"Ratio: {match_count / total_records}")
