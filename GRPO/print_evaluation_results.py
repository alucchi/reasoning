#!/usr/bin/env python
# coding: utf-8

################################################################################
# This code compute the number of correct answers from a HuggingFace dataset
# It compares the columns "Extracted Answer" to "Ground Truth"
# Usage:
# python3 print_evaluations_results.py --hub_dataset_id YOUR_DATASET_ID
################################################################################


import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser(
        description="Load a dataset from the Hugging Face Hub and count how many times the 'Extracted Answer' equals the 'Ground Truth'."
    )
parser.add_argument(
        "--hub_dataset_id",
        type=str,
        required=True,
        help="The dataset ID on the Hugging Face Hub (e.g., 'your-username/your-dataset')."
    )
args = parser.parse_args()

# Load the dataset from the Hugging Face Hub; assuming it has a 'train' split.
dataset = load_dataset(args.hub_dataset_id, split="train")

# Initialize counter and get total number of records
match_count = 0
total_records = len(dataset)

# Iterate over the dataset and count matches between "Extracted Answer" and "Ground Truth"
for record in dataset:
    if record.get("Extracted Answer") == record.get("Ground Truth"):
        match_count += 1

print(f"Total records: {total_records}")
print(f"Correct: {match_count}")
print(f"Ratio: {match_count/total_records}") 
