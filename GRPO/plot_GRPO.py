#!/usr/bin/env python
# coding: utf-8

import argparse
import ast
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process a file for histogram generation.")
parser.add_argument("filename", help="The name of the file to process")
args = parser.parse_args()

# Initialize the results dictionary.
# It will have the form:
#   { prompt_id: {"completions": [list of lists of completions], "rewards": [list of lists of rewards]}, ... }
results = {}

filename = args.filename

# Compute PDF filename: remove the input file extension and add .pdf
base, _ = os.path.splitext(args.filename)
pdf_filename = base + "_histogram.pdf"

# Continue with the rest of your code...
print("Input filename:", filename)
print("PDF filename:", pdf_filename)

# Read the file
with open(filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]  # remove empty lines and whitespace

chunk_size = 6
# Process the file in chunks of chunk_size lines (each iteration)
for i in range(0, len(lines), chunk_size):
    # The order in each iteration is:
    # 0: prompt ids, 3: completions, 4: ground_truth (unused here), 5: out_reward

    # Parse the lines.
    prompt_ids = ast.literal_eval(lines[i].split(":", 1)[1].strip())
    completions = ast.literal_eval(lines[i+3].split(":", 1)[1].strip())
    ground_truths = ast.literal_eval(lines[i+4].split(":", 1)[1].strip())

    # Iterate over each unique prompt id.
    for pid in set(prompt_ids):
        # Find all the indices where this prompt id appears.
        indices = [j for j, p in enumerate(prompt_ids) if p == pid]

        # Gather completions and ground truths corresponding to these indices.
        comps = [completions[j] for j in indices]
        gts = [ground_truths[j] for j in indices]

        # Initialize the prompt id in results if it does not exist yet.
        if pid not in results:
            results[pid] = {"completions": [], "ground_truths": []}

        # Append the lists of completions and ground truths as one element.
        results[pid]["completions"].append(comps)
        results[pid]["ground_truths"].append(gts)
    
# Print the resulting hashmap
print(results)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


with PdfPages(pdf_filename) as pdf:
    # Loop over each prompt id in results.
    for r in results:
        completions = results[r]['completions']
        ground_truths = results[r]['ground_truths']

        # Create a new figure for this prompt id with a 10x10 grid of subplots.
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        axes = axes.flatten()  # Flatten the 2D grid to 1D for easier indexing.

        index = 0  # Counter for the current subplot.
        # Loop over each iteration for this prompt id.
        for k in range(len(completions)):
            if index >= len(axes):
                # If there are more than 100 iterations, stop plotting additional ones.
                break

            # Convert values to float, replacing empty strings with -1.
            ck = [float(x) if x != '' else -1 for x in completions[k]]
            gk = [float(x) if x != '' else -1 for x in ground_truths[k]]

            ax = axes[index]

            # Plot the histogram for completions.
            ax.hist(ck, bins=10, alpha=0.5, label='completions')

            # Check if all values in gk are the same.
            if len(set(gk)) == 1:
                # All ground_truth values are the same.
                value = gk[0]
                # Draw a red dotted vertical line at that value.
                ax.axvline(x=value, color='red', linestyle='dotted', label='ground_truth')
                # Optionally, add a star marker at the top of the line.
                # Get the current y-axis maximum.
                y_max = ax.get_ylim()[1]
                ax.plot(value, y_max, 'r*', markersize=10)
            else:
                # Plot the histogram for ground_truths when values vary.
                ax.hist(gk, bins=10, alpha=0.5, label='ground_truths')

            ax.set_title(f'ID: {r} Iter: {k}', fontsize=8)
            ax.set_xlabel('Value', fontsize=7)
            ax.set_ylabel('Frequency', fontsize=7)
            ax.legend(loc='upper right', fontsize=6)

            index += 1

        # Hide any unused subplots.
        for j in range(index, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        # Save the current figure as a new page in the PDF.
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved histograms to {pdf_filename}")
