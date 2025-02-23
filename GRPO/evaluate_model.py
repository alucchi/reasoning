#!/usr/bin/env python
# coding: utf-8


import torch

not_random = False

if not_random == True:
    import numpy as np
    import random
    from transformers import set_seed

    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
# Parsed parameters

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate evaluation filename.")
parser.add_argument(
    "--dataset_name", 
    type=str, 
    default="AI-MO/NuminaMath-CoT", 
    help="Name of the dataset to use (e.g., 'AI-MO/NuminaMath-CoT' or 'gsm8k')."
)
parser.add_argument(
    "--use_fine_tuned_model",
    type=lambda x: x.lower() in ['true', '1', 'yes'],
    default=False,
    help="Whether to use the fine-tuned model. Accepts 'True', 'False', '1', '0', 'yes', 'no'. Default is True."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for processing. Default is 16."
)
parser.add_argument(
    "--use_training",
    type=lambda x: x.lower() in ['true', '1', 'yes'],
    default=False,
    help="Whether to evaluate on training set. Accepts 'True', 'False', '1', '0', 'yes', 'no'. Default is False."
)
parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
parser.add_argument(
    "--ft_model_name", 
    type=str, 
    default="./llama_ft_NuminaMath-CoT", 
    help="Directory where fine tuned model is, used if --use_fine_tuned_model=True only"
)
parser.add_argument(
    "--model_name", 
    type=str, 
    default="meta-llama/Llama-3.2-1B-Instruct",
    #default="Qwen/Qwen2.5-0.5B-Instruct",
    help="Name of the model"
)


args = parser.parse_args()

# Assign the parsed arguments
dataset_name = args.dataset_name
use_fine_tuned_model = args.use_fine_tuned_model
use_training = args.use_training
batch_size = args.batch_size
ft_model_name = args.ft_model_name
num_return_sequences = args.num_return_sequences

#base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
#base_model_id = "qwen/Qwen2-Math-72B-Instruct"
#base_model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
base_model_id = args.model_name

print("dataset_name", dataset_name)
print("use_fine_tuned_model", use_fine_tuned_model)
print("use_training", use_training)
print("batch_size:", batch_size)
print("ft_model_name", ft_model_name)
print("num_return_sequences:", num_return_sequences)
print('Number of GPUs',torch.cuda.device_count())


################################################################################
# Parameters


full_model_name = base_model_id.split("/")[-1] + ("_ft" if use_fine_tuned_model else "")

n_training = -1
n_training = 100

max_tokens = 512
#max_tokens = 2048

if not_random == True:
    do_sample = False
    temperature = 0
else:
    do_sample = True
    temperature = 1.0

output_filename = "eval_" + full_model_name + "_d" + dataset_name.split('/')[-1]
output_filename += "_batch" + str(batch_size) # debug, remove later
output_filename += "_evaltraining" if use_training else ""
output_filename += "_notrandom" if not_random else ""
output_filename += ("_nseq" + str(num_return_sequences)) if num_return_sequences > 1 else ""
print("output_filename", output_filename)

debug_mode = False # print more information

#system_prompt = "You are a helpful assistant."
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."


################################################################################

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Reload the base model
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", load_in_8bit=True)

if use_fine_tuned_model:

    print('Loading fine tuned model', ft_model_name)
    
    # Reload the LoRA-adapted model if applicable
    model = PeftModel.from_pretrained(model, ft_model_name)

    # Reload the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ft_model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir="./kaggle/working/")


# import dataset
from datasets import load_dataset
if dataset_name == "gsm8k":
    dataset = load_dataset("gsm8k", "main", cache_dir='/tmp')
    print('dataset size:', len(dataset["train"]), len(dataset["test"]))
    data_train = dataset["train"]
    data_test = dataset["test"]
    
    question = 'question'
    answer = 'answer'
    use_messages = False

else:
    # numina                                                                                                       
    dataset = load_dataset(dataset_name)
    print('dataset size:', len(dataset["train"]), len(dataset["test"]))
    data_train = dataset["train"]
    data_test = dataset["test"]
    
    question = 'problem'
    answer = 'solution'

    use_messages = True

if n_training > 0 and n_training < len(data_train):
    data_train = data_train.select(range(n_training))
if n_training > 0 and n_training < len(data_test):
    data_test = data_test.select(range(n_training))
print('loaded dataset size:', len(data_train), len(data_test))
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use `eos_token` as the padding token

model.generation_config.pad_token_id = tokenizer.pad_token_id

# Set the padding side to 'left'
tokenizer.padding_side = 'left'


if use_training:
    print("Evaluating on training set")
    data_test = data_train


################################################################################
# Testing library

import re

def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    # Search for number, possibly negative (hyphen), with thousand separators
    # (comma), and with a decimal point (period inbetween digits).
    numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers


def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
  """Finds the most relevant number in a string."""
  # If model uses the answer delimiter, then select the first number following
  # that format.
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  # In general, select the last number in the string.
  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ''


def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(',', '')



################################################################################
# Main loop comes next...

from collections import Counter

correct_count = 0
incorrect_count = 0
all_responses = {}

PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""
PROMPT = ""

with open(output_filename + "_correct_answers.txt", "w") as correct_file, open(output_filename + "_incorrect_answers.txt", "w") as incorrect_file:
    for batch_start in range(0, len(data_test), batch_size):
        #batch_data = data_test[batch_start:batch_start + batch_size]


        # If `data_test` is a Dataset object, use `select` to slice
        if hasattr(data_test, "select"):
            batch_data = data_test.select(range(batch_start, min(batch_start + batch_size, len(data_test))))
        else:
            # Otherwise, assume it's a list-like object
            batch_data = data_test[batch_start:batch_start + batch_size]

        batch_prompts = []

        for task in batch_data:
            if use_messages:
                messages = task['messages']
            else:
                prompt = task[question]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(text)

        # Tokenize and generate responses in batch
        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

        if debug_mode:
            print("Model inputs shape:", model_inputs["input_ids"].shape)

            # Loop over model inputs and print each sequence
            for i, (input_ids, attention_mask) in enumerate(zip(model_inputs["input_ids"], model_inputs["attention_mask"])):
                print(f"Sequence {i}:")
                print(f"Input IDs: {input_ids}")
                print(f"Attention Mask: {attention_mask}")
        
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_tokens,
            num_return_sequences=num_return_sequences
        )
        
        #generated_ids = [
        #    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        #]

        # `generated_ids` will have shape [batch_size * num_return_sequences, seq_len]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Group responses for each item in the batch
        grouped_responses = [
            responses[i*num_return_sequences:(i+1)*num_return_sequences]
            for i in range(len(batch_data))
        ]

        final_responses = []
        for group in grouped_responses:
            # Extract numeric answers first
            numeric_answers = []
            for r in group:

                # Clean response
                pattern = r"Cutting Knowledge Date: \w+ \d{4}\nToday Date: \d{2} \w+ \d{4}"
                r = re.sub(pattern, "", r).strip()

                # Extract answer from response before counting
                ans = maybe_remove_comma(find_number(r))
                numeric_answers.append(ans)

            # Count the occurrence of each numeric answer
            numeric_answer_counts = Counter(numeric_answers)

            # Find the most common numeric answer
            most_common_numeric_answer, _ = numeric_answer_counts.most_common(1)[0]

            final_responses.append(most_common_numeric_answer)


        for idx, (task, response) in enumerate(zip(batch_data, final_responses)):
            

            try:
                gt_answer = maybe_remove_comma(find_number(task[answer]))
                is_correct = float(response) == float(gt_answer)
                if is_correct:
                    correct_count += 1
                    correct_file.write(f"Task ID: {batch_start + idx}\n")
                    correct_file.write(f"Question: {task[question]}\n")
                    correct_file.write(f"Responses: {responses}\n")
                    correct_file.write(f"Extracted Answer: {response}\n")
                    correct_file.write(f"Ground Truth: {gt_answer}\n\n")
                    correct_file.write("=" * 40 + "\n")
                else:
                    incorrect_count += 1
                    incorrect_file.write(f"Task ID: {batch_start + idx}\n")
                    incorrect_file.write(f"Question: {task[question]}\n")
                    incorrect_file.write(f"Responses: {responses}\n")
                    incorrect_file.write(f"Extracted Answer: {response}\n")
                    incorrect_file.write(f"Ground Truth: {gt_answer}\n\n")
                    incorrect_file.write("=" * 40 + "\n")
            except:
                incorrect_count += 1
                incorrect_file.write(f"Task ID: {batch_start + idx}\n")
                incorrect_file.write(f"Responses: {responses}\n")
                incorrect_file.write(f"Extracted Answer: {response}\n")
                incorrect_file.write(f"Ground Truth: {gt_answer}\n\n")
                incorrect_file.write("=" * 40 + "\n")
                
    # Append totals to the end of the files
    correct_file.write(f"Total Correct Answers: {correct_count}\n")
    incorrect_file.write(f"Total Incorrect Answers: {incorrect_count}\n")

# Synchronize to ensure all operations are done
torch.cuda.synchronize()

# Loop through all available GPUs
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(i)
    print(f"GPU {i}: Peak memory usage: {peak_memory / (1024**2)} MB")
