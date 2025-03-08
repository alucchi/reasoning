#!/usr/bin/env python
# coding: utf-8

################################################################################
# This code evaluates a given model on a given dataset and then upload the
# results to the HuggingFace hub
# Usage:
# python3 evaluate_model_hub.py --batch_size 2 --use_training True --dataset_name MATH-500 --num_return_sequences 1 --model_name Qwen/Qwen2.5-Math-1.5B-Instruct
################################################################################


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
print('Number of GPUs', torch.cuda.device_count())

################################################################################
# Parameters

full_model_name = base_model_id.split("/")[-1] + ("_ft" if use_fine_tuned_model else "")

n_training = -1

max_tokens = 1024

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
revision = output_filename
hub_dataset_id = 'alucchi/' + base_model_id.split("/")[-1] + '_' + dataset_name.split("/")[-1] + '_s' + str(num_return_sequences)

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
from datasets import load_dataset, Dataset

if dataset_name == "gsm8k":
    dataset = load_dataset("gsm8k", "main")
    print('dataset size:', len(dataset["train"]), len(dataset["test"]))
    data_train = dataset["train"]
    data_test = dataset["test"]
    
    question = 'question'
    answer = 'answer'
    use_messages = False

    data_train = data_train.map(lambda x: {"answer": find_number(x["answer"])}) # only extract the number
    data_test = data_test.map(lambda x: {"answer": find_number(x["answer"])}) # only extract the number

elif dataset_name == "MATH-500" or dataset_name == "HuggingFaceH4/MATH-500":
    data_train = load_dataset("HuggingFaceH4/MATH-500", split="test")

    print('dataset size:', len(data_train))
    
    question = 'problem'
    answer = 'answer'
    use_messages = False
    data_test = None
    
elif dataset_name == "AI-MO/NuminaMath-CoT":
    dataset = load_dataset(dataset_name)
    print('dataset size:', len(dataset["train"]), len(dataset["test"]))
    data_train = dataset["train"]
    data_test = dataset["test"]
    
    question = 'problem'
    answer = 'solution'
    use_messages = True
elif dataset_name.split('/')[0] == 'alucchi':
    data_train = load_dataset(dataset_name, "main")["train"]
    data_test = None

    question = "Question"
    answer = "Ground Truth"
    use_messages = False
else:
    print("Unkown dataset")
    raise SystemExit(1)


    
if n_training > 0 and n_training < len(data_train):
    data_train = data_train.select(range(n_training))
if data_test is not None and n_training > 0 and n_training < len(data_test):
    data_test = data_test.select(range(n_training))
if data_test is not None:
    print('loaded dataset size:', len(data_train), len(data_test))
else:
    print('loaded dataset size:', len(data_train))
    
# Adjust batch size if necessary
batch_size = min(batch_size, len(data_train))
print('Updated batch size:', batch_size)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use `eos_token` as the padding token

model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = 'left'

if use_training:
    print("Evaluating on training set")
    data_test = data_train

################################################################################
# Helper functions

import re
import time
from pathlib import Path

from huggingface_hub import (
    create_branch,
    get_full_repo_name,
    list_repo_commits,
    repo_exists,
)

import regex

def extract_answer(response: str) -> str:
    # This pattern does the following:
    # - Optionally matches an opening math-mode delimiter: \(
    # - Matches one or two literal backslashes followed by "boxed"
    # - Matches a named group "braced" which recursively matches balanced braces:
    #       { ( any sequence of non-brace characters or a recursively nested "braced" group )* }
    # - Optionally matches a closing math-mode delimiter: \)
    pattern = (
        r'(?:\\\()?'
        r'\\{1,2}boxed'
        r'(?P<braced>\{(?:[^{}]+|(?P>braced))*\})'
        r'(?:\\\))?'
    )
    # Find all matches in the response
    matches = list(regex.finditer(pattern, response, regex.DOTALL))
    # Iterate over matches in reverse order, returning the last one with non-empty content.
    for m in reversed(matches):
        content = m.group("braced")
        # Remove the outermost braces
        if content.startswith("{") and content.endswith("}"):
            content = content[1:-1]
        if content.strip():
            return content.strip()
    return ""

def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
    """Finds the most relevant (or last) number in a string, in canonical form."""
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ''

def save_dataset(dataset, push_to_hub, hub_dataset_id, revision, output_dir):

    # Might need to authentificate first using by running the command: huggingface-cli login
    # Alternatively:
    # from huggingface_hub import login
    # login(token = 'hf_YOURKEY')
    
    if push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        url = None
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=hub_dataset_id,
                        branch=revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    hub_dataset_id,
                    revision=revision,
                    split="train",
                    private=False,
                    commit_message=f"Add {revision}",
                )
                break
            except Exception as e:
                print(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        print(f"Pushed dataset to {url}")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dataset.to_json(f"{output_dir}/bon_completions.jsonl", lines=True)
        print(f"Saved completions to {output_dir}/bon_completions.jsonl")

################################################################################
# Main loop

from collections import Counter

# This list will collect all evaluation records.
evaluation_records = []

for batch_start in range(0, len(data_test), batch_size):
    # If `data_test` is a Dataset object, use `select` to slice
    if hasattr(data_test, "select"):
        batch_data = data_test.select(range(batch_start, min(batch_start + batch_size, len(data_test))))
    else:
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
    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Group responses for each item in the batch
    grouped_responses = [
        responses[i*num_return_sequences:(i+1)*num_return_sequences]
        for i in range(len(batch_data))
    ]

    final_responses = []
    for group in grouped_responses:
        numeric_answers = []
        for r in group:
            # Clean response
            pattern = r"Cutting Knowledge Date: \w+ \d{4}\nToday Date: \d{2} \w+ \d{4}"
            r = re.sub(pattern, "", r).strip()
            ans = extract_answer(r)
            numeric_answers.append(ans)
        numeric_answer_counts = Counter(numeric_answers)
        most_common_numeric_answer, _ = numeric_answer_counts.most_common(1)[0]
        final_responses.append(most_common_numeric_answer)

    for idx, (task, response) in enumerate(zip(batch_data, final_responses)):
        try:
            gt_answer = task[answer]
            # Create a record for this task
            record = {
                "Task ID": batch_start + idx,
                "Question": task[question],
                "Responses": "\n".join(grouped_responses[idx]),
                "Extracted Answer": response,
                "Ground Truth": gt_answer
            }
            evaluation_records.append(record)
        except Exception as e:
            # In case of error, record available information
            record = {
                "Task ID": batch_start + idx,
                "Question": task.get(question, ""),
                "Responses": "\n".join(grouped_responses[idx]),
                "Extracted Answer": response,
                "Ground Truth": ""
            }
            evaluation_records.append(record)

# Convert the records to a Hugging Face Dataset
eval_dataset = Dataset.from_dict({
    "Task ID": [record["Task ID"] for record in evaluation_records],
    "Question": [record["Question"] for record in evaluation_records],
    "Responses": [record["Responses"] for record in evaluation_records],
    "Extracted Answer": [record["Extracted Answer"] for record in evaluation_records],
    "Ground Truth": [record["Ground Truth"] for record in evaluation_records]
})

# Save the dataset using the provided function.
# Here we set push_to_hub=False and provide an output directory.
save_dataset(eval_dataset, push_to_hub=True, hub_dataset_id=hub_dataset_id, revision=revision, output_dir="./output")

time.sleep(5)
ds = load_dataset(hub_dataset_id, revision=revision)
# Push to Hub as a config for exploration
ds.push_to_hub(hub_dataset_id, config_name=revision)

################################################################################
# GPU memory usage logging

torch.cuda.synchronize()
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(i)
    print(f"GPU {i}: Peak memory usage: {peak_memory / (1024**2)} MB")
