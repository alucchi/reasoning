#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import os
import re
import torch

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


################################################################################
# Helper functions used to define the reward function
################################################################################

def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string, returning them in a canonical form."""
    pattern = re.compile(
        r'-?[\d,]*\.?\d+',  # same as original
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    raw_numbers = pattern.findall(x)

    cleaned_numbers = []
    for num_str in raw_numbers:
        # Remove commas (e.g. "1,234.5" -> "1234.5")
        no_comma = num_str.replace(',', '')

        # Convert to float
        val = float(no_comma)

        # Convert float back to string.
        # This example converts floats like 10.0 to "10" if it is an integer value.
        if val.is_integer():
            cleaned_numbers.append(str(int(val)))
        else:
            cleaned_numbers.append(str(val))

    return cleaned_numbers


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


def maybe_remove_comma(x: str) -> str:
    """Example: 5,600 -> 5600."""
    return x.replace(',', '')


################################################################################
# Hyperparameters & Setup
################################################################################


parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--batch_size", type=int, default=4, # batch_size 8 is too much for Fibonacci
                                    help="Batch size for training")
parser.add_argument("--optim_name", type=str, default="sgd",
                                    help="Name of the optimizer (e.g. sgd, adam, mars)")
parser.add_argument("--n_training", type=int, default=50,
                                    help="Number of training examples to use")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Number of gradient accumulation steps")
parser.add_argument("--beta", type=float, default=0.04,
                                    help="Beta coefficient weights the KL term in GRPO objective")

args = parser.parse_args()

# Update parameters from arguments                                                                                                                                                                                         
batch_size = args.batch_size
optim_name = args.optim_name
n_training = args.n_training
gradient_accumulation_steps = args.gradient_accumulation_steps
beta = args.beta

print("batch_size", batch_size)
print("optim_name", optim_name)
print("n_training", n_training)
print("gradient_accumulation_steps", gradient_accumulation_steps)
print("beta", beta)

num_train_epochs = 100
optim_lr = 5e-4
dataset_name = "gsm8k" # around 8K training datapoints
#dataset_name = "HuggingFaceH4/MATH-500" # 500 datapoints
model_id = "meta-llama/Llama-3.2-1B-Instruct" # Lamma 3.2 model with 1B parameters
#model_id = "Qwen/Qwen2-0.5B-Instruct" # smaller model with 0.5B parameters
max_tokens = 1024
use_chat_template = True # If true, use specific prompt to ask model to reason step by step (CoT approach)

global_new_epoch = False # global variable used to login debug info about rewards

output_dir = "./grpo_Llama-3.2-1B-Instruct_d" + dataset_name + '_n' + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)
output_log = "./grpo_Llama-3.2-1B-Instruct_d" + dataset_name + '_n' + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta) + '.txt'


# (Optionally) adjust or remove if you see conflicts with Accelerate
model_kwargs = dict(
    device_map="auto",     # <--- Keep if you want model sharded by HF
    load_in_8bit=False,
    trust_remote_code=True,
    use_cache=False
)

# LoRA config
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    #target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
    target_modules="all-linear"
)

project_name = "GRPO_training"
run_name = project_name + '_' + dataset_name.split('/')[-1].split('.')[-1] + "_n" + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)

# Initialize wandb
import wandb
wandb.init(project=project_name, # the project I am working on
           job_type="train",
           tags=["hf_sft_lora", "llama"],
           name=run_name) # the Hyperparameters I want to keep track of


################################################################################
# Load Model & Tokenizer
################################################################################

print("Loading base model")
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Wrap model with LoRA
model = get_peft_model(model, peft_config)

################################################################################
# Load Dataset
################################################################################


print("Loading training dataset")
if dataset_name == "gsm8k":
    dataset = load_dataset(dataset_name, "main")
    data_train = dataset["train"].select(range(n_training))  # small subset for demonstration

    data_train = data_train.rename_column("question", "prompt")
    data_train = data_train.rename_column("answer", "ground_truth")
else:
    data_train = load_dataset(dataset_name, split="test").select(range(n_training)) # MATH-500 dataset
    data_train = data_train.rename_column("problem", "prompt")
    data_train = data_train.rename_column("solution", "ground_truth")
    
# Calculate number of steps                                                                                                                                                                                                
steps_per_epoch = math.ceil(len(data_train) / (batch_size * gradient_accumulation_steps))
max_steps = num_train_epochs * steps_per_epoch
logging_steps = math.ceil(steps_per_epoch * 0.25)

print('max_steps', max_steps)


if use_chat_template:

    from trl import apply_chat_template

    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    def add_formatted_prompt(example):
        example["prompt"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]}
        ]
        return example

    data_train = data_train.map(add_formatted_prompt)
    data_train = data_train.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})


################################################################################                                                                                                                                                     
# Optimizer
################################################################################   
if optim_name == "sgd":
    print("Using SGD optimizer\n")
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=optim_lr, momentum=0.9)
elif optim_name == "adam":
    print("Using Adam optimizer\n")
    from torch.optim import Adam as PyTorchAdam
    optimizer = PyTorchAdam(
        model.parameters(),
	lr=optim_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
	weight_decay=1e-4
    )
else:
    print("Using MARS optimizer\n")
    from mars import MARS
    optimizer = MARS(model.parameters(), lr=optim_lr, weight_decay=1e-4, optimize_1d=False)

    

################################################################################
# Reward Function
################################################################################

# Create hashmap to identify prompts
# Convert list to a set to get unique strings (order not preserved)
unique_prompts = set(data_train['prompt'])

# Create a hash map: string -> unique id
stoi = {prompt: idx for idx, prompt in enumerate(unique_prompts)}


import re

def reward_func(prompts, completions, ground_truth, **kwargs):
    global global_new_epoch
        
    ground_truth=[find_number(g) for g in ground_truth]
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    out_reward = [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

    if global_new_epoch:
        global_new_epoch = False
                
        prompt_ids = [stoi[p] for p in prompts]
        
        print('-'*50)
        print('len(completions)', len(completions))
        print('prompt ids', prompt_ids)
        print('prompts', prompts)
        print('completions', completions)
        print('completion numbers', [find_number(c) for c in completions])
        print('ground_truth', [find_number(g) for g in ground_truth])
        print('matches', matches)
        print('out_reward', out_reward)

        with open(output_log, 'a') as f:
            f.write(f'prompt ids: {prompt_ids}\n')
            f.write(f'prompts: {prompts}\n')
            f.write(f'completions: {completions}\n')
            f.write(f'completions: {[find_number(c) for c in completions]}\n')
            f.write(f'ground_truth: {[find_number(g) for g in ground_truth]}\n')
            #f.write(f'matches: {matches}\n')
            f.write(f'out_reward: {out_reward}\n')

        
    return out_reward

# Another reward function, one can combine them by giving reward_funcs=[reward_func1, reward_func2] to GRPOTrainer
def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]

################################################################################
# GRPO Training Setup
################################################################################

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import math
import matplotlib.pyplot as plt

# Compute number of steps per epoch (approximate)
steps_per_epoch = math.ceil(len(data_train) / batch_size)
quarter_epoch_steps = max(1, steps_per_epoch // 4)
print(f"Steps per epoch: {steps_per_epoch}, Quarter epoch steps: {quarter_epoch_steps}")


from transformers.integrations import WandbCallback

class LLMSampleCB(WandbCallback):
    def __init__(self):
        super().__init__()
          
    def on_train_begin(self, args, state, control, **kwargs):
        # You can add any specific logic you need here or leave it as pass
        pass


    def on_epoch_end(self, args, state, control, **kwargs):
        global global_new_epoch
        global_new_epoch = True
        
# Training arguments
training_args = GRPOConfig(
    output_dir=output_dir,
    max_steps=max_steps,
    logging_steps=logging_steps,  # Log every step so our callback can decide when to record
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,            # or fp16 if your GPU doesn't support bf16
    learning_rate=optim_lr,
    beta=beta
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=data_train,
    optimizers=(optimizer, None)
)

# Add our custom loss callback
wandb_callback = LLMSampleCB()
trainer.add_callback(wandb_callback)

################################################################################
# Train
################################################################################

print('Training...')
trainer.train()

################################################################################
# Save model, check GPU consumption
################################################################################

wandb.finish()

# Save final model
trainer.save_model(output_dir)
print("Training complete. Model saved to", output_dir)


# Synchronize to ensure all operations are done
torch.cuda.synchronize()

# Loop through all available GPUs
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(i)
    print(f"GPU {i}: Peak memory usage: {peak_memory / (1024**2)} MB")
