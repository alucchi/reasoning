#!/usr/bin/env python
# coding: utf-8

################################################################################
# This code runs the GRPO training on a given model and dataset
# Usage:
# python3 GRPO_training.py --batch_size 2 --n_training 100 --beta 0.001 --gradient_accumulation_steps 8 --optim_name adam --optim_lr 1e-5 --num_train_epochs 20 Qwen/Qwen2.5-Math-1.5B-Instruct --num_iterations 1 
################################################################################


import argparse
import math
import os
import re
import torch

from datasets import load_dataset
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
parser.add_argument("--optim_name", type=str, default="adam",
                                    help="Name of the optimizer (e.g. sgd, adam, mars)")
parser.add_argument("--n_training", type=int, default=100,
                                    help="Number of training examples to use")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Number of gradient accumulation steps")
parser.add_argument("--beta", type=float, default=0.04,
                                    help="Beta coefficient weights the KL term in GRPO objective")
parser.add_argument("--optim_lr", type=float, default=5e-6,
                                    help="Learning rate")
parser.add_argument("--model_name", type=str, default="llama",
                                    help="llama or qwen")
parser.add_argument("--num_train_epochs", type=int, default=100,
                                    help="Number of training epochs")


args = parser.parse_args()

# Update parameters from arguments
batch_size = args.batch_size
optim_name = args.optim_name
n_training = args.n_training
gradient_accumulation_steps = args.gradient_accumulation_steps
beta = args.beta
optim_lr = args.optim_lr
model_name = args.model_name
num_train_epochs = args.num_train_epochs

print("batch_size", batch_size)
print("optim_name", optim_name)
print("n_training", n_training)
print("gradient_accumulation_steps", gradient_accumulation_steps)
print("beta", beta)
print("optim_lr", optim_lr)
print("model_name", model_name)
print("num_train_epochs", num_train_epochs)

dataset_name = "gsm8k" # around 8K training datapoints
#dataset_name = "HuggingFaceH4/MATH-500" # 500 datapoints
if model_name == "llama":
    model_id = "meta-llama/Llama-3.2-1B-Instruct" # Lamma 3.2 model with 1B parameters
else:
    model_id = "Qwen/Qwen2-0.5B-Instruct" # smaller model with 0.5B parameters
max_tokens = 1024
use_chat_template = True # If true, use specific prompt to ask model to reason step by step (CoT approach)
num_trainable_layers = -1

global_new_epoch = False # global variable used to login debug info about rewards

print_info = False

model_shortname = model_id.split('/')[0]
output_dir = "./grpo_m" + model_shortname + '_d' + dataset_name + '_n' + str(n_training) + '_e' + str(num_train_epochs) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)
output_log =  output_dir + '.txt'

project_name = "GRPO_training2"
run_name = project_name +  "_m" + model_shortname + '_d' + dataset_name.split('/')[-1].split('.')[-1] + "_n" + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)

# Initialize wandb
import wandb
wandb.init(project=project_name, # the project I am working on
           job_type="train",
           tags=["GRPO"],
           name=run_name) # the Hyperparameters I want to keep track of


################################################################################
# Load Model & Tokenizer
################################################################################

print("Loading base model")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
model.generation_config.pad_token_id = tokenizer.pad_token_id

################################################################################
# Load Dataset
################################################################################


print("Loading training dataset")
if dataset_name == "gsm8k":
    dataset = load_dataset(dataset_name, "main")
    data_train = dataset["train"].select(range(n_training))  # small subset for demonstration

    data_train = data_train.rename_column("question", "prompt")
    data_train = data_train.rename_column("answer", "ground_truth")

    data_train = data_train.map(lambda x: {"ground_truth": find_number(x["ground_truth"])})
else:
    data_train = load_dataset(dataset_name, split="test").select(range(n_training)) # MATH-500 dataset
    data_train = data_train.rename_column("problem", "prompt")
    data_train = data_train.rename_column("solution", "ground_truth")
    
    data_train = data_train.map(lambda x: {"ground_truth": find_number(x["ground_truth"])})

# Calculate number of steps                                                                                                                                                                                                
steps_per_epoch = math.ceil(len(data_train) / (batch_size * gradient_accumulation_steps))
max_steps = num_train_epochs * steps_per_epoch
logging_steps = math.ceil(steps_per_epoch * 0.25)
logging_steps = max(logging_steps, 5)

print('max_steps', max_steps)
print('logging_steps', logging_steps)

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
# Optimizer (not used at the moment, but one can change the optimizer as well)
################################################################################   
if optim_name == "sgd":
    print("Using SGD optimizer\n")
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=optim_lr, momentum=0.9)
elif optim_name == "adam":
    print("Using Adam optimizer\n")
    from torch.optim import AdamW as PyTorchAdam
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
        
    # Regular expression to capture content inside \boxed{}
    contents = [find_number(c) for c in completions]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    out_reward = [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

    if global_new_epoch:
        global_new_epoch = False
                
        prompt_ids = [stoi[p] for p in prompts]

        if print_info:
            print('-'*50)
            print('len(completions)', len(completions))
            print('prompt ids', prompt_ids)
            print('prompts', prompts)
            print('completions', completions)
            print('completion numbers', [find_number(c) for c in completions])
            print('ground_truth', [find_number(g) for g in ground_truth])
            print('contents', contents)
            print('out_reward', out_reward)

        with open(output_log, 'a') as f:
            f.write(f'prompt ids: {prompt_ids}\n')
            f.write(f'prompts: {prompts}\n')
            f.write(f'completions: {completions}\n')
            f.write(f'completions: {[find_number(c) for c in completions]}\n')
            f.write(f'ground_truth: {[find_number(g) for g in ground_truth]}\n')
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


# Modify the number of layers to keep trainable
if num_trainable_layers != -1:

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Identify transformer block layers in the model
    if hasattr(model, "model"):  # Some architectures wrap the transformer inside "model"
        transformer_layers = model.model.layers
    elif hasattr(model, "transformer"):  # Another common pattern
        transformer_layers = model.transformer.h
    else:
        raise ValueError("Could not find transformer layers in model. Check model architecture.")

    # Unfreeze the last few layers
    for layer in transformer_layers[-num_trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    print(f"Only the last {num_trainable_layers} layers are trainable.")


        
# Training arguments
training_args = GRPOConfig(
    output_dir=output_dir,
    max_steps=max_steps,
    logging_steps=logging_steps,  # Log every step so our callback can decide when to record
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,            # or fp16 if your GPU doesn't support bf16
    learning_rate=optim_lr,
    beta=beta,
    num_generations=8,
    max_completion_length=512,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
    vllm_device="auto"
    # Other parameters worth considering...
    #weight_decay = 1e-4,
    #warmup_ratio = 0.1,
    #max_grad_norm=0.1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=data_train,
    # Options to set the optimizer
    #optimizers=(optimizer, None)
    #optimizers=(optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=optim_lr), None)
    #optimizers=(torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=optim_lr), None)
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
