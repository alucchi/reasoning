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
from math_verify import parse, verify
import regex

################################################################################
# Helper functions used to define the reward function
################################################################################


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

################################################################################
# Hyperparameters & Setup
################################################################################


parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--batch_size", type=int, default=4, # batch_size 8 is too much for Fibonacci
                                    help="Batch size for training")
parser.add_argument("--optim_name", type=str, default="adam",
                                    help="Name of the optimizer (e.g. sgd, adam, mars)")
parser.add_argument("--n_training", type=int, default=300,
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
parser.add_argument("--num_iterations", type=int, default=1,
                                    help="Number of inner iterations for GRPO")
parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500",
                                    help="Name of the dataset, e.g. HuggingFaceH4/MATH-500 or gsm8k")


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
num_iterations = args.num_iterations
dataset_name = args.dataset_name
#dataset_name = "alucchi/Qwen2.5-Math-1.5B-Instruct_MATH-500_s16_hard"

if model_name == "llama":
    model_id = "meta-llama/Llama-3.2-1B-Instruct" # Lamma 3.2 model with 1B parameters
elif model_name == "qwen":
    model_id = "Qwen/Qwen2-0.5B-Instruct" # smaller model with 0.5B parameters
else:
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct" # smaller model with 0.5B parameters
max_tokens = 1024
use_chat_template = True # If true, use specific prompt to ask model to reason step by step (CoT approach)
num_trainable_layers = -1

global_new_epoch = False # global variable used to login debug info about rewards

print_info = False

use_cues = False # currently experimenting with this, trying to give more cues in prompt to see if that helps

model_shortname = model_id.split('/')[0]
output_dir = "./grpo_m" + model_shortname + '_d' + dataset_name + '_n' + str(n_training) + '_e' + str(num_train_epochs) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)
output_dir += 'mu_' + str(num_iterations) if num_iterations > 1 else ""
output_log =  output_dir + '.txt'

project_name = "GRPO_trl16_r4_" + model_shortname
run_name = "GRPO" +  "_m" + model_shortname + '_d' + dataset_name.split('/')[-1].split('.')[-1] + "_n" + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)
run_name += 'mu_' + str(num_iterations) if num_iterations > 1 else ""

print("batch_size", batch_size)
print("optim_name", optim_name)
print("n_training", n_training)
print("gradient_accumulation_steps", gradient_accumulation_steps)
print("beta", beta)
print("optim_lr", optim_lr)
print("model_name", model_name)
print("num_train_epochs", num_train_epochs)
print("num_iterations", num_iterations)
print("model_id", model_id)
print("dataset_name", dataset_name)
print("run_name", run_name)
print("output_dir", output_dir)
print("output_log", output_log)

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
# The data should be formatted as follows:
# - prompt of the problem
# - ground_truth contains a detailed solution
# - answer if the final answer (without a detailed solution)

print("Loading training dataset")
if dataset_name == "gsm8k":
    dataset = load_dataset(dataset_name, "main")
    data_train = dataset["train"].select(range(n_training))  # small subset for demonstration

    data_train = data_train.rename_column("question", "prompt")
    data_train = data_train.rename_column("answer", "ground_truth")

    #data_train = data_train.map(lambda x: {"ground_truth": find_number(x["ground_truth"])})

    # add a new column called answer that contains the final answer only
    data_train = data_train.map(lambda x: {**x, "answer": find_number(x["ground_truth"])})
    
elif dataset_name == "HuggingFaceH4/MATH-500":
    data_train = load_dataset(dataset_name, split="test").select(range(n_training)) # MATH-500 dataset
    data_train = data_train.rename_column("problem", "prompt")
    data_train = data_train.rename_column("solution", "ground_truth") # the column solution contains a detailed solution
    
elif dataset_name.split('/')[0] == 'alucchi':
    # THIS IS NOT UP-TO-DATE!!!
    data_train = load_dataset(dataset_name, "main")["train"]
    data_train = data_train.rename_column("Question", "prompt")
    data_train = data_train.rename_column("Ground Truth", "answer")
    data_train = data_train.rename_column("Responses", "ground_truth")
else:
    print("Unkown dataset")
    raise SystemExit(1)
    
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
        example_prompt = example["prompt"]
        if use_cues:
            example_prompt += "The final answer you need to find is " + example["ground_truth"] # experimental
        example["prompt"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example_prompt}
        ]
        return example
    
    data_train = data_train.map(add_formatted_prompt)
    data_train = data_train.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})


################################################################################
# Optimizer (not used at the moment)
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

def enclose_if_needed(s):
    # Ensure that s is a string and not None.                                                                                                                                                                              
    if s is None:
        return s
    if not s.startswith('$'):
        s = '$' + s
    if not s.endswith('$'):
        s = s + '$'
    return s

def parse_and_verify(a,b):
    return verify(
            parse(enclose_if_needed(a)),
            parse(enclose_if_needed(b))
    )

def reward_func(prompts, completions, ground_truth, answer, **kwargs):
    global global_new_epoch

    #ground_truth=[find_number(g) for g in ground_truth]
    #ground_truth=[extract_answer(g) for g in ground_truth]
    # Regular expression to capture content inside \boxed{}
    #matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    #contents = [match.group(1) if match else "" for match in matches]
    contents = [extract_answer(c) for c in completions]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    out_reward = [1.0 if parse_and_verify(c, gt) else 0.0 for c, gt in zip(contents, answer)]

    if global_new_epoch:
        global_new_epoch = False
                
        prompt_ids = [stoi[p] for p in prompts]

        if print_info:
            print('-'*50)
            print('len(completions)', len(completions))
            print('prompt ids', prompt_ids)
            print('prompts', prompts)
            print('completions', completions)
            print('completion numbers', [extract_answer(c) for c in completions])
            print('ground_truth', [g for g in ground_truth])
            print('contents', contents)
            print('out_reward', out_reward)

        with open(output_log, 'a') as f:
            f.write(f'prompt ids: {prompt_ids}\n')
            f.write(f'prompts: {prompts}\n')
            f.write(f'completions: {completions}\n')
            f.write(f'ground_truth: {[g for g in ground_truth]}\n')
            f.write(f'completion numbers: {[extract_answer(c) for c in completions]}\n')
            f.write(f'ground_truth numbers: {[a for a in answer]}\n')
            #f.write(f'matches: {matches}\n')
            f.write(f'out_reward: {out_reward}\n')

        
    return out_reward

# Another reward function, one can combine them by giving reward_funcs=[reward_func1, reward_func2] to GRPOTrainer
def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]

def reward_box_format(completions, **kwargs):
    rewards = []
    # Regular expression to check for \boxed{...} pattern.
    pattern = r'\\boxed\{.*?\}'
    
    for completion in completions:
        if re.search(pattern, completion, re.DOTALL):
            rewards.append(1)
        else:
            rewards.append(0)
    return rewards

def reward_steps(completions, **kwargs):
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]
    return [min(1.0, count / 3) for count in matches]

def reward_reasoning_steps(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # Tokenize the texts
        inputs = tokenizer([completion, gt], return_tensors="pt", padding=True).to(model.device)
        
        # Forward pass with hidden states enabled
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract the last hidden state from the hidden_states tuple
        last_hidden_state = outputs.hidden_states[-1]
        
        # Compute sentence embeddings by averaging last hidden states
        embeddings = last_hidden_state.mean(dim=1)
        
        # Calculate cosine similarity between the two embeddings
        cos_sim = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        rewards.append(cos_sim.item())
    return rewards


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
    #weight_decay = 1e-4,
    #warmup_ratio = 0.1 # CHANGE THIS!!!!!!!!!!!!!!!!!!
    #max_grad_norm=0.1,
    num_generations=batch_size,
    num_iterations=num_iterations,
    max_completion_length=max_tokens,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
    vllm_device="auto"
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_func, reward_box_format, reward_steps, reward_reasoning_steps],
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
