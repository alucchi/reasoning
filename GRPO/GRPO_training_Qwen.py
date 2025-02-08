# This is a modified version of https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing#scrollTo=U1ixGbPG0Ni
# which is itself a modified version of https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

import argparse
import math
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


################################################################################
# Hyperparameters & Setup
################################################################################


parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--batch_size", type=int, default=1, # batch_size 8 is too much for Fibonacci
                                    help="Batch size for training")
parser.add_argument("--optim_name", type=str, default="adam",
                                    help="Name of the optimizer (e.g. sgd, adam, mars): feature not implemented yet")
parser.add_argument("--n_training", type=int, default=10,
                                    help="Number of training examples to use")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
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
#print("optim_name", optim_name)
print("n_training", n_training)
print("gradient_accumulation_steps", gradient_accumulation_steps)
print("beta", beta)

num_train_epochs = 5
optim_lr = 5e-4 # 5e-6
dataset_name = "gsm8k" # around 8K training datapoints
#dataset_name = "HuggingFaceH4/MATH-500" # 500 datapoints
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_shortname = model_name.split('/')[0]
#dataset_name = "meta-llama/Llama-3.2-1B-Instruct" # Lamma 3.2 model with 1B parameters
max_tokens = 256
#use_chat_template = True # If true, use specific prompt to ask model to reason step by step (CoT approach)

#global_new_epoch = False # global variable used to login debug info about rewards

output_dir = "./grpo_" + model_shortname + "_d" + dataset_name + '_n' + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)
output_log = "./grpo_" + model_shortname + "_d" + dataset_name + '_n' + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta) + '.txt'

project_name = "GRPO_training"
run_name = project_name + '_' + model_shortname + "_d" + dataset_name.split('/')[-1].split('.')[-1] + "_n" + str(n_training) + '_o' + optim_name + str(optim_lr) + "_b" + str(batch_size) + '_' + str(gradient_accumulation_steps) + '_a' + str(beta)

# Initialize wandb
import wandb
wandb.init(project=project_name, # the project I am working on
           job_type="train",
           tags=["hf_sft_lora", "llama"],
           name=run_name) # the Hyperparameters I want to keep track of

################################################################################
# Dataset Preparation and Helper Functions
################################################################################

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extracts the answer from text formatted in XML."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer if marked with hash symbols."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    """Loads the gsm8k dataset and restructures it into a conversational format."""
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore

# Use a subset for quick testing (select first 10 examples)
dataset = get_gsm8k_questions()
dataset = dataset.select(range(n_training))


################################################################################
# Reward Functions
################################################################################

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20,
          f"Question:\n{q}",
          f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks if the completion has the strict expected format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks if the completion has a loosely expected format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

################################################################################
# Training Configuration
################################################################################


# Calculate number of steps                                                                                                                                                                                                
steps_per_epoch = math.ceil(len(dataset) / (batch_size * gradient_accumulation_steps))
max_steps = num_train_epochs * steps_per_epoch
logging_steps = math.ceil(steps_per_epoch * 0.25)
print('max_steps', max_steps)

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=optim_lr,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=logging_steps,
    bf16=True,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_generations=16,
    max_prompt_length=max_tokens,
    max_completion_length=max_tokens,
    num_train_epochs=num_train_epochs,
    save_steps=steps_per_epoch, # save every epoch
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
    vllm_device="auto",
    #report_to="none"  # Disabling Wandb.
    beta=beta
)

################################################################################
# Model Loading (Using Multi-GPU via device_map="auto")
################################################################################

# Load the model with automatic device mapping to utilize all available GPUs.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # This will shard the model across all available GPUs.
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

################################################################################
# Trainer Initialization and Training
################################################################################

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
)

from transformers.integrations import WandbCallback

class LLMSampleCB(WandbCallback):
    def __init__(self):
        super().__init__()
          
    def on_train_begin(self, args, state, control, **kwargs):
        # You can add any specific logic you need here or leave it as pass
        pass


    def on_epoch_end(self, args, state, control, **kwargs):
        #global global_new_epoch
        #global_new_epoch = True
        pass

# Add our custom loss callback
wandb_callback = LLMSampleCB()
trainer.add_callback(wandb_callback)


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
