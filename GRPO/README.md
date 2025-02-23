## GRPO training

- To train on a small subset of the gsm8k dataset (100 points):
python3 GRPO_training.py --batch_size 2 --n_training 100 --optim_name adam --gradient_accumulation_step 4 --beta 0.005

## Evaluate models
- To evaluate a model stored in the directory GRPO_training_mqwen_dgsm8k_n100_oadam5e-06_b2_8_a0.01:
python3 evaluate_model_GRPO.py --model_name GRPO_training_mqwen_dgsm8k_n100_oadam5e-06_b2_8_a0.01 --batch_size 100 --use_training True --dataset_name gsm8k --num_return_sequences 1

