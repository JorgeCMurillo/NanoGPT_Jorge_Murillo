# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# token-budgeted batch sizing (~0.5M tokens/update target, llm.c-style)
batch_size = 12
block_size = 1024
total_batch_tokens = 524288

# this makes total number of iterations be 20k
max_iters = 20000
lr_decay_iters = 20000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
