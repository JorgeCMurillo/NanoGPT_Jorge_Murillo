"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # legacy/manual fallback when total_batch_tokens <= 0
total_batch_tokens = 524288 # desired global tokens per optimizer step (llm.c-style token budget)
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 20000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 700 # how many steps to warm up for
lr_decay_iters = 20000 # should be ~= max_iters per Chinchilla
min_lr = 0.0 # decay learning rate all the way to 0.0 (llm.c-style q=0.0)
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# extra JSONL logging
log_step_metrics = True # write per-optimizer-step scalar dynamics to scalars.jsonl
log_exposures = True # write per-rank sampled train.bin offsets to exposures/exposures_rankXXXX.jsonl
step_metrics_interval = 10 # optimizer-step interval for scalar dynamics logging
exposure_log_interval = 1 # optimizer-step interval for exposure logging
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


def append_jsonl(path, record):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


@torch.no_grad()
def global_grad_norm_l2(model_obj):
    total_sq = 0.0
    for p in model_obj.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(g.float().pow(2).sum().item())
    return float(total_sq ** 0.5)


@torch.no_grad()
def global_param_norm_l2(model_obj):
    total_sq = 0.0
    for p in model_obj.parameters():
        v = p.detach()
        total_sq += float(v.float().pow(2).sum().item())
    return float(total_sq ** 0.5)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_microstep_global = ddp_world_size * batch_size * block_size
if total_batch_tokens > 0:
    # llm.c-style: derive inner-loop steps from desired global token budget per optimizer step
    gradient_accumulation_steps = max(1, math.ceil(total_batch_tokens / tokens_per_microstep_global))
elif ddp:
    # legacy/manual mode: keep the old interpretation of gradient_accumulation_steps in DDP
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size

tokens_per_iter = gradient_accumulation_steps * tokens_per_microstep_global
if master_process:
    print(f"tokens per microstep (global): {tokens_per_microstep_global:,}")
    if total_batch_tokens > 0:
        print(f"requested total_batch_tokens: {total_batch_tokens:,}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
rank_id = ddp_rank if ddp else 0
scalars_path = os.path.join(out_dir, 'scalars.jsonl')
exposure_path = os.path.join(out_dir, 'exposures', f'exposures_rank{rank_id:04d}.jsonl')

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split, return_meta=False):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data_path = os.path.join(data_dir, 'train.bin')
    else:
        data_path = os.path.join(data_dir, 'val.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if return_meta:
        meta = {
            'split': split,
            'data_path': data_path,
            'start_indices': ix.tolist(),
            'batch_size': int(batch_size),
            'block_size': int(block_size),
        }
        return x, y, meta
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
if log_step_metrics and master_process:
    append_jsonl(scalars_path, {
        'type': 'run_config',
        'iter': int(iter_num),
        'dataset': dataset,
        'data_dir': data_dir,
        'out_dir': out_dir,
        'tokens_per_iter': int(tokens_per_iter),
        'tokens_per_microstep_global': int(tokens_per_microstep_global),
        'total_batch_tokens_target': int(total_batch_tokens),
        'world_size': int(ddp_world_size),
        'gradient_accumulation_steps': int(gradient_accumulation_steps),
        'micro_batch_size': int(batch_size),
        'block_size': int(block_size),
        'decay_lr': bool(decay_lr),
        'learning_rate': float(learning_rate),
        'warmup_iters': int(warmup_iters),
        'lr_decay_iters': int(lr_decay_iters),
        'min_lr': float(min_lr),
        'max_iters': int(max_iters),
    })

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, batch_meta = get_batch('train', return_meta=True) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if log_step_metrics:
            append_jsonl(scalars_path, {
                'type': 'val_loss',
                'iter': int(iter_num),
                'train_loss_eval': float(losses['train']),
                'val_loss': float(losses['val']),
                'lr': float(lr),
                'tokens_seen_global_approx': int(iter_num * tokens_per_iter),
                'eval_iters': int(eval_iters),
            })
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    step_exposure = []
    for micro_step in range(gradient_accumulation_steps):
        if log_exposures:
            step_exposure.append({
                'micro_step': int(micro_step),
                'split': batch_meta['split'],
                'data_path': batch_meta['data_path'],
                'start_indices': batch_meta['start_indices'],
                'batch_size': batch_meta['batch_size'],
                'block_size': batch_meta['block_size'],
            })
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, batch_meta = get_batch('train', return_meta=True)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    should_log_step = log_step_metrics and master_process and (iter_num % step_metrics_interval == 0)
    grad_norm_preclip = None
    grad_norm_postclip = None
    # Unscale gradients once so logged/clipped norms are in true units.
    scaler.unscale_(optimizer)
    # clip the gradient
    if should_log_step:
        grad_norm_preclip = global_grad_norm_l2(model)
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if should_log_step:
        grad_norm_postclip = global_grad_norm_l2(model)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    if should_log_step:
        param_norm = global_param_norm_l2(raw_model)
        append_jsonl(scalars_path, {
            'type': 'train_step',
            'iter': int(iter_num),
            'lr_before_step': float(lr),
            'lr_after_step': float(optimizer.param_groups[0]['lr']),
            'train_loss_raw_last': float(loss.item() * gradient_accumulation_steps),
            'grad_norm_l2_preclip': float(grad_norm_preclip) if grad_norm_preclip is not None else None,
            'grad_norm_l2_postclip': float(grad_norm_postclip) if grad_norm_postclip is not None else None,
            'param_norm_l2': float(param_norm),
            'tokens_seen_global_approx': int((iter_num + 1) * tokens_per_iter),
            'tokens_per_iter': int(tokens_per_iter),
            'world_size': int(ddp_world_size),
            'gradient_accumulation_steps': int(gradient_accumulation_steps),
            'micro_batch_size': int(batch_size),
            'block_size': int(block_size),
        })
    if log_exposures and (iter_num % exposure_log_interval == 0):
        append_jsonl(exposure_path, {
            'type': 'exposure',
            'iter': int(iter_num),
            'rank': int(rank_id),
            'world_size': int(ddp_world_size),
            'tokens_per_iter': int(tokens_per_iter),
            'tokens_seen_local_approx': int((iter_num + 1) * gradient_accumulation_steps * batch_size * block_size),
            'tokens_seen_global_approx': int((iter_num + 1) * tokens_per_iter),
            'micro_batches': step_exposure,
        })
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if log_step_metrics and master_process and (iter_num % step_metrics_interval == 0):
        append_jsonl(scalars_path, {
            'type': 'throughput',
            'iter': int(iter_num),
            'dt_ms': float(dt * 1000.0),
            'tokens_per_second': float(tokens_per_iter / max(dt, 1e-12)),
            'tokens_per_iter': int(tokens_per_iter),
            'mfu': float(running_mfu),
            'lr': float(lr),
        })
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
