import os
import time
import math
import torch
import wandb
from tqdm import tqdm

from src.components import GPT, GPTConfig
from preprocessing import PreprocessingConfig
from src.dataloader import DataLoaderLite
from src.utils import generate_samples

# ------------------------------------------------------------------------------------------------------------
# Wandb
wandb_entity    = "vasudev-gupta-decision-tree-analytics-services"
wandb_project   = "foundation-training-gpt2"
experiment_name = "exp2"                                            # Set a descriptive name for this experiment (used for tracking in W&B dashboards).
                                                                    # When resuming from a checkpoint, this should ideally match the original run's experiment name
                                                                    # to maintain continuity in logs and comparison across runs.
# ------------------------------------------------------------------------------------------------------------
# TRACKING
LOAD_CHECKPOINT        = None               # For a new run set to: None or name of checkpoint file for loading from a ckp 
CHECKPOINTS_PATH       = "checkpoints"      # Directory for saving checkpoints or for loading checkpoint
CHECKPOINTS_STEPS      = 2_000              # Checkpoints are created after these many steps, for a 124M model, size of a checkpoint file is around 1.1GB
VAL_TRACK_STEPS        = 50                 # Validation loop runs after these many steps
VAL_LOSS_STEPS         = 20                 # Average loss is computed for these many micro batches

# ------------------------------------------------------------------------------------------------------------
# PARAMS
MAX_STEPS        = 20_000                  # Total number of training steps (1 step = 1 grad update = many micro steps)
TOTAL_BATCH_SIZE = 262_144                 # This is in tokens.
MICRO_BATCH_SIZE = 8                       # Micro Batch size
BLOCK_SIZE       = 512                     # Sequence length
VOCAB_SIZE       = 50_304                  # Vocabulary size; padded to a power of 2 (e.g., 2^15 + extra) for optimal GPU performance.
                                           # GPU architectures favor power-of-2 memory access patterns; non-aligned vocab sizes can trigger extra padding logic that slows down execution.
WEIGHT_DECAY     = 0.1                     # L2 regularization for decaying weights during optimization.
                                           # Applied only to 2D tensors like linear layer weights and embeddings — not to 1D tensors like biases or LayerNorm parameters.
                                           # This selective decay improves generalization and avoids hurting normalization/bias behavior.
                                           # See: src.components.GPT.configure_optimizers() — separates parameters into decayed vs non-decayed groups.
N_LAYER          = 12                      # Number of transformer layers
N_HEAD           = 12                      # Number of attention heads per multi-head self-attention block
N_EMBD           = 768                     # Dimensionality of token and positional embeddings,
DROPOUT          = 0.0                     # Dropout rate used in attention and MLP blocks (

# ------------------------------------------------------------------------------------------------------------
# LEARNING RATE SCHEDULE
max_lr        = 6e-4                        # Peak learning rate after warmup, used in cosine decay schedule
min_lr        = max_lr * 0.1                # Final learning rate at the end of cosine decay, seen 1/10 lower than peak in other implementations
warmup_steps  = int(MAX_STEPS * 0.035)      # Number of steps for linear LR warmup (3.5% of total training), gradually increases LR from 0 to max_lr

# ------------------------------------------------------------------------------------------------------------
# Device:
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

# ------------------------------------------------------------------------------------------------------------
# Calculate gradient accumulation steps and estimate starting loss
assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * BLOCK_SIZE) == 0, "make sure total batch size is divisible by B * T"
grad_accum_steps       = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * BLOCK_SIZE)
expected_starting_loss = -math.log(1/VOCAB_SIZE)

# ------------------------------------------------------------------------------------------------------------
# Load checkpoint if available
checkpoint_path = os.path.join(CHECKPOINTS_PATH, LOAD_CHECKPOINT) if LOAD_CHECKPOINT else None
if checkpoint_path and os.path.isfile(checkpoint_path):
    checkpoint         = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    step_start         = checkpoint['step'] + 1
    checkpoint_message = f"Loading from checkpoint: {checkpoint_path}. Resuming from step: {step_start}"
else:
    checkpoint_message = "Starting training from scratch."
    checkpoint         = {}
    step_start         = 0

gptconfig                     = checkpoint.get('config')
other_config                  = checkpoint.get('other_config', {}) 
model_state                   = checkpoint.get('model')
train_loader_checkpoint_state = checkpoint.get('train_loader_state')
val_loader_checkpoint_state   = checkpoint.get('val_loader_state')
optimizer_state               = checkpoint.get('optimizer')
init_val_loss                 = checkpoint.get('val_loss', [0.0])
init_val_loss                 = torch.tensor(init_val_loss)

# ------------------------------------------------------------------------------------------------------------
# Start a new wandb run to track this script.
wandbrun = wandb.init(
                        # Set the wandb entity where your project will be logged (generally your team name).
                        entity=wandb_entity,
                        # Set the wandb project where this run will be logged.
                        project=wandb_project,
                        name=experiment_name,
                        # Track hyperparameters and run metadata.
                        config={
                                    "total_batch_size": TOTAL_BATCH_SIZE,
                                    "micro_batch_size": MICRO_BATCH_SIZE,
                                    "block_size": BLOCK_SIZE,
                                    "steps": MAX_STEPS,
                                    "grad_accum_steps": grad_accum_steps,
                                    "vocab_size": VOCAB_SIZE,
                                    "n_layer": N_LAYER,
                                    "n_head": N_HEAD,
                                    "n_embd": N_EMBD,
                                    "dropout": DROPOUT,
                                    "weight_decay": WEIGHT_DECAY,
                                    "lr_max": max_lr,
                                    "lr_min": min_lr,
                                    "lr_warmup_steps": warmup_steps,
                                    "architecture": "gpt2",
                                    "machine": "home-gpu",
                                    "device": device,
                                    "worst_loss": expected_starting_loss,
                                    "training_datasets": ", ".join(f for f in os.listdir("token-shards") if not f.startswith('.'))
                                },
                        )

# ------------------------------------------------------------------------------------------------------------
# SEED
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed(42)
if device == 'mps':
    torch.mps.manual_seed(42)
torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------------------------------------------------
# Dataset
train_loader = DataLoaderLite(B=MICRO_BATCH_SIZE, T=BLOCK_SIZE, split="train", checkpoint_state=train_loader_checkpoint_state)
val_loader   = DataLoaderLite(B=MICRO_BATCH_SIZE, T=BLOCK_SIZE, split="val",   checkpoint_state=val_loader_checkpoint_state)
enc          = PreprocessingConfig.ENC

# ------------------------------------------------------------------------------------------------------------
# Messages
print("-"*100)
print(f"using device: {device}")
print(checkpoint_message)
print(f"total batch size: {TOTAL_BATCH_SIZE}")
print(f"gradient accumulation steps: {grad_accum_steps}")
print(f"worst loss (when all tokens in vocab have uniform prob): {expected_starting_loss}")
print("-"*100)

# ------------------------------------------------------------------------------------------------------------
# Model
def strip_orig_mod_prefix(state_dict):
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

if checkpoint:
    model = GPT(gptconfig)
    model.load_state_dict(strip_orig_mod_prefix(model_state))
else:
    model = GPT(
                GPTConfig(block_size=BLOCK_SIZE,
                          vocab_size=VOCAB_SIZE,
                          n_embd=N_EMBD,
                          n_head=N_HEAD,
                          n_layer=N_LAYER,
                          dropout=DROPOUT)
                )
model.to(device=device)
if device == 'cuda':
    model = torch.compile(model)

# ------------------------------------------------------------------------------------------------------------
# Learning Rate
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > MAX_STEPS:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (MAX_STEPS - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# ------------------------------------------------------------------------------------------------------------
# Optimizer
if checkpoint:
    optimizer = model.configure_optimizers(weight_decay=other_config.get('weight_decay', WEIGHT_DECAY),
                                           learning_rate=other_config.get('lr_max', max_lr),
                                           device_type=device_type)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = model.configure_optimizers(weight_decay=WEIGHT_DECAY,
                                           learning_rate=max_lr,
                                           device_type=device_type)


# ------------------------------------------------------------------------------------------------------------
# Training loop
val_loss_accum = init_val_loss

for step in range(step_start, MAX_STEPS):
    t0 = time.time()
    last_step = (step == MAX_STEPS - 1)
    
    # 1. Evaluate validation loss at some intervals
    if step % VAL_TRACK_STEPS == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = VAL_LOSS_STEPS
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum.item():.4f}")
    
    # 2. Generate some completions from the model at some intervals
    if ((step > 0 and step % 250 == 0) or last_step):
        completions = generate_samples(model=model,
                                        prompt="Hello, I'm a language model,",
                                        enc=enc,
                                        device=device,
                                        device_type=device_type,
                                        max_length=32,
                                        num_return_sequences=4)
        print("-"*50)
        print("\n".join(completions))
        print("-"*50)

    # 3. One step of the optimization
    model.train()
    optimizer.zero_grad()
    
    # gradient accumulation
    loss_accum = 0.0
    for micro_step in tqdm(range(grad_accum_steps), desc="Microbatches"):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):    # check if bfloat16 is supported on the gpu you're running.
            # https://docs.pytorch.org/docs/stable/amp.html
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps                                    #  our loss.backward always deposits gradient (Also, since we are doing in a loop for micro steps; we need to avg them out since loss is always avg (per sample)
        loss_accum += loss.detach()
        loss.backward()

    # 4. Clip norm of the entire gradient vector to 1.0 after backward (as suggested in gpt3) -> check for its stability during the optimisation
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 5. One step of the optimizer by determine the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # 6. Sync queue from cpu to gpu with torch.synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    if device == 'mps':
        torch.mps.synchronize()

    # 7. Verbose
    t1 = time.time()
    dt = (t1 - t0)                # time diff in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step}  | loss: {loss_accum.item()}  | val_loss: {val_loss_accum.item()}  | lr {lr:.4e}  |  norm: {norm: 4f}  | dt: {dt*1000: 2f} ms  | tok/s: {tokens_per_sec: .4f}")

    # 8. Wandb log trace
    wandbrun.log({"step": step, 
                  "loss": loss_accum.item(), 
                  "val_loss": val_loss_accum.item(),
                  "lr": lr, 
                  "norm": norm, 
                  "dt_ms": dt*1000, 
                  "throughput_tokpersec": tokens_per_sec})

    # 9. Model checkpoints
    if step > 0 and (step % CHECKPOINTS_STEPS == 0 or last_step):
        os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINTS_PATH, f"model_{step:05d}.pt")
        checkpoint = {
            'model': model.state_dict(),
            'config': model.config,
            'other_config': {'weight_decay': WEIGHT_DECAY, 'lr_max': max_lr},
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict(),
            'encoding': 'gpt2'
            }

        train_loader_state = {'current_shard': train_loader.current_shard, 'current_position': train_loader.current_position}
        val_loader_state   = {'current_shard': val_loader.current_shard,   'current_position': val_loader.current_position}
        wandb_config       = {'wandb_entity' : wandb_entity, 'wandb_project': wandb_project, 'experiment_name': experiment_name}

        checkpoint['train_loader_state'] = train_loader_state
        checkpoint['val_loader_state']   = val_loader_state
        checkpoint['wandb_config']       = wandb_config
        torch.save(checkpoint, checkpoint_path)
