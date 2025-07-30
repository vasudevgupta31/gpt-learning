import os
import time
import math
import torch
from src.components import GPT, GPTConfig
from src.dataloader import DataLoaderLite
import wandb

# ------------------------------------------------------------------------------------------------------------
# PATHS
CHECKPOINTS_PATH = "checkpoints"

# ------------------------------------------------------------------------------------------------------------
# PARAMS
total_batch_size = 262_144                 # This is in tokens.
B                = 8                       # Micro Batch size
BLOCK_SIZE       = 512                     # Sequence length
VOCAB_SIZE       = 50_304                  # adding fake tokens so our vocab size is a pretty number (power of 2) and this increases performance # this is interesting since we're adding more caluclations but the GPU architecutres (block tiles etc) are in power of 2 and if those doesnt fit with those block tiles and another padding computation is added which takes time.
WEIGHT_DECAY     = 0.1
N_LAYER          = 12                      # number of layers
N_HEAD           = 12                      # number of heads
N_EMBD           = 768                     # embedding dimension
DROPOUT          = 0.0

# ------------------------------------------------------------------------------------------------------------
# LEARNING RATE
max_lr        = 6e-4
max_steps     = 20_000
min_lr        = max_lr * 0.1
warmup_steps  = int(max_steps*0.035)

# ------------------------------------------------------------------------------------------------------------
# Wandb
wandb_entity    = "vasudev-gupta-decision-tree-analytics-services"
wandb_project   = "foundation-training-gpt2"
experiment_name = "test-1"

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
# Start a new wandb run to track this script.
wandbrun = wandb.init(
                        # Set the wandb entity where your project will be logged (generally your team name).
                        entity=wandb_entity,
                        # Set the wandb project where this run will be logged.
                        project=wandb_project,
                        name=experiment_name,
                        # Track hyperparameters and run metadata.
                        config={
                                    "total_batch_size": total_batch_size,
                                    "micro_batch_size": B,
                                    "block_size": BLOCK_SIZE,
                                    "steps": max_steps,
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
                                    "worst_loss": expected_starting_loss
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
train_loader = DataLoaderLite(B=B, T=BLOCK_SIZE, split="train")
val_loader   = DataLoaderLite(B=B, T=BLOCK_SIZE, split="val")
assert total_batch_size % (B * BLOCK_SIZE) == 0, "make sure total batch size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * BLOCK_SIZE)
expected_starting_loss = -math.log(1/VOCAB_SIZE)

# ------------------------------------------------------------------------------------------------------------
# Messages
print("-"*100)
print(f"using device: {device}")
print(f"total batch size: {total_batch_size}")
print(f"gradient accumulation steps: {grad_accum_steps}")
print(f"worst loss (when all tokens in vocab have uniform prob): {expected_starting_loss}")
print("-"*100)

# ------------------------------------------------------------------------------------------------------------
# Model
model = GPT(GPTConfig(block_size=BLOCK_SIZE,
                      vocab_size=VOCAB_SIZE, 
                      n_embd=N_EMBD,
                      n_head=N_HEAD,
                      n_layer=N_LAYER,
                      dropout=DROPOUT))
model.to(device=device)
model = torch.compile(model)

# ------------------------------------------------------------------------------------------------------------
# Learning Rate
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# ------------------------------------------------------------------------------------------------------------
# Optimizer
optimizer = model.configure_optimizers(weight_decay=WEIGHT_DECAY, learning_rate=max_lr, device_type=device_type)

# ------------------------------------------------------------------------------------------------------------
# Training loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    # gradient accumulation
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):    # check if bfloat16 is supported on the gpu you're running.
            # https://docs.pytorch.org/docs/stable/amp.html
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps                                    #  our loss.backward always deposits gradient (Also, since we are doing in a loop for micro steps; we need to avg them out since loss is always avg (per sample)
        loss_accum += loss.detach()
        loss.backward()

    # clip norm of the entire gradient vector to 1.0 after backward (as suggested in gpt3) -> check for its stability during the optimisation
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # sync queue from cpu to gpu with torch.synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    if device == 'mps':
        torch.mps.synchronize()

    # verbose
    t1 = time.time()
    dt = (t1 - t0)                # time diff in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step}  | loss: {loss_accum.item()}  | lr {lr:.4e}  |  norm: {norm: 4f}  | dt: {dt*1000: 2f} ms  | tok/s: {tokens_per_sec: .4f}")

    # Wandb log trace
    wandbrun.log({"step": step, 
                  "loss": loss_accum.item(), 
                  "lr": lr, 
                  "norm": norm, 
                  "dt_ms": dt*1000, 
                  "throughput_tokpersec": tokens_per_sec})


    # write model checkpoints
    if step > 0 and (step % 5000 == 0 or last_step):
        checkpoint_path = os.path.join(CHECKPOINTS_PATH, f"model_{step:05d}.pt")
        checkpoint = {
            'model': model.state_dict(),
            'config': model.config,
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)


torch.save(model, os.path.join("checkpoints", "final_model"))
