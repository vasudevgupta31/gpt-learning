import os
import torch
import tiktoken
from src.components import GPT, GPTConfig
from src.utils import generate_samples

# ------------------------------------------------------------------------------------
# Load from checkpoint
checkpoint_name = "final_model.pt"
checkpoint_path = os.path.join("checkpoints", checkpoint_name)
checkpoint      = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config          = checkpoint["config"]
state_dict      = checkpoint["model"]
enc             = tiktoken.get_encoding(checkpoint['encoding'])

# Remove compile-specific key
def strip_orig_mod_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# ------------------------------------------------------------------------------------
# Setup model
model = GPT(config)
model.load_state_dict(strip_orig_mod_prefix(state_dict))

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device_type = "cuda" if device.startswith("cuda") else "cpu"
model.to(device)
model.eval()

print("-" * 100)
print(f"using device: {device}")
print(f"loaded model from {checkpoint_path}")
print("-" * 100)

# ------------------------------------------------------------------------------------
# Encode prompt and generate
prompt = "In the land of Mordor, where the shadows lie"
completions = generate_samples(model=model,
                               prompt=prompt,
                               enc=enc,
                               device=device,
                               device_type=device_type,
                               max_length=32,
                               num_return_sequences=4)

# Print results
print(f"# Prompt: {prompt}\n")
for i, sample in enumerate(completions):
    print(f"> Sample {i+1}: {sample}")
