import torch
import tiktoken
from torch.nn import functional as F
from components import GPTConfig, GPT, CausalSelfAttention, MLP, Block


model = torch.load("checkpoints/final_model", weights_only=False)
enc = tiktoken.get_encoding("gpt2")

# RUN:
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("-"*100)
print(f"using device: {device}")
print("-"*100)

# generation
model.eval()
num_return_sequences = 4
max_length = 32
tokens = enc.encode("In the land of Mordor, where the shadows lie")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)

while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)


# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {decoded}")
