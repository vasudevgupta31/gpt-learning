import torch
import torch.nn.functional as F

def generate_samples(model, prompt, enc, device, device_type, max_length=32, num_return_sequences=4):
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
    xgen = tokens

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (B, 50), topk_indices is (B, 50)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)

    samples = []
    for i in range(num_return_sequences):
        decoded = enc.decode(xgen[i, :max_length].tolist())
        samples.append(decoded)
    return samples
