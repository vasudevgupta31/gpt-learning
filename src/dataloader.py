import os
import numpy as np
import torch


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, split, checkpoint_state=None):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        # get the shard filenames
        
        data_root = "token-shards"
        shards = []
        for folder in sorted(os.listdir(data_root)):
            folder_path = os.path.join(data_root, folder)
            for f in sorted(os.listdir(folder_path)):
                if split in f:
                    file_path = os.path.join(folder_path, f)
                    shards.append(file_path)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        if checkpoint_state is not None:
            self.load_state(checkpoint_state)
        else:
            self.reset()

    def load_state(self, checkpoint_state):
        self.current_shard = checkpoint_state['current_shard']
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = checkpoint_state['current_position']

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    # def next_batch(self):
    #     B, T = self.B, self.T
    #     buf = self.tokens[self.current_position : self.current_position+B*T+1]
    #     x = (buf[:-1]).view(B, T) # inputs
    #     y = (buf[1:]).view(B, T)  # targets
    #     # advance the position in the tensor
    #     self.current_position += B * T
    #     # if loading the next batch would be out of bounds, advance to next shard
    #     if self.current_position + (B * T + 1) > len(self.tokens):
    #         self.current_shard = (self.current_shard + 1) % len(self.shards)
    #         self.tokens = load_tokens(self.shards[self.current_shard])
    #         self.current_position = B * T
    #     return x, y

    def next_batch(self):
        B, T = self.B, self.T

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T

        return x, y
