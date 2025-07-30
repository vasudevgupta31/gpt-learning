import os
import json
import numpy as np
import tiktoken
from tqdm import tqdm
import multiprocessing as mp
from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    MIN_LEN = 20
    SHARD_SIZE = int(1e8)  # 100M tokens/shard
    VALID_KEYS = {"text", "paragraph", "text_en", "maintext"}
    ENC = tiktoken.get_encoding("gpt2")
    EOT = ENC._special_tokens['<|endoftext|>']


def tokenize(text):
    tokens = [PreprocessingConfig.EOT]
    tokens.extend(PreprocessingConfig.ENC.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def worker_fn(line):
    try:
        l = json.loads(line)
        if isinstance(l, str):
            l = json.loads(l)
        text_col = next((k for k in l if k in PreprocessingConfig.VALID_KEYS), None)
        if not text_col:
            return None
        return tokenize(l[text_col])
    except Exception:
        return None


def tokenize_dataset(dataset_name, input_folder, output_folder):
    print(f"Processing dataset: {dataset_name}")
    os.makedirs(output_folder, exist_ok=True)

    shard_tokens = np.empty((PreprocessingConfig.SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    shard_index = 0

    jsonl_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jsonl")])

    for file in jsonl_files:
        print(f"Reading {file}")
        file_path = os.path.join(input_folder, file)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"{dataset_name}:{file}", unit=" lines"):
                tokens = worker_fn(line)
                if tokens is None:
                    continue

                if token_count + len(tokens) < PreprocessingConfig.SHARD_SIZE:
                    # If tokens fit in current shard
                    shard_tokens[token_count:token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                else:
                    # Fill current shard exactly
                    space_left = PreprocessingConfig.SHARD_SIZE - token_count
                    shard_tokens[token_count:] = tokens[:space_left]

                    # Write the shard
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(output_folder, f"{dataset_name}_{split}_{shard_index:06d}.npy")
                    write_datafile(filename, shard_tokens)
                    shard_index += 1

                    # Start new shard with leftover tokens
                    leftover = tokens[space_left:]
                    shard_tokens[0:len(leftover)] = leftover
                    token_count = len(leftover)

    # Write remaining partial shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(output_folder, f"{dataset_name}_{split}_{shard_index:06d}.npy")
        write_datafile(filename, shard_tokens[:token_count])

    print(f"Done: {dataset_name} → {shard_index + 1} shard(s) saved in {output_folder}")



def tokenize_dataset_mp(dataset_name, input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    shard_tokens = np.empty((PreprocessingConfig.SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    shard_index = 0

    jsonl_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jsonl")])
    nprocs = max(1, mp.cpu_count() // 2)

    with mp.Pool(nprocs) as pool:
        for file in jsonl_files:

            file_path = os.path.join(input_folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                line_iter = f.readlines()
                for tokens in tqdm(pool.imap(worker_fn, line_iter, chunksize=16), total=len(line_iter), desc=f"{dataset_name}:{file}"):
                    if tokens is None:
                        continue

                    if token_count + len(tokens) < PreprocessingConfig.SHARD_SIZE:
                        shard_tokens[token_count:token_count + len(tokens)] = tokens
                        token_count += len(tokens)
                    else:
                        # fill the shard
                        remainder = PreprocessingConfig.SHARD_SIZE - token_count
                        shard_tokens[token_count:] = tokens[:remainder]

                        split = "val" if shard_index == 0 else "train"
                        filename = os.path.join(output_folder, f"{dataset_name}_{split}_{shard_index:06d}.npy")
                        write_datafile(filename, shard_tokens)
                        shard_index += 1

                        # leftover → new shard
                        leftover = tokens[remainder:]
                        shard_tokens[0:len(leftover)] = leftover
                        token_count = len(leftover)

    # Flush final shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(output_folder, f"{dataset_name}_{split}_{shard_index:06d}.npy")
        write_datafile(filename, shard_tokens[:token_count])

    print(f"Done: {dataset_name} → {shard_index + 1} shards in {output_folder}")


if __name__ == "__main__":
    datasets_path = "datasets"
    output_path = "token-shards"

    datasets = [f for f in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, f))]

    for d in datasets:
        d_in_path = os.path.join(datasets_path, d)
        d_op_path = os.path.join(output_path,   d)
        if os.path.exists(d_op_path) and os.listdir(d_op_path):
            print(f"dataset ({d}) shards already exist in {d_op_path}. Skipping, please manually check if you need to reshard them.")
            continue
        tokenize_dataset_mp(dataset_name=d, input_folder=d_in_path, output_folder=d_op_path)
