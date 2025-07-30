"""
Usage:
python download_data.py --hfpath <huggingface_dataset_path> --output_dir <output_dir> [--shard_limit 25000] [--name <subdataname>]

Example:
python download_data.py --hfpath lucadiliello/english_wikipedia --output_dir=datasets/wikipedia-eng2 --shard_limit 25000                               # Done
python download_data.py --hfpath Navanjana/Gutenberg_books --output_dir=datasets/gutenberg-eng --shard_limit 50000                                     # Done
python download_data.py --hfpath britllm/TransWeb-Edu-English --output_dir=datasets/transweb-edu-eng --shard_limit 25000                               # Done
python download_data.py --hfpath HuggingFaceFW/fineweb-edu --output_dir datasets/fineweb-edu-eng --shard_limit 25000 --name sample-10BT                # Yet to download
python download_data.py --hfpath vietgpt/openwebtext_en --output_dir datasets/openwebtext-eng --shard_limit 10000                                      # Done

python download_data.py --hfpath snoop2head/enron_aeslc_emails --output_dir=datasets/enron-emails-eng --shard_limit 25000                              # Done
python download_data.py --hfpath AyoubChLin/CNN_News_Articles_2011-2022 --output_dir=datasets/cnn-articles_2011_2022-eng --shard_limit 10000           # Done
"""
import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset


def save_checkpoint(save_dir, shard_index, row_index, source, shard_limit):
    with open(os.path.join(save_dir, "checkpoint.json"), "w") as f:
        json.dump({"shard_index": shard_index, "row_index": row_index, "source": source, "shard_limit": shard_limit}, f)


def load_checkpoint(save_dir):
    checkpoint_path = os.path.join(save_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"shard_index": 0, "row_index": 0}


def download_data_shards(hugging_face_path: str, name: str, save_dir: str, shard_limit: int):
    print(f"Downloading from: {hugging_face_path}")
    print(f"Saving to: {save_dir}")
    print(f"Shard limit: {shard_limit} examples per shard")
    if name:
        print(f"Using subdataset name: {name}")

    dataset = load_dataset(hugging_face_path, name=name, streaming=True)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = load_checkpoint(save_dir)
    shard_index = checkpoint["shard_index"]
    row_index = checkpoint["row_index"]
    print(f"Resuming from shard {shard_index}, row {row_index}")

    current_shard = []
    current_shard_size = 0
    global_row_index = 0

    for example in tqdm(dataset['train']):
        if global_row_index < row_index:
            global_row_index += 1
            continue

        current_shard.append(example)
        current_shard_size += 1
        global_row_index += 1

        if current_shard_size >= shard_limit:
            shard_path = os.path.join(save_dir, f"shard_{shard_index:04d}.jsonl")
            with open(shard_path, "w") as f:
                for ex in current_shard:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Saved: {shard_path}")

            shard_index += 1
            current_shard = []
            current_shard_size = 0
            save_checkpoint(save_dir, shard_index, global_row_index, hugging_face_path, shard_limit)

    if current_shard:
        shard_path = os.path.join(save_dir, f"shard_{shard_index:04d}.jsonl")
        with open(shard_path, "w") as f:
            for ex in current_shard:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved final shard: {shard_path}")
        save_checkpoint(save_dir, shard_index + 1, global_row_index, hugging_face_path, shard_limit)

    print(f"Finished downloading and saving all data for {hugging_face_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and shard a Hugging Face dataset into JSONL files.")

    parser.add_argument("--hfpath", required=True, help="Path to Hugging Face dataset (e.g., 'lucadiliello/english_wikipedia')")
    parser.add_argument("--output_dir", required=True, help="Directory to save JSONL shards")
    parser.add_argument("--shard_limit", type=int, default=25_000, help="Maximum number of examples per shard, please choose smaller shard size if each row in the dataset is large (default: 50,000)")
    parser.add_argument("--name", type=str, default=None, help="Optional name of data passed to load_dataset (e.g., 'wikipedia')")

    args = parser.parse_args()

    download_data_shards(
        hugging_face_path=args.hfpath,
        name=args.name,
        save_dir=args.output_dir,
        shard_limit=args.shard_limit
    )
