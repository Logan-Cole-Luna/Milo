"""Tokenize pretraining data into uint16 memmap .bin files (nanoGPT format).

Sources:
  fineweb : HuggingFaceFW/fineweb-edu, sample-10BT config (streamed; only the
            requested number of tokens is downloaded/processed).
  shakespeare : tiny-shakespeare, for the 2-minute smoke test.

Examples:
  python scripts/download_data.py --source shakespeare
  python scripts/download_data.py --source fineweb --train-tokens 2e9 --val-tokens 2e7

2e9 GPT-2 tokens -> ~4 GB on disk (uint16). The full sample-10BT is ~20 GB.
"""

import argparse
import os
import urllib.request

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def write_bin(path, token_iter, n_tokens, report_every=50_000_000):
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(int(n_tokens),))
    i = 0
    for toks in token_iter:
        take = min(len(toks), int(n_tokens) - i)
        arr[i:i + take] = toks[:take]
        i += take
        if i % report_every < len(toks):
            print(f"  {path}: {i/1e6:.0f}M / {n_tokens/1e6:.0f}M tokens")
        if i >= n_tokens:
            break
    arr.flush()
    print(f"wrote {path} ({i} tokens)")
    if i < n_tokens:
        print(f"WARNING: source exhausted at {i} tokens; trim --train-tokens")


def fineweb(train_tokens, val_tokens):
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    def tokens():
        for ex in ds:
            yield np.array(enc.encode_ordinary(ex["text"]) + [eot], dtype=np.uint16)

    it = tokens()
    write_bin(os.path.join(OUT, "fineweb_val.bin"), it, val_tokens)
    write_bin(os.path.join(OUT, "fineweb_train.bin"), it, train_tokens)


def shakespeare():
    import tiktoken
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    txt_path = os.path.join(OUT, "shakespeare.txt")
    if not os.path.exists(txt_path):
        urllib.request.urlretrieve(url, txt_path)
    text = open(txt_path).read()
    enc = tiktoken.get_encoding("gpt2")
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    n = int(0.95 * len(ids))
    ids[:n].tofile(os.path.join(OUT, "shakespeare_train.bin"))
    ids[n:].tofile(os.path.join(OUT, "shakespeare_val.bin"))
    print(f"shakespeare: {n} train / {len(ids)-n} val tokens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["fineweb", "shakespeare"], required=True)
    ap.add_argument("--train-tokens", type=float, default=2e9)
    ap.add_argument("--val-tokens", type=float, default=2e7)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.source == "fineweb":
        fineweb(args.train_tokens, args.val_tokens)
    else:
        shakespeare()
