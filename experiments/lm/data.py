"""
FineWeb-Edu data pipeline for LM pretraining.

FineWeb-Edu (HuggingFaceFW/fineweb-edu) is a standard high-quality pretraining
corpus in current LM/optimizer benchmarks. Tokenized with the GPT-2 BPE
(tiktoken, vocab 50257) as in the nanoGPT/modded-nanoGPT speedrun benchmarks.

Two-phase to respect the cluster (login node has internet but no heavy compute;
compute nodes have the reverse):
  download : fetch raw parquet shards to scratch        (login node, I/O only)
  tokenize : parquet -> uint16 .bin shards              (compute-node SLURM job)
  (train)  : FineWebData memmaps the .bin shards         (compute node)

Usage:
  python data.py download --target-gb 6
  python data.py tokenize --target-tokens 1.2e9
"""
import argparse, os, glob, multiprocessing as mp
from pathlib import Path
import numpy as np

ROOT = Path(os.path.expanduser("~/scratch/datasets/fineweb-edu"))
RAW = ROOT / "raw"
TOK = ROOT / "tokenized"
REPO = "HuggingFaceFW/fineweb-edu"
SUBDIR = "sample/10BT"           # 10B-token sample; we tokenize a slice of it
EOT = 50256                      # GPT-2 <|endoftext|>


def download(target_gb):
    """Fetch raw parquet shards (no tokenization) to scratch."""
    from huggingface_hub import list_repo_files, hf_hub_download
    RAW.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith(SUBDIR) and f.endswith(".parquet")]
    files.sort()
    got = 0.0
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(RAW))
        sz = os.path.getsize(p) / 1e9
        got += sz
        print(f"  downloaded {f} ({sz:.2f} GB, total {got:.2f} GB)", flush=True)
        if got >= target_gb:
            break
    print(f"✓ raw data in {RAW} ({got:.2f} GB)")


_enc = None
def _tok(doc):
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("gpt2")
    ids = _enc.encode_ordinary(doc)
    ids.append(EOT)
    return np.array(ids, dtype=np.uint16)


def tokenize(target_tokens):
    """Tokenize downloaded parquet -> uint16 shards (val = first 100M tokens)."""
    import pyarrow.parquet as pq
    TOK.mkdir(parents=True, exist_ok=True)
    parquets = sorted(glob.glob(str(RAW / "**" / "*.parquet"), recursive=True))
    assert parquets, f"no parquet under {RAW}; run `download` first"
    target = int(float(target_tokens))
    shard_size = 100_000_000      # 100M tokens/shard
    written = 0; shard_idx = 0
    buf = np.empty(shard_size, dtype=np.uint16); fill = 0

    def flush(idx, arr):
        split = "val" if idx == 0 else "train"
        path = TOK / f"fineweb_{split}_{idx:04d}.bin"
        arr.tofile(path)
        print(f"  wrote {path.name} ({len(arr):,} tokens)", flush=True)

    with mp.Pool(max(1, os.cpu_count() // 2)) as pool:
        for pqf in parquets:
            if written >= target:
                break
            docs = pq.read_table(pqf, columns=["text"]).column("text").to_pylist()
            for toks in pool.imap(_tok, docs, chunksize=64):
                if fill + len(toks) > shard_size:
                    take = shard_size - fill
                    buf[fill:] = toks[:take]
                    flush(shard_idx, buf.copy())
                    written += shard_size; shard_idx += 1
                    rem = toks[take:]
                    buf[:len(rem)] = rem; fill = len(rem)
                    if written >= target:
                        break
                else:
                    buf[fill:fill + len(toks)] = toks; fill += len(toks)
    if fill > 0 and written < target:
        flush(shard_idx, buf[:fill].copy())
        written += fill
    print(f"✓ tokenized {written:,} tokens -> {TOK}")


class FineWebData:
    """Memmap .bin shards; yield random (x, y) next-token batches."""
    def __init__(self, split, block_size, batch_size, device):
        self.bs, self.T, self.device = batch_size, block_size, device
        shards = sorted(glob.glob(str(TOK / f"fineweb_{split}_*.bin")))
        assert shards, f"no {split} shards in {TOK}; run `tokenize` first"
        self.data = [np.memmap(s, dtype=np.uint16, mode="r") for s in shards]

    def batch(self):
        import torch
        si = np.random.randint(len(self.data))
        d = self.data[si]
        ix = np.random.randint(0, len(d) - self.T - 1, size=self.bs)
        x = np.stack([d[i:i + self.T].astype(np.int64) for i in ix])
        y = np.stack([d[i + 1:i + 1 + self.T].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        return x, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("download"); d.add_argument("--target-gb", type=float, default=6.0)
    t = sub.add_parser("tokenize"); t.add_argument("--target-tokens", default="1.2e9")
    a = ap.parse_args()
    if a.mode == "download":
        download(a.target_gb)
    else:
        tokenize(a.target_tokens)
