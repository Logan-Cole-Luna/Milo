"""Fine-tuning benchmark: RoBERTa-base on GLUE tasks (SST-2, MRPC, RTE).

Covers the regime where optimizers behave very differently from pretraining
(small LRs, few epochs, pretrained init). Reviewers expect at least one
fine-tuning result. Manual training loop so the registry controls the
optimizer exactly.

Example:
  python tasks/glue_finetune.py --glue-task mrpc --optimizer mion --lr 1e-4 --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.registry import build_optimizer, parse_opt_kwargs  # noqa: E402
from tasks.common import (RunLogger, already_done, apply_lr, base_lrs,  # noqa: E402
                          cosine_with_warmup, run_id_from, set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIELDS = {"sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
          "rte": ("sentence1", "sentence2")}


def get_data(task, tokenizer, batch):
    from datasets import load_dataset
    ds = load_dataset("glue", task)
    f1, f2 = FIELDS[task]

    def tok(ex):
        args = (ex[f1],) if f2 is None else (ex[f1], ex[f2])
        return tokenizer(*args, truncation=True, max_length=128)

    ds = ds.map(tok, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds.set_format("torch", columns=cols)
    from transformers import DataCollatorWithPadding
    coll = DataCollatorWithPadding(tokenizer)
    return (torch.utils.data.DataLoader(ds["train"], batch, shuffle=True, collate_fn=coll),
            torch.utils.data.DataLoader(ds["validation"], 128, collate_fn=coll))


@torch.no_grad()
def evaluate(model, loader, task):
    model.eval()
    preds, labels = [], []
    for b in loader:
        out = model(input_ids=b["input_ids"].to(DEVICE),
                    attention_mask=b["attention_mask"].to(DEVICE))
        preds.append(out.logits.argmax(-1).cpu())
        labels.append(b["labels"])
    model.train()
    p, l = torch.cat(preds), torch.cat(labels)
    acc = (p == l).float().mean().item()
    if task == "mrpc":  # report F1 as is standard
        tp = ((p == 1) & (l == 1)).sum().item()
        prec = tp / max(1, (p == 1).sum().item())
        rec = tp / max(1, (l == 1).sum().item())
        return 2 * prec * rec / max(1e-8, prec + rec)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glue-task", choices=list(FIELDS), required=True)
    ap.add_argument("--optimizer", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="results/glue")
    ap.add_argument("--opt-kwargs", default="")
    args = ap.parse_args()

    config = dict(task=f"glue_{args.glue_task}", model="roberta-base",
                  optimizer=args.optimizer, lr=args.lr, wd=args.wd,
                  seed=args.seed, epochs=args.epochs, opt_kwargs=args.opt_kwargs)
    if already_done(args.out_dir, config):
        print("[skip]", run_id_from(config))
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2).to(DEVICE)
    tr, va = get_data(args.glue_task, tokenizer, args.batch)

    optimizer, meta = build_optimizer(args.optimizer, model, args.lr, args.wd,
                                      **parse_opt_kwargs(args.opt_kwargs))
    if meta["sf_train_eval"]:
        optimizer.train()
    lrs0 = base_lrs(optimizer)
    steps_total = args.epochs * len(tr)
    warmup = steps_total // 16

    logger = RunLogger(args.out_dir, run_id_from(config), config)
    step, best, t_steps = 0, 0.0, []
    for epoch in range(args.epochs):
        for b in tr:
            step += 1
            t0 = time.perf_counter()
            if meta["use_schedule"]:
                apply_lr(optimizer, lrs0, cosine_with_warmup(step, steps_total, warmup))
            out = model(input_ids=b["input_ids"].to(DEVICE),
                        attention_mask=b["attention_mask"].to(DEVICE),
                        labels=b["labels"].to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            optimizer.step()
            t_steps.append(time.perf_counter() - t0)
            if step % 50 == 0:
                logger.log(step, f"{out.loss.item():.4f}")
        if meta["sf_train_eval"]:
            optimizer.eval()
        metric = evaluate(model, va, args.glue_task)
        if meta["sf_train_eval"]:
            optimizer.train()
        best = max(best, metric)
        logger.log(step, f"{out.loss.item():.4f}", f"{metric:.4f}")
        print(f"epoch {epoch+1}: val metric {metric:.4f}")

    logger.finish(final_metric=round(metric, 4), best_metric=round(best, 4),
                  ms_per_step=round(1000 * float(np.median(t_steps)), 1))


if __name__ == "__main__":
    main()
