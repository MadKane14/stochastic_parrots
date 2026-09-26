"""p4 (top-50 push, Sun night): fine-tune a CROSS-ENCODER on blocker candidate pairs of a split's TRAIN side.

Usage (repo root, GPU):
    python exploration/p4/ce_train.py --split val_v1@0.1 --config code/business_entity_resolution/configs/exp/p1/T13_sub4.yaml \
        --out /path/ce_model
Model: intfloat/multilingual-e5-small (MIT, 117.65M params) + a 1-logit classification head, reading
"name | address" of the S1 record and of the candidate together (max 128 tokens), trained with BCE on:
all sampled true pairs, the blocker's top-ranked non-matching candidates (hard negatives: the "same address,
different company" confusions) and random candidate negatives. Provided train data only; no labels of the target.
The split's train side is built exactly like the harness (make_split + subsample(DEV_SEED)), so the held-out target
of `ce_eval.py` is never seen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from ber.config import load_config  # noqa: E402
from ber.data import load_part  # noqa: E402
from ber.eval.harness import DEV_SEED, parse_split  # noqa: E402
from ber.eval.splits import make_split, subsample  # noqa: E402
from ber.features.p4_e5 import _clean  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402


def record_text(recs: pd.DataFrame) -> pd.Series:
    """'name | address' per record (cleaned whitespace; raw case/script kept for the multilingual encoder)."""
    n = recs["business_name"].map(_clean)
    a = recs["business_address"].map(_clean)
    return (n + " | " + a).str.strip(" |")


def split_sides(split: str):
    base, frac = parse_split(split)
    full = load_part(os.environ["BER_DATA_DIR"], "train", cache_dir=Path(os.environ["BER_ARTIFACTS"]) / "cache")
    train, target = make_split(full, base)
    if frac < 1:
        train, target = subsample(train, frac, seed=DEV_SEED), subsample(target, frac, seed=DEV_SEED)
    return train, target


def candidates(ds, cfg: dict) -> pd.DataFrame:
    """The config's normalizer + blockers on `ds` (no features); columns s1_id, cand_id, blk_*_rank, label."""
    pipe = Pipeline(cfg, None, use_cache=False)
    pipe.normalizer.fit(ds)
    nds = pipe._normalize(ds)
    for b in pipe.blockers:
        b.fit(nds)
    c = pipe._block(nds)
    c["label"] = ds.labels(c)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_v1@0.1")
    ap.add_argument("--config", required=True, help="pipeline config whose normalizer + blockers make the candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--n-pos", type=int, default=200_000)
    ap.add_argument("--n-hard", type=int, default=200_000)
    ap.add_argument("--n-rand", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    train, _ = split_sides(a.split)
    c = candidates(train, load_config(a.config))
    rank_cols = [x for x in c.columns if x.endswith("_rank")]
    c["best_rank"] = c[rank_cols].min(axis=1)
    pos = c[c["label"] == 1]
    neg = c[c["label"] == 0]
    hard = neg.sort_values(["s1_id", "best_rank"]).groupby("s1_id").head(3)
    rand = neg.drop(hard.index)
    take = lambda d, n: d.iloc[rng.permutation(len(d))[:n]]   # noqa: E731
    tr = pd.concat([take(pos, a.n_pos), take(hard, a.n_hard), take(rand, a.n_rand)], ignore_index=True)
    tr = tr.iloc[rng.permutation(len(tr))].reset_index(drop=True)
    txt = record_text(train.records).to_numpy()
    idx = pd.Index(train.records["entity_id"].to_numpy())
    A, B = txt[idx.get_indexer(tr["s1_id"])], txt[idx.get_indexer(tr["cand_id"])]
    y = tr["label"].to_numpy(np.float32)
    print(f"candidates {len(c):,} (pos {len(pos):,}); training pairs {len(tr):,} (pos {int(y.sum()):,}, hard "
          f"{min(a.n_hard, len(hard)):,}, random {min(a.n_rand, len(rand)):,}); data {time.time() - t0:.0f}s", flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    perm = np.concatenate([rng.permutation(len(tr)) for _ in range(a.epochs)])
    t1, log = time.time(), []
    for s in range(steps):
        i = perm[s * a.batch:(s + 1) * a.batch]
        b = tok(list(A[i]), list(B[i]), padding=True, truncation=True, max_length=a.max_length, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**b).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[i]).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if s % 500 == 0 or s == steps - 1:
            print(f"step {s}/{steps} loss {loss.item():.4f} {(s + 1) * a.batch / (time.time() - t1):,.0f} pairs/s", flush=True)
            log.append({"step": s, "loss": round(loss.item(), 4)})
    out = Path(a.out)
    model.eval().half().save_pretrained(out)
    tok.save_pretrained(out)
    stats = {"split": a.split, "config": a.config, "base_model": a.model, "n_train_pairs": int(len(tr)),
             "n_pos": int(y.sum()), "steps": steps, "batch": a.batch, "lr": a.lr, "max_length": a.max_length,
             "epochs": a.epochs, "train_seconds": round(time.time() - t1, 1), "loss_log": log,
             "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (out / "p4_ce_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "loss_log"}), flush=True)


if __name__ == "__main__":
    main()
