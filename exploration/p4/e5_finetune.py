"""T12 (p4): contrastive fine-tuning of multilingual-e5-small on TRAIN true pairs (provided data only).

Usage (repo root, GPU):
    python exploration/p4/e5_finetune.py --split val_v1 --out /path/ft_val_v1      # train side of val_v1 only
    python exploration/p4/e5_finetune.py --split xc_India --out /path/ft_xc_India  # US-only train side (unseen-country test)
    python exploration/p4/e5_finetune.py --split full --out /path/ft_full          # all train pairs (final model)
Data: for a sample of S1 entities of the split's TRAIN side, one random true match (S1 text -> S2/S3 text) plus a hard
negative = a right record with the same country and the same cleaned address that is NOT a match of that S1 (the
"same address, different company" false positives of I-18); random same-country record if none exists.
Loss: InfoNCE over [positives; hard negatives] of the batch ("query: " prefix, mean pooling, L2 norm = exactly the
p4_e5/p1_e5 encoder), temperature 0.05, AdamW, fp16 autocast. Each batch uses one field (name or full, alternating).
The model is saved in fp16 (`save_pretrained`), loadable by p4_e5 / p1_e5 via `model: <out dir>`.
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
from ber.data import load_part  # noqa: E402
from ber.eval.splits import make_split  # noqa: E402
from ber.features.p4_e5 import _clean, field_texts  # noqa: E402


def training_triples(ds, n_max: int, seed: int) -> pd.DataFrame:
    """(anchor, positive, negative) record rows: one true match per sampled S1 entity + same-address hard negative."""
    rng = np.random.default_rng(seed)
    recs = ds.records.reset_index(drop=True)
    row = pd.Series(np.arange(len(recs)), index=recs["entity_id"].to_numpy())
    s1 = [s for s, m in ds.gt.items() if m]
    s1 = [s1[i] for i in rng.permutation(len(s1))[:n_max]]
    pos = [sorted(ds.gt[s])[rng.integers(len(ds.gt[s]))] for s in s1]
    a, p = row[s1].to_numpy(), row[pos].to_numpy()
    right = np.flatnonzero((recs["source"] != 1).to_numpy())
    country = recs["country"].to_numpy()
    key = pd.Series(country[right], dtype=object) + "|" + recs["business_address"].iloc[right].map(
        lambda x: _clean(x).lower()).to_numpy()
    codes, _ = pd.factorize(key)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    starts = np.r_[0, np.flatnonzero(sc[1:] != sc[:-1]) + 1]
    ends = np.r_[starts[1:], len(sc)]
    grp_of_right = np.empty(len(recs), np.int64); grp_of_right[:] = -1
    grp_of_right[right] = codes
    by_country = {c: right[country[right] == c] for c in np.unique(country[right])}
    has_addr = recs["business_address"].map(lambda x: bool(_clean(x))).to_numpy()
    n = np.empty(len(a), np.int64)
    n_hard = 0
    for i, (s, pr) in enumerate(zip(s1, p)):
        matches = ds.gt[s]
        neg = -1
        if has_addr[pr]:
            g = grp_of_right[pr]
            lo, hi = starts[g], ends[g]
            if hi - lo > 1:
                for _ in range(5):
                    c = right[order[rng.integers(lo, hi)]]
                    if recs["entity_id"].iat[c] not in matches:
                        neg = c
                        n_hard += 1
                        break
        if neg < 0:   # random same-country record that is not a match of this S1
            pool = by_country[country[pr]]
            for _ in range(20):
                c = pool[rng.integers(len(pool))]
                neg = c
                if recs["entity_id"].iat[c] not in matches:
                    break
        n[i] = neg
    print(f"triples: {len(a):,} ({n_hard:,} same-address hard negatives)", flush=True)
    return pd.DataFrame({"a": a, "p": p, "n": n}), n_hard


def blocker_triples(train, frac: float, seed: int, config: str) -> tuple[pd.DataFrame, int]:
    """v2 negatives: run the team blocker (configs/best.yaml) on a `frac` sample of the train side and use, for each S1
    entity with >= 1 match, one random true match and one of its 3 highest-ranked NON-matching candidates (the real
    confusions). Rows index the SAMPLE's records (returned via .attrs['records'])."""
    from ber.config import load_config
    from ber.eval.splits import subsample
    from ber.pipeline import Pipeline
    rng = np.random.default_rng(seed)
    sub = subsample(train, frac, seed=100 + seed)
    pipe = Pipeline(load_config(config), None, use_cache=False)
    pipe.normalizer.fit(sub)
    nds = pipe._normalize(sub)
    for b in pipe.blockers:
        b.fit(nds)
    c = pipe._block(nds)[["s1_id", "cand_id", "blk_0_rank"]]
    c["y"] = sub.labels(c)
    neg = c[c["y"] == 0].sort_values(["s1_id", "blk_0_rank"]).groupby("s1_id").head(3)
    neg = neg.groupby("s1_id")["cand_id"].agg(list)
    recs = sub.records.reset_index(drop=True)
    row = pd.Series(np.arange(len(recs)), index=recs["entity_id"].to_numpy())
    A, P, N = [], [], []
    for s, m in sub.gt.items():
        if not m or s not in neg.index:
            continue
        cands = neg[s]
        A.append(row[s]); P.append(row[sorted(m)[rng.integers(len(m))]]); N.append(row[cands[rng.integers(len(cands))]])
    tri = pd.DataFrame({"a": A, "p": P, "n": N})
    tri.attrs["records"] = recs
    print(f"blocker triples: {len(tri):,} from a {frac} sample ({len(sub.s1):,} S1, {len(c):,} candidates)", flush=True)
    return tri, len(tri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_v1", help="val_v1 | xc_<Country> | full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--n", type=int, default=400_000, help="S1 entities (one pair each)")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg", default="address", choices=["address", "blocker"],
                    help="address = v1 (same-address / random negatives); blocker = v2 (top blocker non-matches)")
    ap.add_argument("--neg-frac", type=float, default=0.25, help="v2: train-side sample fraction that is blocked")
    ap.add_argument("--blk-config", default=str(REPO / "code/business_entity_resolution/configs/best.yaml"))
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(a.seed)
    t0 = time.time()
    full = load_part(os.environ["BER_DATA_DIR"], "train", cache_dir=Path(os.environ["BER_ARTIFACTS"]) / "cache")
    train = full if a.split == "full" else make_split(full, a.split)[0]
    del full
    if a.neg == "blocker":
        tri, n_hard = blocker_triples(train, a.neg_frac, a.seed, a.blk_config)
        recs = tri.attrs.pop("records")
        tri = tri.iloc[np.random.default_rng(a.seed).permutation(len(tri))[:a.n]].reset_index(drop=True)
    else:
        tri, n_hard = training_triples(train, a.n, a.seed)
        recs = train.records.reset_index(drop=True)
    texts = {f: field_texts(recs, f) for f in ("name", "full")}
    del recs
    print(f"data ready in {time.time() - t0:.0f}s; split {a.split} train side: {len(train.s1):,} S1", flush=True)
    del train

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tri) // a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()

    def embed(batch_texts):
        b = tok(["query: " + t for t in batch_texts], padding=True, truncation=True, max_length=a.max_length,
                return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h.float() * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.nn.functional.normalize(v, dim=-1)

    rng = np.random.default_rng(a.seed + 1)
    perm = np.concatenate([rng.permutation(len(tri)) for _ in range(a.epochs)])   # one shuffle per epoch
    A, P, N = tri["a"].to_numpy()[perm], tri["p"].to_numpy()[perm], tri["n"].to_numpy()[perm]
    log, t1, used = [], time.time(), 0
    for s in range(steps):
        field = ("name", "full")[s % 2]
        T = texts[field]
        sl = slice(s * a.batch, (s + 1) * a.batch)
        ta, tp, tn = T[A[sl]], T[P[sl]], T[N[sl]]
        ok = np.array([bool(x) and bool(y) for x, y in zip(ta, tp)])
        if ok.sum() < 8:
            continue
        ta, tp, tn = list(ta[ok]), list(tp[ok]), [t if t else " " for t in tn[ok]]
        q = embed(ta)
        d = embed(tp + tn)
        logits = q @ d.T / a.tau
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ta), device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        used += len(ta)
        if s % 100 == 0 or s == steps - 1:
            el = time.time() - t1
            print(f"step {s}/{steps} field {field} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{used / el:,.0f} pairs/s", flush=True)
            log.append({"step": s, "loss": round(loss.item(), 4)})
    out = Path(a.out)
    model.eval().half().save_pretrained(out)
    tok.save_pretrained(out)
    stats = {"split": a.split, "neg": a.neg, "neg_frac": a.neg_frac if a.neg == "blocker" else None,
             "epochs": a.epochs, "base_model": a.model, "n_triples": int(len(tri)), "n_hard_negatives": int(n_hard),
             "steps": steps, "batch": a.batch, "lr": a.lr, "tau": a.tau, "max_length": a.max_length, "seed": a.seed,
             "train_seconds": round(time.time() - t1, 1), "pairs_used": used, "loss_log": log,
             "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (out / "p4_finetune_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "loss_log"}), flush=True)


if __name__ == "__main__":
    main()
