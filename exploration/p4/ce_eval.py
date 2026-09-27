"""p4: does a cross-encoder (ce_train.py) add to the stack? Proxy on a held-out harness target (val_v1@0.1).

Usage (repo root):  python exploration/p4/ce_eval.py --pairs <harness run dir>/scored_pairs.parquet --split val_v1@0.1 \
                        --model /path/ce_model --out exploration/p4/ce_eval_out.md
`scored_pairs.parquet` = the harness's target-side candidates (after pruning) with the stack's probability `score` and
`label`. The cross-encoder scores the same pairs; decisions use the stack's decider (p1_assign, exclusive right) and
the frozen metric (ber.eval.metrics.evaluate).
PRIMARY comparison, fixed before seeing results: stack score vs mean(stack, CE), both at the config threshold 0.7.
Everything else (CE alone, other weights/thresholds) is secondary and must not be used to pick a setting.
val_v1@0.1 is plumbing-grade since round 3 (D9): a positive result only justifies a dense-CV run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ce_train import record_text, split_sides  # noqa: E402
from ber.eval.metrics import evaluate  # noqa: E402
from ber.registry import build  # noqa: E402


def ce_scores(model_dir: str, A: np.ndarray, B: np.ndarray, batch: int = 512, max_length: int = 128) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
    out = np.empty(len(A), np.float32)
    order = np.argsort([-(len(x) + len(y)) for x, y in zip(A, B)], kind="stable")
    t = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            i = order[s:s + batch]
            b = tok(list(A[i]), list(B[i]), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out[i] = torch.sigmoid(model(**b).logits.squeeze(-1).float()).cpu().numpy()
    print(f"CE scored {len(A):,} pairs in {time.time() - t:.0f}s ({len(A) / (time.time() - t):,.0f}/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", default="val_v1@0.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.7, help="the stack config's decider threshold (not tuned here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pairs = pd.read_parquet(a.pairs)
    _, target = split_sides(a.split)
    txt = record_text(target.records).to_numpy()
    idx = pd.Index(target.records["entity_id"].to_numpy())
    ce = ce_scores(a.model, txt[idx.get_indexer(pairs["s1_id"])], txt[idx.get_indexer(pairs["cand_id"])])
    gb = pairs["score"].to_numpy(np.float32)
    variants = {"stack": gb, "mean(stack,CE) [PRIMARY]": (gb + ce) / 2, "CE alone": ce,
                "0.7*stack+0.3*CE": 0.7 * gb + 0.3 * ce}
    s1 = target.s1
    rows = []
    for name, sc in variants.items():
        for thr in sorted({a.threshold, 0.5}):
            p = pairs[["s1_id", "cand_id", "label"]].copy()
            p["score"] = sc
            m = build("decider", {"name": "p1_assign", "params": {"threshold": thr, "exclusive_right": True}}).decide(target, p)
            r = evaluate(s1, target.gt, p, m[["s1_id", "cand_id"]], n_right=len(target.right))
            rows.append({"variant": name, "threshold": thr, "f05": r["f05"], "ci95": r["f05_ci95"],
                         "pairP": r["pair_precision"], "pairR": r["pair_recall"], "singleton_acc": r["singleton_acc"],
                         "by_country": r["by_country"], "scorer_auc_ap": r.get("scorer")})
            print(json.dumps(rows[-1]), flush=True)
    L = [f"\n## CE proxy — {dt.datetime.now():%Y-%m-%d %H:%M} — pairs `{a.pairs}` ({len(pairs):,}, pos {int(pairs['label'].sum()):,}); "
         f"model `{a.model}`", "| variant | thr | F0.5 [CI95] | pair P / R | singleton acc | India / US |", "|---|---|---|---|---|---|"]
    for r in rows:
        bc = r["by_country"]
        L.append(f"| {r['variant']} | {r['threshold']} | {r['f05']:.4f} {r['ci95']} | {r['pairP']} / {r['pairR']} | "
                 f"{r['singleton_acc']} | {bc.get('India', {}).get('f05', '')} / {bc.get('US', {}).get('f05', '')} |")
    with open(a.out, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
