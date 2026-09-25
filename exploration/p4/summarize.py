"""Print a markdown table of p4 T06 results JSONs (numbers straight from the harness output).

Usage (repo root): python exploration/p4/summarize.py results/p4/*.json
"""
import json
import sys


def main(paths):
    print("| run_id | config | split | status | F0.5 | CI95 | India | US | pairP | pairR | AUC | AP | n_feat | runtime s | commit | dirty | eval ok |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(paths):
        r = json.load(open(p))
        cfg = r["config_path"].replace("\\", "/").split("/")[-1]
        if r["status"] != "ok":
            print(f"| {r['run_id']} | {cfg} | {r['split']} | {r['status']}: {r.get('error', '')[:80]} |" + " |" * 13)
            continue
        m = r.get("metrics", {})
        bc = m.get("by_country", {})
        sc = m.get("scorer") or {}
        print(f"| {r['run_id']} | {cfg} | {r['split']} | ok | {m.get('f05', float('nan')):.4f} | {m.get('f05_ci95')} | "
              f"{bc.get('India', {}).get('f05', '')} | {bc.get('US', {}).get('f05', '')} | {m.get('pair_precision')} | "
              f"{m.get('pair_recall')} | {sc.get('roc_auc', '')} | {sc.get('avg_precision', '')} | "
              f"{r.get('train_pairs', {}).get('n_features')} | {r['runtime_s']} | {r['git']['commit']} | {r['git']['dirty']} | "
              f"{r['eval']['eval_hash_ok']} |")
    print()
    for p in sorted(paths):
        r = json.load(open(p))
        print(f"- `{r['run_id']}` timings_s: {json.dumps(r.get('timings_s'))}  sizes: {json.dumps(r.get('sizes'))}")


if __name__ == "__main__":
    main(sys.argv[1:])
