#!/usr/bin/env python
"""Collect every worker's results + reports from their git branches into sync/ (gitignored)
for the planner. Stdlib only. Read-only on git (does NOT fetch/merge — run `git fetch --all` first).

    git fetch --all --prune
    python scripts/collect.py                 # scans origin/w/* branches + your working tree
    python scripts/collect.py --since 2026-09-25T18:00

Outputs:
    sync/digest.md          leaderboards per split, flags, failures, report index
    sync/results_all.jsonl  one line per run (flattened key fields)
    sync/reports/<branch>/… copies of every report file
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "sync"


def git(*args, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def list_refs(pattern: str) -> list[str]:
    out = git("for-each-ref", "--format=%(refname:short)", f"refs/remotes/{pattern}", check=False)
    return [l.strip() for l in out.splitlines() if l.strip() and not l.strip().endswith("/HEAD")]


def files_at(ref: str, prefix: str) -> list[str]:
    return [l for l in git("ls-tree", "-r", "--name-only", ref, "--", prefix, check=False).splitlines() if l]


def read_at(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}", check=False)


def fmt(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="origin/w/*", help="remote branch pattern(s), comma-separated")
    ap.add_argument("--since", default=None, help="only list runs created at/after this ISO time in 'new runs'")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    lock = {}
    lp = ROOT / "planning" / "eval_lock.json"
    if lp.exists():
        lock = json.loads(lp.read_text(encoding="utf-8"))
    refs = [r for pat in a.refs.split(",") for r in list_refs(pat)]
    runs, reports = {}, []
    SYNC.mkdir(exist_ok=True)

    def ingest(src: str, path: str, text: str):
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            return
        if "run_id" in rec and rec["run_id"] not in runs:
            rec["_source"] = src
            runs[rec["run_id"]] = rec

    for ref in refs:
        for p in files_at(ref, "results"):
            if p.endswith(".json") and "/smoke/" not in p:
                ingest(ref, p, read_at(ref, p))
        for p in files_at(ref, "reports"):
            if p.endswith(".md"):
                dest = SYNC / "reports" / ref.replace("/", "_") / Path(p).relative_to("reports")
                dest.parent.mkdir(parents=True, exist_ok=True)
                txt = read_at(ref, p)
                dest.write_text(txt, encoding="utf-8")
                reports.append((ref, p, txt))
    for p in (ROOT / "results").rglob("*.json"):   # working tree (your own machine)
        if "smoke" not in p.parts:
            ingest("working-tree", str(p.relative_to(ROOT)), p.read_text(encoding="utf-8"))

    locked_hash = lock.get("eval_code_sha256")
    locked_splits = lock.get("splits", {})

    def flags(r):
        f = []
        if r.get("status") != "ok":
            f.append("FAILED")
        ev = r.get("eval", {})
        if not locked_hash:
            f.append("NO_LOCK")
        elif ev.get("eval_code_sha256") != locked_hash:
            f.append("EVAL_MODIFIED")
        sp = r.get("split")
        if sp in locked_splits and ev.get("split_fingerprint") and ev["split_fingerprint"] != locked_splits[sp]["fingerprint"]:
            f.append("SPLIT_MISMATCH")
        if r.get("git", {}).get("dirty"):
            f.append("DIRTY")
        if r.get("smoke"):
            f.append("SAMPLE")
        if r.get("transductive_components"):
            f.append("TRANSDUCTIVE")
        return f

    rows = []
    for r in runs.values():
        m = r.get("metrics") or {}
        b = m.get("blocking") or {}
        rows.append({
            "run_id": r["run_id"], "member": r.get("member"), "task": r.get("task"), "split": r.get("split"),
            "mode": r.get("mode"), "created": r.get("created"), "config": r.get("config_path"),
            "overrides": " ".join(r.get("overrides") or []), "note": r.get("note"), "commit": (r.get("git") or {}).get("commit"),
            "branch": r["_source"], "f05": m.get("f05"), "ci": m.get("f05_ci95"), "singleton_acc": m.get("singleton_acc"),
            "f05_nonsingleton": m.get("f05_nonsingleton"), "oracle_f05": b.get("oracle_f05"),
            "pair_recall_blk": b.get("pair_recall"), "cands": b.get("cands_per_s1_mean"),
            "auc": (m.get("scorer") or {}).get("roc_auc"), "by_country": {k: v["f05"] for k, v in (m.get("by_country") or {}).items()},
            "runtime_s": r.get("runtime_s"), "flags": flags(r), "error": r.get("error"),
            "test_stats": r.get("test_stats"),
        })
    with open(SYNC / "results_all.jsonl", "w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda x: x["created"] or ""):
            f.write(json.dumps(row) + "\n")

    L = [f"# Sync digest\n", f"Branches scanned: {', '.join(refs) or '(none — did you git fetch?)'} + working tree  ",
         f"Runs: {len(rows)} | eval lock: {locked_hash or 'NONE'}\n",
         "Flags: EVAL_MODIFIED/SPLIT_MISMATCH/FAILED/SAMPLE => excluded from leaderboards. "
         "DIRTY = uncommitted code (not reproducible from commit). TRANSDUCTIVE = fits unsupervised stats on target data.\n"]
    ok = [r for r in rows if r["mode"] == "val" and not set(r["flags"]) & {"EVAL_MODIFIED", "SPLIT_MISMATCH", "FAILED", "SAMPLE"}]
    for split in sorted({r["split"] for r in ok}):
        sub = sorted([r for r in ok if r["split"] == split], key=lambda x: -(x["f05"] or 0))[: a.top]
        L.append(f"\n## Leaderboard — {split}\n")
        L.append("| # | F0.5 | CI95 | sing.acc | F0.5 non-sing | oracle F0.5 | blk recall | cands/s1 | AUC | by country | member/task | config + overrides | commit | flags | run_id |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(sub, 1):
            L.append(f"| {i} | {fmt(r['f05'])} | {r['ci']} | {fmt(r['singleton_acc'], 3)} | {fmt(r['f05_nonsingleton'])} | "
                     f"{fmt(r['oracle_f05'])} | {fmt(r['pair_recall_blk'])} | {r['cands']} | {fmt(r['auc'])} | "
                     f"{json.dumps(r['by_country'])} | {r['member']}/{r['task']} | {r['config']} {r['overrides']} | "
                     f"{r['commit']} | {' '.join(r['flags'])} | {r['run_id']} |")
    tests = [r for r in rows if r["mode"] == "test"]
    if tests:
        L.append("\n## Test-mode runs (submission candidates)\n")
        for r in sorted(tests, key=lambda x: x["created"] or "", reverse=True)[:10]:
            L.append(f"- {r['run_id']} ({r['member']}, {r['config']} {r['overrides']}, commit {r['commit']}) "
                     f"flags={r['flags']} stats={json.dumps((r['test_stats'] or {}).get('by_country'))}")
    bad = [r for r in rows if set(r["flags"]) & {"FAILED", "EVAL_MODIFIED", "SPLIT_MISMATCH"}]
    if bad:
        L.append("\n## Problems\n")
        for r in bad:
            L.append(f"- {' '.join(r['flags'])} — {r['run_id']} ({r['branch']}): {r['error'] or ''}")
    if a.since:
        new = [r for r in rows if (r["created"] or "") >= a.since]
        L.append(f"\n## Runs since {a.since}: {len(new)}\n")
        for r in sorted(new, key=lambda x: x["created"]):
            L.append(f"- {r['created']} {r['member']}/{r['task']} {r['split']} F0.5={fmt(r['f05'])} {' '.join(r['flags'])} {r['run_id']}")
    L.append("\n## Reports\n")
    for ref, p, txt in reports:
        head = [l for l in txt.splitlines() if l.lower().startswith(("status:", "verdict:", "**status", "**verdict"))][:2]
        L.append(f"- `{ref}:{p}` — {' | '.join(head) or '(no status line)'}")
    (SYNC / "digest.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {SYNC / 'digest.md'} ({len(rows)} runs, {len(reports)} reports)")


if __name__ == "__main__":
    main()
