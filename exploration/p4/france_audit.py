"""T14b (done by p4 for p2): label-free audit of one country's TEST records (France = unseen in train).

Usage (repo root):  python exploration/p4/france_audit.py --country France --out exploration/p4/france_audit_out.md
Reads ONLY test inputs (no labels exist). Transductive diagnostics: nothing here is used by the pipeline unless the
fix at the end is adopted (then it must be flagged `transductive = True`). The country string is only used to pick
the records to audit; the proposed fix itself is country-agnostic.
  1. How the team's basic `clean` renders typical raw variants (R./RUE, N°, S.A.R.L., ALL., 2EME, accents).
  2. Side-skewed tokens: frequent in S2/S3 addresses/names but rare in S1 (and vice versa) — the variant forms S1
     never uses, found from data (no hand list).
  3. Label-free candidate pairs: S1 x S2/S3 sharing >= 2 (house number, address word) keys, kept as "confident" when
     the names are near-identical (token_set_ratio >= 90). On them, p2's `learn_canonical_map` (the same learner p2
     fits on TRAIN true pairs) mines abbreviation -> full-form maps from the TARGET data.
  4. Effect of the mined address map: token Jaccard of confident pairs vs random S1 x S2/S3 pairs, before/after.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from ber.normalize.basic import clean  # noqa: E402
from ber.normalize.p2_translit import is_abbrev, learn_canonical_map  # noqa: E402

TRAIN_MAP_TOP = {  # the 40 most frequent entries of p2's TRAIN-learned address map (exploration/p2/canon_out.md, w/p2)
    "new": "newc", "st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "ln": "lane", "pl": "place",
    "fl": "floor", "cir": "circle", "ct": "connecticut", "in": "indiana", "or": "oregon", "al": "alabama",
    "ma": "massachusetts", "ca": "california", "va": "virginia", "saint": "saiint", "park": "parrk", "lake": "laake",
    "hill": "hhill", "york": "yorktown"}


def load_country(test_dir: Path, country: str) -> pd.DataFrame:
    parts = []
    for s in (1, 2, 3):
        for ch in pd.read_csv(test_dir / f"test_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False,
                              na_filter=False, chunksize=1_000_000):
            ch = ch[ch["country"] == country]
            ch.insert(1, "source", s)
            parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def skewed(tok_s1: list[list[str]], tok_r: list[list[str]], top: int, min_df: int = 200):
    d1, dr = Counter(), Counter()
    for t in tok_s1:
        d1.update(set(t))
    for t in tok_r:
        dr.update(set(t))
    n1, nr = max(1, len(tok_s1)), max(1, len(tok_r))
    rows = []
    for t in set(d1) | set(dr):
        a, b = d1[t] / n1, dr[t] / nr
        if d1[t] + dr[t] >= min_df:
            rows.append((t, d1[t], dr[t], (b + 1e-5) / (a + 1e-5)))
    df = pd.DataFrame(rows, columns=["token", "df_S1", "df_S2S3", "ratio_S2S3_over_S1"])
    return (df.sort_values("ratio_S2S3_over_S1", ascending=False).head(top),
            df.sort_values("ratio_S2S3_over_S1").head(top))


def candidate_pairs(addr_tok: list[list[str]], is_s1: np.ndarray, max_df: int = 50) -> pd.DataFrame:
    """(house number, address word) keys; pairs S1 x S2/S3 sharing >= 2 keys. Keys with > max_df right records dropped."""
    rid, key = [], []
    for i, toks in enumerate(addr_tok):
        nums = [t for t in toks if t.isdigit()]
        words = {t for t in toks if t.isalpha() and len(t) >= 4}
        if not nums:
            continue
        for w in words:
            rid.append(i); key.append(nums[0] + "|" + w)
    k = pd.DataFrame({"row": np.array(rid, np.int64), "key": key})
    k["s1"] = is_s1[k["row"].to_numpy()]
    L, R = k[k["s1"]], k[~k["s1"]]
    rdf = R["key"].value_counts()
    R = R[R["key"].map(rdf) <= max_df]
    m = L[["row", "key"]].merge(R[["row", "key"]], on="key", suffixes=("_s1", "_r"))
    c = m.groupby(["row_s1", "row_r"]).size().rename("shared").reset_index()
    return c[c["shared"] >= 2].reset_index(drop=True)


def md(df: pd.DataFrame, fmt: str) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    cell = lambda v: format(v, fmt) if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)])


def jacc(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="France")
    ap.add_argument("--data", default=str(REPO / "data" / "dataset" / "test"))
    ap.add_argument("--out", default=str(REPO / "exploration" / "p4" / "france_audit_out.md"))
    ap.add_argument("--min-support", type=int, default=50)
    ap.add_argument("--min-precision", type=float, default=0.5)
    a = ap.parse_args()
    t0 = time.time()
    recs = load_country(Path(a.data), a.country)
    is_s1 = (recs["source"] == 1).to_numpy()
    name = recs["business_name"].map(clean).to_numpy()
    addr = recs["business_address"].map(clean).to_numpy()
    ntok = [s.split() for s in name]
    atok = [s.split() for s in addr]
    L = [f"# {a.country} test audit (label-free) — {dt.datetime.now():%Y-%m-%d %H:%M} — generated by exploration/p4/france_audit.py",
         "", f"- test records with country == {a.country!r}: {len(recs):,} (S1 {is_s1.sum():,}, S2/S3 {(~is_s1).sum():,});"
         f" empty address S1 {np.mean([not x for x in addr[is_s1]]):.3f}, S2/S3 {np.mean([not x for x in addr[~is_s1]]):.3f}"]

    L += ["", "## 1. What basic `clean` makes of typical raw variants", "| raw | clean |", "|---|---|"]
    for raw in ["63 R. DE DIEPPE, LILLE", "N° 6 RUE DU FAISAN, 2EME ETAGE", "17 ALL. RIVE GAUCHE", "126 Av Des Ondines",
                "S.A.S. Connect", "S.A.R.L. Dupont", "Spôrtive établissements", "Hauts-de-France", "53 Bis Rue Firmin",
                "TOURCOING, 9 - R. DU TILLEUL", "CORALIEFETESSASCOM"]:
        L.append(f"| `{raw}` | `{clean(raw)}` |")

    for field, tok in (("address", atok), ("name", ntok)):
        hi, lo = skewed([t for t, s in zip(tok, is_s1) if s], [t for t, s in zip(tok, is_s1) if not s], 30)
        L += ["", f"## 2. {field}: tokens much more frequent on the S2/S3 side than in S1 (variant forms S1 never uses)",
              md(hi, ".1f"),
              "", f"{field}: tokens much more frequent in S1 than on the S2/S3 side (the full forms)",
              md(lo, ".3f")]

    t = time.time()
    cp = candidate_pairs(atok, is_s1)
    s, r = cp["row_s1"].to_numpy(), cp["row_r"].to_numpy()
    sim = process.cpdist(list(name[s]), list(name[r]), scorer=fuzz.token_set_ratio, workers=4)
    conf = cp[sim >= 90].reset_index(drop=True)
    L += ["", "## 3. Label-free candidate pairs (target data only)",
          f"- {len(cp):,} S1 x S2/S3 pairs share >= 2 (house number, address word) keys; {len(conf):,} are confident "
          f"(name token_set_ratio >= 90), covering {conf['row_s1'].nunique():,} S1 ({time.time() - t:.0f}s)"]
    cs, cr = conf["row_s1"].to_numpy(), conf["row_r"].to_numpy()
    maps = {}
    for field, tok in (("address", atok), ("name", ntok)):
        m = learn_canonical_map([tok[i] for i in cs], [tok[i] for i in cr], a.min_support, a.min_precision, 5)
        maps[field] = m
        # support of each mapping on the confident pairs
        sup = Counter()
        for i, j in zip(cs, cr):
            A, B = set(tok[i]), set(tok[j])
            for short, long in m.items():
                if (short in A and long in B) or (short in B and long in A):
                    sup[short] += 1
        rows = sorted(m.items(), key=lambda kv: -sup[kv[0]])
        L += ["", f"### mined {field} map ({len(m)} entries; min_support {a.min_support}, min_precision {a.min_precision})",
              "| short | full form | confident pairs using it | train map says (top-40 only) |", "|---|---|---|---|"]
        for short, long in rows[:40]:
            tr = TRAIN_MAP_TOP.get(short, "")
            flag = " **conflict**" if tr and tr != long else ""
            L.append(f"| {short} | {long} | {sup[short]:,} | {tr}{flag} |")

    am = maps["address"]
    canon = lambda toks: [am.get(t, t) for t in toks]   # noqa: E731
    rng = np.random.default_rng(0)
    rs = rng.choice(np.flatnonzero(is_s1), 20000)
    rr = rng.choice(np.flatnonzero(~is_s1), 20000)
    before_c = np.mean([jacc(atok[i], atok[j]) for i, j in zip(cs, cr)])
    after_c = np.mean([jacc(canon(atok[i]), canon(atok[j])) for i, j in zip(cs, cr)])
    before_r = np.mean([jacc(atok[i], atok[j]) for i, j in zip(rs, rr)])
    after_r = np.mean([jacc(canon(atok[i]), canon(atok[j])) for i, j in zip(rs, rr)])
    L += ["", "## 4. Effect of the mined address map (label-free proxy)",
          "| pairs | mean address token Jaccard before | after |", "|---|---|---|",
          f"| confident candidate pairs ({len(cs):,}) | {before_c:.4f} | {after_c:.4f} |",
          f"| random S1 x S2/S3 pairs (20,000) | {before_r:.4f} | {after_r:.4f} |",
          "", f"- `st` in {a.country} addresses: df S1 {sum('st' in set(x) for x in np.array(atok, dtype=object)[is_s1]):,}, "
          f"S2/S3 {sum('st' in set(x) for x in np.array(atok, dtype=object)[~is_s1]):,}; mined map: st -> {am.get('st', '(none)')}; "
          f"train map: st -> street"]

    # 5. the proposed label-free fix (exploration/p4/unsup_canon.py): mined map incl. 1-letter forms + side-skew junk
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unsup_canon import apply, mine_canon, side_skew_junk
    full = mine_canon([atok[i] for i in cs], [atok[i] for i in cr], a.min_support, a.min_precision)
    canon_all = [[full.get(t, t) for t in x] for x in atok]
    junk = side_skew_junk([x for x, s1 in zip(canon_all, is_s1) if s1], [x for x, s1 in zip(canon_all, is_s1) if not s1])
    fx = lambda toks: apply(toks, full, junk)   # noqa: E731
    fc = np.mean([jacc(fx(atok[i]), fx(atok[j])) for i, j in zip(cs, cr)])
    fr = np.mean([jacc(fx(atok[i]), fx(atok[j])) for i, j in zip(rs, rr)])
    changed = np.mean([fx(x) != x for x, s1 in zip(atok, is_s1) if not s1])
    L += ["", "## 5. Proposed fix (exploration/p4/unsup_canon.py): mined map incl. 1-letter forms + side-skew junk",
          f"- map: {len(full)} entries; single-letter: {sorted((k, v) for k, v in full.items() if len(k) == 1)}",
          f"- junk (after canonicalization): {sorted(junk)}",
          f"- S2/S3 {a.country} addresses changed: {changed:.1%}",
          "| pairs | address token Jaccard: raw | mined map (sec. 4) | full fix |", "|---|---|---|---|",
          f"| confident candidate pairs ({len(cs):,}) | {before_c:.4f} | {after_c:.4f} | {fc:.4f} |",
          f"| random S1 x S2/S3 pairs (20,000) | {before_r:.4f} | {after_r:.4f} | {fr:.4f} |",
          f"- runtime {time.time() - t0:.0f}s"]
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
