"""T22 (p3): which Unicode scripts occur in S1 / S2 / S3 names and addresses of train (methodology fact check).

A record counts for a script if its text contains a letter whose Unicode character name starts with that script
(unicodedata; LATIN excluded). Output: exploration/p3/script_census_out.md.

    python exploration/p3/script_census.py
"""
from __future__ import annotations

import os
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BER_DATA_DIR", REPO / "data/dataset")) / "train"


def scripts(text: str) -> set[str]:
    out = set()
    for ch in text:
        if ch.isalpha() and not ch.isascii():
            name = unicodedata.name(ch, "")
            s = name.split(" ")[0] if name else "UNKNOWN"
            if s != "LATIN":
                out.add(s)
    return out


def main():
    rows = []
    for src in (1, 2, 3):
        df = pd.read_csv(DATA / f"train_source{src}.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                         usecols=["business_name", "business_address"])
        for col in ("business_name", "business_address"):
            vals = df[col]
            nonascii = vals[~vals.map(str.isascii)]
            c = Counter(s for v in nonascii for s in scripts(v))
            rows.append({"source": src, "field": col, "records": len(vals), "non_latin_script_records":
                         int(sum(1 for v in nonascii if scripts(v))), **{k: v for k, v in c.most_common()}})
            print(rows[-1], flush=True)
    tab = pd.DataFrame(rows).fillna(0)
    cols = ["source", "field", "records", "non_latin_script_records"] + sorted(
        [c for c in tab.columns if c not in ("source", "field", "records", "non_latin_script_records")],
        key=lambda c: -tab[c].sum())
    tab = tab[cols]
    for c in cols[2:]:
        tab[c] = tab[c].astype(int)
    out = REPO / "exploration/p3/script_census_out.md"
    out.write_text("# Unicode scripts in train records (letters, LATIN excluded) — exploration/p3/script_census.py\n\n"
                   + "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
                   + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in tab.itertuples(index=False))
                   + "\n", encoding="utf-8")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
