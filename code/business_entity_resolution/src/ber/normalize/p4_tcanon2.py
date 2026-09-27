"""p4 (Sun, lead-approved France follow-up to T23): p4_tcanon + two more label-free, country-agnostic address rules.

Found by the France test audit (reports/p4/T23.md, exploration/p4/france_forms_out.md), none of which p4_tcanon covers:
  * whole address SEGMENTS that only the S2/S3 side uses — France: department names instead of the region
    ("Pas-de-Calais", "Loire-Atlantique": 31.8% of S2/S3 vs 0.17% of S1) and department numbers "(59)" (1.3% vs 0%).
    Rule: split the raw address at , ; and take parenthesised parts as segments; per country-string group, a
    cleaned segment that S2/S3 records use >= 100x more often than S1 records (df >= 1000, S1 df <= 0.1%) is dropped
    (its tokens removed once from addr_norm). Segment level, so a city that is also a department name ("Calais" in
    "…, Calais") stays: S1 uses that segment too.
  * digits glued to one letter ("6B", "12a") are split ("6 b") before mining, so p4_tcanon's miner learns "b -> bis"
    and the record gets a house number for the keys.
Order: split glued tokens -> p4_tcanon's confident pairs + mined abbreviations -> segments canonicalized with that
map, side-skewed segments dropped -> p4_tcanon's side-skew junk tokens.
Fitted on the records being normalized, no labels, no word lists, no country values in code -> transductive.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from ber.normalize.basic import clean
from ber.normalize.p2_translit import TranslitNormalizer
from ber.normalize.p4_tcanon import TargetCanonNormalizer, confident_pairs, merge_maps, mine_canon, side_skew_junk
from ber.registry import register

_SEG = re.compile(r"[,;]")
_PAREN = re.compile(r"\(([^)]*)\)")
_GLUED = re.compile(r"^(\d+)([a-z])$")


def address_segments(raw: str) -> list[str]:
    """Cleaned comma/semicolon segments of a raw address plus its parenthesised parts (deduplicated, non-empty)."""
    raw = str(raw)
    parts = [clean(_PAREN.sub(" ", s)) for s in _SEG.split(raw)] + [clean(p) for p in _PAREN.findall(raw)]
    return list(dict.fromkeys(p for p in parts if p))


def split_glued(tokens: list[str]) -> list[str]:
    """'6b' -> '6', 'b' (digits followed by exactly one letter)."""
    out = []
    for t in tokens:
        m = _GLUED.match(t)
        out.extend(m.groups() if m else (t,))
    return out


def remove_once(tokens: list[str], drop: Counter) -> list[str]:
    """Remove each token of `drop` from `tokens` as many times as it occurs in `drop` (multiset difference), taking the
    LAST occurrences: administrative segments trail the address, so a street such as "rue de calais" stays intact."""
    if not drop:
        return tokens
    left, out = Counter(drop), []
    for t in reversed(tokens):
        if left[t] > 0:
            left[t] -= 1
        else:
            out.append(t)
    return out[::-1]


@register("normalizer", "p4_tcanon2")
class TargetCanon2Normalizer(TargetCanonNormalizer):
    """p4_tcanon + side-skewed address segments dropped + glued number-letter tokens split (see module doc)."""
    transductive = True

    def __init__(self, tc_segments: bool = True, tc_split_glued: bool = True, **kw):
        super().__init__(**kw)
        self.tc_segments, self.tc_split_glued = bool(tc_segments), bool(tc_split_glued)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        out = TranslitNormalizer.transform(self, records)          # p2_translit only; the target steps are below
        addr = out["addr_norm"].to_numpy(dtype=object).copy()
        names = out["name_norm"].to_numpy(dtype=object)
        raw = records["business_address"].to_numpy(dtype=object)
        is_s1 = (out["source"] == 1).to_numpy()
        train_canon = getattr(self, "canon", {}).get("addr", {})
        for g, idx in pd.Series(np.arange(len(out))).groupby(out["country"].to_numpy()).groups.items():
            idx = np.asarray(idx)
            s1 = is_s1[idx]
            tok = [addr[i].split() for i in idx]
            if self.tc_split_glued:
                tok = [split_glued(t) for t in tok]
            if s1.sum() == 0 or (~s1).sum() == 0:
                for k, i in enumerate(idx):
                    addr[i] = " ".join(tok[k])
                continue
            cs, cr = confident_pairs(tok, [names[i] for i in idx], s1)
            cmap = merge_maps(train_canon, mine_canon([tok[i] for i in cs], [tok[i] for i in cr], **self.tc)) \
                if len(cs) else dict(train_canon)
            canon_tok = [[cmap.get(t, t) for t in x] for x in tok]
            seg_junk: set[str] = set()
            if self.tc_segments:      # AFTER the abbreviation map: "st herblain" -> "saint herblain" (used by S1) stays
                segs = [[" ".join(cmap.get(w, w) for w in split_glued(sg.split())) for sg in address_segments(raw[i])]
                        for i in idx]
                seg_junk = side_skew_junk([x for x, a in zip(segs, s1) if a], [x for x, a in zip(segs, s1) if not a],
                                          self.tc_junk_ratio, self.tc_junk_min_df)
                if seg_junk:
                    canon_tok = [remove_once(t, Counter(w for sg in sgs if sg in seg_junk for w in sg.split()))
                                 for t, sgs in zip(canon_tok, segs)]
            junk = side_skew_junk([x for x, a in zip(canon_tok, s1) if a], [x for x, a in zip(canon_tok, s1) if not a],
                                  self.tc_junk_ratio, self.tc_junk_min_df)
            for k, i in enumerate(idx):
                addr[i] = " ".join(t for t in canon_tok[k] if t not in junk)
            self.tc_stats[str(len(self.tc_stats))] = {"records": int(len(idx)), "confident_pairs": int(len(cs)),
                                                      "map_size": len(cmap), "junk": sorted(junk)[:50],
                                                      "segment_junk": sorted(seg_junk)[:50]}
        out["addr_norm"] = addr
        return out
