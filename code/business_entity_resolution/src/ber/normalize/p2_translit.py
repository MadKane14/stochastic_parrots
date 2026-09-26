"""p2 (T04): cross-script + noisy-name normalizer, learned from the training pairs.

Adds the same columns as `basic` (name_norm, addr_norm, name_core, addr_digits) after four extra steps:

1. Native-script tokens -> Latin, with a token map LEARNED from true pairs in fit() (no dictionary, no country logic).
   For every true (S1, S2/S3) pair whose S2/S3 name (address) has non-Latin tokens, the Latin tokens of the S1 name
   (address) are candidate translations. Association = Dice over pair co-occurrence; tokens are then linked
   one-to-one inside each pair by competitive linking (Melamed 2000, "Models of translational equivalence among
   words"), so that `प्राइवेट` links to `private` and not to `limited` (both co-occur with it every time).
   A native token is mapped to its most-linked Latin token if support >= min_support and precision
   (links / occurrences) >= min_precision. Generic: "non-Latin" = a letter whose Unicode name is not LATIN.
2. Script folding (optional): the Brahmic scripts share one code layout (Unicode ch. 12, ISCII), so a letter of any
   of them is folded to the Devanagari letter with the same Unicode name ("BENGALI LETTER KA" -> "DEVANAGARI LETTER
   KA") before counting. Evidence for the same word in different scripts is pooled; no hand-written table.
   Unicode decimal digits of any script become ASCII digits (unicodedata.digit).
3. Web-style names: a trailing web suffix (`delhiconsulting.com`, `Dr dreammarketing.Com`, `x.co.in`; at least 4
   word characters before the dot, suffix not a legal form, so `Pvt.Ltd` is untouched) and a leading `www.` are
   removed; every long token of such a row (up to 4 words of >= 3 chars), and long rare Latin tokens elsewhere
   (only a clean 2-word split, >= 4 chars each, so typos are left alone), are split into known words by unigram
   Viterbi segmentation (`supremepackaging` -> `supreme packaging`). Word counts = per-token max of the train
   counts and (transductively, no labels) the counts on the records being normalized. A token with 1/0 inside
   letters (`cequee1ement`, `agr0`) becomes its 1->l / 0->o variant only if that is a known word or segments into
   known words.
4. Runs of >= 2 single Latin letters in names are joined (`p l l c` -> `pllc`, `m s` -> `ms`).
5. (optional, `canon_map`) Learned abbreviation canonicalization of Latin tokens, per field: tokens a true pair does
   not share are aligned by the same competitive linking, and a link is kept only if one token is an order-preserving
   abbreviation (<= 5 letters) of the other and the long form is not a concatenation of known words
   (`st`->`street`, `pvt`->`private`, `va`->`virginia`; not `eye`->`eyecare`); all records map short -> long.
6. (optional, `junk`) Learned source-noise tokens (frequent on the S2/S3 side of true pairs, almost never in the S1
   partner nor in S1 records) are dropped from name_norm / addr_norm.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from ber.data import Dataset
from ber.interfaces import Normalizer
from ber.normalize.basic import clean
from ber.registry import register

_DIGITS = re.compile(r"\d+")
_WEB_SUFFIX = re.compile(r"(?<=[^\W_]{4})\.([a-z]{2,6})(?:\.[a-z]{2})?\s*$", re.IGNORECASE)   # "...name.com(.in)"
_WWW = re.compile(r"^\s*www\.", re.IGNORECASE)
_HAS_NUM_IN_WORD = re.compile(r"^(?=.*[a-z])(?=.*[01])[a-z01]+$")
_SINGLE_RUN = re.compile(r"(?<!\S)((?:[a-z] ){1,}[a-z])(?!\S)")
_SINGLE_RUN_ANY = re.compile(r"(?<!\S)(?:[a-z] )+[a-z](?!\S)")

_FOLD_CACHE: dict[str, str] = {}
_LATIN_CACHE: dict[str, bool] = {}
_PREP_CACHE: dict = {}


def _fold_char(ch: str) -> str:
    """Script-agnostic canonical form of one character (see module docstring, step 2)."""
    out = _FOLD_CACHE.get(ch)
    if out is not None:
        return out
    out = ch
    d = unicodedata.digit(ch, None)
    if d is not None and not ch.isascii():
        out = str(d)
    else:
        name = unicodedata.name(ch, "")
        script, _, rest = name.partition(" ")
        if rest and script not in ("LATIN", "DEVANAGARI"):
            try:
                out = unicodedata.lookup("DEVANAGARI " + rest)
            except KeyError:
                out = ch
    _FOLD_CACHE[ch] = out
    return out


def fold(token: str) -> str:
    return "".join(_fold_char(c) for c in token) if not token.isascii() else token


def _is_native(token: str) -> bool:
    """True if the token contains a letter outside the Latin script."""
    if token.isascii():
        return False
    for ch in token:
        if ch.isascii():
            continue
        v = _LATIN_CACHE.get(ch)
        if v is None:
            v = (not ch.isalpha()) or unicodedata.name(ch, "").startswith("LATIN")
            _LATIN_CACHE[ch] = v
        if not v:
            return True
    return False


def _digits_to_ascii(s: str) -> str:
    if s.isascii():
        return s
    return "".join(str(unicodedata.digit(c)) if (not c.isascii() and unicodedata.digit(c, None) is not None) else c
                   for c in s)


def _group_max(keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Max of `vals` within each group of equal `keys`, broadcast back to every element."""
    order = np.lexsort((-vals, keys))
    ks = keys[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    out = np.empty_like(vals)
    out[order] = vals[order][first][np.cumsum(first) - 1]
    return out


def learn_token_map(native_side: list[list[str]], latin_side: list[list[str]], min_support: int,
                    min_precision: float, rounds: int = 3) -> dict[str, str]:
    """Token translation map native -> Latin from aligned bags of tokens (one pair per list element).

    Dice association over pair co-occurrence, then competitive linking inside each pair (vectorised as `rounds`
    of mutual-best linking: an (n, l) edge is linked when it is the best remaining edge for both n and l; ties are
    broken by first occurrence). Integer token codes + numpy throughout (memory ~ 40 bytes per within-pair edge)."""
    pid_n, tok_n, pid_l, tok_l = [], [], [], []
    for i, (ns, ls) in enumerate(zip(native_side, latin_side)):
        ns, ls = dict.fromkeys(ns), dict.fromkeys(ls)          # dedupe, keep order (deterministic codes)
        if ns and ls:
            tok_n.extend(ns); pid_n.extend([i] * len(ns))
            tok_l.extend(ls); pid_l.extend([i] * len(ls))
    if not tok_n:
        return {}
    cn_codes, uniq_n = pd.factorize(pd.Series(tok_n, dtype=object))
    cl_codes, uniq_l = pd.factorize(pd.Series(tok_l, dtype=object))
    del tok_n, tok_l
    pid_n, pid_l = np.asarray(pid_n, np.int64), np.asarray(pid_l, np.int64)
    Vn, Vl = len(uniq_n), len(uniq_l)
    cn = np.bincount(cn_codes, minlength=Vn)                   # pairs containing each native token
    cl = np.bincount(cl_codes, minlength=Vl)
    # all within-pair edges (cross product per pair; both sides are grouped by pair id in increasing order)
    n_pairs = int(max(pid_n.max(), pid_l.max())) + 1
    cnt_l = np.bincount(pid_l, minlength=n_pairs)
    start_l = np.r_[0, np.cumsum(cnt_l)[:-1]]
    rep = cnt_l[pid_n]
    e_p = np.repeat(pid_n, rep)
    e_n = np.repeat(cn_codes.astype(np.int64), rep)
    off = np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep)
    e_l = cl_codes.astype(np.int64)[np.repeat(start_l[pid_n], rep) + off]
    del off, rep
    key_nl = e_n * Vl + e_l
    uk, inv, co = np.unique(key_nl, return_inverse=True, return_counts=True)
    dice = (2.0 * co / (cn[uk // Vl] + cl[uk % Vl]))[inv]
    del inv
    kpn, kpl = e_p * Vn + e_n, e_p * Vl + e_l
    del e_p, e_n, e_l
    alive = np.ones(len(dice), bool)
    linked = []
    for _ in range(rounds):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        d, a_pn, a_pl = dice[idx], kpn[idx], kpl[idx]
        cand = np.flatnonzero((d == _group_max(a_pn, d)) & (d == _group_max(a_pl, d)))
        cand = cand[np.unique(a_pn[cand], return_index=True)[1]]
        cand = cand[np.unique(a_pl[cand], return_index=True)[1]]
        linked.append(key_nl[idx[cand]])
        kill = np.isin(a_pn, a_pn[cand]) | np.isin(a_pl, a_pl[cand])
        alive[idx[kill]] = False
    lk, links = np.unique(np.concatenate(linked), return_counts=True)
    ln, ll = lk // Vl, lk % Vl
    order = np.lexsort((-links, ln))                            # per native token: most-linked Latin token first
    ln, ll, links = ln[order], ll[order], links[order]
    first = np.r_[True, ln[1:] != ln[:-1]]
    ln, ll, links = ln[first], ll[first], links[first]
    ok = (links >= min_support) & (links / cn[ln] >= min_precision)
    return dict(zip(np.asarray(uniq_n)[ln[ok]].tolist(), np.asarray(uniq_l)[ll[ok]].tolist()))


def is_abbrev(short: str, long: str) -> bool:
    """`short` is an order-preserving abbreviation of `long`: same first letter, strictly shorter, and its letters occur
    in `long` in order (st/street, pvt/private, ltd/limited, va/virginia, rd/road). Blocks unrelated words and typos."""
    if not short or len(short) >= len(long) or short[0] != long[0]:
        return False
    it = iter(long)
    return all(c in it for c in short)


def _is_word(t: str) -> bool:
    return len(t) >= 2 and t.isascii() and t.isalpha()


def learn_canonical_map(left: list[list[str]], right: list[list[str]], min_support: int,
                        min_precision: float, max_short_len: int = 5, is_compound=None) -> dict[str, str]:
    """Abbreviation -> full form map learned from true pairs (S1 tokens `left`, S2/S3 tokens `right`).

    Only the tokens a pair does NOT share are candidates (Latin words, >= 2 letters); they are aligned by the same
    Dice + competitive linking as the native map (Arasu et al., VLDB 2009 learn such transformations from matching
    examples). A link is kept only if one token abbreviates the other (`is_abbrev`), the short form has at most
    `max_short_len` letters, and the long form is not a concatenation of known words (`is_compound`, e.g. a web name
    `eyecare` = eye + care, which `eye` would otherwise "abbreviate"). The map points from the short to the long form,
    so both sides of any pair end up in the same canonical form. Chains are collapsed."""
    ur, ul = [], []
    for L, R in zip(left, right):
        sl, sr = set(L), set(R)
        ur.append([t for t in R if t not in sl and _is_word(t)])
        ul.append([t for t in L if t not in sr and _is_word(t)])
    table = learn_token_map(ur, ul, min_support, min_precision)
    out: dict[str, str] = {}
    for r, l in table.items():
        short, long = (r, l) if is_abbrev(r, l) else (l, r) if is_abbrev(l, r) else (None, None)
        if short is None or len(short) > max_short_len or (is_compound is not None and is_compound(long)):
            continue
        out.setdefault(short, long)

    def canon(t: str) -> str:
        seen = set()
        while t in out and t not in seen:
            seen.add(t)
            t = out[t]
        return t
    return {k: canon(k) for k in out if canon(k) != k}


def learn_junk(left: list[list[str]], right: list[list[str]], df_left: Counter, df_right: Counter,
               min_count: int, max_hit: float, max_ratio: float) -> dict[str, dict]:
    """Source-specific noise tokens: frequent on the S2/S3 side of true pairs (>= min_count pairs), almost never present
    in the S1 partner (hit rate <= max_hit) and almost absent from S1 records overall (S1 df <= max_ratio * S2/S3 df).
    They add weight to IDF norms without ever matching. Returns {token: stats}."""
    c, h = Counter(), Counter()
    for L, R in zip(left, right):
        sl = set(L)
        for t in set(R):
            c[t] += 1
            if t in sl:
                h[t] += 1
    junk = {}
    for t, n in c.items():
        if n >= min_count and not t.isdigit() and h[t] <= max_hit * n and df_left.get(t, 0) <= max_ratio * df_right.get(t, 1):
            junk[t] = {"pairs": n, "hit_rate": round(h[t] / n, 4), "s1_df": df_left.get(t, 0), "right_df": df_right.get(t, 0)}
    return junk


class Segmenter:
    """Unigram Viterbi word segmentation of concatenated Latin tokens (e.g. `supremepackaging`)."""

    def __init__(self, counts: Counter, min_word_freq: int, min_word_len: int = 3, max_word_len: int = 20):
        words = {w: c for w, c in counts.items()
                 if c >= min_word_freq and min_word_len <= len(w) <= max_word_len and w.isalpha() and w.isascii()}
        total = float(sum(words.values())) or 1.0
        self.cost = {w: -math.log(c / total) for w, c in words.items()}
        self.maxlen = max((len(w) for w in words), default=0)
        self.counts = counts
        self._cache: dict[str, list[str] | None] = {}

    def known(self, w: str) -> bool:
        return w in self.cost

    def split(self, tok: str, max_parts: int = 4, min_part_len: int = 3) -> list[str] | None:
        """Best split into 2..max_parts known words of length >= min_part_len, or None."""
        key = (tok, max_parts, min_part_len)
        if key in self._cache:
            return self._cache[key]
        n = len(tok)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(max(0, j - self.maxlen), j):
                c = self.cost.get(tok[i:j])
                if c is not None and best[i] + c < best[j]:
                    best[j], back[j] = best[i] + c, i
        out = None
        if best[n] < math.inf:
            parts, j = [], n
            while j > 0:
                parts.append(tok[back[j]:j])
                j = back[j]
            parts.reverse()
            ok = 2 <= len(parts) <= max_parts and min(map(len, parts)) >= min_part_len
            out = parts if ok else None
        self._cache[key] = out
        return out


@register("normalizer", "p2_translit")
class TranslitNormalizer(Normalizer):
    """basic normalization + learned native->Latin token map + web-name / initials cleanup (see module doc)."""

    transductive = True   # segmentation vocabulary also counts name tokens of the records being normalized

    def __init__(self, legal_tokens: list[str] | None = None, mode: str = "replace", min_support: int = 3,
                 min_precision: float = 0.5, fold_scripts: bool = True, map_address: bool = True,
                 web_names: bool = True, seg_min_len: int = 8, seg_rare_max: int = 2, seg_word_freq: int = 20,
                 seg_target_vocab: bool = True, join_initials: bool = True, canon_map: bool = False,
                 canon_min_support: int = 50, canon_min_precision: float = 0.5, canon_max_short_len: int = 5,
                 junk: bool = False,
                 junk_min_count: int = 1000, junk_max_hit: float = 0.01, junk_max_ratio: float = 0.01):
        if mode not in ("replace", "append"):
            raise ValueError("mode must be 'replace' or 'append'")
        self.legal = set(legal_tokens or [])
        self.mode, self.min_support, self.min_precision = mode, int(min_support), float(min_precision)
        self.fold_scripts, self.map_address, self.web_names = bool(fold_scripts), bool(map_address), bool(web_names)
        self.seg_min_len, self.seg_rare_max, self.seg_word_freq = int(seg_min_len), int(seg_rare_max), int(seg_word_freq)
        self.seg_target_vocab, self.join_initials = bool(seg_target_vocab), bool(join_initials)
        self.canon_map, self.canon_min_support, self.canon_min_precision = bool(canon_map), int(canon_min_support), float(canon_min_precision)
        self.canon_max_short_len = int(canon_max_short_len)
        self.junk, self.junk_min_count = bool(junk), int(junk_min_count)
        self.junk_max_hit, self.junk_max_ratio = float(junk_max_hit), float(junk_max_ratio)
        self.name_map: dict[str, str] = {}
        self.addr_map: dict[str, str] = {}
        self.canon: dict[str, dict[str, str]] = {"name": {}, "addr": {}}
        self.junk_tokens: dict[str, set[str]] = {"name": set(), "addr": set()}
        self.fit_counts: Counter = Counter()
        self.stats: dict = {}

    # ---------------- fit ----------------
    @staticmethod
    def _prep(records: pd.DataFrame, col: str) -> pd.Series:
        """basic `clean` + any-script digits -> ASCII of records[col]. Memoised per DataFrame (a reference is kept, so
        the id cannot be reused) because fit() and transform() normally see the same records."""
        key = (id(records), col, len(records))
        hit = _PREP_CACHE.get(key)
        if hit is None or hit[0] is not records:
            if len(_PREP_CACHE) >= 4:
                _PREP_CACHE.clear()
            hit = (records, records[col].astype(str).map(clean).map(_digits_to_ascii))
            _PREP_CACHE[key] = hit
        return hit[1].copy()

    def _native_tokens(self, s: str) -> list[str]:
        return [fold(t) if self.fold_scripts else t for t in s.split() if _is_native(t)]

    def _learn(self, raw: pd.Series, s_rows: np.ndarray, c_rows: np.ndarray, key: str) -> dict[str, str]:
        """Token map from true pairs whose right-side `raw` text is non-ASCII (only those can hold native tokens)."""
        vals = raw.to_numpy()
        sel = np.fromiter((not s.isascii() for s in vals[c_rows]), bool, len(c_rows))
        right = [_digits_to_ascii(clean(s)) for s in vals[c_rows[sel]]]
        left = [_digits_to_ascii(clean(s)) for s in vals[s_rows[sel]]]
        nat = [self._native_tokens(s) for s in right]
        lat = [[t for t in s.split() if t.isascii()] for s in left]
        self.stats[f"{key}_pairs_with_native"] = int(sum(1 for x in nat if x))
        table = learn_token_map(nat, lat, self.min_support, self.min_precision)
        self.stats[f"{key}_map_size"] = len(table)
        return table

    def fit(self, train: Dataset) -> "TranslitNormalizer":
        recs = train.records
        self.stats = {}
        if self.web_names:
            self.fit_counts = Counter(t for s in self._prep(recs, "business_name") for t in s.split())
        if train.gt is None:
            return self
        row = pd.Series(np.arange(len(recs)), index=recs["entity_id"].to_numpy())
        pairs = [(s, c) for s, lst in train.gt.items() for c in lst]
        if not pairs:
            return self
        s_rows = row.reindex([p[0] for p in pairs]).to_numpy()
        c_rows = row.reindex([p[1] for p in pairs]).to_numpy()
        ok = ~(np.isnan(s_rows) | np.isnan(c_rows))
        s_rows, c_rows = s_rows[ok].astype(np.int64), c_rows[ok].astype(np.int64)
        self.name_map = self._learn(recs["business_name"], s_rows, c_rows, "name")
        if self.map_address:
            self.addr_map = self._learn(recs["business_address"], s_rows, c_rows, "addr")
        if self.canon_map or self.junk:
            is_s1 = (recs["source"] == 1).to_numpy()
            for key, col, table, fb in (("name", "business_name", self.name_map, self.addr_map),
                                        ("addr", "business_address", self.addr_map, self.name_map)):
                prepped = self._prep(recs, col).to_numpy()
                left = [prepped[i].split() for i in s_rows]
                right = [self._map_native(prepped[i], table, fb).split() for i in c_rows]
                if self.canon_map:
                    seg = Segmenter(Counter(t for v in prepped for t in v.split()), self.seg_word_freq)
                    cmap = learn_canonical_map(left, right, self.canon_min_support, self.canon_min_precision,
                                               self.canon_max_short_len, lambda w: seg.split(w, 4, 3) is not None)
                    self.canon[key] = cmap
                    self.stats[f"{key}_canon_size"] = len(cmap)
                    left = [[cmap.get(t, t) for t in L] for L in left]
                    right = [[cmap.get(t, t) for t in R] for R in right]
                if self.junk:
                    df_l, df_r = Counter(), Counter()
                    for v, s1 in zip(prepped, is_s1):
                        (df_l if s1 else df_r).update(set(v.split()))
                    jk = learn_junk(left, right, df_l, df_r, self.junk_min_count, self.junk_max_hit, self.junk_max_ratio)
                    self.junk_tokens[key] = set(jk)
                    self.stats[f"{key}_junk"] = sorted(jk)[:100]
                del left, right, prepped
        return self

    # ---------------- transform ----------------
    def _map_native(self, s: str, table: dict[str, str], fallback: dict[str, str]) -> str:
        if s.isascii():
            return s
        out, extra = [], []
        for t in s.split():
            if not _is_native(t):
                out.append(t)
                continue
            f = fold(t) if self.fold_scripts else t
            lat = table.get(f) or fallback.get(f)
            if lat is None:
                out.append(t)
            elif self.mode == "replace":
                out.append(lat)
            else:
                out.append(t)
                extra.append(lat)
        return " ".join(out + extra)

    def _fix_name_token(self, t: str, seg: Segmenter, counts: Counter, web: bool = False) -> str:
        """1/0 -> l/o variant or word segmentation of one name token; on web-style rows every long token is tried."""
        if not t.isascii():
            return t
        if _HAS_NUM_IN_WORD.match(t):
            v = t.replace("1", "l").replace("0", "o")
            if seg.known(v):
                return v
            parts = seg.split(v, *self._seg_limits(web)) if len(v) >= self.seg_min_len else None
            return " ".join(parts) if parts else t
        if len(t) >= self.seg_min_len and t.isalpha() and (web or counts.get(t, 0) <= self.seg_rare_max):
            parts = seg.split(t, *self._seg_limits(web))
            if parts:
                return " ".join(parts)
        return t

    @staticmethod
    def _canonical(v: str, cmap: dict[str, str], junk: set[str]) -> str:
        """Abbreviations -> learned full form; learned junk tokens dropped (the string is kept if nothing would remain)."""
        toks = [cmap.get(t, t) for t in v.split() if t not in junk]
        return " ".join(toks) if toks else v

    @staticmethod
    def _seg_limits(web: bool) -> tuple[int, int]:
        """(max_parts, min_part_len): web names are concatenations by construction (up to 4 words of >= 3 chars);
        elsewhere a rare long token is more often a typo, so only a clean 2-word split (>= 4 chars each) is accepted."""
        return (4, 3) if web else (2, 4)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        out = records.copy()
        raw = records["business_name"].astype(str)
        name = self._prep(records, "business_name")
        addr = self._prep(records, "business_address")
        n_changed = {}
        web = pd.Series(False, index=name.index)
        if self.web_names:                              # "...xyzname.com" -> "xyzname" (suffix never a legal form)
            suf = raw.str.extract(_WEB_SUFFIX, expand=False).str.lower()
            web = suf.notna() & ~suf.isin(self.legal)
            web |= raw.str.contains(_WWW)
            if web.any():
                stripped = raw[web].str.replace(_WWW, "", regex=True).str.replace(_WEB_SUFFIX, "", regex=True)
                name.loc[web] = stripped.map(clean).map(_digits_to_ascii)
            n_changed["web_rows"] = int(web.sum())
        if self.name_map:
            nz = ~name.map(str.isascii)
            name.loc[nz] = name[nz].map(lambda s: self._map_native(s, self.name_map, self.addr_map))
            n_changed["name_native_rows"] = int(nz.sum())
        if self.addr_map:
            nz = ~addr.map(str.isascii)
            addr.loc[nz] = addr[nz].map(lambda s: self._map_native(s, self.addr_map, self.name_map))
            n_changed["addr_native_rows"] = int(nz.sum())
        if self.web_names:                              # segmentation + 1/0 variants on long / mixed tokens
            counts = Counter(self.fit_counts)
            if self.seg_target_vocab:   # per-token max, not sum: same records as in fit() are not double-counted,
                target = Counter(t for s in name for t in s.split())   # and train / val / test see ~equal counts
                for tok, c in target.items():
                    if c > counts.get(tok, 0):
                        counts[tok] = c
            seg = Segmenter(counts, self.seg_word_freq)
            cand = name.str.contains(r"[a-z]{%d,}|[a-z][01]|[01][a-z]" % self.seg_min_len, regex=True) | web
            before = name[cand]
            is_web = web[cand].to_numpy()
            name.loc[cand] = [" ".join(self._fix_name_token(t, seg, counts, w) for t in s.split())
                              for s, w in zip(before, is_web)]
            n_changed["segmented_rows"] = int((name[cand] != before).sum())
        if self.join_initials:
            m = name.str.contains(_SINGLE_RUN_ANY)
            name.loc[m] = name[m].str.replace(_SINGLE_RUN, lambda mo: mo.group(1).replace(" ", ""), regex=True)
            n_changed["initials_rows"] = int(m.sum())
        for key, series in (("name", name), ("addr", addr)):
            cmap, junk = self.canon[key], self.junk_tokens[key]
            if cmap or junk:
                series[:] = [self._canonical(v, cmap, junk) for v in series]
                n_changed[f"{key}_canon_applied"] = True
        self.stats["transform"] = n_changed
        _PREP_CACHE.clear()
        out["name_norm"] = name
        out["addr_norm"] = addr
        legal = self.legal
        out["name_core"] = name.map(lambda s: " ".join(t for t in s.split() if t not in legal) or s)
        out["addr_digits"] = addr.map(lambda s: " ".join(_DIGITS.findall(s)))
        return out
