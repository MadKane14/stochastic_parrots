"""Script-agnostic romanization + consonant skeletons (p1), stdlib only (unicodedata character names).

Non-Latin text is romanized from Unicode character names (`DEVANAGARI LETTER KA` -> "k" + inherent "a",
`... VOWEL SIGN E` -> "e", virama drops the inherent vowel, anusvara -> "n", digits -> 0-9). Works identically
for every script that follows these naming conventions (Devanagari, Gujarati, Bengali, Tamil, Telugu, Kannada,
Malayalam, Odia, Gurmukhi, ...); no dictionaries, no language or country logic. The consonant skeleton (first
letter of each token + its consonants, aspiration and doubling collapsed) makes romanized and Latin spellings
comparable: "Red Consultants" and "रेड कंसल्टेंट्स" both become "rd knsltnts".
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
_LET = re.compile(r"^[A-Z\- ]*? LETTER (?:VOCALIC )?([A-Z]+)$")
_VS = re.compile(r"^[A-Z\- ]*? VOWEL SIGN (?:VOCALIC )?([A-Z]+)$")
_DIG = re.compile(r"DIGIT ([A-Z]+)$")
_NUM = {"ZERO":"0","ONE":"1","TWO":"2","THREE":"3","FOUR":"4","FIVE":"5","SIX":"6","SEVEN":"7","EIGHT":"8","NINE":"9"}
VOW = set("AEIOU")
@lru_cache(maxsize=None)
def char_rom(ch: str) -> tuple[str, str]:
    """(kind, text) for one character; kind in ascii/cons/vowel/vsign/virama/nasal/digit/other/skip."""
    if ch.isascii(): return ("ascii", ch.lower())
    nm = unicodedata.name(ch, "")
    if not nm: return ("other", " ")
    m = _DIG.search(nm)
    if m and m.group(1) in _NUM: return ("digit", _NUM[m.group(1)])
    if "VIRAMA" in nm or nm.endswith("SIGN HALANT") or "PULLI" in nm or "CHANDRAKKALA" in nm: return ("virama", "")
    if "ANUSVARA" in nm or "CANDRABINDU" in nm: return ("nasal", "n")
    if "VISARGA" in nm: return ("other", "h")
    if "NUKTA" in nm or "ZERO WIDTH" in nm or "AVAGRAHA" in nm: return ("skip", "")
    m = _VS.match(nm)
    if m: return ("vsign", m.group(1).lower())
    m = _LET.match(nm)
    if m:
        x = m.group(1)
        if "LATIN" in nm or "CYRILLIC" in nm or "GREEK" in nm: return ("ascii", unicodedata.normalize("NFKD", ch)[0].lower())
        if x[0] in VOW: return ("vowel", x.lower())
        return ("cons", x[:-1].lower() if x.endswith("A") and len(x) > 1 else x.lower())
    cat = unicodedata.category(ch)
    if cat[0] == "M": return ("skip", "")
    return ("other", " " if cat[0] in "PZS" else "")
def romanize(s: str) -> str:
    """Romanize non-ASCII characters of `s` (ASCII passes through, lowercased)."""
    out, pend = [], False   # pend: previous consonant still owns its inherent 'a'
    for ch in s:
        k, t = char_rom(ch)
        if k == "cons":
            if pend: out.append("a")
            out.append(t); pend = True
        elif k == "vsign": out.append(t); pend = False
        elif k == "virama": pend = False
        else:
            if pend: out.append("a")
            pend = False; out.append(t)
    if pend: out.append("a")
    return "".join(out)
_SK = [(re.compile(p), r) for p, r in [(r"[^a-z0-9 ]", ""), (r"(?<=[a-z])h", ""), (r"ph", "f"), (r"[cq]", "k"), (r"x", "ks"),
       (r"w", "v"), (r"z", "j"), (r"y", "i"), (r"(.)\1+", r"\1")]]
def skeleton(s: str) -> str:
    """Consonant skeleton of a (romanized) string: keep a token's first letter, drop later vowels."""
    s = s.lower()
    for p, r in _SK: s = p.sub(r, s)
    return " ".join(t[0] + re.sub(r"[aeiou]", "", t[1:]) for t in s.split())


def name_skeleton(s: str) -> str:
    """Skeleton of a normalized field, romanizing it first only when it contains non-ASCII characters."""
    return skeleton(s if s.isascii() else romanize(s))
