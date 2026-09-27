"""Unit checks for p4_tcanon2's helpers (run: python exploration/p4/test_tcanon2.py)."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.normalize.p4_tcanon2 import address_segments, remove_once, split_glued  # noqa: E402

def test_segments():
    assert address_segments("63 R. DE DIEPPE, LILLE (59), Pas-de-Calais") == ["63 r de dieppe", "lille", "pas de calais", "59"]
    assert address_segments("") == []

def test_split_glued():
    assert split_glued(["6b", "rue", "12", "a1", "23805c", "bb"]) == ["6", "b", "rue", "12", "a1", "23805", "c", "bb"]

def test_remove_once():
    assert remove_once(["9", "rue", "de", "calais", "calais", "pas", "de", "calais"], Counter("pas de calais".split())) == \
        ["9", "rue", "de", "calais", "calais"]

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
