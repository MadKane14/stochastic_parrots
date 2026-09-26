"""Unit checks for exploration/p4/unsup_canon.py (run: python exploration/p4/test_unsup_canon.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unsup_canon import apply, confident_pairs, merge_maps, mine_canon, side_skew_junk  # noqa: E402


def test_single_letter_and_conflict():
    left = ([["12", "rue", "de", "dieppe"]] * 600 + [["4", "place", "saint", "denis"]] * 300
            + [["3", "avenue", "foch"]] * 60)
    right = ([["12", "r", "de", "dieppe"]] * 600 + [["4", "place", "st", "denis"]] * 300
             + [["3", "av", "foch"]] * 60)
    m = mine_canon(left, right, min_support=50, min_precision=0.5, min_support_1=500, min_precision_1=0.8)
    assert m.get("r") == "rue", m
    assert m.get("st") == "saint", m
    assert m.get("av") == "avenue", m
    merged = merge_maps({"st": "street", "rd": "road"}, m)
    assert merged["st"] == "saint" and merged["rd"] == "road", merged   # target wins, train kept elsewhere


def test_single_letter_needs_support():
    left, right = [["5", "rue", "x"]] * 100, [["5", "r", "x"]] * 100
    assert "r" not in mine_canon(left, right, min_support_1=500)


def test_side_skew_junk():
    s1 = [["12", "rue", "foch"]] * 5000
    right = [["n", "12", "rue", "foch"]] * 3000 + [["12", "rue", "foch"]] * 2000
    junk = side_skew_junk(s1, right, min_ratio=100, min_df_right=1000)
    assert junk == {"n"}, junk
    assert apply(["n", "12", "r", "foch"], {"r": "rue"}, junk) == ["12", "rue", "foch"]


def test_confident_pairs():
    addr = [["12", "rue", "dieppe", "lille"], ["12", "r", "dieppe", "lille"], ["12", "r", "dieppe", "lille"],
            ["99", "rue", "dieppe", "lille"]]
    names = ["pix union sas", "pix union", "other company", "pix union sas"]
    is_s1 = np.array([True, False, False, False])
    s, r = confident_pairs(addr, names, is_s1)
    assert list(zip(s, r)) == [(0, 1)], list(zip(s, r))   # same number+words AND same name; not the other company


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
