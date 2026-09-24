"""Sanity tests for the frozen metric + splits. Run: python tests/test_metrics.py (no pytest needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ber.data import Dataset  # noqa: E402
from ber.eval.metrics import entity_fbeta, macro_fbeta  # noqa: E402
from ber.eval.splits import make_split  # noqa: E402


def test_entity_fbeta():
    # worked example from the problem statement: P=2/3, R=1 -> 0.714
    assert abs(entity_fbeta({"a", "b", "c"}, {"a", "c"}) - 0.7142857) < 1e-6
    assert entity_fbeta(set(), set()) == 1.0          # correct singleton
    assert entity_fbeta({"a"}, set()) == 0.0          # false merge on a singleton
    assert entity_fbeta(set(), {"a"}) == 0.0          # missed everything
    assert entity_fbeta({"x"}, {"a"}) == 0.0          # wrong match
    assert abs(entity_fbeta({"a"}, {"a", "b"}) - (1.25 * 1 * 0.5 / (0.25 * 1 + 0.5))) < 1e-9


def test_macro_includes_singletons():
    gt = {"S1-1": frozenset(), "S1-2": frozenset({"S2-1"})}
    assert macro_fbeta({"S1-2": {"S2-1"}}, gt, ["S1-1", "S1-2"]) == 1.0
    assert macro_fbeta({"S1-1": {"S2-9"}, "S1-2": {"S2-1"}}, gt, ["S1-1", "S1-2"]) == 0.5


def test_split_grouping():
    rows = [[f"S1-{i}", 1, "n", "a", "US" if i % 2 else "India"] for i in range(200)]
    rows += [[f"S2-{i}", 2, "n", "a", "US"] for i in range(300)]
    recs = pd.DataFrame(rows, columns=["entity_id", "source", "business_name", "business_address", "country"])
    gt = {f"S1-{i}": frozenset({f"S2-{i}"} if i < 150 else set()) for i in range(200)}
    full = Dataset("train", recs, gt)
    tr, va = make_split(full, "val_v1")
    assert not set(tr.records.entity_id) & set(va.records.entity_id)
    for s, lst in va.gt.items():                       # val matches live in the val pool
        assert lst <= set(va.records.entity_id) and lst == gt[s]
    assert 0.15 < len(va.s1) / 200 < 0.25
    tr2, va2 = make_split(full, "val_v1")               # deterministic
    assert set(va2.records.entity_id) == set(va.records.entity_id)
    tr3, va3 = make_split(full, "xc_India")
    assert set(va3.s1.country) == {"India"} and set(tr3.s1.country) == {"US"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
