"""Unit checks for ber.normalize.p2_translit (run: python exploration/p2/test_translit.py)."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.data import Dataset  # noqa: E402
from ber.normalize.p2_translit import Segmenter, TranslitNormalizer, fold, learn_token_map, _is_native  # noqa: E402


def test_linking():
    # प्राइवेट and लिमिटेड always co-occur with both "private" and "limited"; only competitive linking separates them
    nat = [["सूर्य", "प्राइवेट", "लिमिटेड"], ["स्टार", "प्राइवेट", "लिमिटेड"], ["सूर्य", "लिमिटेड"],
           ["स्टार", "प्राइवेट", "लिमिटेड"], ["सूर्य", "प्राइवेट", "लिमिटेड"]]
    lat = [["surya", "private", "limited"], ["star", "private", "limited"], ["surya", "limited"],
           ["star", "private", "limited"], ["surya", "private", "limited"]]
    m = learn_token_map(nat, lat, min_support=2, min_precision=0.5)
    assert m == {"सूर्य": "surya", "स्टार": "star", "प्राइवेट": "private", "लिमिटेड": "limited"}, m


def test_fold():
    # Bengali / Odia / Devanagari "shri" fold to the same Devanagari string
    assert fold("শ্রী") == fold("श्री") == fold("ଶ୍ରୀ"), (fold("শ্রী"), fold("श्री"), fold("ଶ୍ରୀ"))
    assert fold("१२३") == "123"
    assert _is_native("श्री") and not _is_native("tránsformer") and not _is_native("abc")


def test_segmenter():
    seg = Segmenter(Counter({"supreme": 50, "packaging": 40, "pediatric": 30, "medicine": 60, "sup": 25}), 20)
    assert seg.split("supremepackaging") == ["supreme", "packaging"]
    assert seg.split("pediatricmedicine") == ["pediatric", "medicine"]
    assert seg.split("qwertyuiop") is None


def test_transform():
    recs = pd.DataFrame({
        "entity_id": ["S1-1", "S2-1", "S1-2", "S2-2", "S2-3", "S2-4", "S2-5"],
        "source": [1, 2, 1, 2, 2, 2, 2],
        "business_name": ["Supreme Packaging LLC", "supremepackaging.com", "King Gamco PLLC", "king gamco p l l c",
                          "Dr dreammarketing.Com", "Shree Traders Pvt.Ltd", "cequee1ement"],
        "business_address": ["1 Main St", "", "2 Oak Rd", "", "", "", ""],
        "country": ["US"] * 7})
    for i in range(30):   # vocabulary for segmentation
        recs.loc[len(recs)] = [f"S3-{i}", 3, "dream marketing supreme packaging element", "", "US"]
    n = TranslitNormalizer(legal_tokens=["llc", "pllc", "ltd", "pvt"])
    n.fit(Dataset("t", recs, None))
    out = n.transform(recs).set_index("entity_id")["name_norm"]
    assert out["S2-1"] == "supreme packaging", out["S2-1"]
    assert out["S2-2"] == "king gamco pllc", out["S2-2"]
    assert out["S2-3"] == "dr dream marketing", out["S2-3"]
    assert out["S2-4"] == "shree traders pvt ltd", out["S2-4"]      # 'Pvt.Ltd' is not a web suffix
    assert out["S2-5"] == "cequeelement" or out["S2-5"] == "cequee1ement", out["S2-5"]


def test_subset_consistency():
    # a record must normalize the same whether the target set is all records or a subset (train vs val vs test)
    recs = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(60)], "source": [2] * 60,
                         "business_name": ["healthline.com", "Healthline Care LLC", "supremepackaging"] * 20,
                         "business_address": [""] * 60, "country": ["US"] * 60})
    for i in range(25):
        recs.loc[len(recs)] = [f"S3-{i}", 3, "health line care supreme packaging", "", "US"]
    n = TranslitNormalizer(legal_tokens=["llc"])
    n.fit(Dataset("t", recs, None))
    full = n.transform(recs)["name_norm"].iloc[:30].tolist()
    part = n.transform(recs.iloc[:30].copy())["name_norm"].tolist()
    assert full == part, (full[:3], part[:3])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
