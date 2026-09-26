"""Unit checks for ber.normalize.p2_translit (run: python exploration/p2/test_translit.py)."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.data import Dataset  # noqa: E402
from ber.normalize.p2_translit import (Segmenter, TranslitNormalizer, fold, is_abbrev, learn_canonical_map,  # noqa: E402
                                      learn_junk, learn_token_map, _is_native)


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


def test_abbrev():
    for a, b in [("st", "street"), ("pvt", "private"), ("ltd", "limited"), ("va", "virginia"), ("rd", "road"),
                 ("ave", "avenue"), ("tn", "tennessee")]:
        assert is_abbrev(a, b), (a, b)
    for a, b in [("frhght", "freight"), ("inc", "llc"), ("street", "st"), ("road", "rd"), ("ts", "street"), ("va", "va")]:
        assert not is_abbrev(a, b), (a, b)


def test_canonical_map():
    left = [["10", "forest", "avenue", "virginia"], ["2", "oak", "street", "virginia"], ["5", "elm", "avenue", "ohio"]] * 30
    right = [["10", "forest", "ave", "va"], ["2", "oak", "st", "va"], ["5", "elm", "ave", "ohio"]] * 30
    right[0] = ["10", "forrest", "ave", "va"]          # a typo must not be learned
    m = learn_canonical_map(left, right, min_support=20, min_precision=0.5)
    assert m == {"ave": "avenue", "va": "virginia", "st": "street"}, m
    # glued web names are not full forms: "eye" must not map to "eyecare" (= eye + care), long "short" forms rejected
    left = [["eye", "care", "clinic"], ["pediatric", "dental"]] * 30
    right = [["eyecare", "clinic"], ["pediatricdental"]] * 30
    seg = Segmenter(Counter({"eye": 60, "care": 60, "clinic": 60, "pediatric": 60, "dental": 60}), 20)
    m = learn_canonical_map(left, right, 20, 0.5, 5, lambda w: seg.split(w, 4, 3) is not None)
    assert m == {}, m


def test_junk():
    left = [["abc", "traders"]] * 50 + [["xyz", "co"]] * 50
    right = [["abc", "traders", "null"]] * 50 + [["xyz", "co", "null"]] * 50
    df_l, df_r = {"abc": 50, "traders": 50}, {"null": 5000, "abc": 50}
    j = learn_junk(left, right, df_l, df_r, min_count=50, max_hit=0.01, max_ratio=0.01)
    assert set(j) == {"null"}, j


def test_transform_canon_junk():
    rows = []
    for i in range(60):
        rows += [(f"S1-{i}", 1, f"Acme{i} Private Limited", f"{i} Forest Avenue, Virginia"),
                 (f"S2-{i}", 2, f"Acme{i} Pvt Ltd", f"{i} Forest Ave, VA, NULL")]
    recs = pd.DataFrame(rows, columns=["entity_id", "source", "business_name", "business_address"])
    recs["country"] = "US"
    for i in range(3000):
        recs.loc[len(recs)] = [f"S3-{i}", 3, f"other{i}", f"{i} null", "US"]
    gt = {f"S1-{i}": frozenset([f"S2-{i}"]) for i in range(60)}
    n = TranslitNormalizer(legal_tokens=["pvt", "private", "ltd", "limited"], canon_map=True, canon_min_support=20,
                           junk=True, junk_min_count=50)
    n.fit(Dataset("t", recs, gt))
    out = n.transform(recs).set_index("entity_id")
    assert out.loc["S2-7", "name_norm"] == "acme7 private limited", out.loc["S2-7", "name_norm"]
    assert out.loc["S2-7", "addr_norm"] == "7 forest avenue virginia", out.loc["S2-7", "addr_norm"]
    assert out.loc["S1-7", "addr_norm"] == "7 forest avenue virginia"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
