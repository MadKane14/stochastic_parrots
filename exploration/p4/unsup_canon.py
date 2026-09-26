"""T14b (p4 for p2): label-free target canonicalization — the implementation lives in
code/business_entity_resolution/src/ber/normalize/p4_tcanon.py (normalizer `p4_tcanon`); re-exported here for the
exploration scripts and tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "business_entity_resolution" / "src"))
from ber.normalize.p4_tcanon import apply, confident_pairs, merge_maps, mine_canon, side_skew_junk  # noqa: E402,F401
