"""Feature 5 runtime checks with the exported models (skipped if they are missing)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from core.region.tissue_map import MagnificationEstimator, TissueRegionModel, draw_region_overlay

REGION = ROOT / "models/region_effb0.onnx"
MAG = ROOT / "models/magnification_mnv3.onnx"
BACH = ROOT / "training/data/bach"
OBJ = {"4x": 2.5, "10x": 1.0, "20x": 0.5, "40x": 0.25}


def _photo(label):
    ph = [p for p in json.loads((BACH / "index.json").read_text())["photos"] if p["split"] == "test" and p["label"] == label]
    return cv2.cvtColor(cv2.imread(str(BACH / "photos" / ph[0]["file"])), cv2.COLOR_BGR2RGB)   # 1 µm/px


@pytest.mark.skipif(not REGION.exists() or not (BACH / "index.json").exists(), reason="region model / BACH missing")
def test_region_model_on_held_out_photos():
    rm = TissueRegionModel(REGION)
    inv = rm.predict(_photo(0), 1.0)
    normal = rm.predict(_photo(3), 1.0)
    assert inv.fractions()["invasive"] > 0.4
    assert normal.fractions()["invasive"] + normal.fractions()["in_situ"] < 0.3
    ov = draw_region_overlay(_photo(0), inv)
    assert ov.shape == _photo(0).shape
    assert inv.tumour_prob((50, 60)).shape == (50, 60)


@pytest.mark.skipif(not MAG.exists() or not (BACH / "index.json").exists(), reason="magnification model / BACH missing")
def test_magnification_snaps_to_objective():
    me = MagnificationEstimator(MAG, OBJ)
    img = _photo(0)                                       # 1 µm/px ~ "10x"
    assert me.estimate(img)[0] == "10x"
    four = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)   # 2.5 µm/px
    assert me.estimate(four)[0] == "4x"
    assert me.estimate(np.full((600, 800, 3), 240, np.uint8))[0] is None        # blank glass
