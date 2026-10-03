"""Feature 8 sanity tests on a real BreCaHAD field (skipped if the data / model are missing)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

MODEL = ROOT / "models/tubule_lumen_lgbm.txt"
IMG = ROOT / "training/data/brecahad/jpg/Case_6-01.jpg"


@pytest.mark.skipif(not (MODEL.exists() and IMG.exists()), reason="lumen model / BreCaHAD missing")
def test_tubule_field_and_session():
    from core.tubules.lumen import LumenClassifier, analyze_field
    from core.tubules.score import TubuleSession
    clf = LumenClassifier(MODEL)
    im = cv2.cvtColor(cv2.imread(str(IMG)), cv2.COLOR_BGR2RGB)        # gland-forming tumour, 0.318 µm/px
    f = analyze_field(im, 0.318, clf)
    assert 0 < f.tubule_area_mm2 <= f.tumour_area_mm2
    assert f.lumens.sum() >= 3                                         # the field has ~11 annotated lumens
    s = TubuleSession(clf, None, min_tumour_mm2=0.01)
    assert s.add_field(im, 0.318)["accepted"]
    assert not s.add_field(im, 0.318)["accepted"]                      # same field again
    r = s.result()
    assert r.score in (1, 2, 3) and 0 < r.tubule_pct <= 100
    blank = analyze_field(np.full((600, 800, 3), 245, np.uint8), 1.0, clf)
    assert blank.tumour_area_mm2 == 0 and blank.lumens.sum() == 0
