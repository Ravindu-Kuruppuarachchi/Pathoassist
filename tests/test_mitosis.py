"""Feature 6 logic tests: HPF counting, cross-frame de-duplication, mitotic score lookup."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from core.mitosis.counting import TABLE, HPFCounter, mitotic_score
from core.mitosis.detector import MitosisCandidate


def test_score_table_monotonic():
    rows = np.array(TABLE["rows"])
    assert (np.diff(rows[:, 0]) > 0).all() and (np.diff(rows[:, 2]) >= 0).all() and (np.diff(rows[:, 3]) >= 0).all()
    assert (rows[:, 2] < rows[:, 3]).all()


@pytest.mark.parametrize("count,diam,score", [(8, 0.55, 1), (9, 0.55, 2), (17, 0.55, 2), (18, 0.55, 3),
                                              (4, 0.40, 1), (10, 0.40, 3), (13, 0.69, 1), (28, 0.69, 3)])
def test_mitotic_score_lookup(count, diam, score):
    assert mitotic_score(count, diam)["score"] == score


def test_small_camera_field_scaled_and_flagged():
    r = mitotic_score(3, 0.30)                     # camera field smaller than any eyepiece row
    assert r["score1_max"] < 4 and any("outside the table" in n for n in r["notes"])


@pytest.fixture(scope="module")
def slide():
    p = ROOT / "training/data/pannuke/fold3_images.npy"
    if not p.exists():
        pytest.skip("PanNuke not available")
    X = np.load(p, mmap_mode="r")
    rng = np.random.default_rng(1)
    return np.vstack([np.hstack([X[i] for i in rng.integers(0, len(X), 12)]) for _ in range(8)])


def _cands(slide_pts, y0, x0, fh, fw):
    """Candidates for the slide points that fall inside the frame at (y0, x0)."""
    return [MitosisCandidate(x - x0, y - y0, 0.9, 0.9, True) for x, y in slide_pts
            if x0 <= x < x0 + fw and y0 <= y < y0 + fh]


def test_hpf_counting_dedups_across_overlapping_frames(slide):
    fh, fw = 600, 800
    pts = [(300, 300), (700, 350), (1000, 400), (1500, 900), (2400, 1500)]   # slide px (x, y)
    c = HPFCounter(um_per_px=0.25, frame_shape=(fh, fw))
    r1 = c.add_frame(slide[100:100 + fh, 100:100 + fw], _cands(pts, 100, 100, fh, fw))
    assert r1["accepted"] and len(r1["new"]) == 2                          # (300,300), (700,350)
    # second frame shifted right by 500 px: 37.5% overlap, contains (700,350) again and (1000,400)
    r2 = c.add_frame(slide[100:100 + fh, 600:600 + fw], _cands(pts, 100, 600, fh, fw))
    assert r2["accepted"] and len(r2["new"]) == 1 and len(r2["duplicates"]) == 1
    # >50% overlap with frame 1 -> not a new HPF
    r3 = c.add_frame(slide[150:150 + fh, 200:200 + fw], _cands(pts, 150, 200, fh, fw))
    assert not r3["accepted"]
    # unrelated field -> new HPF, new island
    r4 = c.add_frame(slide[1300:1300 + fh, 2000:2000 + fw], _cands(pts, 1300, 2000, fh, fw))
    assert r4["accepted"] and len(r4["new"]) == 1
    res = c.result()
    assert res["hpfs_counted"] == 3 and res["mitoses"] == 4 and res["warnings"]


def test_accept_reject_add(slide):
    c = HPFCounter(0.25, (600, 800))
    c.add_frame(slide[:600, :800], [MitosisCandidate(100, 100, .9, .9, True), MitosisCandidate(400, 300, .8, .7, True),
                                    MitosisCandidate(500, 500, .3, .2, False)])
    assert c.result()["mitoses"] == 2
    c.set_status(0, "rejected")
    c.add_manual(0, 700, 100)
    r = c.result()
    assert r["mitoses"] == 2 and r["rejected"] == 1 and r["added"] == 1
    assert abs(c.hpf_area_mm2 - 0.03) < 1e-9
