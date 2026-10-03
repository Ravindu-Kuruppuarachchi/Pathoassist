"""Feature 4 tests: ASCO/CAP rule engine, cell categories, membrane measurement, calibration."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from core.her2.membrane import cells_from_nuclei, membrane_sectors
from core.her2.scoring import HER2Thresholds, calibrate_cells, classify_cells, her2_score
from core.ihc.stain import StainMatrix


# ------------------------------------------------------------------ rule engine
@pytest.mark.parametrize("counts,score,sub", [
    ([1000, 0, 0, 0], "0", "null"),
    ([950, 50, 0, 0], "0", "ultralow"),        # 5% faint incomplete
    ([900, 100, 0, 0], "0", "ultralow"),       # exactly 10% -> not > 10%
    ([890, 110, 0, 0], "1+", ""),
    ([700, 200, 100, 0], "1+", ""),            # complete exactly 10%
    ([700, 190, 110, 0], "2+", ""),
    ([600, 200, 150, 50], "2+", ""),           # strong 5%, complete 20%
    ([600, 200, 100, 100], "2+", ""),          # strong exactly 10%
    ([600, 200, 90, 110], "3+", ""),
])
def test_rule_engine(counts, score, sub):
    r = her2_score(counts)
    assert (r.score, r.subgroup) == (score, sub)
    assert r.reflex_ish == (score == "2+")


def test_flags_and_categories():
    r = her2_score([880, 120, 0, 0])                         # 12% any staining
    assert r.score == "1+" and r.her2_low == "HER2-low" and any("0 / 1+ boundary" in f for f in r.flags)
    r = her2_score([950, 20, 0, 30])                         # strong in 3%
    assert r.score == "0" and any("heterogeneous" in f for f in r.flags)
    r = her2_score([800, 200, 0, 0], n_intense_incomplete=150)  # 15% intense incomplete
    assert any("unusual pattern" in f for f in r.flags)
    r = her2_score([100, 10, 0, 0])
    assert any("< 500" in f for f in r.flags)
    assert her2_score([600, 200, 150, 50]).her2_low.startswith("HER2-low if ISH negative")
    assert not her2_score([1000, 0, 0, 0]).rules_verified


def test_classify_cells():
    th = HER2Thresholds(stain_od=0.1, strong_min=0.5, complete_min=0.75, faint_max=0.2)
    S = np.full((5, 16), 0.02, np.float32)
    S[1, :4] = 0.15                      # faint, 25% of ring -> faint/incomplete
    S[2, :] = 0.3                        # complete, moderate -> 2
    S[3, :] = 0.8                        # complete, strong -> 3
    S[4, :6] = 0.9                       # strong but incomplete -> 1 + intense_incomplete
    cat, ii, comp, inten, meas = classify_cells(S, th)
    assert cat.tolist() == [0, 1, 2, 3, 1] and ii.tolist() == [False, False, False, False, True]
    S2 = S.copy()
    S2[0, :10] = np.nan                  # ring mostly outside the frame
    assert not classify_cells(S2, th)[4][0]


# ------------------------------------------------------------------ membrane measurement on synthetic cells
def _synthetic(um_per_px=0.5):
    """Grid of 6x6 cells: nucleus (H) + membrane ring (DAB) with known completeness/intensity."""
    M = StainMatrix()
    H, W = 420, 420
    h_od, d_od = np.zeros((H, W)), np.zeros((H, W))
    nuclei = np.zeros((H, W), np.int32)
    truth, centres = [], []
    rng = np.random.default_rng(0)
    k = 0
    for gy in range(6):
        for gx in range(6):
            cy, cx = 35 + gy * 70, 35 + gx * 70
            k += 1
            cv2.circle(nuclei, (cx, cy), 9, k, -1)
            cv2.circle(h_od, (cx, cy), 9, 0.5, -1)
            comp = [0, 0.25, 0.5, 1.0, 1.0, 1.0][gx]
            inten = [0, 0.15, 0.3, 0.15, 0.35, 0.8][gy] if comp else 0
            if comp:
                cv2.ellipse(d_od, (cx, cy), (22, 22), 0, 0, 360 * comp, inten, 3)
            truth.append((comp, inten))
            centres.append((cy, cx))
    od = np.stack([h_od, d_od, np.zeros_like(h_od)], -1) @ M.rgb_from_stain
    rgb = np.clip(255 * 10 ** (-od) + rng.normal(0, 1.5, od.shape), 0, 255).astype(np.uint8)
    return rgb, nuclei, np.array(centres, float), truth, M


def test_membrane_completeness_and_intensity():
    rgb, nuclei, centres, truth, M = _synthetic()
    dab = M.concentrations(rgb)[..., 1]
    cells = cells_from_nuclei(nuclei, dab, um_per_px=0.5, max_um=9)   # isolated cells (as if bordered by stroma)
    S, ring = membrane_sectors(cells, nuclei, dab, centres, um_per_px=0.5, ring_um=1.5)
    th = HER2Thresholds(stain_od=0.07, strong_min=0.5, complete_min=0.75, faint_max=0.2)
    cat, ii, comp, inten, meas = classify_cells(S, th)
    for i, (c, it) in enumerate(truth):
        if c == 0 or it == 0:
            assert cat[i] == 0, i
        else:
            assert abs(comp[i] - c) <= 0.2, (i, comp[i], c)
            expect = 1 if c < 0.75 or it < 0.2 else (3 if it >= 0.5 else 2)
            assert cat[i] == expect, (i, cat[i], expect, comp[i], inten[i])


def test_calibrate_cells_recovers_thresholds():
    rng = np.random.default_rng(3)
    true = HER2Thresholds(stain_od=0.12, strong_min=0.55, complete_min=0.75, faint_max=0.2)
    S = []
    for _ in range(400):
        s = np.full(16, rng.uniform(0, 0.06))
        n_st = rng.integers(0, 17)
        s[rng.permutation(16)[:n_st]] = rng.uniform(0.1, 1.0)
        S.append(s.astype(np.float32))
    y = classify_cells(np.stack(S), true, min_valid_sectors=0)[0]
    th, rep = calibrate_cells(S, y)
    assert rep["weighted_kappa"] > 0.95
