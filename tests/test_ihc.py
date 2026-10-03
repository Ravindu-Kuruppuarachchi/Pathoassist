"""Feature 2 logic tests: scoring rules, calibration, stain separation, de-duplication, case session."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from core.ihc.case import CaseSession
from core.ihc.dedup import FieldDeduplicator
from core.ihc.pipeline import FieldAnalysis
from core.ihc.scoring import IntensityThresholds, allred_proportion, calibrate, score_counts, status
from core.ihc.stain import StainMatrix, estimate_hdab_matrix, per_nucleus_stain


# ----------------------------------------------------------------- scoring
@pytest.mark.parametrize("pct,ps,st", [(0, 0, "Negative"), (0.5, 1, "Negative"), (1, 2, "Low positive"),
                                       (10, 2, "Low positive"), (10.5, 3, "Positive"), (33, 3, "Positive"),
                                       (50, 4, "Positive"), (66.5, 5, "Positive"), (100, 5, "Positive")])
def test_proportion_and_status(pct, ps, st):
    assert allred_proportion(pct) == ps and status(pct) == st


def test_score_counts():
    s = score_counts([400, 200, 200, 200], min_cells=500)   # 1000 tumour nuclei, 60% positive
    assert s.pct_positive == 60 and s.h_score == 120        # 20 + 2*20 + 3*20
    assert s.allred_ps == 4 and s.allred_is == 2 and s.allred_total == 6
    assert s.status == "Positive" and not s.warnings
    low = score_counts([300, 3, 0, 0])
    assert low.status == "Negative" and low.allred_ps == 1 and any("< 500" in w for w in low.warnings)
    assert score_counts([0, 0, 0, 0]).status == "Insufficient"


def test_calibration_recovers_thresholds():
    rng = np.random.default_rng(0)
    true = IntensityThresholds(0.12, 0.3, 0.55)
    od = rng.uniform(0, 0.9, 240)
    y = true.bin(od)
    y[rng.choice(240, 12, replace=False)] = rng.integers(0, 4, 12)   # 5% pathologist disagreement
    th, rep = calibrate(od, y)
    assert th.calibrated and rep["weighted_kappa"] > 0.9
    assert abs(th.weak - 0.12) < 0.05 and abs(th.moderate - 0.3) < 0.05 and abs(th.strong - 0.55) < 0.05


# ----------------------------------------------------------------- stain
def _synth(M, h, d):
    od = np.stack([h, d, np.zeros_like(h)], -1) @ M.rgb_from_stain
    return np.clip(255 * 10 ** (-od), 0, 255).round().astype(np.uint8)


def test_deconvolution_roundtrip():
    M = StainMatrix()
    rng = np.random.default_rng(1)
    h, d = rng.uniform(0.05, 0.8, (64, 64)), rng.uniform(0, 0.8, (64, 64))
    c = M.concentrations(_synth(M, h, d))
    assert np.abs(c[..., 0] - h).mean() < 0.01 and np.abs(c[..., 1] - d).mean() < 0.01


def test_estimate_matrix_recovers_lab_stains():
    lab = StainMatrix(h=[0.55, 0.75, 0.37], dab=[0.30, 0.60, 0.74])   # a lab's own (non-Ruifrok) vectors
    rng = np.random.default_rng(2)
    h = rng.uniform(0.1, 0.9, (200, 200))
    d = rng.uniform(0, 0.9, (200, 200)) * (rng.random((200, 200)) < 0.4)
    est = estimate_hdab_matrix([_synth(lab, h, d)])
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    assert cos(est.rgb_from_stain[0], lab.rgb_from_stain[0]) > 0.99
    assert cos(est.rgb_from_stain[1], lab.rgb_from_stain[1]) > 0.99


def test_per_nucleus_stain():
    labels = np.zeros((10, 10), np.int32)
    labels[:5, :5], labels[5:, 5:] = 1, 2
    conc = np.zeros((10, 10, 3), np.float32)
    conc[:5, :5, 1], conc[5:, 5:, 1] = 0.2, 0.7
    h, d = per_nucleus_stain(labels, conc)
    assert np.allclose(d, [0.2, 0.7])


# ----------------------------------------------------------------- dedup
@pytest.fixture(scope="module")
def slide():
    p = ROOT / "training/data/pannuke/fold3_images.npy"
    if not p.exists():
        pytest.skip("PanNuke not available")
    X = np.load(p, mmap_mode="r")
    rng = np.random.default_rng(0)
    rows = [np.hstack([X[i] for i in rng.integers(0, len(X), 12)]) for _ in range(8)]
    return np.vstack(rows)   # 2048 x 3072 tissue mosaic


def test_dedup_overlap_rule(slide):
    fh, fw = 720, 960
    frame = lambda y, x: slide[y:y + fh, x:x + fw]
    dd = FieldDeduplicator(0.5)
    dd.add(frame(100, 100))
    assert dd.check(frame(100, 100)) is not None                    # same field
    ov = dd.check(frame(100 + 150, 100 + 200))                      # ~0.64 overlap
    assert ov is not None and abs(ov.fraction - (1 - 150 / fh) * (1 - 200 / fw)) < 0.05
    assert dd.check(frame(100 + 500, 100 + 100)) is None           # ~0.27 overlap -> new field
    assert dd.check(frame(1200, 2000)) is None                     # unrelated field


# ----------------------------------------------------------------- case session
def _analysis(n_tumour_pos, n_tumour_neg, n_other, seed=0):
    rng = np.random.default_rng(seed)
    n = n_tumour_pos + n_tumour_neg + n_other
    labels = np.zeros((200, 300), np.int32)
    pts = np.stack([rng.uniform(10, 190, n), rng.uniform(10, 290, n)], 1)
    for i, (y, x) in enumerate(pts.astype(int)):
        labels[y - 2:y + 3, x - 2:x + 3] = i + 1
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    coords = pts[:, :, None] + 3 * np.stack([np.sin(ang), np.cos(ang)])[None]
    ct = np.array([0] * (n_tumour_pos + n_tumour_neg) + [1] * n_other, np.int8)
    inten = np.array([2] * n_tumour_pos + [0] * (n_tumour_neg + n_other), np.int8)
    prob = np.eye(4, dtype=np.float32)[ct]
    return FieldAnalysis(labels=labels, points=pts, coords=coords, cell_type=ct, type_prob=prob,
                         dab_od=np.where(inten > 0, 0.45, 0.05).astype(np.float32),
                         h_od=np.full(n, 0.3, np.float32), intensity=inten, um_per_px=0.25)


def test_case_session_totals_corrections_and_report(tmp_path):
    rng = np.random.default_rng(0)
    s = CaseSession(tmp_path, "TEST-001", "ER", min_cells=500)
    f1 = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    f2 = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    assert s.add_field(f1, _analysis(30, 60, 10)).accepted
    assert not s.add_field(f1, _analysis(30, 60, 10)).accepted      # duplicate frame is skipped
    assert s.add_field(f2, _analysis(10, 80, 10, seed=1)).accepted
    sc = s.score()
    assert sc.n_tumour == 180 and sc.counts == [140, 0, 40, 0] and any("< 500" in w for w in sc.warnings)

    a = s.field(0).analysis
    i = s.nucleus_at(0, *a.points[0])                                # click on a positive tumour nucleus
    assert i == 0
    s.toggle_positive(0, i)                                          # -> negative
    s.toggle_tumour(0, 1)                                            # tumour -> lymphocyte
    s.toggle_tumour(0, 95)                                           # lymphocyte -> tumour
    sc2 = s.score()
    assert sc2.counts == [142, 0, 38, 0]   # +1 neg (flipped), -1 mod (flipped), -1 mod (now lymph), +1 neg (now tumour)
    log = [json.loads(l) for l in open(tmp_path / "corrections.jsonl")]
    assert [r["kind"] for r in log] == ["intensity", "cell_type", "cell_type"]
    assert log[1]["old"] == "tumour" and log[1]["new"] == "lymphocyte"

    from core.ihc.report import write_report
    pdf = write_report(s, tmp_path / "report.pdf", lab="Test lab")
    assert pdf.stat().st_size > 5000 and pdf.read_bytes()[:4] == b"%PDF"
