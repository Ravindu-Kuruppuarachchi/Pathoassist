"""Feature 3 tests: Ki-67 positivity calibration, global / hotspot index, session and report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from core.ihc.pipeline import FieldAnalysis
from core.ki67.scoring import band, calibrate_positivity, global_index, hotspot_index
from core.ki67.session import Ki67Session


@pytest.mark.parametrize("pct,b", [(0, "Low"), (5, "Low"), (5.1, "Intermediate"), (29.9, "Intermediate"),
                                   (30, "High"), (80, "High")])
def test_ikwg_bands(pct, b):
    assert band(pct).startswith(b)


def test_global_pooled_and_weighted():
    counts = [[90, 10], [50, 50], [300, 100]]          # [neg, pos] per field
    g = global_index(counts)
    assert g.n_counted == 600 and g.n_positive == 160 and g.index_pct == pytest.approx(26.67, abs=0.01)
    assert not g.warnings
    w = global_index(counts, weights=[0.5, 0.25, 0.25])  # 0.5*10 + 0.25*50 + 0.25*25
    assert w.mode == "global (weighted)" and w.index_pct == pytest.approx(23.75)
    assert global_index([[100, 20]]).warnings             # < 500 nuclei


def test_hotspot_takes_highest_fields_until_500():
    counts = [[180, 20], [100, 100], [150, 150], [400, 10]]   # field indices 10%, 50%, 50%, 2.4%
    h = hotspot_index(counts, field_ids=[0, 1, 2, 3])
    assert h.fields_used == [1, 2] and h.n_counted == 500 and h.index_pct == 50.0
    assert any("pooled over 2 fields" in w for w in h.warnings)
    single = hotspot_index([[100, 500], [500, 5]])
    assert single.fields_used == [0] and single.index_pct == pytest.approx(83.33, abs=0.01)
    small = hotspot_index([[50, 50]])
    assert any("unreliable" in w for w in small.warnings)


def test_calibrate_positivity():
    rng = np.random.default_rng(0)
    od = np.concatenate([rng.normal(0.25, 0.08, 150), rng.normal(0.8, 0.15, 60)])
    y = np.r_[np.zeros(150, bool), np.ones(60, bool)]
    th, rep = calibrate_positivity(od, y)
    assert 0.4 < th.cut < 0.6 and rep["balanced_accuracy"] > 0.97 and th.n_calibration == 210


def _analysis(n_pos, n_neg, n_other=0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg + n_other
    labels = np.zeros((300, 400), np.int32)
    pts = np.stack([rng.uniform(10, 290, n), rng.uniform(10, 390, n)], 1)
    for i, (y, x) in enumerate(pts.astype(int)):
        labels[y - 2:y + 3, x - 2:x + 3] = i + 1
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    coords = pts[:, :, None] + 3 * np.stack([np.sin(ang), np.cos(ang)])[None]
    ct = np.array([0] * (n_pos + n_neg) + [1] * n_other, np.int8)
    inten = np.array([1] * n_pos + [0] * (n_neg + n_other), np.int8)
    return FieldAnalysis(labels=labels, points=pts, coords=coords, cell_type=ct, type_prob=np.eye(4, dtype=np.float32)[ct],
                         dab_od=np.where(inten > 0, 0.8, 0.1).astype(np.float32), h_od=np.full(n, 0.3, np.float32),
                         intensity=inten, um_per_px=0.25)


def test_session_global_hotspot_corrections_report(tmp_path):
    rng = np.random.default_rng(1)
    s = Ki67Session(tmp_path, "K-TEST", min_cells=300)
    frames = [rng.integers(0, 255, (300, 400, 3), dtype=np.uint8) for _ in range(3)]
    for f, a in zip(frames, [_analysis(20, 180, 30, 0), _analysis(120, 130, 0, 1), _analysis(60, 90, 10, 2)]):
        assert s.add_field(f, a).accepted
    assert not s.add_field(frames[0], _analysis(20, 180)).accepted          # same field again
    assert s.field_counts().tolist() == [[180, 20], [130, 120], [90, 60]]
    g = s.global_result()
    assert g.n_counted == 600 and g.n_positive == 200
    h = s.hotspot_result()                                                   # 48%, then 40% -> 400 nuclei
    assert h.fields_used == [1, 2] and h.n_counted == 400 and h.index_pct == 45.0

    s.toggle_positive(0, 0)                                                  # positive -> negative
    s.toggle_tumour(0, 220)                                                  # lymphocyte -> tumour (negative)
    assert s.field_counts()[0].tolist() == [182, 19]
    log = [json.loads(l) for l in open(tmp_path / "corrections.jsonl")]
    assert [r["stain"] for r in log] == ["Ki67", "Ki67"]

    s.set_field_weight(0, 0.5), s.set_field_weight(1, 0.25), s.set_field_weight(2, 0.25)
    assert s.global_result().mode == "global (weighted)"
    s.save()
    summ = json.loads((tmp_path / "session.json").read_text())
    assert summ["hotspot"]["fields_used"] == [1, 2] and len(summ["fields"]) == 3

    from core.ki67.report import write_ki67_report
    pdf = write_ki67_report(s, tmp_path / "r.pdf", primary="hotspot")
    assert pdf.read_bytes()[:4] == b"%PDF" and pdf.stat().st_size > 5000
