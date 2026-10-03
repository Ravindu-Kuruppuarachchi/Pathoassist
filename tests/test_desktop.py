"""Desktop app tests (Qt-free parts): flat-field, READY gate, calibration profile, case store + patient details,
case reopening / signing, analysis modules, annotation documents, export and HER2 cross-check."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from app.desktop.annotation import TASK_BY_KEY, AnnotationStore, Item, render_mask
from app.desktop.casestore import CaseStore, FrameLogger, valid_case_id
from app.desktop.microscope import FlatField, MicroscopeProfile, QualityGate, um_per_px_from_ticks
from core.grading.case import BreastCase
from core.grading.report import write_breast_report
from core.her2.field_classifier import cross_check
from core.ihc.scoring import score_counts

SAMPLE = ROOT / "app/desktop/assets/sample_field.jpg"


def test_flat_field_removes_vignetting():
    h, w = 120, 160
    yy, xx = np.mgrid[:h, :w]
    vig = 1 - 0.8 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)       # 40% darker corners
    blank = np.clip(np.stack([vig * 220, vig * 200, vig * 180], -1), 0, 255).astype(np.uint8)
    ff = FlatField()
    ff.set_blank(blank)
    out = ff.apply(blank)
    assert out.std() < 6 and abs(float(out.mean()) - 245) < 6        # flat and white
    assert ff.apply(np.zeros((10, 10, 3), np.uint8)).shape == (10, 10, 3)   # other size passes through
    with pytest.raises(ValueError):
        ff.set_blank(np.zeros((50, 50, 3), np.uint8))


def test_quality_gate_ready_and_single_dwell():
    rng = np.random.default_rng(0)
    sharp = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    g = QualityGate(still_frames=3)
    states = [g.update(sharp) for _ in range(6)]
    assert not states[0].ready and states[-1].ready and sum(s.dwell for s in states) == 1
    moved = np.roll(sharp, 40, axis=1)
    assert not g.update(moved).still
    blurred = cv2.GaussianBlur(moved, (0, 0), 6)
    states = [g.update(blurred) for _ in range(6)]
    assert not any(s.ready for s in states)                         # still but not sharp


def test_calibration_profile(tmp_path):
    assert um_per_px_from_ticks((100, 100), (1100, 100), 250) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        um_per_px_from_ticks((0, 0), (5, 5), 10)
    p = MicroscopeProfile.new("Scope A/1", tmp_path)
    p.set_calibration("40x", 0.25, 1920, "stage micrometer")
    p.save_blank("40x", np.full((20, 30, 3), 200, np.uint8))
    p.save()
    q = MicroscopeProfile.load(p.path)
    assert q.um_per_px("40x", 1920) == pytest.approx(0.25)
    assert q.um_per_px("40x", 960) == pytest.approx(0.5)            # half the resolution, twice the µm/px
    assert q.um_per_px("10x", 1920) is None
    assert q.load_blank("40x").shape == (20, 30, 3)
    assert q.objective_names() == ["4x", "10x", "20x", "40x"]
    fov = MicroscopeProfile.field_of_view(0.25, 1920, 1080)
    assert fov["area_mm2"] == pytest.approx(0.1296) and fov["equiv_diameter_mm"] == pytest.approx(0.406, abs=1e-3)


def test_case_store_patient_and_frames(tmp_path):
    store = CaseStore(tmp_path)
    assert valid_case_id("B26-001.a") and not valid_case_id("../x") and not valid_case_id("")
    with pytest.raises(ValueError):
        store.create("B1", {"patient_id": ""}, "core biopsy")
    c = store.create("B1", {"patient_id": "P-9", "name": "A B", "dob": ""}, "excision", "tester")
    with pytest.raises(FileExistsError):
        store.create("B1", {"patient_id": "P-9"}, "excision")
    assert c.patient == {"patient_id": "P-9", "name": "A B"}
    png = FrameLogger("t").save(FrameLogger.case_frames_dir(c, "H&E"), np.zeros((8, 8, 3), np.uint8),
                                dict(case_id="B1", patient_id="P-9", name="A B", objective="40x"))
    meta = json.loads(png.with_suffix(".json").read_text())
    assert meta["case_id"] == "B1" and "patient_id" not in meta and "name" not in meta     # de-identified
    assert len(FrameLogger.list_frames(c.dir, "H&E")) == 1
    s = store.list()[0]
    assert (s.case_id, s.patient_id, s.specimen) == ("B1", "P-9", "excision")


def test_case_reopen_sign_and_report(tmp_path):
    store = CaseStore(tmp_path)
    c = store.create("B2", {"patient_id": "P-1", "name": "<Name> & Co"}, "core biopsy", "u")
    c.set_biomarker("ER", score_counts([100, 100, 100, 300]))
    c.set_extra("Tumour region", {"fields": 3}, "3 fields")
    c.grade.set_auto("tubules", 3, "x")
    c.save()
    r = store.open("B2", "v")
    assert r.patient["name"] == "<Name> & Co" and r.grade.c["tubules"].auto_score == 3 and "ER" in r.biomarkers
    assert r.extras["Tumour region"]["summary"] == "3 fields"
    with pytest.raises(ValueError):           # partial grade can never be signed
        r.sign("Dr", allow_no_grade=True)
    r.set_patient(clinician="Dr Who")
    assert r.patient["clinician"] == "Dr Who"
    events = [e["event"] for e in r.audit.entries()]
    assert "case_reopened" in events and r.audit.verify()[0]

    ihc = store.create("B3", {"patient_id": "P-2"}, "core biopsy")
    ihc.set_biomarker("PR", score_counts([900, 50, 30, 20]))
    with pytest.raises(ValueError):
        ihc.sign("Dr")
    ihc.sign("Dr", allow_no_grade=True)       # IHC-only case
    pdf = write_breast_report(ihc, tmp_path / "r.pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"
    pdf2 = write_breast_report(r, tmp_path / "r2.pdf")         # escaped patient text, extras section
    assert pdf2.stat().st_size > 2000


def test_her2_cross_check():
    assert cross_check("2+", [])["agrees"] is None
    r = cross_check("1+", [np.array([0.1, 0.2, 0.6, 0.1]), np.array([0.1, 0.3, 0.5, 0.1])])
    assert r["classifier_score"] == "2+" and not r["agrees"] and "review carefully" in r["note"]


def test_annotation_store_export(tmp_path):
    st = AnnotationStore(tmp_path)
    rgb = np.full((60, 80, 3), 200, np.uint8)
    d = st.new("nuclei", rgb, 0.25, "Dr A", stain="H&E", case_id="B1", objective="40x")
    d.items += [Item("polygon", "nucleus", [[5, 5], [20, 5], [20, 20], [5, 20]]),
                Item("polygon", "nucleus", [[30, 30], [50, 30], [40, 50]]),
                Item("polygon", "nucleus", [[60, 5], [70, 5], [65, 15]], "model")]
    d.field_labels["x"] = "1"
    st.save(d)
    m = cv2.imread(str(d.dir / "mask.png"), cv2.IMREAD_UNCHANGED)
    assert m.dtype == np.uint16 and m[10, 10] == 1 and m[35, 40] == 2 and m[8, 65] == 0   # unconfirmed not in mask
    gj = json.loads((d.dir / "annotation.geojson").read_text())
    assert len(gj["features"]) == 3 and gj["features"][0]["geometry"]["type"] == "Polygon"
    assert gj["features"][0]["properties"]["classification"]["name"] == "nucleus"
    assert d.item_at(10, 10, 3) == 0 and d.item_at(2, 50, 3) is None
    re = st.load(d.dir)
    assert len(re.items) == 3 and re.items[2].source == "model" and re.n_unconfirmed() == 1

    p = st.new("ki67_positivity", rgb, 0.25, "Dr B")
    p.items = [Item("point", "positive", [[10, 10]]), Item("point", "negative", [[30, 30]])]
    p.status = "complete"
    st.save(p)
    rows = list(csv.DictReader(open(p.dir / "points.csv")))
    assert [r["class"] for r in rows] == ["positive", "negative"]
    man = list(csv.DictReader(open(st.export_manifest("ki67_positivity"))))
    assert man[0]["status"] == "complete" and man[0]["n_positive"] == "1"

    t = TASK_BY_KEY["region"]
    r = st.new("region", rgb, 1.0, "Dr C")
    r.items = [Item("polygon", "stroma", [[0, 0], [40, 0], [40, 40], [0, 40]])]
    assert render_mask(r, t)[10, 10] == 4                          # class index + 1


def test_er_module_end_to_end(tmp_path):
    from app.desktop.hub import ModelHub
    from app.desktop.modules import ERModule
    hub = ModelHub(ROOT / "models", tmp_path / "calib")
    case = CaseStore(tmp_path / "cases").create("E1", {"patient_id": "P"}, "core biopsy", "u")
    m = ERModule(hub, case, "Dr", {"save_frames": True})
    rgb = cv2.cvtColor(cv2.imread(str(SAMPLE)), cv2.COLOR_BGR2RGB)
    out = m.add_frame(rgb, 0.25, {"objective": "40x"})
    assert out.accepted and out.field_id == 0 and m.fields[0].thumb is not None
    dup = m.add_frame(rgb, 0.25)
    assert not dup.accepted and "overlaps" in dup.message
    y, x = m.session.field(0).analysis.points[0]
    before = int(m.session.field(0).analysis.intensity[0])
    assert m.click(0, float(x), float(y), set()).startswith("intensity")
    assert int(m.session.field(0).analysis.intensity[0]) == (before + 1) % 4
    assert (m.run_dir / "corrections.jsonl").is_file()
    assert "Status" in m.summary_html()
    msg = m.save_to_case()
    assert "ER" in case.biomarkers and "PDF" in msg and case.snapshots
    assert len(FrameLogger.list_frames(case.dir, "ER")) == 1
