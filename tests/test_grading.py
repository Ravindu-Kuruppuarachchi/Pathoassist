"""Feature 9 tests: Nottingham grade, audit chain, grading wizard, signing lock, combined report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from core.grading.audit import AuditLog
from core.grading.case import BreastCase, GradingWizard
from core.grading.nottingham import NottinghamGrade, grade_from_total
from core.grading.report import write_breast_report
from core.her2.scoring import her2_score
from core.ihc.scoring import score_counts
from core.ki67.scoring import global_index, hotspot_index
from core.tubules.score import tubule_score


@pytest.mark.parametrize("total,grade", [(3, 1), (5, 1), (6, 2), (7, 2), (8, 3), (9, 3)])
def test_grade_mapping(total, grade):
    assert grade_from_total(total) == grade


@pytest.mark.parametrize("pct,score", [(90, 1), (75.1, 1), (75, 2), (10, 2), (9.9, 3), (0, 3)])
def test_tubule_score(pct, score):
    assert tubule_score(pct) == score


def test_grade_components_and_edits(tmp_path):
    a = AuditLog(tmp_path / "a.jsonl", user="tester")
    g = NottinghamGrade(a)
    g.set_auto("tubules", 3, "tubules 4%")
    g.set_auto("pleomorphism", 2)
    assert not g.complete and g.grade is None
    g.set_manual("mitoses", 3)
    assert g.c["mitoses"].source == "manual" and g.total == 8 and g.grade == 3
    g.set_manual("pleomorphism", 3, reason="marked atypia")
    assert g.c["pleomorphism"].edited and g.c["pleomorphism"].auto_score == 2 and g.total == 9
    with pytest.raises(ValueError):
        g.set_manual("tubules", 4)
    assert a.verify()[0] and [e["event"] for e in a.entries()][-1] == "component_manual"


def test_audit_detects_tampering(tmp_path):
    p = tmp_path / "a.jsonl"
    a = AuditLog(p, user="x")
    for i in range(4):
        a.log("event", i=i)
    assert a.verify() == (True, "ok")
    lines = p.read_text().splitlines()
    e = json.loads(lines[1])
    e["details"]["i"] = 99
    lines[1] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n")
    ok, msg = AuditLog(p).verify()
    assert not ok and "entry 2" in msg
    p.write_text("\n".join([lines[0]] + lines[2:]) + "\n")          # delete an entry
    assert not AuditLog(p).verify()[0]


class _Stub:
    """Minimal stand-ins for the step engines."""
    def __init__(self, result):
        self.r = result
        self.frames = 0
        self.scorer = type("S", (), {"validated": False})()

    def add_field(self, *a, **k):
        self.frames += 1
        return dict(accepted=True)

    def add_frame(self, *a, **k):
        self.frames += 1
        return dict(accepted=True)

    def result(self):
        return self.r


def test_wizard_flow_lock_and_report(tmp_path):
    from core.pleomorphism.scorer import PleoResult
    from core.tubules.score import TubuleResult
    case = BreastCase(tmp_path / "case", "B-001", "core biopsy", user="dr_test")
    tub = _Stub(TubuleResult(3, 6.0, 4.2, 0.25, 12, 30))
    pleo = _Stub(PleoResult(3, [0.05, 0.2, 0.75], 2.7, 4, 2100, [3, 3, 2, 3]))
    det = type("D", (), {"detect": lambda self, f, u: ([], {})})()
    hpf = _Stub(dict(score=2, mitoses=9, hpfs_counted=10, field_diameter_mm=0.5))
    w = GradingWizard(case, tub, pleo, hpf, det)
    frame = np.zeros((64, 64, 3), np.uint8)
    assert w.step == "tubules" and w.add_frame(frame, 1.0)["accepted"]
    w.finish_step()
    assert w.step == "pleomorphism"
    w.add_frame(frame, 0.25)
    w.finish_step()
    w.add_frame(frame, 0.25)
    w.finish_step()
    assert w.step == "review" and case.grade.total == 8 and case.grade.grade == 3
    w.edit("mitoses", 3, "missed mitoses in HPF 7")
    assert case.grade.grade == 3 and case.grade.c["mitoses"].edited

    er = score_counts([200, 100, 200, 500])
    case.set_biomarker("ER", er, validated=False, method="DAB nuclear OD")
    case.set_biomarker("PR", score_counts([900, 50, 30, 20]))
    case.set_biomarker("HER2", her2_score([700, 200, 80, 20]))
    case.set_biomarker("Ki67", dict(global_=None, **{"global": global_index([[700, 300]]).as_dict(),
                                                    "hotspot": hotspot_index([[300, 200], [400, 100]]).as_dict()}))
    case.add_snapshot(np.full((120, 160, 3), 200, np.uint8), "Tumour at 10x", "tubules")
    w.sign("Dr Test")
    assert w.step == "signed" and case.signed_by == "Dr Test"
    with pytest.raises(PermissionError):
        w.edit("tubules", 1)
    with pytest.raises(PermissionError):
        case.set_biomarker("ER", er)
    pdf = write_breast_report(case, tmp_path / "report.pdf", lab="Test lab")
    assert pdf.read_bytes()[:4] == b"%PDF" and pdf.stat().st_size > 5000
    d = case.save()
    assert d["grade"]["grade"] == 3 and d["signed_by"] == "Dr Test"
    ok, _ = case.audit.verify()
    events = [e["event"] for e in case.audit.entries()]
    assert ok and "change_refused_after_signing" in events and events[-1] == "report_written"


def test_cannot_sign_incomplete(tmp_path):
    case = BreastCase(tmp_path / "c", "B-002", "excision")
    w = GradingWizard(case)
    with pytest.raises(RuntimeError):
        w.sign("x")
    w.enter_manual(2)
    w.enter_manual(2)
    w.enter_manual(1)
    assert case.grade.grade == 1
    w.sign("Dr X")
