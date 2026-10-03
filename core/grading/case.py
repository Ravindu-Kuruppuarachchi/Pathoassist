"""Breast case: Nottingham grade (via the grading wizard) + ER / PR / HER2 / Ki-67 results,
snapshots and the audit log, for the combined breast report (Feature 9).

Wizard: Tubules (10x sweep) -> Pleomorphism (40x tumour fields) -> Mitoses (10 HPF at 40x) ->
review -> sign. Each step feeds frames to its engine (Features 8, 7, 6); finishing a step stores
the automatic component score; the pathologist can override any component or skip a step and
enter the score directly. After signing, the case is locked (further changes are refused and logged).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2

from core.grading.audit import AuditLog
from core.grading.nottingham import COMPONENTS, NottinghamGrade

SPECIMENS = ("core biopsy", "excision", "other")


def _plain(x):
    if is_dataclass(x):
        return asdict(x)
    if hasattr(x, "as_dict"):
        return x.as_dict()
    return x


class BreastCase:
    def __init__(self, case_dir, case_id, specimen="core biopsy", user=None):
        if specimen not in SPECIMENS:
            raise ValueError(f"specimen must be one of {SPECIMENS}")
        self.dir = Path(case_dir)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.case_id, self.specimen = case_id, specimen
        self.audit = AuditLog(self.dir / "audit.jsonl", user)
        self.grade = NottinghamGrade(self.audit)
        self.biomarkers: dict = {}       # "ER", "PR", "HER2", "Ki67" -> result dict
        self.snapshots: list = []        # [{file, caption, module}]
        self.signed_by = None
        self.audit.log("case_opened", case_id=case_id, specimen=specimen)

    def _check_open(self, what):
        if self.signed_by:
            self.audit.log("change_refused_after_signing", what=what)
            raise PermissionError(f"case {self.case_id} is signed by {self.signed_by}; changes are not allowed")

    def set_biomarker(self, name, result, validated=False, method=""):
        """name: 'ER', 'PR', 'HER2', 'Ki67'; result: the module's result object or dict."""
        self._check_open(f"biomarker {name}")
        if name not in ("ER", "PR", "HER2", "Ki67"):
            raise KeyError(name)
        d = _plain(result)
        self.biomarkers[name] = dict(result=d, validated=validated, method=method)
        self.audit.log("biomarker_set", marker=name, summary=_summary(name, d), validated=validated)

    def add_snapshot(self, rgb, caption, module=""):
        self._check_open("snapshot")
        fn = f"snapshots/{len(self.snapshots) + 1:02d}_{module or 'image'}.jpg"
        cv2.imwrite(str(self.dir / fn), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.snapshots.append(dict(file=fn, caption=caption, module=module))
        self.audit.log("snapshot", file=fn, caption=caption, module=module)

    def sign(self, pathologist: str):
        if not self.grade.complete:
            missing = [k for k in COMPONENTS if self.grade.c[k].score is None]
            raise ValueError(f"cannot sign: grade components missing {missing}")
        self._check_open("sign")
        self.signed_by = pathologist
        self.audit.log("signed", pathologist=pathologist, grade=self.grade.grade, total=self.grade.total,
                       biomarkers=sorted(self.biomarkers))

    def save(self):
        d = dict(case_id=self.case_id, specimen=self.specimen, grade=self.grade.as_dict(), biomarkers=self.biomarkers,
                 snapshots=self.snapshots, signed_by=self.signed_by, audit_head=self.audit.last_hash,
                 saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (self.dir / "case.json").write_text(json.dumps(d, indent=2, default=str))
        return d


def _summary(name, d):
    try:
        if name in ("ER", "PR"):
            return f"{d['status']}, {d['pct_positive']}% positive, Allred {d['allred_total']}, H-score {d['h_score']}"
        if name == "HER2":
            return f"{d['score']}{' (' + d['subgroup'] + ')' if d.get('subgroup') else ''}{', reflex ISH' if d.get('reflex_ish') else ''}"
        if name == "Ki67":
            g = d.get("global", d)
            return f"{g['index_pct']}% ({g.get('mode', 'global')})"
    except (KeyError, TypeError):
        pass
    return ""


class GradingWizard:
    STEPS = [("tubules", "10x sweep of the tumour"), ("pleomorphism", "40x tumour fields (at least 3)"),
             ("mitoses", "10 high-power fields at 40x"), ("review", "check and edit the components"), ("signed", "")]

    def __init__(self, case: BreastCase, tubules=None, pleomorphism=None, hpf_counter=None, mitosis_detector=None):
        """tubules: core.tubules.score.TubuleSession; pleomorphism: core.pleomorphism.scorer.PleomorphismSession;
        hpf_counter: core.mitosis.counting.HPFCounter; mitosis_detector: core.mitosis.detector.MitosisDetector.
        Any engine may be None - that step is then entered manually."""
        self.case = case
        self.engines = dict(tubules=tubules, pleomorphism=pleomorphism, mitoses=hpf_counter)
        self.detector = mitosis_detector
        self.i = 0
        case.audit.log("wizard_started")

    @property
    def step(self):
        return self.STEPS[self.i][0]

    @property
    def instruction(self):
        return self.STEPS[self.i][1]

    def add_frame(self, frame_rgb, um_per_px, placement=None) -> dict:
        s = self.step
        eng = self.engines.get(s)
        if eng is None:
            return dict(accepted=False, reason=f"no automatic engine for {s}; enter the score manually")
        if s == "tubules":
            return eng.add_field(frame_rgb, um_per_px, placement)
        if s == "pleomorphism":
            return eng.add_field(frame_rgb, um_per_px)
        if s == "mitoses":
            cands, _ = self.detector.detect(frame_rgb, um_per_px)
            return eng.add_frame(frame_rgb, cands)
        return dict(accepted=False, reason=f"step '{s}' takes no frames")

    def finish_step(self):
        """Store the engine's automatic score for the current component and move on."""
        s = self.step
        if s not in COMPONENTS:
            raise RuntimeError(f"step '{s}' is not a grading component")
        eng = self.engines.get(s)
        if eng is None:
            raise RuntimeError(f"no engine for {s}: use enter_manual()")
        if s == "tubules":
            r = eng.result()
            self.case.grade.set_auto("tubules", r.score, f"tubules {r.tubule_pct:.0f}% of {r.tumour_area_mm2:.1f} mm² invasive tumour",
                                     validated=False, details=_plain(r))
        elif s == "pleomorphism":
            r = eng.result()
            self.case.grade.set_auto("pleomorphism", r.score, f"P(1/2/3) = {r.probs}, {r.n_fields} fields, {r.n_tumour} tumour nuclei",
                                     validated=bool(eng.scorer.validated), details=_plain(r))
        else:
            r = eng.result()
            self.case.grade.set_auto("mitoses", r["score"], f"{r['mitoses']} mitoses in {r['hpfs_counted']} HPF "
                                     f"(field diameter {r['field_diameter_mm']} mm)", validated=False, details=r)
        self.i += 1

    def enter_manual(self, score: int, reason="entered by pathologist"):
        s = self.step
        if s not in COMPONENTS:
            raise RuntimeError(f"step '{s}' is not a grading component")
        self.case.grade.set_manual(s, score, reason)
        self.i += 1

    def edit(self, component, score, reason=""):
        """Override any component at review time."""
        self.case._check_open(f"edit {component}")
        self.case.grade.set_manual(component, score, reason)

    def back(self):
        if self.step != "signed" and self.i > 0:
            self.i -= 1

    def sign(self, pathologist):
        if self.step != "review":
            raise RuntimeError("finish all grading steps and review before signing")
        self.case.sign(pathologist)
        self.i += 1
