"""Cases on disk and the frame logger (Qt-free).

    <cases>/<case_id>/
        case.json          patient / request details, results of every module, grade, snapshots, signature
        audit.jsonl        hash-chained audit log (core.grading.audit)
        frames/<stain>/    every analysed / captured frame: PNG (lossless) + JSON sidecar
        modules/<module>/<run>/   per-run working files (field images, session.json, corrections.jsonl, module PDF)
        snapshots/         images placed in the combined report
        reports/           combined PDF reports (draft and signed)

Frame sidecars hold the case ID, stain, objective, µm/px and capture details - never patient identifiers -
so frames can be shared for training as they are.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.grading.case import BreastCase

STAINS = ["H&E", "ER", "PR", "Ki67", "HER2"]
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def valid_case_id(case_id: str) -> bool:
    return bool(_ID.match(case_id or ""))


def stain_dir(stain: str) -> str:
    return stain.replace("&", "")          # "H&E" -> "HE"


@dataclass
class CaseSummary:
    case_id: str
    patient_id: str
    name: str
    specimen: str
    created: str
    saved: str
    signed_by: str | None
    grade: int | None
    results: list
    path: Path


class CaseStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def case_dir(self, case_id: str) -> Path:
        return self.root / case_id

    def exists(self, case_id: str) -> bool:
        return (self.case_dir(case_id) / "case.json").is_file()

    def create(self, case_id: str, patient: dict, specimen: str, user: str | None = None) -> BreastCase:
        case_id = case_id.strip()
        if not valid_case_id(case_id):
            raise ValueError("Case ID: 1-64 characters, letters, digits, '.', '_' or '-' (it becomes a folder name)")
        if self.exists(case_id):
            raise FileExistsError(f"case {case_id} already exists - open it instead")
        if not (patient or {}).get("patient_id", "").strip():
            raise ValueError("Patient ID is required")
        case = BreastCase(self.case_dir(case_id), case_id, specimen, user=user, patient=patient)
        case.save()
        return case

    def open(self, case_id: str, user: str | None = None) -> BreastCase:
        if not self.exists(case_id):
            raise FileNotFoundError(f"no case {case_id} in {self.root}")
        return BreastCase.open(self.case_dir(case_id), user)

    def list(self) -> list[CaseSummary]:
        out = []
        for p in self.root.glob("*/case.json"):
            try:
                d = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            pat = d.get("patient", {})
            results = sorted(d.get("biomarkers", {}))
            results += [k for k, c in d.get("grade", {}).get("components", {}).items() if c.get("score") is not None]
            results += sorted(d.get("extras", {}))
            out.append(CaseSummary(d["case_id"], pat.get("patient_id", ""), pat.get("name", ""), d.get("specimen", ""),
                                   d.get("created", ""), d.get("saved", ""), d.get("signed_by"),
                                   d.get("grade", {}).get("grade"), results, p.parent))
        return sorted(out, key=lambda c: c.saved or c.created, reverse=True)


class FrameLogger:
    """Saves frames as PNG + JSON sidecar into a case (or a dated captures folder without a case)."""

    def __init__(self, app_version: str = ""):
        self.app_version = app_version

    def save(self, folder: Path, rgb: np.ndarray, meta: dict) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        png = folder / f"{stem}.png"
        cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        m = {k: v for k, v in meta.items() if k not in ("patient", "patient_id", "name", "dob")}
        m.update(width=int(rgb.shape[1]), height=int(rgb.shape[0]), app_version=self.app_version,
                 captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        png.with_suffix(".json").write_text(json.dumps(m, indent=2, default=str))
        return png

    @staticmethod
    def case_frames_dir(case: BreastCase, stain: str) -> Path:
        return case.dir / "frames" / stain_dir(stain)

    @staticmethod
    def list_frames(case_dir: Path, stain: str | None = None) -> list[tuple[Path, dict]]:
        base = Path(case_dir) / "frames"
        dirs = [base / stain_dir(stain)] if stain else sorted(d for d in base.glob("*") if d.is_dir())
        out = []
        for d in dirs:
            for png in sorted(d.glob("*.png")):
                try:
                    meta = json.loads(png.with_suffix(".json").read_text())
                except (OSError, json.JSONDecodeError):
                    meta = {}
                out.append((png, meta))
        return out


def read_meta(image_path: Path) -> dict:
    """Sidecar of a saved frame (empty if none)."""
    try:
        return json.loads(Path(image_path).with_suffix(".json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_rgb(path: Path) -> np.ndarray | None:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
