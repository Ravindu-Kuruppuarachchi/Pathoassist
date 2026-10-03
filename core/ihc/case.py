"""ER/PR case session: running totals over dwell frames, de-duplication and click-to-correct.

Every correction is appended to <case_dir>/corrections.jsonl as a training label
(frame file, nucleus centre, stain, DAB OD, old/new value), ready for retraining the
cell-type classifier and re-calibrating the intensity thresholds.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.ihc.dedup import FieldDeduplicator
from core.ihc.pipeline import CELL_TYPES, TUMOUR, FieldAnalysis
from core.ihc.scoring import Score, score_counts


@dataclass
class Field:
    index: int
    frame_file: str
    analysis: FieldAnalysis
    timestamp: float
    corrected: set = field(default_factory=set)   # nucleus indices touched by the user

    def counts(self) -> np.ndarray:
        a = self.analysis
        t = a.cell_type == TUMOUR
        return np.bincount(a.intensity[t], minlength=4)[:4]


@dataclass
class AddResult:
    accepted: bool
    field_index: int | None
    duplicate_of: int | None = None
    overlap: float = 0.0


class CaseSession:
    def __init__(self, case_dir: str | Path, case_id: str, stain: str, min_cells: int = 500,
                 max_overlap: float = 0.5):
        if stain not in ("ER", "PR"):
            raise ValueError("stain must be 'ER' or 'PR'")
        self.dir = Path(case_dir)
        (self.dir / "fields").mkdir(parents=True, exist_ok=True)
        self.case_id, self.stain, self.min_cells = case_id, stain, min_cells
        self.fields: list[Field] = []
        self.dedup = FieldDeduplicator(max_overlap)
        self._next = 0

    # ------------------------------------------------------------- fields
    def add_field(self, frame_rgb: np.ndarray, analysis: FieldAnalysis, force: bool = False) -> AddResult:
        """Count a dwell frame unless it overlaps an already-counted field by > max_overlap."""
        if not force:
            ov = self.dedup.check(frame_rgb)
            if ov is not None:
                return AddResult(False, None, duplicate_of=self.fields[ov.index].index, overlap=ov.fraction)
        idx = self._next
        self._next += 1
        fn = f"fields/{idx:03d}.jpg"
        cv2.imwrite(str(self.dir / fn), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.fields.append(Field(idx, fn, analysis, time.time()))
        self.dedup.add(frame_rgb)
        return AddResult(True, idx)

    def remove_field(self, field_index: int):
        pos = self._pos(field_index)
        self.fields.pop(pos)
        self.dedup.remove(pos)

    def _pos(self, field_index):
        for i, f in enumerate(self.fields):
            if f.index == field_index:
                return i
        raise KeyError(field_index)

    def field(self, field_index) -> Field:
        return self.fields[self._pos(field_index)]

    # ------------------------------------------------------------- totals
    def score(self) -> Score:
        counts = sum((f.counts() for f in self.fields), np.zeros(4, int))
        return score_counts(counts, self.min_cells)

    def field_scores(self) -> list[Score]:
        return [score_counts(f.counts(), min_cells=0) for f in self.fields]

    # ------------------------------------------------------------- click-to-correct
    def nucleus_at(self, field_index: int, y: float, x: float, radius: float = 6) -> int | None:
        """Nucleus index under a click (label map), else the nearest centre within `radius` px."""
        a = self.field(field_index).analysis
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < a.labels.shape[0] and 0 <= xi < a.labels.shape[1] and a.labels[yi, xi] > 0:
            return int(a.labels[yi, xi]) - 1
        if not len(a):
            return None
        d = np.hypot(a.points[:, 0] - y, a.points[:, 1] - x)
        i = int(d.argmin())
        return i if d[i] <= radius else None

    def set_cell_type(self, field_index: int, nucleus: int, cell_type: int, user: str = ""):
        f = self.field(field_index)
        old = int(f.analysis.cell_type[nucleus])
        f.analysis.cell_type[nucleus] = cell_type
        self._log(f, nucleus, "cell_type", CELL_TYPES[old], CELL_TYPES[cell_type], user)

    def set_intensity(self, field_index: int, nucleus: int, intensity: int, user: str = ""):
        f = self.field(field_index)
        old = int(f.analysis.intensity[nucleus])
        f.analysis.intensity[nucleus] = intensity
        self._log(f, nucleus, "intensity", old, int(intensity), user)

    def toggle_tumour(self, field_index: int, nucleus: int, user: str = ""):
        """Tumour <-> non-tumour. A non-tumour nucleus becomes tumour; a tumour nucleus becomes
        its most likely non-tumour class."""
        a = self.field(field_index).analysis
        if a.cell_type[nucleus] == TUMOUR:
            new = int(np.argmax(a.type_prob[nucleus][1:])) + 1
        else:
            new = TUMOUR
        self.set_cell_type(field_index, nucleus, new, user)

    def toggle_positive(self, field_index: int, nucleus: int, user: str = ""):
        """Negative -> positive (weak, or higher if the measured OD says so); positive -> negative."""
        a = self.field(field_index).analysis
        new = 0 if a.intensity[nucleus] > 0 else 1
        self.set_intensity(field_index, nucleus, new, user)

    def _log(self, f: Field, nucleus: int, kind: str, old, new, user: str):
        f.corrected.add(nucleus)
        a = f.analysis
        rec = dict(case=self.case_id, stain=self.stain, field=f.index, frame=f.frame_file,
                   y=round(float(a.points[nucleus, 0]), 1), x=round(float(a.points[nucleus, 1]), 1),
                   um_per_px=a.um_per_px, dab_od=round(float(a.dab_od[nucleus]), 4),
                   kind=kind, old=old, new=new, user=user, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.dir / "corrections.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------- persistence
    def summary(self) -> dict:
        return dict(case_id=self.case_id, stain=self.stain, n_fields=len(self.fields),
                    score=self.score().as_dict(),
                    fields=[dict(index=f.index, frame=f.frame_file, n_nuclei=len(f.analysis),
                                 n_corrected=len(f.corrected), **s.as_dict())
                            for f, s in zip(self.fields, self.field_scores())])

    def save(self):
        (self.dir / "session.json").write_text(json.dumps(self.summary(), indent=2))
