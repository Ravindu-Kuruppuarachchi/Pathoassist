"""Guided 10-HPF mitosis counting at 40x.

Each accepted dwell frame is one HPF (up to 10). A frame overlapping an earlier HPF by
more than 50% is not a new HPF. Frames are registered to earlier ones by phase
correlation, so a mitosis seen in two partially overlapping frames is counted once.
The pathologist can accept / reject each detection or add a missed one; the count and
the mitotic score (from the calibrated field diameter) update immediately.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.ihc.dedup import THUMB, overlap, thumbnail
from core.mitosis.detector import MitosisCandidate

TABLE = json.loads((Path(__file__).parent / "mitotic_score_table.json").read_text())


def mitotic_score(count_10hpf: float, field_diameter_mm: float) -> dict:
    """Nottingham mitotic score (1-3) for a count over 10 HPFs of the given field diameter.

    In the table's range the nearest row is used. Outside it (e.g. a camera field smaller
    than a 0.40 mm eyepiece field) the thresholds are scaled by field area from the nearest
    row - thresholds in the table are ~proportional to area - and this is flagged.
    """
    rows = np.array(TABLE["rows"], float)
    d = rows[:, 0]
    i = int(np.abs(d - field_diameter_mm).argmin())
    s1, s3 = rows[i, 2], rows[i, 3]
    notes = []
    if field_diameter_mm < d.min() - 0.005 or field_diameter_mm > d.max() + 0.005:
        f = (field_diameter_mm / d[i]) ** 2
        s1, s3 = s1 * f, s3 * f
        notes.append(f"Field diameter {field_diameter_mm:.3f} mm is outside the table "
                     f"({d.min():.2f}-{d.max():.2f} mm); thresholds scaled by field area.")
    if not TABLE.get("verified"):
        notes.append("Score table not yet verified by a pathologist (see mitotic_score_table.json).")
    score = 1 if count_10hpf <= s1 else (3 if count_10hpf >= s3 else 2)
    return dict(score=score, score1_max=round(float(s1), 1), score3_min=round(float(s3), 1),
                table_diameter_mm=float(d[i]), notes=notes)


@dataclass
class Mitosis:
    id: int
    hpf: int                      # HPF where it was first seen
    x: float                      # frame px in that HPF
    y: float
    gx: float | None              # position in the registered island frame (µm), None if unregistered
    gy: float | None
    island: int
    det_score: float
    ver_score: float
    status: str                   # "auto", "accepted", "rejected", "added"

    @property
    def counted(self):
        return self.status in ("auto", "accepted", "added")


@dataclass
class HPF:
    index: int
    island: int
    offset_um: tuple              # frame origin in island coordinates (µm)
    um_per_px: float
    thumb: np.ndarray = field(repr=False)
    shape: tuple = (0, 0)


class HPFCounter:
    def __init__(self, um_per_px: float, frame_shape: tuple, n_hpf: int = 10, max_overlap: float = 0.5,
                 dedup_radius_um: float = 8.0):
        """um_per_px / frame_shape: calibrated 40x camera field -> HPF area and equivalent diameter."""
        self.um_per_px = um_per_px
        h, w = frame_shape[:2]
        self.hpf_area_mm2 = h * w * (um_per_px / 1000) ** 2
        self.field_diameter_mm = 2 * math.sqrt(self.hpf_area_mm2 / math.pi)
        self.n_hpf, self.max_overlap, self.dedup_um = n_hpf, max_overlap, dedup_radius_um
        self.hpfs: list[HPF] = []
        self.mitoses: list[Mitosis] = []
        self._islands = 0

    # --------------------------------------------------------- frames
    def add_frame(self, frame_rgb: np.ndarray, candidates: list[MitosisCandidate]) -> dict:
        """Count a dwell frame as the next HPF. Returns {accepted, reason, hpf, new, duplicates}."""
        if len(self.hpfs) >= self.n_hpf:
            return dict(accepted=False, reason=f"{self.n_hpf} HPFs already counted")
        t = thumbnail(frame_rgb)
        k = THUMB / max(frame_rgb.shape[:2])               # thumbnail px per frame px
        best = None
        for h in self.hpfs:
            frac, (dx, dy), resp = overlap(h.thumb, t)
            if frac > self.max_overlap:
                return dict(accepted=False, reason=f"overlaps HPF {h.index + 1} by {frac:.0%}")
            if frac > 0 and (best is None or resp > best[1]):
                best = (h, resp, dx, dy)
        if best is not None:
            h, _, dx, dy = best
            # new frame's origin lies at (dx, dy) thumbnail px in HPF h
            off = (h.offset_um[0] + dx / k * self.um_per_px, h.offset_um[1] + dy / k * self.um_per_px)
            island = h.island
        else:
            off, island = (0.0, 0.0), self._islands
            self._islands += 1
        idx = len(self.hpfs)
        self.hpfs.append(HPF(idx, island, off, self.um_per_px, t, frame_rgb.shape[:2]))
        new, dup = [], []
        for c in candidates:
            if not c.accepted:
                continue
            gx, gy = off[0] + c.x * self.um_per_px, off[1] + c.y * self.um_per_px
            same = [m for m in self.mitoses if m.island == island and m.gx is not None
                    and math.hypot(m.gx - gx, m.gy - gy) <= self.dedup_um]
            if same:
                dup.append(same[0].id)
                continue
            m = Mitosis(len(self.mitoses), idx, c.x, c.y, gx, gy, island, c.det_score, c.ver_score, "auto")
            self.mitoses.append(m)
            new.append(m.id)
        return dict(accepted=True, hpf=idx, new=new, duplicates=dup)

    # --------------------------------------------------------- review
    def set_status(self, mitosis_id: int, status: str):
        if status not in ("accepted", "rejected"):
            raise ValueError(status)
        self.mitoses[mitosis_id].status = status

    def add_manual(self, hpf: int, x: float, y: float) -> int:
        h = self.hpfs[hpf]
        m = Mitosis(len(self.mitoses), hpf, x, y, h.offset_um[0] + x * self.um_per_px,
                    h.offset_um[1] + y * self.um_per_px, h.island, 0.0, 0.0, "added")
        self.mitoses.append(m)
        return m.id

    def in_hpf(self, hpf: int):
        return [m for m in self.mitoses if m.hpf == hpf]

    # --------------------------------------------------------- result
    def result(self) -> dict:
        n = len(self.hpfs)
        count = sum(m.counted for m in self.mitoses)
        area = n * self.hpf_area_mm2
        res = dict(hpfs_counted=n, mitoses=count, area_mm2=round(area, 3),
                   per_mm2=round(count / area, 2) if area else None,
                   field_diameter_mm=round(self.field_diameter_mm, 3),
                   rejected=sum(m.status == "rejected" for m in self.mitoses),
                   added=sum(m.status == "added" for m in self.mitoses), warnings=[])
        if n < self.n_hpf:
            res["warnings"].append(f"Only {n} of {self.n_hpf} HPFs counted.")
        if n:
            # score on the count scaled to 10 HPFs (identical when all 10 are counted)
            res.update(mitotic_score(count * self.n_hpf / n, self.field_diameter_mm))
        return res
