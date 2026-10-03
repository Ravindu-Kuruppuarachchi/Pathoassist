"""Feature 0 building blocks, Qt-free so they can be tested headless.

FlatField      : "capture blank" reference; every live frame is divided by it (removes vignetting,
                 uneven illumination and the colour cast of the lamp; the blank becomes white).
QualityGate    : motion (frame difference) + focus (Laplacian variance) -> READY when the stage is
                 still and the frame is sharp. A *dwell* is the first READY frame after movement.
MicroscopeProfile : per microscope, µm/px of every objective from the stage-micrometer calibration
                 (stored with the camera width it was measured at, so it follows resolution changes),
                 the field of view, and the flat-field blank of each objective.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OBJECTIVES = ["4x", "10x", "20x", "40x"]
WHITE = 245.0                       # the blank is mapped to this grey level (not 255: keeps highlights)


# ---------------------------------------------------------------------------------------------- flat field
class FlatField:
    def __init__(self):
        self.gain: np.ndarray | None = None      # float32 (H, W, 3)
        self.shape: tuple | None = None

    @property
    def active(self):
        return self.gain is not None

    def set_blank(self, blank_rgb: np.ndarray):
        """blank_rgb: frame of an empty area of the slide (no tissue), same light and focus as for work."""
        b = blank_rgb.astype(np.float32)
        sigma = max(b.shape[:2]) / 150             # smooth out dust and noise; vignetting is low-frequency
        b = cv2.GaussianBlur(b, (0, 0), sigma)
        if float(b.mean()) < 40:
            raise ValueError("the blank frame is too dark - check the lamp and that the light path is open")
        self.gain = np.clip(WHITE / np.maximum(b, 1.0), 0.5, 4.0).astype(np.float32)
        self.shape = blank_rgb.shape

    def clear(self):
        self.gain, self.shape = None, None

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        g = self.gain
        if g is None or rgb.shape != self.shape:
            return rgb
        return cv2.convertScaleAbs(cv2.multiply(rgb.astype(np.float32), g))   # saturates to 0-255 uint8


# ---------------------------------------------------------------------------------------------- gate
@dataclass
class GateState:
    motion: float = 0.0       # mean abs grey difference to the previous frame (0-255)
    focus: float = 0.0        # Laplacian variance on the downscaled grey frame
    peak: float = 0.0         # recent best focus (decays), so "sharp" is relative to this slide/objective
    still: bool = False
    sharp: bool = False
    ready: bool = False
    dwell: bool = False       # True exactly once per still period (the frame to analyse / save)


class QualityGate:
    def __init__(self, motion_thresh: float = 1.6, still_frames: int = 4, focus_rel: float = 0.7,
                 min_focus: float = 15.0, width: int = 320):
        self.motion_thresh, self.still_frames = motion_thresh, still_frames
        self.focus_rel, self.min_focus, self.width = focus_rel, min_focus, width
        self._prev = None
        self._still_count = 0
        self._peak = 0.0
        self._dwelled = False

    def reset(self):
        self._prev, self._still_count, self._peak, self._dwelled = None, 0, 0.0, False

    def update(self, rgb: np.ndarray) -> GateState:
        h, w = rgb.shape[:2]
        s = self.width / w
        g = cv2.cvtColor(cv2.resize(rgb, (self.width, max(1, round(h * s))), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        motion = float(np.abs(g - self._prev).mean()) if self._prev is not None and self._prev.shape == g.shape else 99.0
        self._prev = g
        focus = float(cv2.Laplacian(g, cv2.CV_32F).var())
        moving = motion > self.motion_thresh
        if moving:
            self._still_count = 0
            self._dwelled = False
            self._peak = max(focus, self._peak * 0.9)      # new field: forget the old peak quickly
        else:
            self._still_count += 1
            self._peak = max(focus, self._peak * 0.995)
        still = self._still_count >= self.still_frames
        sharp = focus >= self.min_focus and focus >= self.focus_rel * self._peak
        ready = still and sharp
        dwell = ready and not self._dwelled
        if dwell:
            self._dwelled = True
        return GateState(motion, focus, self._peak, still, sharp, ready, dwell)


# ---------------------------------------------------------------------------------------------- calibration
def um_per_px_from_ticks(p1, p2, distance_um: float) -> float:
    """Two clicked tick marks of a stage micrometer (frame px, (x, y)) and their true distance."""
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if d < 20:
        raise ValueError("the two points are too close - click tick marks far apart (most of the field)")
    if distance_um <= 0:
        raise ValueError("distance must be positive")
    return distance_um / d


@dataclass
class Objective:
    name: str
    um_per_px: float | None = None      # at ref_width
    ref_width: int | None = None        # camera frame width (px) the calibration was measured at
    method: str = ""                    # "stage micrometer" or "entered manually"
    calibrated_at: str = ""
    blank_file: str = ""                # flat-field blank (PNG, next to the profile)

    @property
    def calibrated(self):
        return bool(self.um_per_px and self.ref_width)


@dataclass
class MicroscopeProfile:
    name: str
    camera: str = ""
    objectives: dict = field(default_factory=dict)     # name -> Objective
    path: Path | None = None

    @staticmethod
    def slug(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "microscope"

    @classmethod
    def new(cls, name: str, folder: Path) -> "MicroscopeProfile":
        p = cls(name, objectives={o: Objective(o) for o in DEFAULT_OBJECTIVES})
        p.path = Path(folder) / f"{cls.slug(name)}.json"
        return p

    @classmethod
    def load(cls, path: Path) -> "MicroscopeProfile":
        d = json.loads(Path(path).read_text())
        objs = {k: Objective(**v) for k, v in d.get("objectives", {}).items()}
        return cls(d["name"], d.get("camera", ""), objs, Path(path))

    @classmethod
    def list(cls, folder: Path) -> list["MicroscopeProfile"]:
        out = []
        for p in sorted(Path(folder).glob("*.json")):
            try:
                out.append(cls.load(p))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return out

    def save(self):
        d = dict(name=self.name, camera=self.camera, objectives={k: asdict(v) for k, v in self.objectives.items()},
                 saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2))

    @property
    def blank_dir(self) -> Path:
        return self.path.with_suffix("")

    # -------------------------------------------------------------- objectives
    def objective_names(self) -> list[str]:
        def mag(n):
            m = re.match(r"(\d+(?:\.\d+)?)", n)
            return float(m.group(1)) if m else 1e9
        return sorted(self.objectives, key=mag)

    def set_calibration(self, objective: str, um_per_px: float, frame_width: int, method: str):
        o = self.objectives.setdefault(objective, Objective(objective))
        o.um_per_px, o.ref_width, o.method = float(um_per_px), int(frame_width), method
        o.calibrated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def um_per_px(self, objective: str, frame_width: int) -> float | None:
        """Scale of a frame of this width (µm/px follows the camera resolution in use)."""
        o = self.objectives.get(objective)
        if o is None or not o.calibrated:
            return None
        return o.um_per_px * o.ref_width / frame_width

    def calibrated_scales(self, frame_width: int) -> dict:
        return {n: self.um_per_px(n, frame_width) for n in self.objective_names() if self.objectives[n].calibrated}

    @staticmethod
    def field_of_view(um_per_px: float, width: int, height: int) -> dict:
        w_um, h_um = width * um_per_px, height * um_per_px
        area = w_um * h_um / 1e6
        return dict(width_um=round(w_um, 1), height_um=round(h_um, 1), area_mm2=round(area, 4),
                    equiv_diameter_mm=round(2 * math.sqrt(area / math.pi), 3))

    # -------------------------------------------------------------- flat field
    def save_blank(self, objective: str, blank_rgb: np.ndarray):
        self.blank_dir.mkdir(parents=True, exist_ok=True)
        fn = f"blank_{self.slug(objective)}.png"
        cv2.imwrite(str(self.blank_dir / fn), cv2.cvtColor(blank_rgb, cv2.COLOR_RGB2BGR))
        self.objectives.setdefault(objective, Objective(objective)).blank_file = fn

    def load_blank(self, objective: str) -> np.ndarray | None:
        o = self.objectives.get(objective)
        if o is None or not o.blank_file:
            return None
        im = cv2.imread(str(self.blank_dir / o.blank_file))
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def clear_blank(self, objective: str):
        o = self.objectives.get(objective)
        if o is not None and o.blank_file:
            (self.blank_dir / o.blank_file).unlink(missing_ok=True)
            o.blank_file = ""
