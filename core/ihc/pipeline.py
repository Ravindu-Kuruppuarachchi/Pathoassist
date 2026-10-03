"""Per-frame ER/PR analysis: nuclei -> cell type -> DAB intensity."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.ihc.scoring import IntensityThresholds
from core.ihc.stain import StainMatrix, per_nucleus_stain
from core.nuclei import NucleiDetector

CELL_TYPES = ["tumour", "lymphocyte", "stroma", "other"]
TUMOUR = 0


@dataclass
class FieldAnalysis:
    labels: np.ndarray          # (H,W) int32 instance map, label i+1 <-> nucleus i
    points: np.ndarray          # (n,2) y,x
    coords: np.ndarray          # (n,2,rays) polygons for drawing
    cell_type: np.ndarray       # (n,) int8 index into CELL_TYPES
    type_prob: np.ndarray       # (n,4) classifier probabilities (uniform if no classifier)
    dab_od: np.ndarray          # (n,) mean nuclear DAB optical density
    h_od: np.ndarray            # (n,) mean nuclear haematoxylin OD
    intensity: np.ndarray       # (n,) int8 0..3
    um_per_px: float | None = None
    timings: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.points)


class ERPRAnalyzer:
    def __init__(self, detector: NucleiDetector, classifier=None, stain: StainMatrix | None = None,
                 thresholds: IntensityThresholds | None = None):
        self.detector = detector
        self.classifier = classifier            # core.ihc.celltype.CellTypeClassifier or None
        self.stain = stain or StainMatrix()
        self.thresholds = thresholds or IntensityThresholds()

    def analyze(self, frame_rgb: np.ndarray, um_per_px: float | None = None) -> FieldAnalysis:
        det = self.detector.detect(frame_rgb, um_per_px=um_per_px)
        n = len(det)
        conc = self.stain.concentrations(frame_rgb)
        h_od, dab_od = per_nucleus_stain(det.labels, conc, n)
        # nuclei fully hidden by neighbours in the label map have no pixels -> use the centre pixel
        miss = np.isnan(dab_od)
        if miss.any():
            yy, xx = np.clip(det.points[miss].round().astype(int).T, 0, np.array(frame_rgb.shape[:2])[:, None] - 1)
            dab_od[miss], h_od[miss] = conc[yy, xx, 1], conc[yy, xx, 0]
        if self.classifier is not None and n:
            prob = self.classifier.predict(frame_rgb, det.points, um_per_px)
        else:
            prob = np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1))
        return FieldAnalysis(labels=det.labels, points=det.points, coords=det.coords,
                             cell_type=prob.argmax(1).astype(np.int8) if n else np.zeros(0, np.int8),
                             type_prob=prob, dab_od=dab_od, h_od=h_od,
                             intensity=self.thresholds.bin(dab_od), um_per_px=um_per_px, timings=det.timings)
