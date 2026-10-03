"""HER2 frame analysis: nuclei -> (cell type) -> cells -> membrane ring -> per-cell category,
plus the cytoplasmic / background staining artefact check.

Membrane ODs are measured above the field's background (median extracellular DAB), so a
colour cast or light non-specific staining does not turn every cell 'positive'."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.her2.membrane import cells_from_nuclei, membrane_sectors, region_medians
from core.her2.scoring import HER2Thresholds, classify_cells
from core.ihc.pipeline import TUMOUR, FieldAnalysis
from core.ihc.stain import StainMatrix
from core.nuclei import NucleiDetector

BACKGROUND_FLAG_OD = 0.25        # extracellular DAB above this flags non-specific background staining


@dataclass
class HER2FieldAnalysis(FieldAnalysis):
    """`labels` holds the CELL map (click anywhere in a cell selects it); `intensity` holds the
    membrane category 0-3; `dab_od` the membrane intensity."""
    nuclei: np.ndarray = None
    sector_od: np.ndarray = None
    completeness: np.ndarray = None
    intense_incomplete: np.ndarray = None
    measurable: np.ndarray = None
    artefact: dict = field(default_factory=dict)


class HER2Analyzer:
    def __init__(self, detector: NucleiDetector, classifier=None, stain: StainMatrix | None = None,
                 thresholds: HER2Thresholds | None = None, cell_method: str = "watershed", max_cell_um: float = 8.0):
        self.detector, self.classifier = detector, classifier
        self.stain = stain or StainMatrix()
        self.th = thresholds or HER2Thresholds()
        self.cell_method, self.max_cell_um = cell_method, max_cell_um

    def analyze(self, frame_rgb: np.ndarray, um_per_px: float) -> HER2FieldAnalysis:
        det = self.detector.detect(frame_rgb, um_per_px=um_per_px)
        n = len(det)
        conc = self.stain.concentrations(frame_rgb)
        dab = conc[..., 1]
        cells = cells_from_nuclei(det.labels, dab, um_per_px, self.max_cell_um, self.cell_method)
        sector_abs, ring = membrane_sectors(cells, det.labels, dab, det.points, um_per_px)
        tissue = conc[..., :2].sum(-1) > 0.15
        cyto, extra = region_medians(cells, det.labels, ring, dab, tissue)
        # membrane staining is judged against the field's background DAB level (stain tint,
        # scanner colour cast, non-specific background), as a pathologist does
        base = extra if np.isfinite(extra) else 0.0
        sector_od = np.clip(sector_abs - base, 0, None)
        cat, ii, comp, inten, measurable = classify_cells(sector_od, self.th)
        # nuclei too close to the frame edge have truncated rings -> not counted
        h, w = frame_rgb.shape[:2]
        m = self.max_cell_um / um_per_px
        inside = ((det.points[:, 0] > m) & (det.points[:, 0] < h - m) & (det.points[:, 1] > m)
                  & (det.points[:, 1] < w - m)) if n else np.zeros(0, bool)
        measurable = measurable & inside
        if self.classifier is not None and n:
            prob = self.classifier.predict(frame_rgb, det.points, um_per_px)
        else:
            prob = np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1))
        stained = cat > 0
        ring_med = float(np.median(inten[stained])) if stained.sum() >= 10 else float("nan")   # above background
        cyto_rel = cyto - base if np.isfinite(cyto) else float("nan")
        art = dict(cytoplasm_dab=cyto, extracellular_dab=extra, membrane_dab_above_background=ring_med, flags=[])
        if np.isfinite(cyto_rel) and cyto_rel >= self.th.stain_od and np.isfinite(ring_med) and cyto_rel >= 0.6 * ring_med:
            art["flags"].append(f"Cytoplasmic DAB ({cyto_rel:.2f} OD above background) is close to the membrane staining "
                                f"({ring_med:.2f}): possible cytoplasmic staining or artefact - interpret with caution.")
        if np.isfinite(extra) and extra >= BACKGROUND_FLAG_OD:
            art["flags"].append(f"High background DAB ({extra:.2f} OD outside cells): possible non-specific staining.")
        return HER2FieldAnalysis(labels=cells, points=det.points, coords=det.coords,
                                 cell_type=prob.argmax(1).astype(np.int8) if n else np.zeros(0, np.int8),
                                 type_prob=prob, dab_od=inten, h_od=np.zeros(n, np.float32), intensity=cat,
                                 um_per_px=um_per_px, timings=det.timings, nuclei=det.labels, sector_od=sector_od,
                                 completeness=comp, intense_incomplete=ii, measurable=measurable, artefact=art)


def field_counts(a: HER2FieldAnalysis):
    """Counted cells (tumour + measurable ring) per category, and the intense-incomplete count."""
    use = (a.cell_type == TUMOUR) & a.measurable
    return np.bincount(a.intensity[use], minlength=4)[:4], int((a.intense_incomplete & use).sum())
