"""Ki-67 analysis: the Feature 2 pipeline (nuclei -> cell type -> DAB) with binary positivity,
and a case session that reports both the global and the hotspot index.

Field de-duplication, running totals, click-to-correct and correction logging all come from
core.ihc.case.CaseSession. Here `intensity` is 0 (negative) or 1 (positive).
"""
from __future__ import annotations

import numpy as np

from core.ihc.case import CaseSession
from core.ihc.pipeline import TUMOUR, ERPRAnalyzer
from core.ihc.scoring import IntensityThresholds
from core.ki67.scoring import Ki67Result, PositivityThreshold, global_index, hotspot_index


def ki67_analyzer(detector, classifier=None, stain=None, positivity: PositivityThreshold | None = None) -> ERPRAnalyzer:
    """classifier=None counts every detected nucleus as tumour."""
    p = positivity or PositivityThreshold()
    return ERPRAnalyzer(detector, classifier, stain, IntensityThresholds(p.cut, np.inf, np.inf))


class Ki67Session(CaseSession):
    def __init__(self, case_dir, case_id, min_cells=500, max_overlap=0.5, celltype_filter=True):
        super().__init__(case_dir, case_id, "Ki67", min_cells=min_cells, max_overlap=max_overlap)
        self.celltype_filter = celltype_filter
        self.field_weights: dict[int, float] = {}   # optional per-field tissue share (IKWG global)

    def field_counts(self) -> np.ndarray:
        """(n_fields, 2): [negative, positive] tumour nuclei per field."""
        rows = []
        for f in self.fields:
            a = f.analysis
            t = a.cell_type == TUMOUR
            pos = int((a.intensity[t] > 0).sum())
            rows.append([int(t.sum()) - pos, pos])
        return np.array(rows, int).reshape(-1, 2)

    def set_field_weight(self, field_index: int, weight: float):
        self.field(field_index)
        self.field_weights[field_index] = float(weight)

    def global_result(self) -> Ki67Result:
        ids = [f.index for f in self.fields]
        w = None
        if self.field_weights:
            w = [self.field_weights.get(i, 0.0) for i in ids]
        r = global_index(self.field_counts(), ids, weights=w, min_cells=self.min_cells)
        if not self.celltype_filter:
            r.warnings.append("Cell-type filter off: all detected nuclei were counted, not only tumour nuclei.")
        return r

    def hotspot_result(self) -> Ki67Result:
        r = hotspot_index(self.field_counts(), [f.index for f in self.fields], min_cells=self.min_cells)
        if not self.celltype_filter:
            r.warnings.append("Cell-type filter off: all detected nuclei were counted, not only tumour nuclei.")
        return r

    def summary(self) -> dict:
        fc = self.field_counts()
        return dict(case_id=self.case_id, stain="Ki67", n_fields=len(self.fields), celltype_filter=self.celltype_filter,
                    global_=self.global_result().as_dict(), hotspot=self.hotspot_result().as_dict(),
                    fields=[dict(index=f.index, frame=f.frame_file, negative=int(c[0]), positive=int(c[1]),
                                 index_pct=round(100 * c[1] / c.sum(), 2) if c.sum() else None,
                                 weight=self.field_weights.get(f.index), n_corrected=len(f.corrected))
                            for f, c in zip(self.fields, fc)])
