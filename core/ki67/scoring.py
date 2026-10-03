"""Ki-67 index: binary positivity, global vs hotspot averaging.

Any nuclear DAB staining above the cut counts as positive (intensity is not graded for Ki-67).
Only tumour nuclei are counted (when the cell-type classifier is enabled).

  global  : all counted fields pooled - positive / all tumour nuclei. If the pathologist gives
            per-field weights (e.g. estimated share of the tumour area each field represents,
            the IKWG "global" method), the index is the weighted mean of the field indices.
  hotspot : fields sorted by index, highest first, taken until >= min_cells tumour nuclei
            (default 500) - the index over those fields.

Bands follow the International Ki67 in Breast Cancer Working Group (IKWG, Nielsen et al.,
JNCI 2021): <= 5% low and >= 30% high are the reproducible ranges; 6-29% is intermediate,
where inter-observer agreement is poor and the value should be interpreted with caution.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LOW_MAX, HIGH_MIN = 5.0, 30.0


@dataclass
class PositivityThreshold:
    """Mean nuclear DAB OD >= cut -> Ki-67 positive. Default 0.52 is fitted on BC-DeepLIIF
    training fields (Ruifrok stain vectors); re-fit on local slides with calibrate_positivity()."""
    cut: float = 0.52
    calibrated_on: str = "BC-DeepLIIF training set (public)"
    n_calibration: int = 0

    def positive(self, dab_od) -> np.ndarray:
        return np.asarray(dab_od, float) >= self.cut

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def calibrate_positivity(dab_od, is_positive, source="local pathologist labels") -> tuple[PositivityThreshold, dict]:
    """Fit the cut that maximises balanced accuracy against pathologist positive/negative labels."""
    od = np.asarray(dab_od, float)
    y = np.asarray(is_positive, bool)
    if y.sum() < 10 or (~y).sum() < 10:
        raise ValueError("need at least 10 positive and 10 negative labelled nuclei (aim for ~200)")
    cands = np.unique(np.quantile(od, np.linspace(0.01, 0.99, 197)))

    def bacc(t):
        p = od >= t
        return ((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2

    scores = np.array([bacc(t) for t in cands])
    t = float(cands[scores.argmax()])
    p = od >= t
    return PositivityThreshold(t, source, len(od)), dict(
        balanced_accuracy=float(scores.max()), sensitivity=float((p & y).sum() / y.sum()),
        specificity=float((~p & ~y).sum() / (~y).sum()), n=int(len(od)))


def band(index_pct: float) -> str:
    if index_pct <= LOW_MAX:
        return "Low (<=5%)"
    if index_pct >= HIGH_MIN:
        return "High (>=30%)"
    return "Intermediate (6-29%)"


@dataclass
class Ki67Result:
    mode: str                 # "global", "global (weighted)" or "hotspot"
    index_pct: float
    n_positive: int
    n_counted: int
    fields_used: list         # field indices contributing
    band: str
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _counts(field_counts):
    c = np.asarray(field_counts, float).reshape(-1, 2)     # per field: [negative, positive]
    return c[:, 0], c[:, 1]


def global_index(field_counts, field_ids=None, weights=None, min_cells=500) -> Ki67Result:
    neg, pos = _counts(field_counts)
    ids = list(range(len(neg))) if field_ids is None else list(field_ids)
    tot = neg + pos
    n_pos, n = int(pos.sum()), int(tot.sum())
    warnings = []
    if weights is not None:
        w = np.asarray(weights, float)
        ok = tot > 0
        if not ok.any() or w[ok].sum() <= 0:
            idx = 0.0
        else:
            idx = float((w[ok] * pos[ok] / tot[ok]).sum() / w[ok].sum() * 100)
        mode = "global (weighted)"
    else:
        idx = 100 * n_pos / n if n else 0.0
        mode = "global"
    if n < min_cells:
        warnings.append(f"Only {n} tumour nuclei counted (< {min_cells}); score more fields.")
    return Ki67Result(mode, round(idx, 2), n_pos, n, ids, band(idx) if n else "Insufficient", warnings)


def hotspot_index(field_counts, field_ids=None, min_cells=500) -> Ki67Result:
    neg, pos = _counts(field_counts)
    ids = list(range(len(neg))) if field_ids is None else list(field_ids)
    tot = neg + pos
    idx_f = np.where(tot > 0, pos / np.maximum(tot, 1), -1)
    order = [i for i in np.argsort(-idx_f, kind="stable") if tot[i] > 0]
    used, n, n_pos = [], 0, 0
    for i in order:
        used.append(ids[i])
        n += int(tot[i])
        n_pos += int(pos[i])
        if n >= min_cells:
            break
    idx = 100 * n_pos / n if n else 0.0
    warnings = []
    if n < min_cells:
        warnings.append(f"Hotspot has only {n} tumour nuclei (< {min_cells}); the hotspot index is unreliable.")
    if len(used) > 1:
        warnings.append(f"Hotspot pooled over {len(used)} fields to reach {min_cells} nuclei; "
                        "make sure they are adjacent fields from the same hotspot.")
    return Ki67Result("hotspot", round(idx, 2), n_pos, n, used, band(idx) if n else "Insufficient", warnings)
