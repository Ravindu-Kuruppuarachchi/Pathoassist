"""ER / PR scoring rules: intensity bins, % positive, H-score, Allred, status.

Only tumour nuclei are counted. Definitions:
  intensity bin : 0 negative, 1 weak, 2 moderate, 3 strong (from mean nuclear DAB OD)
  % positive    : (weak + moderate + strong) / all tumour nuclei * 100
  H-score       : 1*%weak + 2*%moderate + 3*%strong                        (0-300)
  Allred PS     : 0 none, 1 <1%, 2 1-10%, 3 >10-33%, 4 >33-66%, 5 >66%
  Allred IS     : average intensity of the *positive* nuclei, rounded (1-3; 0 if none)
  Allred TS     : PS + IS (0, 2-8)
  status        : Negative <1%, Low positive 1-10%, Positive >10%  (ASCO/CAP 2020;
                  "Low positive" is the ER category - PR reports use the same cut-offs)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

INTENSITY_NAMES = ["negative", "weak", "moderate", "strong"]


@dataclass
class IntensityThresholds:
    """Upper-open cut points on mean nuclear DAB OD: od < weak -> 0, < moderate -> 1, < strong -> 2, else 3.
    Defaults are placeholders from the literature range for Ruifrok DAB OD; they MUST be
    calibrated against a pathologist on ~200 labelled nuclei from the lab (calibrate())."""
    weak: float = 0.15
    moderate: float = 0.35
    strong: float = 0.60
    calibrated: bool = False
    n_calibration: int = 0

    def bin(self, dab_od) -> np.ndarray:
        return np.digitize(np.asarray(dab_od, float), [self.weak, self.moderate, self.strong]).astype(np.int8)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def calibrate(dab_od, labels) -> tuple[IntensityThresholds, dict]:
    """Fit the three cut points to pathologist labels (0-3) per nucleus.

    Exhaustive search over candidate cut points (midpoints of sorted ODs) maximising the
    quadratic-weighted kappa, which respects the ordinal nature of the bins.
    Returns thresholds and a report with accuracy, weighted kappa and the confusion matrix.
    """
    od = np.asarray(dab_od, float)
    y = np.asarray(labels, int)
    if len(od) < 40 or len(set(y)) < 4:
        raise ValueError("need labelled nuclei covering all four intensity classes (aim for ~200)")
    qs = np.unique(np.quantile(od, np.linspace(0.02, 0.98, 49)))
    best = (-np.inf, None)
    for i, a in enumerate(qs):
        for j in range(i + 1, len(qs)):
            b = qs[j]
            for c in qs[j + 1:]:
                k = weighted_kappa(y, np.digitize(od, [a, b, c]))
                if k > best[0]:
                    best = (k, (a, b, c))
    a, b, c = best[1]
    th = IntensityThresholds(float(a), float(b), float(c), calibrated=True, n_calibration=len(od))
    pred = th.bin(od)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, pred), 1)
    return th, dict(accuracy=float((pred == y).mean()), weighted_kappa=float(best[0]), confusion=cm.tolist())


def weighted_kappa(a, b, n=4) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal labels 0..n-1."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((n, n))
    np.add.at(O, (a, b), 1)
    w = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (w * E).sum()
    return 1.0 - (w * O).sum() / den if den > 0 else 1.0


def allred_proportion(pct: float) -> int:
    if pct <= 0:
        return 0
    if pct < 1:
        return 1
    if pct <= 10:
        return 2
    if pct <= 33:
        return 3
    if pct <= 66:
        return 4
    return 5


def status(pct: float) -> str:
    if pct < 1:
        return "Negative"
    if pct <= 10:
        return "Low positive"
    return "Positive"


@dataclass
class Score:
    n_tumour: int
    counts: list            # tumour nuclei per intensity bin [neg, weak, mod, strong]
    pct_positive: float
    pct_by_bin: list
    h_score: float
    allred_ps: int
    allred_is: int
    allred_total: int
    status: str
    warnings: list

    def as_dict(self):
        return asdict(self)


def score_counts(counts, min_cells: int = 500) -> Score:
    """counts: tumour nuclei per intensity bin [negative, weak, moderate, strong]."""
    c = np.asarray(counts, float)
    n = int(c.sum())
    pct = (100 * c / n) if n else np.zeros(4)
    pos = c[1:].sum()
    pct_pos = float(pct[1:].sum())
    h = float(pct[1] + 2 * pct[2] + 3 * pct[3])
    is_ = int(np.floor((c[1] + 2 * c[2] + 3 * c[3]) / pos + 0.5)) if pos else 0
    ps = allred_proportion(pct_pos)
    warnings = []
    if n < min_cells:
        warnings.append(f"Only {n} tumour nuclei counted (< {min_cells}); score more fields.")
    if n and 0.5 <= pct_pos <= 1.5:
        warnings.append("Close to the 1% negative / low-positive cut-off; review carefully.")
    if n and 8 <= pct_pos <= 12:
        warnings.append("Close to the 10% low-positive / positive cut-off; review carefully.")
    return Score(n_tumour=n, counts=c.astype(int).tolist(), pct_positive=round(pct_pos, 2),
                 pct_by_bin=[round(float(p), 2) for p in pct], h_score=round(h, 1), allred_ps=ps,
                 allred_is=is_, allred_total=ps + is_ if ps else 0, status=status(pct_pos) if n else "Insufficient",
                 warnings=warnings)
