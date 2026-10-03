"""HER2 per-cell categories and the ASCO/CAP scoring rules.

Per-cell membrane category (from the ring sector DAB ODs):
  0 none                    : no stained ring sector
  1 faint / incomplete      : some staining, but not complete, or complete but only faint
  2 weak-moderate complete  : complete ring (>= complete_min of sectors), intensity < strong_min
  3 strong complete         : complete ring, intensity >= strong_min
A sector is stained when its mean DAB OD >= stain_od; intensity = mean OD of the stained
sectors. Cells with moderate-strong but incomplete rings are counted as category 1 and
reported separately (`intense_incomplete`) because ASCO/CAP treats that pattern as unusual.

Rule engine (ASCO/CAP 2018 focused update, with the 0 subgroups used since the 2023 CAP
template for HER2-low / HER2-ultralow), on % of counted tumour cells:
  3+           strong complete staining in > 10%
  2+           weak-moderate complete staining in > 10% (complete incl. strong > 10%, strong <= 10%)
               -> reflex ISH required
  1+           faint / barely perceptible incomplete staining in > 10%  (any staining > 10%)
  0 ultralow   any membrane staining in > 0% and <= 10% ("0 with membrane staining")
  0 null       no membrane staining ("0 without membrane staining")
These rules are written from the guideline text and MUST be checked by a pathologist against
the current ASCO/CAP document before clinical use (RULES_VERIFIED below).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CELL_CATEGORIES = ["none", "faint/incomplete", "weak-moderate complete", "strong complete"]
RULES_VERSION = "ASCO/CAP 2018 focused update + CAP 2023 HER2-low/ultralow wording"
RULES_VERIFIED = False


@dataclass
class HER2Thresholds:
    """Placeholders fitted on public data (BCI, see training/her2/calibrate_bci.py);
    re-calibrate with a pathologist on local slides (calibrate_cells)."""
    stain_od: float = 0.12        # sector counts as stained
    strong_min: float = 0.45      # mean stained-sector OD for "strong"
    complete_min: float = 0.75    # fraction of sectors stained for "complete"
    faint_max: float = 0.18       # complete rings fainter than this count as faint (category 1)
    source: str = "default placeholders"

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def cell_features(sector_od: np.ndarray, stain_od: float):
    """completeness (fraction of measurable sectors stained) and intensity per cell."""
    valid = ~np.isnan(sector_od)
    stained = np.nan_to_num(sector_od, nan=-1) >= stain_od
    n_valid = valid.sum(1)
    comp = np.where(n_valid > 0, stained.sum(1) / np.maximum(n_valid, 1), 0.0)
    with np.errstate(invalid="ignore"):
        inten = np.where(stained.any(1), np.nansum(np.where(stained, sector_od, 0), 1) / np.maximum(stained.sum(1), 1), 0.0)
    return comp.astype(np.float32), inten.astype(np.float32), n_valid


def classify_cells(sector_od: np.ndarray, th: HER2Thresholds, min_valid_sectors: int = 8):
    """Returns (category (n,), intense_incomplete (n,) bool, completeness, intensity, measurable (n,) bool)."""
    comp, inten, n_valid = cell_features(sector_od, th.stain_od)
    cat = np.zeros(len(comp), np.int8)
    any_stain = comp > 0
    complete = comp >= th.complete_min
    cat[any_stain] = 1
    cat[complete & (inten >= th.faint_max)] = 2
    cat[complete & (inten >= th.strong_min)] = 3
    intense_incomplete = any_stain & ~complete & (inten >= th.strong_min)
    measurable = n_valid >= min_valid_sectors        # ring mostly visible (not cut by the frame edge)
    return cat, intense_incomplete, comp, inten, measurable


@dataclass
class HER2Result:
    score: str                    # "0", "1+", "2+", "3+"
    subgroup: str                 # "null" / "ultralow" for 0, "" otherwise
    n_cells: int
    pct: list                     # % of cells per category [none, faint, weak-mod complete, strong complete]
    pct_any_staining: float
    pct_complete: float
    pct_intense_incomplete: float
    reflex_ish: bool
    her2_low: str                 # "HER2-low (if ISH negative)" etc.
    flags: list = field(default_factory=list)
    rules_version: str = RULES_VERSION
    rules_verified: bool = RULES_VERIFIED

    def as_dict(self):
        return asdict(self)


def her2_score(counts, n_intense_incomplete: int = 0, min_cells: int = 500, near: float = 3.0) -> HER2Result:
    """counts: tumour cells per category [none, faint/incomplete, weak-mod complete, strong complete]."""
    c = np.asarray(counts, float)
    n = int(c.sum())
    pct = 100 * c / n if n else np.zeros(4)
    p_any, p_complete, p_strong = float(pct[1:].sum()), float(pct[2:].sum()), float(pct[3])
    p_ii = 100 * n_intense_incomplete / n if n else 0.0
    flags = []
    if p_strong > 10:
        score, sub = "3+", ""
    elif p_complete > 10:
        score, sub = "2+", ""
    elif p_any > 10:
        score, sub = "1+", ""
    elif p_any > 0:
        score, sub = "0", "ultralow"
    else:
        score, sub = "0", "null"
    reflex = score == "2+"
    if reflex:
        flags.append("IHC 2+ (equivocal): reflex in situ hybridisation (ISH) required.")
    if 0 < p_strong <= 10:
        flags.append(f"Strong complete staining in {p_strong:.1f}% of cells (<= 10%): heterogeneous - review; "
                     "consider reporting the proportion and testing the area by ISH.")
    if p_ii > 10 and score in ("0", "1+"):
        flags.append(f"Moderate-intense but incomplete membrane staining in {p_ii:.1f}% of cells: unusual pattern - "
                     "ASCO/CAP advises considering 2+ (equivocal) and ISH.")
    for name, val, cut in [("0 / 1+", p_any, 10), ("1+ / 2+", p_complete, 10), ("2+ / 3+", p_strong, 10)]:
        if abs(val - cut) <= near:
            flags.append(f"Near the {name} boundary ({val:.1f}% vs 10% cut-off): review carefully.")
    if score == "0" and p_any < 1:
        flags.append(f"{'No' if p_any == 0 else 'Very little'} membrane staining ({p_any:.2f}% of cells): "
                     "the null vs ultralow distinction rests on very few cells - confirm at 40x.")
    if n < min_cells:
        flags.append(f"Only {n} tumour cells assessed (< {min_cells}); score more fields.")
    low = {"1+": "HER2-low", "2+": "HER2-low if ISH negative; HER2-positive if ISH positive",
           "3+": "HER2-positive", "0": "HER2-ultralow" if sub == "ultralow" else "HER2-null (0)"}[score]
    return HER2Result(score, sub, n, [round(float(p), 2) for p in pct], round(p_any, 2), round(p_complete, 2),
                      round(p_ii, 2), reflex, low, flags)


def calibrate_cells(sector_ods: list, labels: list, grid=None) -> tuple[HER2Thresholds, dict]:
    """Fit thresholds to pathologist per-cell categories (0-3) by maximising quadratic-weighted
    kappa. sector_ods: list of (S,) arrays (one per labelled cell); labels: ints 0-3."""
    from core.ihc.scoring import weighted_kappa
    S = np.stack(sector_ods)
    y = np.asarray(labels, int)
    grid = grid or dict(stain_od=np.linspace(0.05, 0.4, 15), strong_min=np.linspace(0.2, 1.0, 17),
                        complete_min=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9], faint_max=np.linspace(0.08, 0.4, 9))
    best = (-np.inf, None)
    for so in grid["stain_od"]:
        for sm in grid["strong_min"]:
            for cm in grid["complete_min"]:
                for fm in grid["faint_max"]:
                    if fm >= sm or fm < so:
                        continue
                    th = HER2Thresholds(float(so), float(sm), float(cm), float(fm), "calibrated on pathologist cell labels")
                    k = weighted_kappa(y, classify_cells(S, th, min_valid_sectors=0)[0])
                    if k > best[0]:
                        best = (k, th)
    th = best[1]
    pred = classify_cells(S, th, min_valid_sectors=0)[0]
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, pred), 1)
    return th, dict(weighted_kappa=float(best[0]), accuracy=float((pred == y).mean()), confusion=cm.tolist(), n=len(y))
