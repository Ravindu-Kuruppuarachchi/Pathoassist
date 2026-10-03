"""HER2 cell geometry and membrane measurement.

cells_from_nuclei : expand StarDist nuclei into cells. Watershed on the smoothed DAB image
                    (membranes are DAB-rich, so cell borders settle on them), seeded by the
                    nuclei and limited to `max_um` (default 8 µm) from the nearest nucleus.
                    method="voronoi" gives plain distance-limited expansion (skimage
                    expand_labels) - more robust on unstained (HER2 0) slides.
membrane_sectors  : the cell (minus its nucleus) is split into `n_sectors` angular sectors
                    around the nucleus centre; in each sector the membrane is taken as the
                    DAB peak (lightly smoothed max) between the nucleus and the cell border ->
                    (n_cells, n_sectors). Taking the peak, rather than a fixed band at the cell
                    border, also finds the membrane of cells bordering stroma, whose outer
                    border is only the distance limit. `ring` (pixels near the cell border)
                    is still returned for the cytoplasm / background artefact check.
Per-cell completeness / intensity are derived from the sector ODs in core.her2.scoring, so
thresholds can be re-fitted later without re-running images.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels, find_boundaries, watershed

N_SECTORS = 16


def cells_from_nuclei(nuclei: np.ndarray, dab: np.ndarray, um_per_px: float, max_um: float = 8.0,
                      method: str = "watershed") -> np.ndarray:
    r = max_um / um_per_px
    if method == "voronoi":
        return expand_labels(nuclei, distance=r)
    dist = ndi.distance_transform_edt(nuclei == 0)
    elev = ndi.gaussian_filter(dab.astype(np.float32), sigma=max(0.5, 0.5 / um_per_px))
    return watershed(elev, markers=nuclei, mask=dist <= r, compactness=0.001)


def membrane_sectors(cells: np.ndarray, nuclei: np.ndarray, dab: np.ndarray, centres_yx: np.ndarray,
                     um_per_px: float, ring_um: float = 1.0, n_sectors: int = N_SECTORS):
    """Returns (sector_od (n, S) with NaN for empty sectors, ring_mask)."""
    n = len(centres_yx)
    border = find_boundaries(cells, mode="inner") & (cells > 0)
    near = ndi.distance_transform_edt(~border) <= max(1.0, ring_um / um_per_px)
    ring = near & (cells > 0) & (nuclei == 0)
    nuc = ndi.binary_dilation(nuclei > 0, iterations=max(1, int(round(0.5 / um_per_px))))
    sm = ndi.gaussian_filter(dab.astype(np.float32), sigma=max(0.5, 0.35 / um_per_px))
    ys, xs = np.nonzero((cells > 0) & ~nuc)
    lab = cells[ys, xs] - 1
    ok = (lab >= 0) & (lab < n)
    ys, xs, lab = ys[ok], xs[ok], lab[ok]
    ang = np.arctan2(ys - centres_yx[lab, 0], xs - centres_yx[lab, 1])
    sec = ((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    key = lab * n_sectors + sec
    od = np.full(n * n_sectors, -np.inf, np.float32)
    np.maximum.at(od, key, sm[ys, xs])
    cnt = np.bincount(key, minlength=n * n_sectors)
    od[cnt < 3] = np.nan                       # sector with (almost) no cytoplasm visible
    return od.reshape(n, n_sectors), ring


def region_medians(cells: np.ndarray, nuclei: np.ndarray, ring: np.ndarray, dab: np.ndarray, tissue: np.ndarray):
    """Median DAB OD of cytoplasm (cell, not ring, not nucleus) and of extracellular tissue -
    used for the cytoplasmic / background staining artefact check."""
    cyto = (cells > 0) & ~ring & (nuclei == 0)
    extra = tissue & (cells == 0)
    med = lambda m: float(np.median(dab[m])) if m.sum() > 200 else float("nan")
    return med(cyto), med(extra)
