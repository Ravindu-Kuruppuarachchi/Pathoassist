"""Haematoxylin-DAB colour deconvolution and per-nucleus DAB optical density.

Default stain vectors are Ruifrok & Johnston's (the same ones skimage.rgb2hed uses).
`estimate_hdab_matrix` fits the lab's own H and DAB vectors from a few representative
frames (Macenko-style: principal plane of tissue OD, then the extreme angles), which
matters because DAB hue and counterstain strength vary between labs and batches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# rows = stain OD vectors (R, G, B); from skimage.color.colorconv.rgb_from_hed
RUIFROK_H = np.array([0.65, 0.70, 0.29])
RUIFROK_DAB = np.array([0.27, 0.57, 0.78])


def _unit(v):
    return v / np.linalg.norm(v)


class StainMatrix:
    def __init__(self, h=RUIFROK_H, dab=RUIFROK_DAB, background=(255.0, 255.0, 255.0)):
        h, dab = _unit(np.asarray(h, float)), _unit(np.asarray(dab, float))
        res = _unit(np.cross(h, dab))
        self.rgb_from_stain = np.stack([h, dab, res])          # (3 stains, 3 channels)
        self.stain_from_rgb = np.linalg.inv(self.rgb_from_stain)
        self.background = np.asarray(background, float)       # white level (flat-field)

    def od(self, rgb: np.ndarray) -> np.ndarray:
        I0 = self.background
        return (-np.log10(np.clip(rgb.astype(np.float32), 1, None) / I0)).clip(0)

    def concentrations(self, rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 RGB -> (H,W,3) float32 [haematoxylin, DAB, residual] optical density."""
        return (self.od(rgb) @ self.stain_from_rgb).astype(np.float32)

    def to_json(self) -> str:
        return json.dumps(dict(h=self.rgb_from_stain[0].tolist(), dab=self.rgb_from_stain[1].tolist(),
                               background=self.background.tolist()))

    @classmethod
    def from_json(cls, s: str | Path):
        d = json.loads(Path(s).read_text() if isinstance(s, Path) or str(s).endswith(".json") else s)
        return cls(d["h"], d["dab"], d.get("background", (255, 255, 255)))


def estimate_hdab_matrix(frames, beta=0.15, alpha=1.0, background=(255, 255, 255)) -> StainMatrix:
    """Macenko estimate of H and DAB vectors from representative IHC frames (RGB uint8).

    Pixels with total OD < beta (glass) are dropped; OD is projected onto its two main
    eigenvectors; the alpha / 100-alpha percentile angles give the two stain directions.
    The bluer one (higher R-channel OD relative to B) is haematoxylin.
    """
    I0 = np.asarray(background, float)
    od = np.concatenate([-np.log10(np.clip(f.reshape(-1, 3).astype(float), 1, None) / I0) for f in frames])
    od = od[(od > beta).all(1)]
    if len(od) < 1000:
        raise ValueError("not enough stained tissue to estimate stain vectors")
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, 1:3]
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, (alpha, 100 - alpha))
    v1 = _unit(plane @ np.array([np.cos(lo), np.sin(lo)]))
    v2 = _unit(plane @ np.array([np.cos(hi), np.sin(hi)]))
    v1, v2 = np.abs(v1), np.abs(v2)
    # haematoxylin absorbs most in red; DAB (brown) absorbs most in blue
    h, dab = (v1, v2) if v1[0] / v1[2] > v2[0] / v2[2] else (v2, v1)
    return StainMatrix(h, dab, background)


def per_nucleus_stain(labels: np.ndarray, conc: np.ndarray, n: int | None = None):
    """Mean haematoxylin and DAB OD inside each labelled nucleus.

    labels: (H,W) int instance map (1..n); conc: (H,W,3) from StainMatrix.concentrations.
    Returns (mean_h, mean_dab), each shape (n,), index i <-> label i+1.
    """
    n = int(labels.max()) if n is None else n
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n + 1)[1:].astype(np.float64)
    area[area == 0] = np.nan
    h = np.bincount(flat, weights=conc[..., 0].ravel(), minlength=n + 1)[1:] / area
    d = np.bincount(flat, weights=conc[..., 1].ravel(), minlength=n + 1)[1:] / area
    return h.astype(np.float32), d.astype(np.float32)
