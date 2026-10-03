"""Tubule formation (Feature 8): gland lumen detection and tubular area at ~10x.

Works at 1 µm/px (a 10x camera frame is resized there from its calibrated µm/px).
  candidates : near-white connected regions (low total H&E optical density), 15 µm² .. 0.2 mm²
  features   : shape (µm), contents (secretion), and three rings around the space (0-6, 6-15,
               15-25 µm): haematoxylin / eosin density and nuclear-pixel fraction - a true lumen is
               lined by epithelium (dense ring of nuclei); stromal clefts, vessels, fat and tears
               are not - plus how many other white spaces are nearby (fat comes in clusters)
  classifier : LightGBM lumen vs non-lumen (models/tubule_lumen_lgbm.txt), trained on BreCaHAD
  tubule area: each accepted lumen + its epithelial lining (dilated by `lining_um` within tissue)
Tubule % = tubular area / invasive tumour area, accumulated over the mosaic (core.tubules.score).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

from core.ihc.stain import RUIFROK_H, StainMatrix

UM = 1.0                                      # working scale, µm/px
HE = StainMatrix(RUIFROK_H, np.array([0.07, 0.99, 0.11]))
MIN_AREA, MAX_AREA = 15.0, 200_000.0          # µm²
WHITE_OD = 0.22                               # total OD below this = unstained space
RINGS = [(0, 6), (6, 15), (15, 25)]           # µm


def white_balance(img):
    """Scale each channel so the brightest background (99.5th percentile) reads as white: camera
    white balance otherwise makes glass look 'stained' and hides every lumen."""
    ref = np.percentile(img.reshape(-1, 3), 99.5, axis=0).clip(120, 255)
    return np.clip(img.astype(np.float32) * (250.0 / ref), 0, 255).astype(np.uint8)


def to_working(frame_rgb, um_per_px):
    frame_rgb = white_balance(frame_rgb)
    s = um_per_px / UM
    if abs(s - 1) < 0.02:
        return frame_rgb, 1.0
    return cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), s


def field_mask(img):
    """Inside the microscope field: excludes the black area outside the eyepiece circle of camera
    photos (it would read as extremely dense 'stain')."""
    v = img.max(-1)
    m = cv2.morphologyEx((v > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.erode(m, np.ones((9, 9), np.uint8)) > 0


def white_mask(img):
    od = (-np.log10(np.clip(img.astype(np.float32), 1, 255) / 255)).sum(-1)
    m = ((od < WHITE_OD) & field_mask(img)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m, od


@dataclass
class Candidates:
    labels: np.ndarray            # (h, w) component map at working scale (1..n)
    feats: np.ndarray             # (n, F)
    centroids: np.ndarray         # (n, 2) x, y at working scale
    areas: np.ndarray             # µm²
    tissue: np.ndarray            # (h, w) bool tissue mask
    scale: float                  # working px per frame px
    names: list = field(default_factory=list)


def candidates(frame_rgb, um_per_px) -> Candidates:
    img, s = to_working(frame_rgb, um_per_px)
    wm, od = white_mask(img)
    conc = HE.concentrations(img)
    h_od, e_od = conc[..., 0], conc[..., 1]
    lab, n = ndi.label(wm)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = (sizes >= MIN_AREA / UM ** 2) & (sizes <= MAX_AREA / UM ** 2)
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    lab = remap[lab]
    n = int(keep.sum())
    tissue = (od >= WHITE_OD) & field_mask(img)
    big_white = (wm > 0) & (lab == 0)                       # very large white areas (glass, big fat) - not candidates
    H, W = lab.shape
    nuc = h_od > np.percentile(h_od[tissue], 75) if tissue.any() else np.zeros_like(tissue)
    # distance from each pixel to the nearest candidate, and which candidate (for ring statistics)
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    owner = lab[iy, ix]
    names = ["area", "perimeter", "solidity", "eccentricity", "extent", "eq_diam", "major", "minor", "touches_border",
             "inside_od_mean", "inside_od_std"]
    for a, b in RINGS:
        names += [f"r{a}_{b}_h", f"r{a}_{b}_e", f"r{a}_{b}_nuc", f"r{a}_{b}_white", f"r{a}_{b}_h_std"]
    names += ["ring_h_contrast", "white_neighbours_40um", "big_white_near"]
    F = np.zeros((n, len(names)), np.float32)
    cents = np.zeros((n, 2), np.float32)
    areas = np.zeros(n, np.float32)
    if n == 0:
        return Candidates(lab, F, cents, areas, tissue, s, names)
    props = regionprops(lab)
    flat_owner, flat_d = owner.ravel(), dist.ravel()
    ring_stats = {}
    for a, b in RINGS:
        sel = (flat_d > a / UM) & (flat_d <= b / UM)
        o = flat_owner[sel]
        cnt = np.bincount(o, minlength=n + 1).astype(np.float32)
        def m(v):
            return np.bincount(o, weights=v.ravel()[sel], minlength=n + 1) / np.maximum(cnt, 1)
        hm = m(h_od)
        ring_stats[(a, b)] = (hm, m(e_od), m(nuc.astype(np.float32)), m(wm.astype(np.float32)),
                              np.sqrt(np.maximum(m(h_od ** 2) - hm ** 2, 0)))
    inside_cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float32)
    in_mean = np.bincount(lab.ravel(), weights=od.ravel(), minlength=n + 1) / np.maximum(inside_cnt, 1)
    in_sq = np.bincount(lab.ravel(), weights=(od ** 2).ravel(), minlength=n + 1) / np.maximum(inside_cnt, 1)
    big_dist = ndi.distance_transform_edt(~big_white) if big_white.any() else np.full(lab.shape, 1e6)
    for i, p in enumerate(props):
        k = p.label
        y0, x0, y1, x1 = p.bbox
        cy, cx = p.centroid
        cents[i] = (cx, cy)
        areas[i] = p.area * UM ** 2
        row = [p.area * UM ** 2, p.perimeter * UM, p.solidity, p.eccentricity, p.extent, p.equivalent_diameter_area * UM,
               p.major_axis_length * UM, p.minor_axis_length * UM, float(y0 == 0 or x0 == 0 or y1 == H or x1 == W),
               in_mean[k], np.sqrt(max(in_sq[k] - in_mean[k] ** 2, 0))]
        for a, b in RINGS:
            row += [v[k] for v in ring_stats[(a, b)]]
        row += [ring_stats[RINGS[0]][0][k] - ring_stats[RINGS[2]][0][k]]
        F[i, :len(row)] = row
    # neighbourhood: other candidates whose centroid lies within 40 µm of this candidate's border
    from scipy.spatial import cKDTree
    tree = cKDTree(cents)
    radii = 40 / UM + F[:, 5] / 2
    F[:, names.index("white_neighbours_40um")] = [len(tree.query_ball_point(cents[i], radii[i])) - 1 for i in range(n)]
    ci = np.clip(cents.astype(int), 0, [W - 1, H - 1])
    F[:, names.index("big_white_near")] = (big_dist[ci[:, 1], ci[:, 0]] < 30 / UM).astype(np.float32)
    return Candidates(lab, F, cents, areas, tissue, s, names)


class LumenClassifier:
    def __init__(self, model_path, threshold=None):
        import json
        import lightgbm as lgb
        self.m = lgb.Booster(model_file=str(model_path))
        meta = json.loads(Path(model_path).with_suffix(".json").read_text())
        self.threshold = threshold if threshold is not None else meta.get("threshold", 0.5)
        self.names = meta["features"]

    def predict(self, c: Candidates) -> np.ndarray:
        if len(c.feats) == 0:
            return np.zeros(0)
        return self.m.predict(c.feats)


def epithelium_mask(img_working, tissue, sigma_um=6.0, min_density=0.30):
    """Cellular (epithelial) tissue: local density of haematoxylin-rich (nuclear) pixels. Tumour
    epithelium is crowded with nuclei; stroma has sparse nuclei and much eosin."""
    h = HE.concentrations(img_working)[..., 0]
    nuc = (h > np.percentile(h[tissue], 70)) if tissue.any() else np.zeros_like(tissue)
    dens = cv2.GaussianBlur(nuc.astype(np.float32), (0, 0), sigma_um / UM)
    return (dens >= min_density) & tissue


@dataclass
class TubuleField:
    lumen_prob: np.ndarray
    lumens: np.ndarray            # (n,) bool accepted
    tubule_mask: np.ndarray       # (h, w) bool at working scale
    tissue: np.ndarray
    tumour: np.ndarray            # (h, w) bool invasive tumour at working scale
    scale: float

    @property
    def tumour_area_mm2(self):
        return float(self.tumour.sum()) * (UM / 1000) ** 2

    @property
    def tubule_area_mm2(self):
        return float((self.tubule_mask & self.tumour).sum()) * (UM / 1000) ** 2


def analyze_field(frame_rgb, um_per_px, clf: LumenClassifier, tumour_prob=None, lining_um=25.0, tumour_thresh=0.5,
                  min_lumen_um2=30.0, min_enclosed=0.6) -> TubuleField:
    """Tubule formation is the share of TUMOUR EPITHELIUM arranged around lumina (stroma excluded):
      tumour  = epithelium (crowded nuclei) inside the invasive region (+ the accepted lumina)
      tubule  = epithelium within `lining_um` (1-2 cell layers) of an accepted lumen (+ the lumina)
    tumour_prob: invasive-tumour probability map from the Feature 5 region model (any resolution,
    frame extent); None = the whole field is tumour (e.g. the pathologist is sweeping the tumour)."""
    c = candidates(frame_rgb, um_per_px)
    p = clf.predict(c)
    img, _ = to_working(frame_rgb, um_per_px)
    epi = epithelium_mask(img, c.tissue)
    # a tubule lumen is enclosed by tumour epithelium: share of the 0-6 µm ring that is epithelium
    n = len(p)
    if n:
        dist, (iy, ix) = ndi.distance_transform_edt(c.labels == 0, return_indices=True)
        ring = (dist > 0) & (dist <= 6 / UM)
        own = c.labels[iy, ix][ring]
        cnt = np.bincount(own, minlength=n + 1)[1:]
        enclosed = np.bincount(own, weights=epi[ring].astype(float), minlength=n + 1)[1:] / np.maximum(cnt, 1)
    else:
        enclosed = np.zeros(0)
    acc = (p >= clf.threshold) & (c.areas >= min_lumen_um2) & (enclosed >= min_enclosed)
    lum = np.isin(c.labels, np.nonzero(acc)[0] + 1)
    k = int(round(lining_um / UM))
    near = cv2.dilate(lum.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    if tumour_prob is None:
        region = np.ones_like(epi)
    else:
        tp = cv2.resize(tumour_prob.astype(np.float32), c.labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
        region = tp >= tumour_thresh
    tumour = (epi | lum) & region
    tubule = ((near & epi) | lum) & region
    return TubuleField(p, acc, tubule, c.tissue, tumour, c.scale)
