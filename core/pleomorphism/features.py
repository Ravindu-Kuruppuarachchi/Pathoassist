"""Nuclear pleomorphism features (Feature 7).

Per nucleus (StarDist label map, calibrated µm/px):
  shape     : area (µm²), perimeter, major / minor axis, eccentricity, solidity, roundness
  chromatin : mean and std of haematoxylin OD inside the nucleus (density, unevenness), dark-spot
              fraction (coarse chromatin / nucleoli proxy) and GLCM texture of the haematoxylin
              channel (contrast, dissimilarity, homogeneity, energy, correlation). Nuclei are
              resampled to a fixed 0.25 µm/px before GLCM so texture is comparable across scanners.
Per field (tumour nuclei only), the distribution of those values: size and its spread
(anisonucleosis), size relative to lymphocytes in the same field and to a reference lymphocyte
nucleus (~38 µm², 7 µm diameter), shape irregularity and chromatin statistics.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops_table

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
REF_LYMPHOCYTE_UM2 = 38.0
GLCM_UM = 0.25                       # texture is computed at this scale
MAX_GLCM_NUCLEI = 150                # per field (speed)


def nucleus_table(labels: np.ndarray, h_od: np.ndarray, um_per_px: float, glcm_ids=None) -> dict:
    """Per-nucleus measurements. labels: (H,W) int (1..n); h_od: (H,W) haematoxylin OD."""
    n = int(labels.max())
    if n == 0:
        return {}
    props = regionprops_table(labels, intensity_image=h_od,
                              properties=("label", "area", "perimeter", "major_axis_length", "minor_axis_length",
                                          "eccentricity", "solidity", "bbox", "intensity_mean"))
    lab = props["label"]
    t = dict(label=lab,
             area=props["area"] * um_per_px ** 2,
             perimeter=props["perimeter"] * um_per_px,
             major=props["major_axis_length"] * um_per_px,
             minor=props["minor_axis_length"] * um_per_px,
             eccentricity=props["eccentricity"],
             solidity=props["solidity"],
             h_mean=props["intensity_mean"])
    t["roundness"] = 4 * np.pi * t["area"] / np.maximum(t["perimeter"] ** 2, 1e-6)
    # per-nucleus std of haematoxylin and dark-spot fraction (via bincount, all nuclei at once)
    flat_l = labels.ravel()
    flat_h = h_od.ravel().astype(np.float64)
    cnt = np.bincount(flat_l, minlength=n + 1)
    s1 = np.bincount(flat_l, weights=flat_h, minlength=n + 1)
    s2 = np.bincount(flat_l, weights=flat_h ** 2, minlength=n + 1)
    mean = s1 / np.maximum(cnt, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0))
    dark = np.bincount(flat_l, weights=(flat_h > mean[flat_l] + 1.5 * std[flat_l]).astype(float), minlength=n + 1)
    t["h_std"] = std[lab]
    t["dark_frac"] = (dark / np.maximum(cnt, 1))[lab]
    # GLCM on a subset of nuclei
    for p in GLCM_PROPS:
        t["glcm_" + p] = np.full(len(lab), np.nan)
    ids = np.arange(len(lab)) if glcm_ids is None else np.asarray(glcm_ids)
    s = um_per_px / GLCM_UM
    hq = np.clip(h_od / 1.2 * 31, 0, 31).astype(np.uint8)          # 32 grey levels over 0-1.2 OD
    for i in ids:
        y0, x0, y1, x1 = (props[f"bbox-{k}"][i] for k in range(4))
        m = labels[y0:y1, x0:x1] == lab[i]
        if m.sum() < 9:
            continue
        crop = np.where(m, hq[y0:y1, x0:x1], 0)
        if abs(s - 1) > 0.05:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            m = cv2.resize(m.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0
        crop = np.where(m, crop + 1, 0).astype(np.uint8)            # 0 = outside the nucleus
        g = graycomatrix(crop, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33, symmetric=True, normed=False)
        g = g[1:, 1:].astype(np.float64)                             # drop pairs touching the background
        tot = g.sum(axis=(0, 1), keepdims=True)
        if tot.min() == 0:
            continue
        g = g / tot
        for p in GLCM_PROPS:
            t["glcm_" + p][i] = float(graycoprops(g, p).mean())
    return t


def field_features(t: dict, tumour: np.ndarray, lymph: np.ndarray) -> dict:
    """Distribution features over tumour nuclei of one field (or pooled fields)."""
    f = {}
    n = int(tumour.sum())
    f["n_tumour"] = n
    if n < 5:
        return f
    a = t["area"][tumour]
    f.update(area_mean=a.mean(), area_median=np.median(a), area_std=a.std(), area_cv=a.std() / a.mean(),
             area_p10=np.percentile(a, 10), area_p90=np.percentile(a, 90),
             area_iqr_rel=(np.percentile(a, 75) - np.percentile(a, 25)) / np.median(a),
             area_max_rel=np.percentile(a, 95) / np.median(a),
             area_rel_ref=np.median(a) / REF_LYMPHOCYTE_UM2)
    la = t["area"][lymph]
    f["area_rel_lymph"] = np.median(a) / np.median(la) if len(la) >= 5 else np.nan
    for k in ("eccentricity", "solidity", "roundness", "major", "minor", "h_mean", "h_std", "dark_frac"):
        v = t[k][tumour]
        f[f"{k}_mean"], f[f"{k}_std"] = float(np.mean(v)), float(np.std(v))
    f["solidity_p10"] = float(np.percentile(t["solidity"][tumour], 10))
    f["major_p90"] = float(np.percentile(t["major"][tumour], 90))
    for p in GLCM_PROPS:
        v = t["glcm_" + p][tumour]
        v = v[~np.isnan(v)]
        f[f"glcm_{p}_mean"] = float(v.mean()) if len(v) else np.nan
        f[f"glcm_{p}_std"] = float(v.std()) if len(v) else np.nan
    return f


def pool_tables(tables):
    """Concatenate per-nucleus tables (several x40 frames -> one x20 field / case)."""
    keys = [k for k in tables[0] if k != "label"]
    return {k: np.concatenate([t[k] for t in tables]) for k in keys}
