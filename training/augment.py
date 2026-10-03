"""Training augmentation for microscope-camera robustness.

Works on raw RGB in [0,1] (before normalisation), then applies the same percentile
normalisation used at inference.

- geometry: flips / rot90 / scale jitter (±20%, covers calibration error)
- stain: HED jitter (Tellez et al.), brightness / contrast / gamma / saturation
- optics: defocus blur, vignetting, sensor noise, JPEG artefacts
- pseudo-IHC: strip eosin and add DAB to a random subset of nuclei, so the detector
  sees brown (positive) and blue (negative) nuclei on a pale background. This is a
  stand-in until real annotated IHC fields exist - verify on real ER/PR/Ki-67 frames.
"""
import cv2
import numpy as np
from skimage.color import hed2rgb, rgb2hed

from training.data import normalize

rng = np.random.default_rng()


def _geometry(x, y):
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    k = rng.integers(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    if rng.random() < 0.5:
        s = rng.uniform(0.8, 1.25)
        h, w = y.shape
        nh, nw = round(h * s), round(w * s)
        xs = cv2.resize(np.ascontiguousarray(x), (nw, nh), interpolation=cv2.INTER_LINEAR)
        ys = cv2.resize(np.ascontiguousarray(y).astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
        if s >= 1:
            oy, ox = rng.integers(nh - h + 1), rng.integers(nw - w + 1)
            x, y = xs[oy:oy + h, ox:ox + w], ys[oy:oy + h, ox:ox + w]
        else:  # pad with blank glass, never mirror labels
            py, px = h - nh, w - nw
            t, l = rng.integers(py + 1), rng.integers(px + 1)
            glass = np.percentile(xs.reshape(-1, 3), 99, axis=0)
            x = np.empty((h, w, 3), np.float32)
            x[:] = glass
            x[t:t + nh, l:l + nw] = xs
            y = np.zeros((h, w), np.float32)
            y[t:t + nh, l:l + nw] = ys
    return np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y).astype(np.uint16)


def _stain(x, y, p_ihc=0.3):
    hed = rgb2hed(np.clip(x, 1e-4, 1))
    if rng.random() < p_ihc:  # pseudo-IHC
        hed[..., 1] *= rng.uniform(0.0, 0.3)
        ids = np.unique(y)
        ids = ids[ids > 0]
        if len(ids):
            pos = ids[rng.random(len(ids)) < rng.uniform(0, 0.8)]
            mask = np.isin(y, pos).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
            hed[..., 2] += mask * rng.uniform(0.05, 0.5)
            hed[..., 0] *= 1 - 0.5 * mask * rng.random()  # DAB partly masks haematoxylin
        hed[..., 2] += rng.uniform(0, 0.03)  # light background DAB
    alpha = rng.uniform(0.85, 1.15, 3)
    beta = rng.uniform(-0.03, 0.03, 3)
    x = hed2rgb(hed * alpha + beta).astype(np.float32)

    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)          # brightness/contrast
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)                      # gamma
    g = x.mean(-1, keepdims=True)
    x = g + (x - g) * rng.uniform(0.6, 1.4)                            # saturation
    return np.clip(x, 0, 1).astype(np.float32)


def _stain_ihc_only(x, y):
    """Always convert to the pseudo-IHC look (used to evaluate on an IHC-like test set)."""
    return _stain(x, y, p_ihc=1.0)


def _optics(x):
    h, w = x.shape[:2]
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 2.0))
    if rng.random() < 0.5:
        cy, cx = h * rng.uniform(0.2, 0.8), w * rng.uniform(0.2, 0.8)
        yy, xx = np.mgrid[:h, :w]
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.5 * max(h, w)) ** 2
        x = x * (1 - rng.uniform(0.05, 0.35) * r2)[..., None]
    if rng.random() < 0.5:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape)
    x = np.clip(x, 0, 1).astype(np.float32)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    return x


def augmenter(x, y):
    """StarDist augmenter: x raw RGB float32 [0,1], y instance labels."""
    x, y = _geometry(x, y)
    x = _stain(x, y)
    x = _optics(x)
    return normalize(x), y
