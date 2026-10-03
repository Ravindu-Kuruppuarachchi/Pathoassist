"""Simple field de-duplication: skip a new dwell frame that overlaps a counted one by > 50%.

Phase correlation on small grey thumbnails gives candidate translations between two
frames (zero-padded, so large shifts do not wrap); each candidate is checked by normalised
cross-correlation (NCC) over the overlapping area and the best one wins. With the frame
size the shift gives the overlap fraction. Low NCC means the frames share no content,
so they are treated as non-overlapping.
Full mosaicking (Feature 5) replaces this later.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

THUMB = 256          # longest side of the thumbnail used for registration
MIN_RESPONSE = 0.5   # NCC over the overlap below this = unrelated fields (tests/test_ihc.py, test_mitosis.py)
MIN_AREA = 0.05      # ignore candidate shifts leaving < 5% overlap (NCC unreliable)


def thumbnail(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    s = THUMB / max(h, w)
    g = cv2.cvtColor(cv2.resize(frame_rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return (g - g.mean()) / (g.std() + 1e-6)


@dataclass
class Overlap:
    index: int          # which previous field
    fraction: float     # overlapping area / frame area
    shift: tuple        # (dx, dy) in thumbnail pixels
    response: float     # NCC over the overlap


def register(a: np.ndarray, b: np.ndarray, n_peaks: int = 5) -> tuple[tuple, float]:
    """Translation of thumbnail b relative to a: b's origin lies at (dx, dy) in a's pixels,
    i.e. b[y, x] ~ a[y + dy, x + dx]. Returns ((dx, dy), ncc)."""
    h, w = a.shape
    Fa = np.fft.rfft2(a, s=(2 * h, 2 * w))
    Fb = np.fft.rfft2(b, s=(2 * h, 2 * w))
    R = Fa * np.conj(Fb)
    r = np.fft.irfft2(R / (np.abs(R) + 1e-9), s=(2 * h, 2 * w))
    flat = np.argpartition(r.ravel(), -n_peaks)[-n_peaks:]
    best = ((0.0, 0.0), -1.0)
    for idx in flat:
        dy, dx = np.unravel_index(idx, r.shape)
        dy = dy - 2 * h if dy >= h else dy
        dx = dx - 2 * w if dx >= w else dx
        oh, ow = h - abs(dy), w - abs(dx)
        if oh <= 0 or ow <= 0 or oh * ow < MIN_AREA * h * w:
            continue
        pa = a[max(dy, 0):max(dy, 0) + oh, max(dx, 0):max(dx, 0) + ow]
        pb = b[max(-dy, 0):max(-dy, 0) + oh, max(-dx, 0):max(-dx, 0) + ow]
        pa, pb = pa - pa.mean(), pb - pb.mean()
        ncc = float((pa * pb).sum() / (np.sqrt((pa ** 2).sum() * (pb ** 2).sum()) + 1e-9))
        if ncc > best[1]:
            best = ((float(dx), float(dy)), ncc)
    return best


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple, float]:
    """Overlap fraction between two same-size thumbnails, b's offset (dx, dy) in a, and NCC."""
    if a.shape != b.shape:
        return 0.0, (0.0, 0.0), 0.0
    (dx, dy), ncc = register(a, b)
    if ncc < MIN_RESPONSE:
        return 0.0, (dx, dy), ncc
    h, w = a.shape
    frac = max(0.0, w - abs(dx)) * max(0.0, h - abs(dy)) / (w * h)
    return float(frac), (dx, dy), ncc


class FieldDeduplicator:
    def __init__(self, max_overlap: float = 0.5):
        self.max_overlap = max_overlap
        self.thumbs: list[np.ndarray] = []

    def check(self, frame_rgb: np.ndarray) -> Overlap | None:
        """Largest overlap with any accepted field, or None if the frame is new enough."""
        t = thumbnail(frame_rgb)
        best = None
        for i, prev in enumerate(self.thumbs):
            frac, shift, resp = overlap(prev, t)
            if frac > self.max_overlap and (best is None or frac > best.fraction):
                best = Overlap(i, frac, shift, resp)
        return best

    def add(self, frame_rgb: np.ndarray):
        self.thumbs.append(thumbnail(frame_rgb))

    def remove(self, index: int):
        self.thumbs.pop(index)
