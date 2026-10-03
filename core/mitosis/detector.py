"""CPU mitosis detection: heatmap detector -> candidate peaks -> EfficientNet-B0 verifier.

Both models are ONNX with a sidecar JSON (native µm/px, thresholds). Frames are resized to
the models' 0.25 µm/px (40x) using the calibrated µm/px.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class MitosisCandidate:
    x: float            # frame pixels
    y: float
    det_score: float
    ver_score: float
    accepted: bool      # passes both thresholds (the automatic call)


def _prep(rgb):
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]


def _session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def heatmap_peaks(hm, thresh, stride, nms=5):
    mx = cv2.dilate(hm, np.ones((nms, nms), np.uint8))
    ys, xs = np.nonzero((hm == mx) & (hm >= thresh))
    s = hm[ys, xs]
    o = np.argsort(-s)
    return np.stack([(xs[o] + 0.5) * stride, (ys[o] + 0.5) * stride, s[o]], 1) if len(o) else np.zeros((0, 3))


class MitosisDetector:
    def __init__(self, det_path: str | Path, ver_path: str | Path | None = None, num_threads: int | None = None,
                 tile: int = 1024, margin: int = 64):
        det_path = Path(det_path)
        meta = json.loads(det_path.with_suffix(".json").read_text())
        self.um_per_px = float(meta["um_per_px"])
        self.stride = int(meta["stride"])
        self.cand_thresh = float(meta["cand_thresh"])
        self.det_thresh = float(meta["det_thresh"])
        self.ver_thresh = float(meta.get("ver_thresh", 0.5))
        self.vcrop = int(meta.get("ver_crop", 96))
        self.det = _session(det_path, num_threads)
        self.ver = _session(ver_path, num_threads) if ver_path else None
        if self.ver is None:  # detector-only mode uses the detector-only threshold
            self.det_thresh = float(meta.get("det_only_thresh", self.det_thresh))
        self.tile, self.margin = tile, margin

    def heatmap(self, img):
        """Stride-s probability map for an image already at model scale."""
        H, W = img.shape[:2]
        T, m = self.tile, self.margin
        step = T - 2 * m
        ny, nx = max(1, -(-H // step)), max(1, -(-W // step))
        Hp, Wp = ny * step + 2 * m, nx * step + 2 * m
        pad = np.pad(img, ((m, Hp - H - m), (m, Wp - W - m), (0, 0)), mode="reflect")
        s = self.stride
        out = np.zeros((ny * step // s, nx * step // s), np.float32)
        for i in range(ny):
            for j in range(nx):
                logit = self.det.run(None, {"image": _prep(pad[i * step:i * step + T, j * step:j * step + T])})[0][0, 0]
                p = 1 / (1 + np.exp(-logit))
                out[i * step // s:(i + 1) * step // s, j * step // s:(j + 1) * step // s] = \
                    p[m // s:(m + step) // s, m // s:(m + step) // s]
        return out[:-(-H // s), :-(-W // s)]

    def verify(self, img, pts):
        if self.ver is None or not len(pts):
            return np.ones(len(pts), np.float32)
        r = self.vcrop // 2
        pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]
        xy = np.round(pts[:, :2]).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)   # peaks at the border can sit up to stride/2 outside
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        crops = np.stack([pad[y:y + 2 * r, x:x + 2 * r] for x, y in xy])
        x = ((crops.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)
        logit = self.ver.run(None, {"crops": np.ascontiguousarray(x)})[0][:, 0]
        return (1 / (1 + np.exp(-logit))).astype(np.float32)

    def detect(self, frame_rgb: np.ndarray, um_per_px: float | None = None):
        """Returns (candidates, timings). Candidates below cand_thresh are dropped."""
        t0 = time.perf_counter()
        s = 1.0 if um_per_px is None else um_per_px / self.um_per_px
        img = frame_rgb if abs(s - 1) < 0.02 else cv2.resize(
            frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        s = 1.0 if abs(s - 1) < 0.02 else s
        hm = self.heatmap(img)
        t1 = time.perf_counter()
        pts = heatmap_peaks(hm, self.cand_thresh, self.stride)
        v = self.verify(img, pts)
        t2 = time.perf_counter()
        cands = [MitosisCandidate(float(x / s), float(y / s), float(d), float(vv),
                                  bool(d >= self.det_thresh and vv >= self.ver_thresh))
                 for (x, y, d), vv in zip(pts, v)]
        return cands, dict(detector=t1 - t0, verifier=t2 - t1, total=t2 - t0)
