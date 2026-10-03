"""Feature 5 runtime: tissue-region map on a 10x frame, and auto-magnification.

TissueRegionModel: the frame is resized to the model scale (~0.93 µm/px), cut into 224 px
patches with 50% overlap, classified by the distilled EfficientNet-B0 (ONNX, CPU), and the
patch probabilities are averaged back to a coarse map (one cell per 112 px).

MagnificationEstimator: predicts µm/px from tissue crops of a frame and snaps it to the nearest
objective of the microscope's calibration (replaces the objective hotkeys).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REGION_COLORS = {0: (220, 0, 0), 1: (150, 0, 200), 2: (0, 170, 0)}     # invasive, in situ, benign


def _session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@dataclass
class RegionMap:
    probs: np.ndarray          # (gh, gw, 5) class probabilities on the patch grid
    cell_px: float             # grid cell size in frame px
    classes: list

    def upsample(self, shape):
        h, w = shape[:2]
        return np.stack([cv2.resize(self.probs[..., k], (w, h), interpolation=cv2.INTER_LINEAR)
                         for k in range(self.probs.shape[-1])], -1)

    def tumour_prob(self, shape=None):
        p = self.probs[..., 0] + self.probs[..., 1]
        return p if shape is None else cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def fractions(self):
        """Share of tissue (non-background cells) predicted as each class."""
        c = self.probs.argmax(-1)
        tissue = c != 4
        return {self.classes[k]: float((c[tissue] == k).mean()) if tissue.any() else 0.0 for k in range(4)}


class TissueRegionModel:
    def __init__(self, model_path, num_threads=None, batch=32):
        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text())
        self.um, self.patch, self.classes = meta["um_per_px"], meta["patch"], meta["classes"]
        self.sess, self.batch = _session(model_path, num_threads), batch

    def predict(self, frame_rgb, um_per_px, overlap: bool = False) -> RegionMap:
        """overlap=False (live view): non-overlapping patches, ~4x faster; overlap=True: 50% overlap,
        smoother map (reports / still frames)."""
        s = um_per_px / self.um
        img = cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        P = self.patch
        st = P // 2 if overlap else P
        H, W = img.shape[:2]
        if H < P or W < P:
            img = cv2.copyMakeBorder(img, 0, max(0, P - H), 0, max(0, P - W), cv2.BORDER_REFLECT)
            H, W = img.shape[:2]
        ys = list(range(0, H - P + 1, st)) or [0]
        xs = list(range(0, W - P + 1, st)) or [0]
        tiles = np.stack([img[y:y + P, x:x + P] for y in ys for x in xs]).astype(np.float32) / 255
        x = ((tiles - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
        logits = np.concatenate([self.sess.run(None, {"patches": x[i:i + self.batch]})[0] for i in range(0, len(x), self.batch)])
        e = np.exp(logits - logits.max(1, keepdims=True))
        pr = (e / e.sum(1, keepdims=True)).reshape(len(ys), len(xs), -1)
        if not overlap:
            return RegionMap(pr.astype(np.float32), st / s, self.classes)
        # each patch covers 2x2 grid cells of size `st`; average overlapping patches per cell
        gh, gw = len(ys) + 1, len(xs) + 1
        acc = np.zeros((gh, gw, pr.shape[-1]), np.float32)
        cnt = np.zeros((gh, gw, 1), np.float32)
        for dy in (0, 1):
            for dx in (0, 1):
                acc[dy:dy + len(ys), dx:dx + len(xs)] += pr
                cnt[dy:dy + len(ys), dx:dx + len(xs)] += 1
        return RegionMap(acc / np.maximum(cnt, 1), st / s, self.classes)


def draw_region_overlay(frame_rgb, rm: RegionMap, alpha=0.35, min_prob=0.5):
    out = frame_rgb.astype(np.float32)
    full = rm.upsample(frame_rgb.shape)
    c = full.argmax(-1)
    for k, col in REGION_COLORS.items():
        m = (c == k) & (full[..., k] >= min_prob)
        out[m] = out[m] * (1 - alpha) + np.array(col) * alpha
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)
    return out.astype(np.uint8)


class MagnificationEstimator:
    def __init__(self, model_path, objectives: dict, num_threads=None, n_crops=8):
        """objectives: calibration {name: µm/px}, e.g. {"4x": 2.42, "10x": 0.97, "20x": 0.48, "40x": 0.24}."""
        model_path = Path(model_path)
        self.crop = json.loads(model_path.with_suffix(".json").read_text())["crop"]
        self.sess, self.objectives, self.n_crops = _session(model_path, num_threads), objectives, n_crops

    def estimate(self, frame_rgb, seed=0):
        """Returns (objective name, estimated µm/px, confidence 0-1 from crop agreement) or (None, nan, 0)
        when the frame has no tissue."""
        rng = np.random.default_rng(seed)
        H, W = frame_rgb.shape[:2]
        C = self.crop
        cs = []
        for _ in range(self.n_crops * 4):
            y, x = rng.integers(0, H - C + 1), rng.integers(0, W - C + 1)
            c = frame_rgb[y:y + C, x:x + C]
            if (cv2.cvtColor(c, cv2.COLOR_RGB2HSV)[..., 1] > 20).mean() > 0.25:
                cs.append(c)
            if len(cs) == self.n_crops:
                break
        if len(cs) < 2:
            return None, float("nan"), 0.0
        x = ((np.stack(cs).astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
        est = self.sess.run(None, {"crops": x})[0][:, 0]
        med = float(np.median(est))
        names = list(self.objectives)
        logs = np.log2(np.array([self.objectives[n] for n in names]))
        best = names[int(np.abs(logs - med).argmin())]
        conf = float(np.mean(np.abs(est - med) < 0.5))
        return best, float(2 ** med), conf
