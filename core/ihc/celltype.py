"""Cell-type classifier runtime: 64x64 crops around nuclei -> tumour / lymphocyte / stroma / other.

The model (MobileNetV3-Small, ONNX) expects crops at its native scale (sidecar JSON
`um_per_px`, 0.25 = 40x), so frames are rescaled with the calibrated µm/px first.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class CellTypeClassifier:
    def __init__(self, model_path: str | Path, num_threads: int | None = None, batch: int = 256):
        import onnxruntime as ort
        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text())
        self.crop = int(meta["crop"])
        self.um_per_px = float(meta["um_per_px"])
        self.classes = meta["classes"]
        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.batch = batch

    def crops(self, frame_rgb: np.ndarray, points_yx: np.ndarray, um_per_px: float | None) -> np.ndarray:
        s = 1.0 if um_per_px is None else um_per_px / self.um_per_px
        img = frame_rgb if abs(s - 1) < 0.02 else cv2.resize(
            frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        s = s if abs(s - 1) >= 0.02 else 1.0
        r = self.crop // 2
        pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
        c = np.round(points_yx * s).astype(int) + r
        out = np.empty((len(c), self.crop, self.crop, 3), np.uint8)
        for i, (y, x) in enumerate(c):
            y = min(max(y, r), pad.shape[0] - r)
            x = min(max(x, r), pad.shape[1] - r)
            out[i] = pad[y - r:y + r, x - r:x + r]
        return out

    def predict_crops(self, crops: np.ndarray) -> np.ndarray:
        x = ((crops.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)
        probs = []
        for i in range(0, len(x), self.batch):
            logits = self.sess.run(None, {self.inp: np.ascontiguousarray(x[i:i + self.batch])})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs.append(e / e.sum(1, keepdims=True))
        return np.concatenate(probs).astype(np.float32) if probs else np.zeros((0, len(self.classes)), np.float32)

    def predict(self, frame_rgb, points_yx, um_per_px=None) -> np.ndarray:
        return self.predict_crops(self.crops(frame_rgb, points_yx, um_per_px))
