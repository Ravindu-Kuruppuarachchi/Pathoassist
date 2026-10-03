"""HER2 field classifier runtime (Feature 4, optional cross-check of the rule engine).

EfficientNet-B0 (ONNX) trained on 512 µm crops at 1 µm/px fed at 384 px (training/her2/train_field_classifier.py).
A frame is resized to 1 µm/px, cut into non-overlapping 512 µm tiles (one centred tile when the field is
smaller), and the tile softmax outputs are averaged. Across fields the case probabilities are the mean
of the field probabilities. When the classifier and the rule engine disagree the case is flagged
"review carefully"; the classifier never replaces the rule-engine score.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class HER2FieldClassifier:
    def __init__(self, model_path, num_threads: int | None = None, max_tiles: int = 8):
        import onnxruntime as ort
        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text())
        self.um, self.crop_um, self.size = float(meta["um_per_px"]), int(meta["crop_um"]), int(meta["input"])
        self.classes = meta["classes"]
        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.max_tiles = max_tiles

    def tiles(self, frame_rgb, um_per_px):
        s = um_per_px / self.um
        img = cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        C = int(round(self.crop_um / self.um))
        h, w = img.shape[:2]
        if h < C or w < C:   # small field: pad by reflection to one tile
            img = cv2.copyMakeBorder(img, 0, max(0, C - h), 0, max(0, C - w), cv2.BORDER_REFLECT)
            h, w = img.shape[:2]
        ny, nx = h // C, w // C
        oy, ox = (h - ny * C) // 2, (w - nx * C) // 2
        out = [img[oy + i * C:oy + (i + 1) * C, ox + j * C:ox + (j + 1) * C] for i in range(ny) for j in range(nx)]
        if len(out) > self.max_tiles:
            out = [out[i] for i in np.linspace(0, len(out) - 1, self.max_tiles).round().astype(int)]
        return [cv2.resize(t, (self.size, self.size), interpolation=cv2.INTER_AREA) for t in out]

    def predict(self, frame_rgb, um_per_px) -> np.ndarray:
        """Field probabilities over the classes ("0", "1+", "2+", "3+")."""
        t = np.stack(self.tiles(frame_rgb, um_per_px)).astype(np.float32) / 255
        x = ((t - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
        logits = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).mean(0)


def cross_check(rule_score: str, field_probs: list) -> dict:
    """Compare the rule-engine score with the mean classifier probabilities over the counted fields."""
    if not field_probs:
        return dict(classifier_score=None, probs=None, agrees=None, note="")
    p = np.mean(np.stack(field_probs), 0)
    classes = ["0", "1+", "2+", "3+"]
    cls = classes[int(p.argmax())]
    agrees = cls == rule_score
    note = "" if agrees else (f"Field classifier suggests {cls} (p={p.max():.2f}) but the rule engine gives {rule_score}: "
                              "review carefully.")
    return dict(classifier_score=cls, probs=[round(float(v), 3) for v in p], agrees=bool(agrees), note=note)
