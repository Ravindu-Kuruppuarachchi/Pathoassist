"""Hotspot guidance at 10x: where should the pathologist go for the 10-HPF count?

A low-magnification copy of the mitosis heatmap detector (trained at 1 µm/px) gives a
candidate-mitosis density for a 10x frame. Summing it over a disc the size of one 40x HPF
(from calibration) ranks locations; the top non-overlapping discs are the "go here"
targets. Across frames, `HotspotMap` accumulates targets in stage coordinates when frame
offsets are known (from mosaicking, Feature 5), otherwise per frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Hotspot:
    x: float                 # frame px (10x frame)
    y: float
    expected_count: float    # summed candidate density in one HPF disc (relative score, not a count)
    rank: int


class HotspotGuide:
    def __init__(self, model_path: str | Path, num_threads: int | None = None):
        import onnxruntime as ort
        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text())
        self.um_per_px = float(meta["um_per_px"])        # 1.0
        self.stride = int(meta["stride"])
        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

    def density(self, frame_rgb, um_per_px):
        """Candidate density at stride resolution + the scale (density px per frame px)."""
        s = um_per_px / self.um_per_px
        img = cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if abs(s - 1) > 0.02 else frame_rgb
        s = s if abs(s - 1) > 0.02 else 1.0
        h, w = img.shape[:2]
        H, W = -(-h // 32) * 32, -(-w // 32) * 32          # network needs multiples of 32
        pad = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
        x = ((pad.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        logit = self.sess.run(None, {"image": x})[0][0, 0]
        p = 1 / (1 + np.exp(-logit))
        return p[:-(-h // self.stride), :-(-w // self.stride)], s / self.stride

    def hotspots(self, frame_rgb, um_per_px: float, hpf_diameter_um: float, top_k: int = 3) -> list[Hotspot]:
        d, k = self.density(frame_rgb, um_per_px)
        r = max(1, int(round(hpf_diameter_um / 2 / um_per_px * k)))          # HPF radius in density px
        disc = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(disc, (r, r), r, 1, -1)
        score = cv2.filter2D(d, -1, disc, borderType=cv2.BORDER_CONSTANT)
        # keep the HPF inside the frame
        score[:r], score[-r:], score[:, :r], score[:, -r:] = 0, 0, 0, 0
        out = []
        for rank in range(top_k):
            y, x = np.unravel_index(int(score.argmax()), score.shape)
            if score[y, x] <= 0:
                break
            out.append(Hotspot(float((x + 0.5) / k), float((y + 0.5) / k), float(score[y, x]), rank + 1))
            cv2.circle(score, (int(x), int(y)), 2 * r, 0, -1)                # next target must not overlap
        return out


class HotspotMap:
    """Collects hotspots from many 10x frames; with stage offsets (µm) they share one map."""

    def __init__(self):
        self.spots = []   # (stage_x_um or None, stage_y_um or None, frame_id, Hotspot)

    def add(self, frame_id, spots: list[Hotspot], um_per_px: float, offset_um: tuple | None = None):
        for s in spots:
            if offset_um is None:
                self.spots.append((None, None, frame_id, s))
            else:
                self.spots.append((offset_um[0] + s.x * um_per_px, offset_um[1] + s.y * um_per_px, frame_id, s))

    def ranked(self, k: int = 5):
        return sorted(self.spots, key=lambda t: -t[3].expected_count)[:k]
