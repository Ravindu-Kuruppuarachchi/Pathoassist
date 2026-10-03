"""Phikon-v2 tile embeddings for the pleomorphism score (Feature 7).

A 40x field is resampled to 0.5 µm/px, cut into 224 px tiles (112 µm), glass / eyepiece-edge
tiles are dropped, and each tile is embedded with Phikon-v2 (owkin, ViT-L, open weights; INT8
weights via OpenVINO, ~0.2-0.3 s per tile on CPU). A field is represented by its mean tile
embedding; a case by the mean over its fields. A small model (standardise -> PCA -> multinomial
logistic regression, models/pleomorphism_phikon.npz) maps that to P(score 1/2/3).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

UM, TILE = 0.5, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def field_tiles(frame_rgb: np.ndarray, um_per_px: float) -> list:
    s = um_per_px / UM
    im = cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = im.shape[:2]
    out = []
    for y in range(0, H - TILE + 1, TILE):
        for x in range(0, W - TILE + 1, TILE):
            t = im[y:y + TILE, x:x + TILE]
            hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
            if ((hsv[..., 1] > 25) & (hsv[..., 2] < 230)).mean() > 0.5:
                out.append(t)
    return out


class TileEmbedder:
    def __init__(self, model_path, num_threads: int | None = None):
        import openvino as ov
        core = ov.Core()
        cfg = {"INFERENCE_NUM_THREADS": num_threads} if num_threads else {}
        self.net = core.compile_model(core.read_model(str(model_path)), "CPU", cfg)

    def embed_field(self, frame_rgb, um_per_px, batch=8, max_tiles=24):
        """Mean tile embedding of a field (1024,), or None if the field has no tissue tiles.
        Large frames are capped at `max_tiles` evenly spaced tiles (~0.2-0.3 s per tile on CPU)."""
        ts = field_tiles(frame_rgb, um_per_px)
        if not ts:
            return None
        if len(ts) > max_tiles:
            ts = [ts[i] for i in np.linspace(0, len(ts) - 1, max_tiles).round().astype(int)]
        x = ((np.stack(ts).astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)
        e = np.concatenate([self.net(np.ascontiguousarray(x[i:i + batch]))[0] for i in range(0, len(x), batch)])
        return e.mean(0)


class EmbeddingPleoModel:
    def __init__(self, path):
        z = np.load(path)
        self.mu, self.sd = z["emb_mean"], z["emb_std"]
        self.pca_mean, self.pca_comp = z["pca_mean"], z["pca_components"]
        self.coef, self.intercept = z["coef"], z["intercept"]
        meta = Path(path).with_suffix(".json")
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    def predict(self, emb: np.ndarray) -> np.ndarray:
        x = ((emb - self.mu) / self.sd - self.pca_mean) @ self.pca_comp.T
        lg = x @ self.coef.T + self.intercept
        p = np.exp(lg - lg.max())
        return p / p.sum()
