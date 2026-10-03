"""CPU nuclei detection engine.

StarDist network runs through OpenVINO (default) or ONNX Runtime; post-processing
(polygon NMS + label rendering) uses StarDist's compiled C routines, so the app
does not need TensorFlow at runtime.

Pipeline:
    frame (any µm/px) -> resize to the model's native µm/px -> percentile normalise
    -> fixed-size overlapping tiles -> network -> stitch prob/dist maps
    -> one NMS over the whole frame -> polygons + label image in frame coordinates

Stitching the raw network maps (keeping only each tile's centre) and running a
single NMS avoids duplicate nuclei at tile seams, so no detection merging is needed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from stardist.geometry import dist_to_coord, polygons_to_label
from stardist.nms import non_maximum_suppression


@dataclass
class Detections:
    labels: np.ndarray        # (H, W) int32 instance map in frame pixels, 0 = background
    points: np.ndarray        # (n, 2) nucleus centres (y, x) in frame pixels
    coords: np.ndarray        # (n, 2, n_rays) polygon vertices (y, x) in frame pixels
    prob: np.ndarray          # (n,) detection score
    timings: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.prob)


class NucleiDetector:
    def __init__(self, model_path: str | Path, backend: str | None = None, num_threads: int | None = None,
                 tile: int = 512, margin: int = 64, prob_thresh: float | None = None,
                 nms_thresh: float | None = None):
        """model_path: .xml (OpenVINO IR) or .onnx. A sibling `<stem>.json` holds the metadata
        (grid, native µm/px, thresholds) written by training/export.py."""
        model_path = Path(model_path)
        meta = json.loads(model_path.with_suffix(".json").read_text())
        self.grid = tuple(meta["grid"])
        self.model_um_per_px = float(meta["um_per_px"])
        self.norm_pmin, self.norm_pmax = meta.get("norm_percentiles", (1.0, 99.8))
        self.prob_thresh = prob_thresh if prob_thresh is not None else meta["prob_thresh"]
        self.nms_thresh = nms_thresh if nms_thresh is not None else meta["nms_thresh"]
        g = max(self.grid)
        if tile % 16 or margin % g or (tile - 2 * margin) <= 0:
            raise ValueError("tile must be a multiple of 16 and margin a multiple of the grid")
        self.tile, self.margin = tile, margin

        backend = backend or ("openvino" if model_path.suffix == ".xml" else "onnx")
        self.backend = backend
        if backend == "openvino":
            import openvino as ov
            core = ov.Core()
            model = core.read_model(model_path)
            model.reshape({model.inputs[0]: [1, tile, tile, 3]})
            cfg = {"PERFORMANCE_HINT": "LATENCY"}
            if num_threads:
                cfg["INFERENCE_NUM_THREADS"] = num_threads
            self._compiled = core.compile_model(model, "CPU", cfg)
            self._request = self._compiled.create_infer_request()
            outs = {o.get_any_name(): o for o in self._compiled.outputs}
            self._out_prob, self._out_dist = outs["prob"], outs["dist"]
        elif backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            if num_threads:
                so.intra_op_num_threads = num_threads
            self._sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
            self._in_name = self._sess.get_inputs()[0].name
        else:
            raise ValueError(f"unknown backend {backend}")

    # ------------------------------------------------------------------ network
    def _run_net(self, x: np.ndarray):
        """x: (1, T, T, 3) float32 -> prob (T/g, T/g), dist (T/g, T/g, n_rays)."""
        if self.backend == "openvino":
            res = self._request.infer({0: x})
            prob, dist = res[self._out_prob], res[self._out_dist]
        else:
            prob, dist = self._sess.run(["prob", "dist"], {self._in_name: x})
        return prob[0, ..., 0], dist[0]

    def _predict_maps(self, img: np.ndarray):
        """Tiled prediction on a normalised float32 image (H, W, 3) at model resolution."""
        H, W = img.shape[:2]
        gy, gx = self.grid
        T, m = self.tile, self.margin
        step = T - 2 * m
        ny, nx = max(1, -(-H // step)), max(1, -(-W // step))
        Hp, Wp = ny * step + 2 * m, nx * step + 2 * m
        padded = np.pad(img, ((m, Hp - H - m), (m, Wp - W - m), (0, 0)), mode="reflect")

        prob = dist = None
        for i in range(ny):
            for j in range(nx):
                y0, x0 = i * step, j * step
                p, d = self._run_net(padded[None, y0:y0 + T, x0:x0 + T])
                if prob is None:
                    prob = np.zeros((ny * step // gy, nx * step // gx), np.float32)
                    dist = np.zeros(prob.shape + (d.shape[-1],), np.float32)
                cy, cx = slice(m // gy, (m + step) // gy), slice(m // gx, (m + step) // gx)
                oy, ox = slice(y0 // gy, (y0 + step) // gy), slice(x0 // gx, (x0 + step) // gx)
                prob[oy, ox] = p[cy, cx]
                dist[oy, ox] = d[cy, cx]
        hh, ww = -(-H // gy), -(-W // gx)
        return prob[:hh, :ww], dist[:hh, :ww]

    # ------------------------------------------------------------------ public
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Percentile normalisation over the whole frame (never per tile)."""
        sub = img[::4, ::4] if img.shape[0] * img.shape[1] > 1_000_000 else img
        lo, hi = np.percentile(sub, (self.norm_pmin, self.norm_pmax))
        return ((img.astype(np.float32) - lo) / max(hi - lo, 1e-6)).astype(np.float32)

    def detect(self, frame_rgb: np.ndarray, um_per_px: float | None = None,
               return_labels: bool = True) -> Detections:
        """frame_rgb: (H, W, 3) uint8 RGB. um_per_px: frame scale from calibration
        (None = assume the model's native scale)."""
        t0 = time.perf_counter()
        H0, W0 = frame_rgb.shape[:2]
        s = 1.0 if um_per_px is None else um_per_px / self.model_um_per_px
        img = frame_rgb
        if abs(s - 1.0) > 0.02:
            img = cv2.resize(frame_rgb, (round(W0 * s), round(H0 * s)),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        else:
            s = 1.0
        x = self.normalize(img)
        t1 = time.perf_counter()

        prob, dist = self._predict_maps(x)
        dist = np.maximum(dist, 1e-3)  # as StarDist: zero rays (ReLU) give degenerate polygons in NMS
        t2 = time.perf_counter()

        points, probi, disti = non_maximum_suppression(
            dist, prob, grid=self.grid, prob_thresh=self.prob_thresh, nms_thresh=self.nms_thresh)
        labels = None
        if return_labels:
            labels = polygons_to_label(disti, points, shape=img.shape[:2], prob=probi)
            if s != 1.0:
                labels = cv2.resize(labels.astype(np.int32), (W0, H0), interpolation=cv2.INTER_NEAREST)
            labels = labels.astype(np.int32)
        # label id k+1 <-> entry k (polygons_to_label keeps the given order for ids)
        coords = dist_to_coord(disti, points) / s if len(points) else np.zeros((0, 2, dist.shape[-1]))
        t3 = time.perf_counter()
        return Detections(labels=labels, points=points / s, coords=coords, prob=probi,
                          timings=dict(preprocess=t1 - t0, network=t2 - t1, postprocess=t3 - t2,
                                       total=t3 - t0))


def draw_overlay(frame_rgb: np.ndarray, det: Detections, color=(0, 255, 0), thickness=1) -> np.ndarray:
    """Nucleus outlines drawn on a copy of the frame (for the app's overlay toggle)."""
    out = frame_rgb.copy()
    if len(det):
        polys = [np.round(c[::-1].T).astype(np.int32) for c in det.coords]  # (y,x) rows -> (x,y) points
        cv2.polylines(out, polys, isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out
