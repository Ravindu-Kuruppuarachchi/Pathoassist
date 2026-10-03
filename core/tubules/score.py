"""Tubule formation score over the slide (Feature 8).

Tubule % = tubular area / invasive tumour area, accumulated over everything the pathologist has
swept at 10x - not per frame. With a Mosaic, each frame's tumour and tubule masks are painted into
map coordinates (union, so overlapping frames are not double counted); without one, frames
overlapping an earlier frame by > 50% are skipped (core.ihc.dedup).

Nottingham tubule score: > 75% of tumour forming tubules -> 1, 10-75% -> 2, < 10% -> 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.ihc.dedup import FieldDeduplicator
from core.region.mosaic import TiledLayer
from core.tubules.lumen import UM, LumenClassifier, TubuleField, analyze_field


def tubule_score(pct: float) -> int:
    if pct > 75:
        return 1
    if pct >= 10:
        return 2
    return 3


@dataclass
class TubuleResult:
    score: int
    tubule_pct: float
    tumour_area_mm2: float
    tubule_area_mm2: float
    n_fields: int
    n_lumens: int
    warnings: list = field(default_factory=list)


class TubuleSession:
    def __init__(self, classifier: LumenClassifier, region_model=None, mosaic=None, min_tumour_mm2=1.0, map_um=4.0):
        self.clf, self.region, self.mosaic = classifier, region_model, mosaic
        self.min_tumour_mm2, self.map_um = min_tumour_mm2, map_um
        self.dedup = FieldDeduplicator(0.5)
        self.layers = {}                 # island -> (tumour layer, tubule layer) in map pixels
        self.tum_area = self.tub_area = 0.0
        self.n_fields = self.n_lumens = 0
        self.fields: list[TubuleField] = []

    def add_field(self, frame_rgb, um_per_px, placement=None) -> dict:
        if placement is None and self.dedup.check(frame_rgb) is not None:
            return dict(accepted=False, reason="overlaps an earlier field by > 50%")
        tp = None
        if self.region is not None:
            rm = self.region.predict(frame_rgb, um_per_px)
            tp = rm.probs[..., 0]                         # invasive only (in situ / benign ducts excluded)
        f = analyze_field(frame_rgb, um_per_px, self.clf, tp)
        self.fields.append(f)
        self.n_fields += 1
        self.n_lumens += int(f.lumens.sum())
        if placement is not None and placement.island >= 0:
            self._paint(f, placement)
        else:
            self.dedup.add(frame_rgb)
            self.tum_area += f.tumour_area_mm2
            self.tub_area += f.tubule_area_mm2
        return dict(accepted=True, tumour_mm2=f.tumour_area_mm2, tubule_mm2=f.tubule_area_mm2, lumens=int(f.lumens.sum()))

    def _paint(self, f: TubuleField, p):
        k = UM / self.map_um
        tum = cv2.resize(f.tumour.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
        tub = cv2.resize((f.tubule_mask & f.tumour).astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
        x0, y0 = int(round(p.x * self.mosaic.map_um / self.map_um)), int(round(p.y * self.mosaic.map_um / self.map_um))
        lt, lb = self.layers.setdefault(p.island, (TiledLayer(1, np.uint8), TiledLayer(1, np.uint8)))
        old_t = lt.get(x0, y0, tum.shape[1], tum.shape[0])[..., 0]
        old_b = lb.get(x0, y0, tub.shape[1], tub.shape[0])[..., 0]
        lt.put(x0, y0, np.maximum(old_t, tum)[..., None])
        lb.put(x0, y0, np.maximum(old_b, tub)[..., None])

    def _areas(self):
        if not self.layers:
            return self.tum_area, self.tub_area
        px = (self.map_um / 1000) ** 2
        t = sum(int(tile.sum()) for lt, _ in self.layers.values() for tile in lt.tiles.values()) * px
        b = sum(int(tile.sum()) for _, lb in self.layers.values() for tile in lb.tiles.values()) * px
        return t + self.tum_area, b + self.tub_area

    def result(self) -> TubuleResult:
        t, b = self._areas()
        pct = 100 * b / t if t > 0 else 0.0
        warn = []
        if t < self.min_tumour_mm2:
            warn.append(f"Only {t:.2f} mm² of invasive tumour assessed (< {self.min_tumour_mm2} mm²); sweep more of the tumour at 10x.")
        for cut in (10, 75):
            if abs(pct - cut) <= 5:
                warn.append(f"Tubule formation {pct:.0f}% is near the {cut}% cut-off; review.")
        warn.append("Tubule model trained on public data (BreCaHAD lumens); not validated against local tubule scores.")
        return TubuleResult(tubule_score(pct) if t > 0 else 0, round(pct, 1), round(t, 3), round(b, 3), self.n_fields, self.n_lumens, warn)


def draw_tubule_overlay(frame_rgb, f: TubuleField, alpha=0.35):
    h, w = frame_rgb.shape[:2]
    tub = cv2.resize((f.tubule_mask & f.tumour).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    tum = cv2.resize(f.tumour.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out = frame_rgb.astype(np.float32)
    out[tub] = out[tub] * (1 - alpha) + np.array([0, 170, 0]) * alpha
    cnts, _ = cv2.findContours(tum.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (220, 0, 0), 2)
    return out.astype(np.uint8)
