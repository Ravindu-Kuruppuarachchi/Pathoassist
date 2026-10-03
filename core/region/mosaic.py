"""Running slide map from 4-10x camera frames (Feature 5 mosaicking + mini-map).

Every frame is resized to the map scale (default 4 µm/px) and high-pass filtered (robust to
vignetting / brightness changes). Placement:
  1. track   : phase correlation + NCC against the previous tissue frame (core.ihc.dedup.register)
  2. refine  : register against the map already built around the predicted position, so
               errors do not accumulate frame to frame (no drift over a sweep)
  3. relocate: if tracking fails (stage jump, gap of bare glass), template-match the frame
               against every island built so far
  4. island  : if that fails too, start a new island (an unconnected piece of the map)
  5. merge   : whenever a frame also matches another island, the two islands are joined -
               a sweep that crosses glass ends up as one map once later rows link the pieces
Each island has its own tiled layers (coordinates may grow in any direction). An overlay layer
(e.g. tumour probability from the region model, mitosis hotspots) uses the same coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.ihc.dedup import register

TILE = 512


def _prep(gray):
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 8)            # remove illumination / vignetting
    return g / (g.std() + 1e-6)


def tissue_fraction(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(((hsv[..., 1] > 20) & (hsv[..., 2] < 235)).mean())


class TiledLayer:
    def __init__(self, channels, dtype):
        self.c, self.dtype, self.tiles = channels, dtype, {}

    def _tile(self, ty, tx, create):
        t = self.tiles.get((ty, tx))
        if t is None and create:
            t = self.tiles[(ty, tx)] = np.zeros((TILE, TILE, self.c), self.dtype)
        return t

    def get(self, x0, y0, w, h):
        out = np.zeros((h, w, self.c), self.dtype)
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                t = self._tile(ty, tx, False)
                if t is None:
                    continue
                ya, yb = max(y0, ty * TILE), min(y0 + h, (ty + 1) * TILE)
                xa, xb = max(x0, tx * TILE), min(x0 + w, (tx + 1) * TILE)
                out[ya - y0:yb - y0, xa - x0:xb - x0] = t[ya - ty * TILE:yb - ty * TILE, xa - tx * TILE:xb - tx * TILE]
        return out

    def put(self, x0, y0, arr, where=None):
        h, w = arr.shape[:2]
        arr = arr.reshape(h, w, self.c)
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                t = self._tile(ty, tx, True)
                ya, yb = max(y0, ty * TILE), min(y0 + h, (ty + 1) * TILE)
                xa, xb = max(x0, tx * TILE), min(x0 + w, (tx + 1) * TILE)
                src = arr[ya - y0:yb - y0, xa - x0:xb - x0]
                dst = t[ya - ty * TILE:yb - ty * TILE, xa - tx * TILE:xb - tx * TILE]
                if where is None:
                    dst[:] = src
                else:
                    m = where[ya - y0:yb - y0, xa - x0:xb - x0]
                    dst[m] = src[m]

    def bounds(self):
        if not self.tiles:
            return 0, 0, 0, 0
        ys = [k[0] for k in self.tiles]
        xs = [k[1] for k in self.tiles]
        return min(xs) * TILE, min(ys) * TILE, (max(xs) + 1) * TILE, (max(ys) + 1) * TILE


class Island:
    def __init__(self):
        self.rgb = TiledLayer(3, np.uint8)
        self.gray = TiledLayer(1, np.float32)
        self.cov = TiledLayer(1, np.uint8)
        self.overlay = TiledLayer(1, np.float32)


@dataclass
class Placement:
    status: str             # "first", "tracked", "relocated", "new_island", "no_tissue"
    x: float                # frame origin in map px within its island
    y: float
    island: int
    ncc: float
    w: int = 0              # frame size in map px
    h: int = 0
    index: int = -1         # frame number in the mosaic

    def um(self, map_um_per_px):
        return self.x * map_um_per_px, self.y * map_um_per_px


class Mosaic:
    def __init__(self, map_um_per_px: float = 4.0, min_ncc: float = 0.3, relocate_ncc: float = 0.35,
                 merge_ncc: float = 0.45):
        self.map_um = map_um_per_px
        self.min_ncc, self.relocate_ncc, self.merge_ncc = min_ncc, relocate_ncc, merge_ncc
        self.islands: dict[int, Island] = {}
        self._next_island = 0
        self.frames = []            # [small rgb, prepped gray, Placement, overlay values or None]
        self.prev = None            # (prepped gray, Placement) of the last tissue frame
        self.placements: list[Placement] = []
        self.merges = 0

    @property
    def n_islands(self):
        return len(self.islands)

    # ------------------------------------------------------------------ placement
    def add_frame(self, frame_rgb: np.ndarray, um_per_px: float) -> Placement:
        s = um_per_px / self.map_um
        small = cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        if tissue_fraction(small) < 0.05:
            p = Placement("no_tissue", np.nan, np.nan, -1, 0.0, w, h, len(self.placements))
            self.placements.append(p)
            return p
        g = _prep(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
        p = None
        if self.prev is None:
            p = Placement("first", 0.0, 0.0, self._new_island(), 1.0, w, h)
        else:
            pg, pp = self.prev
            if pg.shape == g.shape and pp.island in self.islands:
                (dx, dy), ncc = register(pg, g)
                if ncc >= self.min_ncc:
                    p = self._refine(g, Placement("tracked", pp.x + dx, pp.y + dy, pp.island, ncc, w, h))
            if p is None:
                p = self._relocate(g, w, h)
        p.index = len(self.placements)
        self._paste(self.islands[p.island], small, g, p.x, p.y)
        self.frames.append([small, g, p, None])
        self.prev = (g, p)
        self.placements.append(p)
        self._try_merge(g, p)
        return p

    def _new_island(self):
        k = self._next_island
        self._next_island += 1
        self.islands[k] = Island()
        return k

    def _refine(self, g, p, isl=None):
        isl = isl if isl is not None else self.islands[p.island]
        h, w = g.shape
        x, y = int(round(p.x)), int(round(p.y))
        cov = isl.cov.get(x, y, w, h)[..., 0] > 0
        if cov.mean() < 0.1:
            return p
        ref = isl.gray.get(x, y, w, h)[..., 0]
        (dx, dy), ncc = register(np.where(cov, ref, 0), g * cov)
        if ncc >= self.min_ncc and abs(dx) < w / 4 and abs(dy) < h / 4:
            p.x, p.y = p.x + dx, p.y + dy
        return p

    def _match(self, g, isl: Island, min_overlap=0.15, search_max=768):
        """Where does frame g overlap an island, allowing PARTIAL overlap (a later sweep row
        touching an earlier piece)? Phase correlation of the zero-padded frame against the
        island canvas (downscaled for the search), candidate peaks checked by NCC over the
        covered overlap, then refined at full resolution. Returns (ncc, x, y) in island coords."""
        x0, y0, x1, y1 = isl.gray.bounds()
        C = isl.gray.get(x0, y0, x1 - x0, y1 - y0)[..., 0]
        V = isl.cov.get(x0, y0, x1 - x0, y1 - y0)[..., 0] > 0
        h, w = g.shape
        f = min(1.0, search_max / max(C.shape))
        Cs = cv2.resize(C * V, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        Vs = cv2.resize(V.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) > 0.5
        gs = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        hs, ws = gs.shape
        Hc, Wc = Cs.shape
        PH, PW = Hc + hs, Wc + ws
        Fa = np.fft.rfft2(Cs, s=(PH, PW))
        Fb = np.fft.rfft2(gs, s=(PH, PW))
        R = Fa * np.conj(Fb)
        r = np.fft.irfft2(R / (np.abs(R) + 1e-9), s=(PH, PW))
        best = (-1.0, 0, 0)
        for idx in np.argpartition(r.ravel(), -8)[-8:]:
            dy, dx = np.unravel_index(idx, r.shape)
            dy = dy - PH if dy > Hc else dy            # frame origin in canvas (search) px, may be negative
            dx = dx - PW if dx > Wc else dx
            ya, yb, xa, xb = max(0, dy), min(Hc, dy + hs), max(0, dx), min(Wc, dx + ws)
            if yb <= ya or xb <= xa:
                continue
            m = Vs[ya:yb, xa:xb]
            if m.sum() < min_overlap * hs * ws:
                continue
            a = Cs[ya:yb, xa:xb][m]
            b = gs[ya - dy:yb - dy, xa - dx:xb - dx][m]
            a, b = a - a.mean(), b - b.mean()
            ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
            if ncc > best[0]:
                best = (ncc, x0 + dx / f, y0 + dy / f)
        if best[0] < 0:
            return best
        # refine at full resolution on the covered part of the overlap
        p = self._refine(g, Placement("tmp", best[1], best[2], -1, 0, w, h), isl)
        return best[0], p.x, p.y

    def _relocate(self, g, w, h):
        best = (-1.0, None, 0, 0)
        for k, isl in self.islands.items():
            sc, x, y = self._match(g, isl)
            if sc > best[0]:
                best = (sc, k, x, y)
        if best[0] >= self.relocate_ncc:
            return self._refine(g, Placement("relocated", best[2], best[3], best[1], best[0], w, h))
        return Placement("new_island", 0.0, 0.0, self._new_island(), 0.0, w, h)

    def _try_merge(self, g, p):
        """If this frame also overlaps another island, join that island into this frame's one."""
        for k in [k for k in self.islands if k != p.island]:
            sc, x, y = self._match(g, self.islands[k])
            if sc < self.merge_ncc:
                continue
            # frame sits at (x, y) in island k and at (p.x, p.y) in its own island -> shift k onto p.island
            dx, dy = p.x - x, p.y - y
            dst = self.islands[p.island]
            for small, fg, fp, ov in self.frames:
                if fp.island == k:
                    fp.x, fp.y, fp.island = fp.x + dx, fp.y + dy, p.island
                    self._paste(dst, small, fg, fp.x, fp.y)
                    if ov is not None:
                        self._paste_overlay(dst, fp, ov)
            del self.islands[k]
            self.merges += 1

    @staticmethod
    def _paste(isl, small, g, x, y):
        x, y = int(round(x)), int(round(y))
        h, w = g.shape
        new = isl.cov.get(x, y, w, h)[..., 0] == 0
        isl.rgb.put(x, y, small, where=new)
        isl.gray.put(x, y, g[..., None], where=new)
        isl.cov.put(x, y, np.ones((h, w, 1), np.uint8))

    # ------------------------------------------------------------------ overlays / queries
    @staticmethod
    def _paste_overlay(isl, p, values):
        x, y = int(round(p.x)), int(round(p.y))
        v = cv2.resize(values.astype(np.float32), (p.w, p.h), interpolation=cv2.INTER_LINEAR)
        old = isl.overlay.get(x, y, p.w, p.h)[..., 0]
        isl.overlay.put(x, y, np.maximum(old, v)[..., None])

    def put_overlay(self, p: Placement, values: np.ndarray):
        """Write a per-pixel map (any resolution, frame extent) at a placement (max-combine)."""
        self.frames[[f[2] for f in self.frames].index(p)][3] = values
        self._paste_overlay(self.islands[p.island], p, values)

    def frame_point_to_map_um(self, p: Placement, x_px, y_px, frame_um_per_px):
        """A point in a frame (frame px) -> island map coordinates in µm."""
        return p.x * self.map_um + x_px * frame_um_per_px, p.y * self.map_um + y_px * frame_um_per_px

    def coverage_mm2(self):
        return sum(int(t.sum()) for isl in self.islands.values() for t in isl.cov.tiles.values()) * (self.map_um / 1000) ** 2

    def minimap(self, width: int = 400, overlay_color=(255, 0, 0), markers=(), gap=40) -> np.ndarray:
        """RGB mini-map: islands side by side, covered tissue, overlay, current field box, markers
        [(x_um, y_um, island, (r,g,b))]."""
        if not self.islands:
            return np.full((width // 2, width, 3), 240, np.uint8)
        parts, offs, cur = [], {}, 0
        for k, isl in sorted(self.islands.items()):
            x0, y0, x1, y1 = isl.rgb.bounds()
            parts.append((k, isl, x0, y0, x1, y1))
            offs[k] = (cur - x0, -y0)
            cur += (x1 - x0) + gap
        W, H = cur - gap, max(y1 - y0 for _, _, _, y0, _, y1 in parts)
        k_s = width / W
        out = np.full((max(1, int(H * k_s)), width, 3), 235, np.uint8)
        for k, isl, x0, y0, x1, y1 in parts:
            ww, hh = x1 - x0, y1 - y0
            sz = (max(1, int(ww * k_s)), max(1, int(hh * k_s)))
            img = cv2.resize(isl.rgb.get(x0, y0, ww, hh), sz, interpolation=cv2.INTER_AREA)
            cov = cv2.resize(isl.cov.get(x0, y0, ww, hh)[..., 0].astype(np.float32), sz) > 0
            ov = np.clip(cv2.resize(isl.overlay.get(x0, y0, ww, hh)[..., 0], sz), 0, 1)[..., None] * 0.55
            piece = np.where(cov[..., None], img, np.uint8(235))
            piece = (piece * (1 - ov) + np.array(overlay_color) * ov).astype(np.uint8)
            ox = int((offs[k][0] + x0) * k_s)
            hh2, ww2 = min(piece.shape[0], out.shape[0]), min(piece.shape[1], out.shape[1] - ox)
            out[:hh2, ox:ox + ww2] = piece[:hh2, :ww2]
        for xu, yu, isl, col in markers:
            if isl in offs:
                cv2.drawMarker(out, (int((xu / self.map_um + offs[isl][0]) * k_s), int((yu / self.map_um + offs[isl][1]) * k_s)),
                               col, cv2.MARKER_TRIANGLE_DOWN, 14, 2)
        last = next((p for p in reversed(self.placements) if p.island in self.islands), None)
        if last is not None:
            ox, oy = offs[last.island]
            cv2.rectangle(out, (int((last.x + ox) * k_s), int((last.y + oy) * k_s)),
                          (int((last.x + last.w + ox) * k_s), int((last.y + last.h + oy) * k_s)), (0, 160, 0), 2)
        return out
