"""Feature 5 mosaic tests: a simulated serpentine sweep over a real slide image with camera
effects (vignetting, brightness drift, noise, blur), a stage jump and a blank frame."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from core.region.mosaic import Mosaic

FULL = ROOT / "training/data/midog/full/100.jpg"


@pytest.fixture(scope="module")
def slide():
    if not FULL.exists():
        pytest.skip("MIDOG slide not available")
    return cv2.cvtColor(cv2.imread(str(FULL)), cv2.COLOR_BGR2RGB)     # 7084 x 5311 at 0.25 µm/px


def camera(rgb, rng):
    """Vignetting + brightness drift + noise + slight blur, like a microscope camera."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    vig = 1 - 0.35 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    x = rgb.astype(np.float32) * vig[..., None] * rng.uniform(0.9, 1.1) + rng.normal(0, 4, rgb.shape)
    return cv2.GaussianBlur(np.clip(x, 0, 255).astype(np.uint8), (0, 0), rng.uniform(0.3, 1.2))


def test_serpentine_sweep_no_drift(slide):
    """10x-like frames (treat the 0.25 µm/px slide as 1 µm/px, frames 1280x960 = 1.3 x 1 mm)."""
    rng = np.random.default_rng(0)
    fw, fh, um = 1280, 960, 1.0
    H, W = slide.shape[:2]
    m = Mosaic(map_um_per_px=4.0)
    truth, est = [], []
    ys = list(range(0, H - fh, 700))
    for r, y in enumerate(ys):
        xs = list(range(0, W - fw, 900))
        for x in (xs if r % 2 == 0 else xs[::-1]):
            x2 = int(np.clip(x + rng.integers(-60, 61), 0, W - fw))
            y2 = int(np.clip(y + rng.integers(-60, 61), 0, H - fh))
            p = m.add_frame(camera(slide[y2:y2 + fh, x2:x2 + fw], rng), um)
            assert p.status in ("first", "tracked"), p
            truth.append((x2 * um, y2 * um))
            est.append(p.um(m.map_um))
    truth, est = np.array(truth), np.array(est)
    err = np.linalg.norm((est - est[0]) - (truth - truth[0]), axis=1)
    print(f"{len(truth)} frames, max position error {err.max():.1f} µm, final {err[-1]:.1f} µm")
    assert err.max() < 16          # < 4 map px at 4 µm/px over the whole sweep
    mm = m.minimap(300)
    assert mm.shape[1] == 300 and mm.dtype == np.uint8


def test_jump_relocates_and_blank_frame(slide):
    rng = np.random.default_rng(1)
    m = Mosaic(map_um_per_px=4.0)
    fw, fh = 1280, 960
    for x in range(0, 4000, 900):
        m.add_frame(camera(slide[2000:2000 + fh, x:x + fw], rng), 1.0)
    p = m.add_frame(camera(slide[2000:2000 + fh, 200:200 + fw], rng), 1.0)    # jump back to the start
    assert p.status in ("tracked", "relocated") and abs(p.x * 4 - 200) < 16 and abs(p.y * 4) < 16
    blank = np.full((fh, fw, 3), 242, np.uint8)
    assert m.add_frame(blank, 1.0).status == "no_tissue"
    far = m.add_frame(camera(slide[4300:4300 + fh, 5600:5600 + fw], rng), 1.0)   # unseen area, no overlap
    assert far.status == "new_island" and m.n_islands == 2


def test_glass_gap_islands_merge(slide):
    """Row 1 is interrupted by bare glass (no overlap across the gap -> 2 islands); row 2 overlaps
    both pieces, so the islands are joined and the positions stay consistent."""
    rng = np.random.default_rng(2)
    fw, fh = 1280, 960
    m = Mosaic(map_um_per_px=4.0)
    blank = np.full((fh, fw, 3), 242, np.uint8)
    truth = {}
    for x in (0, 900):                                    # left piece
        truth[m.add_frame(camera(slide[1000:1000 + fh, x:x + fw], rng), 1.0).index] = (x, 1000)
    m.add_frame(blank, 1.0)                               # glass
    for x in (3600, 4500):                                # right piece, no overlap with the left one
        truth[m.add_frame(camera(slide[1000:1000 + fh, x:x + fw], rng), 1.0).index] = (x, 1000)
    assert m.n_islands == 2
    for x in range(4500, -1, -900):                       # second row, overlapping both pieces
        truth[m.add_frame(camera(slide[1700:1700 + fh, x:x + fw], rng), 1.0).index] = (x, 1700)
    assert m.n_islands == 1 and m.merges == 1
    ps = {p.index: p for p in m.placements if p.island >= 0}
    ref = ps[0]
    for i, (tx, ty) in truth.items():
        ex, ey = (ps[i].x - ref.x) * 4, (ps[i].y - ref.y) * 4
        assert abs(ex - tx) < 20 and abs(ey - (ty - 1000)) < 20, (i, ex, ey, tx, ty)
