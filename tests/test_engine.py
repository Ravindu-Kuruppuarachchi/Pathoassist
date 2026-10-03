"""Engine behaviour tests (run: pytest tests/). Uses whichever exported model exists."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi
from stardist.data import test_image_he_2d as sample_he_image
from stardist.matching import matching

from core.nuclei import NucleiDetector, draw_overlay

MODEL = next((p for p in [ROOT / "models/stardist_he_ihc_ft_int8.xml", ROOT / "models/stardist_he_ft_int8.xml", ROOT / "models/stardist_he_pretrained.onnx"]
              if p.exists()), None)
pytestmark = pytest.mark.skipif(MODEL is None, reason="no exported model in models/")


@pytest.fixture(scope="module")
def img():
    im = sample_he_image()
    return np.vstack([np.hstack([im, im[:, ::-1]]), np.hstack([im[::-1], im[::-1, ::-1]])])  # 600x1000


def test_tiling_invariance(img):
    a = NucleiDetector(MODEL, tile=256, margin=64).detect(img)
    b = NucleiDetector(MODEL, tile=1024, margin=64).detect(img)
    assert matching(a.labels, b.labels, 0.5).f1 > 0.99


def test_labels_match_polygons(img):
    d = NucleiDetector(MODEL).detect(img)
    assert d.labels.shape == img.shape[:2] and d.labels.max() == len(d)
    c = np.array(ndi.center_of_mass(d.labels > 0, d.labels, range(1, len(d) + 1)))
    ok = ~np.isnan(c).any(1)  # a polygon can be fully covered by higher-scoring neighbours
    assert np.median(np.linalg.norm(c[ok] - d.points[ok], axis=1)) < 2.0


def test_um_per_px_rescaling(img):
    """A frame captured at half the resolution, with correct µm/px, gives the same nuclei."""
    det = NucleiDetector(MODEL)
    full = det.detect(img, um_per_px=0.25)
    half_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    half = det.detect(half_img, um_per_px=0.5)
    assert half.labels.shape == half_img.shape[:2]
    assert abs(len(half) - len(full)) / len(full) < 0.15
    up = cv2.resize(half.labels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    assert matching(full.labels, up, 0.3).f1 > 0.8
    assert np.allclose(np.median(half.points, 0) * 2, np.median(full.points, 0), atol=10)


def test_blank_frame():
    d = NucleiDetector(MODEL).detect(np.full((480, 640, 3), 235, np.uint8))
    assert len(d) == 0 and d.labels.max() == 0
    draw_overlay(np.zeros((480, 640, 3), np.uint8), d)


def test_overlay(img):
    d = NucleiDetector(MODEL).detect(img)
    ov = draw_overlay(img, d)
    assert ov.shape == img.shape and (ov != img).any()
