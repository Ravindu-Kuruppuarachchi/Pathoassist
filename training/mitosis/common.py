"""Shared pieces for mitosis training: data, heatmap detector, augmentation, matching.

Detector: MIDOG++ labels are points (every box is a fixed 50x50 px), so instead of a
box detector (RTMDet/YOLOX) we train the centre-heatmap core of such detectors directly
(CenterNet-style): MobileNetV3-Large features -> FPN-lite -> stride-4 heatmap, trained
with the CenterNet focal loss; imposters (mimics) get extra negative weight.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import training.augment as A

D = ROOT / "training" / "data" / "midog"
STRIDE = 4
SIGMA = 2.5            # heatmap gaussian, in stride-4 cells (= 10 px = 2.5 µm)
MATCH_RADIUS = 30      # px at 0.25 µm/px = 7.5 µm (MIDOG evaluation protocol)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(rgb01):
    return torch.from_numpy(((rgb01 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))


class HeatmapDetector(nn.Module):
    def __init__(self, backbone="mobilenetv3_large_100", pretrained=True, ch=64):
        super().__init__()
        self.body = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        chs = self.body.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in chs])
        self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 1, 1))
        nn.init.constant_(self.head[-1].bias, -4.0)   # prior ~0.018: most cells are background

    def forward(self, x):
        feats = self.body(x)                          # strides 4, 8, 16, 32
        y = self.lat[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(self.lat[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + lat(f)
        return self.head(self.smooth(y))              # logits at stride 4


def render_targets(points, shape, stride=STRIDE, sigma=SIGMA):
    """points: [(x, y, cat)], cat 1 = mitosis, 2 = imposter -> (heatmap, imposter weight) at stride."""
    h, w = shape[0] // stride, shape[1] // stride
    hm = np.zeros((h, w), np.float32)
    neg_w = np.ones((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w]
    for x, y, c in points:
        cx, cy = x / stride, y / stride
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        if c == 1:
            hm = np.maximum(hm, g)
        else:
            neg_w += 3.0 * (g > 0.1)
    for x, y, c in points:   # exact peak = 1 at mitosis centres
        if c == 1:
            iy, ix = int(y / stride), int(x / stride)
            if 0 <= iy < h and 0 <= ix < w:
                hm[iy, ix] = 1.0
    return hm, neg_w


def focal_loss(logits, hm, neg_w, alpha=2, beta=4):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = hm.eq(1).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - hm) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos) * neg_w
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def augment_points(x, pts, rng, scale=(0.85, 1.15)):
    """Flip/rot90/scale an image (H,W,3 float) and its points [(x,y,c)]. Output same size."""
    h, w = x.shape[:2]
    P = np.array([[p[0], p[1]] for p in pts], np.float32).reshape(-1, 2)
    C = [p[2] for p in pts]
    if rng.random() < 0.5:
        x = x[:, ::-1]
        P[:, 0] = w - 1 - P[:, 0]
    for _ in range(rng.integers(4)):   # rot90 counter-clockwise: (x, y) -> (y, w-1-x)
        x = np.rot90(x)
        P = np.stack([P[:, 1], w - 1 - P[:, 0]], 1)
        h, w = w, h
    s = rng.uniform(*scale)
    if abs(s - 1) > 0.01:
        M = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]], np.float32)
        x = cv2.warpAffine(np.ascontiguousarray(x), M, (w, h), borderMode=cv2.BORDER_REFLECT)
        P = P * s + M[:, 2]
    keep = (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
    return np.ascontiguousarray(x), [(float(px), float(py), c) for (px, py), c, k in zip(P, C, keep) if k]


def photometric(x, rng):
    A.rng = rng
    x = A._stain(x, np.zeros(x.shape[:2], np.uint16), p_ihc=0.0)
    return A._optics(x)


class CropDataset(Dataset):
    def __init__(self, records, train=True, scale=1.0):
        """scale < 1 trains a low-magnification model (e.g. 0.25 -> 1 µm/px for 10x hotspot guidance)."""
        self.r, self.train, self.scale = records, train, scale

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        rec = self.r[i]
        x = cv2.cvtColor(cv2.imread(str(D / "crops" / rec["file"])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        pts = [tuple(p) for p in rec["points"]]
        if self.train:
            rng = np.random.default_rng()
            x, pts = augment_points(x, pts, rng)
            x = photometric(x, rng)
        if self.scale != 1.0:
            x = cv2.resize(x, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            pts = [(px * self.scale, py * self.scale, c) for px, py, c in pts]
        hm, nw = render_targets(pts, x.shape[:2])
        return to_tensor(x), torch.from_numpy(hm)[None], torch.from_numpy(nw)[None]


def peaks(hm: np.ndarray, thresh: float, stride=STRIDE, nms=5):
    """Local maxima of a probability heatmap -> (n,3) array of x, y (input px), score."""
    mx = cv2.dilate(hm, np.ones((nms, nms), np.uint8))
    ys, xs = np.nonzero((hm == mx) & (hm >= thresh))
    s = hm[ys, xs]
    o = np.argsort(-s)
    return np.stack([(xs[o] + 0.5) * stride, (ys[o] + 0.5) * stride, s[o]], 1) if len(o) else np.zeros((0, 3))


@torch.no_grad()
def predict_full(model, rgb_u8, device="cuda", tile=1024, margin=64):
    """Stride-4 probability heatmap for a large image (tiled, centre-crop stitching)."""
    H, W = rgb_u8.shape[:2]
    step = tile - 2 * margin
    ny, nx = -(-H // step), -(-W // step)
    Hp, Wp = ny * step + 2 * margin, nx * step + 2 * margin
    img = np.pad(rgb_u8, ((margin, Hp - H - margin), (margin, Wp - W - margin), (0, 0)), mode="reflect")
    out = np.zeros((ny * step // STRIDE, nx * step // STRIDE), np.float32)
    m4, s4 = margin // STRIDE, step // STRIDE
    for i in range(ny):
        for j in range(nx):
            t = img[i * step:i * step + tile, j * step:j * step + tile].astype(np.float32) / 255
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(to_tensor(t)[None].to(device)).float())[0, 0].cpu().numpy()
            out[i * s4:(i + 1) * s4, j * s4:(j + 1) * s4] = p[m4:m4 + s4, m4:m4 + s4]
    return out[:-(-H // STRIDE), :-(-W // STRIDE)]


def match(pred_xy, gt_xy, radius=MATCH_RADIUS):
    """Greedy (score-ordered) 1:1 matching, as in the MIDOG evaluation. Returns tp, fp, fn, matched mask."""
    matched = np.zeros(len(pred_xy), bool)
    used = np.zeros(len(gt_xy), bool)
    for i, p in enumerate(pred_xy):
        if not len(gt_xy):
            break
        d = np.hypot(gt_xy[:, 0] - p[0], gt_xy[:, 1] - p[1])
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= radius:
            used[j] = matched[i] = True
    tp = int(matched.sum())
    return tp, len(pred_xy) - tp, len(gt_xy) - tp, matched


def load_full(split):
    recs = [r for r in json.loads((D / "full.json").read_text()) if r["split"] == split]
    return recs
