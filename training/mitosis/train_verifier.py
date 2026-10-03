"""Train the mitosis verifier (EfficientNet-B0, 96x96 crops at 0.25 µm/px): mitosis vs mimic.

Negatives = MIDOG imposters + hard negatives mined from the trained detector's false
positives on the training crops. Then tune the (detector, verifier) thresholds on the
validation slides.

    ~/venvs/pathoassist-torch/bin/python training/mitosis/train_verifier.py
"""
import argparse
import json
import time

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (D, MATCH_RADIUS, ROOT, HeatmapDetector, load_full, match, peaks, photometric,
                    predict_full, to_tensor)

VCROP = 96
CAND_THRESH = 0.05     # detector candidates passed to the verifier


def _read(file, cache={}):
    if file not in cache:
        cache[file] = cv2.cvtColor(cv2.imread(str(D / "crops" / file)), cv2.COLOR_BGR2RGB)
    return cache[file]


def crop_at(img, x, y, size=VCROP, angle=0.0):
    """size x size crop centred at (x, y), optional rotation, reflect padding."""
    M = cv2.getRotationMatrix2D((float(x), float(y)), angle, 1.0)
    M[:, 2] += [size / 2 - x, size / 2 - y]
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@torch.no_grad()
def mine(det, crops, device):
    """Candidates on every training crop: (file, x, y, label, det_score)."""
    det.eval()
    out = []
    for i in range(0, len(crops), 32):
        batch = crops[i:i + 32]
        x = torch.stack([to_tensor(_read(r["file"]).astype(np.float32) / 255) for r in batch]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            hm = torch.sigmoid(det(x).float()).cpu().numpy()[:, 0]
        for r, h in zip(batch, hm):
            gt = np.array([[p[0], p[1]] for p in r["points"] if p[2] == 1]).reshape(-1, 2)
            for px, py, s in peaks(h, CAND_THRESH):
                d = np.hypot(gt[:, 0] - px, gt[:, 1] - py).min() if len(gt) else np.inf
                out.append((r["file"], px, py, int(d <= MATCH_RADIUS), float(s)))
    return out


class VerifierData(Dataset):
    def __init__(self, samples, train):
        self.s, self.train = samples, train

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        f, x, y, lab = self.s[i][:4]
        img = _read(f)
        if self.train:
            rng = np.random.default_rng()
            c = crop_at(img, x + rng.normal(0, 3), y + rng.normal(0, 3), angle=rng.uniform(0, 360))
            if rng.random() < 0.5:
                c = c[:, ::-1]
            c = photometric(np.ascontiguousarray(c).astype(np.float32) / 255, rng)
        else:
            c = crop_at(img, x, y).astype(np.float32) / 255
        return to_tensor(c), torch.tensor(lab, dtype=torch.float32)


@torch.no_grad()
def verify_points(ver, img, pts, device):
    if not len(pts):
        return np.zeros(0)
    x = torch.stack([to_tensor(crop_at(img, px, py).astype(np.float32) / 255) for px, py, _ in pts]).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.sigmoid(ver(x).float()[:, 0]).cpu().numpy()


def tune(det, ver, recs, device):
    """Candidates + verifier scores on whole slides; sweep verifier threshold. Returns best + rows."""
    cand, gts = [], []
    for r in recs:
        img = cv2.cvtColor(cv2.imread(str(D / "full" / r["file"])), cv2.COLOR_BGR2RGB)
        p = peaks(predict_full(det, img, device), CAND_THRESH)
        v = verify_points(ver, img, p, device)
        cand.append(np.column_stack([p, v]) if len(p) else np.zeros((0, 4)))
        gts.append(np.array([[x, y] for x, y, c in r["points"] if c == 1]).reshape(-1, 2))
    rows = []
    for dt in (0.05, 0.1, 0.2, 0.3):
        for vt in np.arange(0.1, 0.96, 0.05):
            tp = fp = fn = 0
            for c, g in zip(cand, gts):
                k = c[(c[:, 2] >= dt) & (c[:, 3] >= vt)]
                k = k[np.argsort(-k[:, 3])]
                a, b, cc, _ = match(k[:, :2], g)
                tp, fp, fn = tp + a, fp + b, fn + cc
            rows.append(dict(det_thresh=dt, ver_thresh=round(float(vt), 3), tp=tp, fp=fp, fn=fn,
                             f1=2 * tp / max(2 * tp + fp + fn, 1)))
    return max(rows, key=lambda r: r["f1"]), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"
    out = ROOT / "models"

    det = HeatmapDetector(pretrained=False).to(device)
    det.load_state_dict(torch.load(out / "mitosis_det.pt"))
    crops = json.loads((D / "crops.json").read_text())
    cands = mine(det, crops, device)
    # add every labelled mitosis (positive) and imposter (negative) that the detector did not propose
    for r in crops:
        for px, py, c in r["points"]:
            cands.append((r["file"], px, py, int(c == 1), -1.0))
    # hold out 10% of slides for early stopping
    slides = sorted({int(f[:3]) for f, *_ in cands})
    rng = np.random.default_rng(0)
    hold = set(rng.choice(slides, max(1, len(slides) // 10), replace=False).tolist())
    tr = [c for c in cands if int(c[0][:3]) not in hold]
    va = [c for c in cands if int(c[0][:3]) in hold]
    n_pos = sum(c[3] for c in tr)
    print(f"verifier samples: train {len(tr)} ({n_pos} mitosis), holdout {len(va)}", flush=True)

    w = np.array([1.0 if c[3] else n_pos / (len(tr) - n_pos) for c in tr])   # balanced sampling
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=min(len(tr), 40000), replacement=True)
    dl = DataLoader(VerifierData(tr, True), args.batch, sampler=sampler, num_workers=args.workers, drop_last=True)
    dv = DataLoader(VerifierData(va, False), 512, num_workers=args.workers)
    ver = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss()
    best = -1
    for ep in range(args.epochs):
        ver.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(ver(x)[:, 0].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        ver.eval()
        P, Y = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x, y in dv:
                P.append(torch.sigmoid(ver(x.to(device))[:, 0].float()).cpu().numpy())
                Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        f1s = [2 * ((P >= t) & (Y == 1)).sum() / max((P >= t).sum() + (Y == 1).sum(), 1) for t in np.arange(0.1, 0.95, 0.05)]
        print(f"epoch {ep + 1}: loss {tot / len(dl):.4f}  holdout best F1 {max(f1s):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if max(f1s) > best:
            best = max(f1s)
            torch.save(ver.state_dict(), out / "mitosis_ver.pt")

    ver.load_state_dict(torch.load(out / "mitosis_ver.pt"))
    ver.eval()
    b, rows = tune(det, ver, load_full("val"), device)
    print("val (detector + verifier):", b, flush=True)
    json.dump(dict(best=b, rows=rows), open(out / "mitosis_ver_val.json", "w"), indent=2)


if __name__ == "__main__":
    main()
