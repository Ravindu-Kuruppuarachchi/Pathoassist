"""Train the mitosis heatmap detector on MIDOG++ human-breast crops.

    ~/venvs/pathoassist-torch/bin/python training/mitosis/train_detector.py --epochs 40

Model selection: F1 on the 15 whole validation slides (threshold swept), MIDOG matching.
Saves models/mitosis_det.pt (+ .onnx, .json with the chosen threshold).
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (D, MATCH_RADIUS, ROOT, CropDataset, HeatmapDetector, focal_loss, load_full, match, peaks,
                    predict_full)


def evaluate_full(model, recs, device, thresholds=np.arange(0.1, 0.95, 0.05), return_preds=False, scale=1.0):
    model.eval()
    preds, gts = [], []
    for r in recs:
        img = cv2.cvtColor(cv2.imread(str(D / "full" / r["file"])), cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hm = predict_full(model, img, device)
        preds.append(peaks(hm, thresholds.min()))
        gts.append(scale * np.array([[x, y] for x, y, c in r["points"] if c == 1]).reshape(-1, 2))
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for p, g in zip(preds, gts):
            a, b, c, _ = match(p[p[:, 2] >= t][:, :2], g, radius=MATCH_RADIUS * scale)
            tp, fp, fn = tp + a, fp + b, fn + c
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rows.append(dict(thresh=round(float(t), 3), tp=tp, fp=fp, fn=fn, f1=f1,
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
    best = max(rows, key=lambda r: r["f1"])
    return (best, rows, preds, gts) if return_preds else (best, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "models" / "mitosis_det"))
    ap.add_argument("--um-per-px", type=float, default=0.25, help="1.0 trains the 10x hotspot model")
    args = ap.parse_args()
    device = "cuda"

    crops = json.loads((D / "crops.json").read_text())
    val = load_full("val")
    print(f"{len(crops)} training crops, {sum(p[2] == 1 for r in crops for p in r['points'])} mitosis points; "
          f"{len(val)} val slides", flush=True)
    scale = 0.25 / args.um_per_px
    dl = DataLoader(CropDataset(crops, train=True, scale=scale), args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=True)
    model = HeatmapDetector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    best_f1, log = -1, []
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, hm, nw in dl:
            x, hm, nw = x.to(device, non_blocking=True), hm.to(device), nw.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = focal_loss(logits.float(), hm, nw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep + 1}: loss {tot / len(dl):.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % 2 == 0 or ep + 1 == args.epochs:
            best, _ = evaluate_full(model, val, device, scale=scale)
            msg += f"  val F1 {best['f1']:.3f} @ {best['thresh']} (P {best['precision']:.2f} R {best['recall']:.2f})"
            log.append(dict(epoch=ep + 1, **best))
            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                torch.save(model.state_dict(), args.out + ".pt")
                json.dump(dict(thresh=best["thresh"], val=best, epoch=ep + 1), open(args.out + "_val.json", "w"), indent=2)
        print(msg, flush=True)
    json.dump(log, open(args.out + "_trainlog.json", "w"), indent=2)


if __name__ == "__main__":
    main()
