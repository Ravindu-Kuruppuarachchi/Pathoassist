"""Optional HER2 cross-check: EfficientNet-B0 field classifier (0 / 1+ / 2+ / 3+) on BCI IHC.

When it disagrees with the rule engine the app shows "review carefully". Trained on BCI-train
(1500 images, ~1 µm/px), evaluated on BCI-test (977).
LICENCE: BCI is non-commercial - this model is for research / validation only; retrain on
local data before any commercial use.

    CUDA_VISIBLE_DEVICES=2 ~/venvs/pathoassist-torch/bin/python training/her2/train_field_classifier.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import training.augment as A
from core.ihc.scoring import weighted_kappa

D = ROOT / "training" / "data" / "bci"
LABELS = {"0": 0, "1+": 1, "2+": 2, "3+": 3}
CROP, SIZE = 512, 384           # 512 µm crops at 1 µm/px, fed at 384 px
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class BCI(Dataset):
    def __init__(self, recs, train):
        self.r, self.train = recs, train
        self.cache = {}

    def __len__(self):
        return len(self.r)

    def img(self, i):
        return cv2.cvtColor(cv2.imread(str(D / "ihc" / self.r[i]["file"])), cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        im = self.img(i).astype(np.float32) / 255
        if self.train:
            rng = np.random.default_rng()
            A.rng = rng
            y, x = rng.integers(0, 1024 - CROP + 1, 2)
            im = im[y:y + CROP, x:x + CROP]
            if rng.random() < 0.5:
                im = im[:, ::-1]
            im = np.ascontiguousarray(np.rot90(im, rng.integers(4)))
            im = A._optics(np.clip(im * rng.uniform(0.9, 1.1, 3) + rng.uniform(-0.04, 0.04), 0, 1).astype(np.float32))
            im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (SIZE * 2, SIZE * 2), interpolation=cv2.INTER_AREA)   # 4 tiles of 512 µm
        x = (im - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)), LABELS[self.r[i]["label"]]


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    P, Y = [], []
    for x, y in dl:
        x = x.to(device)
        tiles = torch.cat([x[..., :SIZE, :SIZE], x[..., :SIZE, SIZE:], x[..., SIZE:, :SIZE], x[..., SIZE:, SIZE:]])
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(tiles).float().softmax(1)
        P.append(p.view(4, len(x), 4).mean(0).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    device = "cuda"
    recs = json.loads((D / "index.json").read_text())
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    va, tr = [tr[i] for i in perm[:200]], [tr[i] for i in perm[200:]]
    y_tr = np.array([LABELS[r["label"]] for r in tr])
    w = 1.0 / np.bincount(y_tr, minlength=4)[y_tr]
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr) * 2, replacement=True)
    dl = DataLoader(BCI(tr, True), 32, sampler=sampler, num_workers=12, drop_last=True, persistent_workers=True)
    dv = DataLoader(BCI(va, False), 16, num_workers=8)
    dt = DataLoader(BCI(te, False), 16, num_workers=8)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4).to(device)
    epochs = 15
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best = -1
    out = ROOT / "models" / "her2_field_effb0"
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        P, Y = predict(model, dv, device)
        k = weighted_kappa(Y, P.argmax(1))
        print(f"epoch {ep + 1}: loss {tot / len(dl):.4f} val kappa {k:.3f} acc {(P.argmax(1) == Y).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
        if k > best:
            best = k
            torch.save(model.state_dict(), str(out) + ".pt")
    model.load_state_dict(torch.load(str(out) + ".pt"))
    P, Y = predict(model, dt, device)
    p = P.argmax(1)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (Y, p), 1)
    res = dict(n=int(len(Y)), weighted_kappa=float(weighted_kappa(Y, p)), accuracy=float((p == Y).mean()),
               within_one=float((np.abs(p - Y) <= 1).mean()), confusion=cm.tolist())
    print("BCI test:", json.dumps(res), flush=True)
    np.save(ROOT / "eval" / "results" / "her2_field_test_probs.npy", P)
    (ROOT / "eval" / "results" / "her2_field_classifier.json").write_text(json.dumps(res, indent=2))
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(out) + ".onnx", input_names=["field"],
                      output_names=["logits"], dynamic_axes={"field": {0: "n"}}, opset_version=17, dynamo=False)
    Path(str(out) + ".json").write_text(json.dumps(dict(um_per_px=1.0, crop_um=CROP, input=SIZE, classes=["0", "1+", "2+", "3+"],
                                                        licence="trained on BCI (non-commercial) - research/validation only"), indent=2))


if __name__ == "__main__":
    main()
