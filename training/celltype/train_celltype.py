"""Train the nucleus cell-type classifier (MobileNetV3-Small, 64x64 crops at 0.25 µm/px).

Classes: tumour / lymphocyte / stroma / other.
Public stand-in data (until SHIDC-B-Ki-67 or local QuPath labels are available):
PanNuke nuclei with their class labels, mapped
    neoplastic -> tumour, inflammatory -> lymphocyte, connective -> stroma,
    dead + non-neoplastic epithelial -> other.
PanNuke is H&E, so most training crops are converted to an IHC look (pseudo-IHC:
eosin removed, haematoxylin counterstain, DAB on random nuclei - see training/augment.py)
and the classifier is pushed to ignore DAB colour when deciding the cell type.

    ~/venvs/pathoassist-torch/bin/python training/celltype/train_celltype.py
"""
import argparse
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
import torch.nn as nn
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

import training.augment as A
from training.data import OUT as PANNUKE, load_split

CLASSES = ["tumour", "lymphocyte", "stroma", "other"]
PANNUKE_TO_OURS = np.array([-1, 0, 1, 2, 3, 3])  # background, neoplastic, inflammatory, connective, dead, epithelial
CROP, CTX = 64, 96                               # model crop; context crop augmented before centre-cropping
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_classes(split):
    """Class map aligned with load_split(split)."""
    if split == "test":
        return np.load(PANNUKE / "fold3_classes.npy")
    C2 = np.load(PANNUKE / "fold2_classes.npy")
    perm = np.random.default_rng(0).permutation(len(C2))   # same permutation as training.data.load_split
    n_val = len(C2) // 10
    if split == "val":
        return C2[perm[:n_val]]
    return np.concatenate([np.load(PANNUKE / "fold1_classes.npy"), C2[perm[n_val:]]])


def index_nuclei(Y, C):
    """(image, cy, cx, class) for every labelled nucleus, class = majority vote inside the mask."""
    rows = []
    for i, (y, c) in enumerate(zip(Y, C)):
        ids = np.unique(y)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        com = ndi.center_of_mass(np.ones_like(y), y, ids)
        for lid, (cy, cx) in zip(ids, com):
            vals = c[y == lid]
            cls = PANNUKE_TO_OURS[np.bincount(vals, minlength=6)[1:].argmax() + 1]
            rows.append((i, int(round(cy)), int(round(cx)), int(cls)))
    return np.array(rows, np.int32)


class NucleusCrops(Dataset):
    def __init__(self, X, Y, index, train, p_ihc=0.7, seed=None, zoom=1.0):
        """zoom = model µm/px / 0.25: 2.0 gives 64 px crops covering 32 µm (more tissue context)."""
        self.X, self.Y, self.idx, self.train, self.p_ihc, self.zoom = X, Y, index, train, p_ihc, zoom
        self.seed = seed   # fixed seed -> deterministic pseudo-IHC for evaluation

    def __len__(self):
        return len(self.idx)

    def _crop(self, a, cy, cx, size):
        r = size // 2
        pad = [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2)
        a = np.pad(a, pad, mode="reflect" if a.ndim == 3 else "constant")
        return a[cy:cy + size, cx:cx + size]

    def __getitem__(self, k):
        i, cy, cx, cls = self.idx[k]
        if self.seed is not None:
            A.rng = np.random.default_rng(self.seed + k)
        elif not hasattr(self, "_seeded"):
            A.rng = np.random.default_rng()   # fresh stream per DataLoader worker
            self._seeded = True
        n = int(round(CTX * self.zoom))
        x = self._crop(self.X[i], cy, cx, n).astype(np.float32) / 255
        y = self._crop(self.Y[i], cy, cx, n)
        if self.zoom != 1.0:
            x = cv2.resize(x, (CTX, CTX), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y.astype(np.float32), (CTX, CTX), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        if self.train:
            if A.rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            k4 = A.rng.integers(4)
            x, y = np.ascontiguousarray(np.rot90(x, k4)), np.ascontiguousarray(np.rot90(y, k4))
            x = A._stain(x, y, p_ihc=self.p_ihc)
            x = A._optics(x)
            s = CTX // 2 - CROP // 2 + A.rng.integers(-3, 4, 2)   # small centre jitter (detector error)
        elif self.seed is not None:   # evaluation in pseudo-IHC appearance
            x = A._stain_ihc_only(x, y)
            s = np.array([CTX // 2 - CROP // 2] * 2)
        else:
            s = np.array([CTX // 2 - CROP // 2] * 2)
        x = x[s[0]:s[0] + CROP, s[1]:s[1] + CROP]
        x = (np.clip(x, 0, 1) - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)), int(cls)


def evaluate(model, loader, device):
    model.eval()
    P, T = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, t in loader:
            P.append(model(x.to(device)).float().softmax(1).cpu().numpy())
            T.append(t.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    pred = P.argmax(1)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (T, pred), 1)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    tp = cm[0, 0]
    tum_p, tum_r = tp / max(cm[:, 0].sum(), 1), tp / max(cm[0].sum(), 1)
    return dict(accuracy=float((pred == T).mean()), balanced_accuracy=float(recall.mean()),
                recall=dict(zip(CLASSES, recall.round(3).tolist())), precision=dict(zip(CLASSES, prec.round(3).tolist())),
                tumour_f1=float(2 * tum_p * tum_r / max(tum_p + tum_r, 1e-9)), confusion=cm.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "models" / "celltype_mnv3s"))
    ap.add_argument("--um-per-px", type=float, default=0.25, help="crop scale; 0.5 = 32 µm context")
    args = ap.parse_args()
    device = "cuda"

    data = {}
    for split in ("train", "val", "test"):
        X, Y, _ = load_split(split)
        C = load_classes(split)
        idx = index_nuclei(Y, C)
        data[split] = (X, Y, idx)
        print(split, len(idx), "nuclei, class counts", np.bincount(idx[:, 3], minlength=4).tolist(), flush=True)

    zoom = args.um_per_px / 0.25
    tr = NucleusCrops(*data["train"], train=True, zoom=zoom)
    freq = np.bincount(data["train"][2][:, 3], minlength=4).astype(float)
    weights = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32)
    weights = weights / weights.mean()
    dl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)
    mk = lambda split, seed=None: DataLoader(NucleusCrops(*data[split], train=False, seed=seed, zoom=zoom), 1024,
                                             num_workers=args.workers)

    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    crit = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    scaler = torch.amp.GradScaler()
    best = -1
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, t in dl:
            x, t = x.to(device, non_blocking=True), t.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        v = evaluate(model, mk("val", seed=123), device)   # select on pseudo-IHC val (the target look)
        print(f"epoch {ep + 1}: loss {tot / len(dl):.4f}  val(IHC-look) bal.acc {v['balanced_accuracy']:.3f} "
              f"tumour F1 {v['tumour_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if v["balanced_accuracy"] > best:
            best = v["balanced_accuracy"]
            torch.save(model.state_dict(), args.out + ".pt")

    model.load_state_dict(torch.load(args.out + ".pt"))
    res = {"test_HE": evaluate(model, mk("test"), device), "test_pseudoIHC": evaluate(model, mk("test", seed=7), device)}
    for k, r in res.items():
        print(k, json.dumps({kk: r[kk] for kk in ("accuracy", "balanced_accuracy", "tumour_f1", "recall")}))

    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, CROP, CROP), args.out + ".onnx", input_names=["crops"],
                      output_names=["logits"], dynamic_axes={"crops": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    Path(args.out + ".json").write_text(json.dumps(dict(crop=CROP, um_per_px=args.um_per_px, classes=CLASSES,
                                                        source="PanNuke folds 1-2 (+pseudo-IHC)"), indent=2))
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(args.out).name}_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
