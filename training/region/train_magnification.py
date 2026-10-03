"""Auto-magnification: tiny CNN that predicts log2(µm/px) from 160x160 crops of a camera frame.

Training images of known scale are resampled to random target scales (0.2-4 µm/px, i.e.
roughly 50x down to 2.5x), so the model learns texture scale (nucleus size, fibre width)
rather than a fixed set of objectives. The app averages several crops per frame and snaps
to the nearest objective in the microscope's calibration.

Sources (native µm/px): MIDOG++ crops + slides (H&E, 0.25), PanNuke (H&E, 0.25), DeepLIIF
(Ki-67 IHC, 0.25), BCI (HER2 IHC, ~1.0), BACH photos (H&E, 1.0). Held out: MIDOG test slides,
PanNuke fold 3, DeepLIIF test set, BCI test, BACH photo test split.

    CUDA_VISIBLE_DEVICES=1 ~/venvs/pathoassist-torch/bin/python training/region/train_magnification.py
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

CROP = 160
LO, HI = 0.2, 4.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
OBJECTIVES = {"40x": 0.25, "20x": 0.5, "10x": 1.0, "4x": 2.5}   # typical camera scales for evaluation


def deepliif_ihc(zip_name, originals_only=False, every=1):
    """IHC panel (first square) of each DeepLIIF image, without the stardist-dependent eval helpers."""
    import zipfile
    with zipfile.ZipFile(ROOT / "training/data/deepliif" / zip_name) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png"))
        if originals_only:
            names = [n for n in names if "_" not in Path(n).stem]
        for n in names[::every]:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            yield cv2.cvtColor(im[:, :im.shape[0]], cv2.COLOR_BGR2RGB)


def gather():
    """Returns {split: [(image uint8 RGB, native µm/px, source)]}."""
    data = {"train": [], "test": []}
    midog = json.loads((ROOT / "training/data/midog/full.json").read_text())
    for r in midog:                                   # val + test slides; store at 0.5 µm/px to save RAM
        im = cv2.cvtColor(cv2.imread(str(ROOT / "training/data/midog/full" / r["file"])), cv2.COLOR_BGR2RGB)
        data["test" if r["split"] == "test" else "train"].append((cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 0.5, "midog_slide"))
    crops = sorted((ROOT / "training/data/midog/crops").glob("*.jpg"))[::4]
    data["train"] += [(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), 0.25, "midog_crop") for f in crops]
    from training.data import OUT as PN
    for k, split in ((1, "train"), (3, "test")):
        X = np.load(PN / f"fold{k}_images.npy", mmap_mode="r")
        data[split] += [(np.asarray(X[i]), 0.25, "pannuke") for i in range(0, len(X), 6)]
    data["train"] += [(x, 0.25, "deepliif") for x in deepliif_ihc("BC-DeepLIIF_Training_Set.zip", originals_only=True)]
    data["test"] += [(x, 0.25, "deepliif") for x in deepliif_ihc("DeepLIIF_Testing_Set.zip", every=4)]
    bci = json.loads((ROOT / "training/data/bci/index.json").read_text())
    for split in ("train", "test"):
        fs = [r for r in bci if r["split"] == split][:400]
        data[split] += [(cv2.cvtColor(cv2.imread(str(ROOT / "training/data/bci/ihc" / r["file"])), cv2.COLOR_BGR2RGB), 1.0, "bci") for r in fs]
    bach = json.loads((ROOT / "training/data/bach/index.json").read_text())["photos"]
    for r in bach:
        im = cv2.cvtColor(cv2.imread(str(ROOT / "training/data/bach/photos" / r["file"])), cv2.COLOR_BGR2RGB)
        data["test" if r["split"] == "test" else "train"].append((im, 1.0, "bach_photo"))
    return data


def sample(img, um0, um, rng):
    """CROP x CROP crop at target scale `um` (>= um0), or None if the image is too small."""
    f = um0 / um
    need = int(np.ceil(CROP / f)) + 2
    H, W = img.shape[:2]
    if need > min(H, W):
        return None
    for _ in range(5):
        y, x = rng.integers(0, H - need + 1), rng.integers(0, W - need + 1)
        c = img[y:y + need, x:x + need]
        if (cv2.cvtColor(c, cv2.COLOR_RGB2HSV)[..., 1] > 20).mean() > 0.25:
            break
    c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    return c[:CROP, :CROP]


class ScaleSet(Dataset):
    def __init__(self, items, n, train):
        self.items, self.n, self.train = items, n, train

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        while True:
            img, um0, _ = self.items[rng.integers(len(self.items))]
            um = float(np.exp(rng.uniform(np.log(max(LO, um0)), np.log(HI))))
            c = sample(img, um0, um, rng)
            if c is not None:
                break
        x = c.astype(np.float32) / 255
        if self.train:
            A.rng = rng
            x = np.ascontiguousarray(np.rot90(x, rng.integers(4)))
            x = A._stain(x, np.zeros(x.shape[:2], np.uint16), p_ihc=0.0)
            x = A._optics(x)
        x = (np.clip(x, 0, 1) - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)), torch.tensor(np.log2(um), dtype=torch.float32)


@torch.no_grad()
def frame_estimate(model, img, device, n_crops=8, seed=0):
    """Median log2(µm/px) over tissue crops of a frame (already at its own scale)."""
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    cs = []
    for _ in range(n_crops * 3):
        y, x = rng.integers(0, H - CROP + 1), rng.integers(0, W - CROP + 1)
        c = img[y:y + CROP, x:x + CROP]
        if (cv2.cvtColor(c, cv2.COLOR_RGB2HSV)[..., 1] > 20).mean() > 0.25:
            cs.append(c)
        if len(cs) == n_crops:
            break
    if not cs:
        cs = [img[:CROP, :CROP]]
    x = torch.from_numpy(((np.stack(cs).astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)).to(device)
    return float(np.median(model(x)[:, 0].cpu().numpy()))


def main():
    device = "cuda"
    data = gather()
    print({k: len(v) for k, v in data.items()}, flush=True)
    dl = DataLoader(ScaleSet(data["train"], 60000, True), 256, shuffle=True, num_workers=16, drop_last=True)
    model = timm.create_model("mobilenetv3_small_050", pretrained=True, num_classes=1).to(device)
    epochs = 12
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * len(dl), pct_start=0.1)
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, t in dl:
            x, t = x.to(device, non_blocking=True), t.to(device)
            loss = F.smooth_l1_loss(model(x)[:, 0], t, beta=0.25)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}: loss {tot / len(dl):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), ROOT / "models" / "magnification_mnv3.pt")

    # frame-level evaluation: held-out images resampled to each objective's scale
    model.eval()
    rng = np.random.default_rng(1)
    rows = []
    names, scales = list(OBJECTIVES), np.array(list(OBJECTIVES.values()))
    for img, um0, src in data["test"]:
        for name, um in OBJECTIVES.items():
            if um < um0 - 1e-6:
                continue
            f = um0 / um
            fr = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
            if min(fr.shape[:2]) < CROP:
                continue
            est = frame_estimate(model, fr, device, seed=int(rng.integers(1 << 30)))
            pred = names[int(np.abs(np.log2(scales) - est).argmin())]
            rows.append(dict(source=src, objective=name, true_log2=float(np.log2(um)), est_log2=est, pred=pred))
    acc = np.mean([r["pred"] == r["objective"] for r in rows])
    err = np.array([abs(r["est_log2"] - r["true_log2"]) for r in rows])
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r["pred"] == r["objective"])
    by_obj = {o: float(np.mean([r["pred"] == o for r in rows if r["objective"] == o])) for o in OBJECTIVES}
    res = dict(n=len(rows), objective_accuracy=float(acc), median_abs_error_factor=float(2 ** np.median(err)),
               p95_abs_error_factor=float(2 ** np.percentile(err, 95)),
               by_source={k: float(np.mean(v)) for k, v in by.items()}, by_objective=by_obj)
    print(json.dumps(res, indent=2), flush=True)
    model.cpu()
    torch.onnx.export(model, torch.zeros(1, 3, CROP, CROP), ROOT / "models" / "magnification_mnv3.onnx",
                      input_names=["crops"], output_names=["log2_um_per_px"], dynamic_axes={"crops": {0: "n"}},
                      opset_version=17, dynamo=False)
    (ROOT / "models" / "magnification_mnv3.json").write_text(json.dumps(dict(crop=CROP, range_um_per_px=[LO, HI]), indent=2))
    (ROOT / "eval" / "results" / "magnification.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
