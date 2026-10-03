"""Feature 5 tissue-region model: Phikon-v2 teacher -> EfficientNet-B0 student (distillation).

Classes: 0 invasive, 1 in_situ, 2 benign, 3 normal_other (stroma/fat/normal), 4 background.
Patches: 224 px at ~0.93 µm/px (224 µm fields), from training/region/prepare_bach.py.

1. embed   : Phikon-v2 (owkin, ViT-L DINOv2, open weights) CLS embeddings for every patch -
             labelled BACH WSI patches, tiles of the BACH microscope photos (camera domain),
             unlabelled BACH WSI patches, and unlabelled tiles from the MIDOG++ breast slides.
2. teacher : logistic regression on the embeddings (slides A01-A08 + training photos). A
             slide-only teacher gets ~33% of held-out camera photos right, so photo tiles are
             essential; local camera patches will matter even more.
3. distil  : the student learns the teacher's soft labels (T=2) on ALL patches, plus hard labels
             on the labelled ones, with heavy stain / blur / vignetting augmentation.
4. evaluate: held-out slides A09-A10 (patch level; binary tumour = invasive + in situ) and the
             BACH photo test split (image level: Normal / Benign / InSitu / Invasive).
5. export  : models/region_effb0.onnx (+ .json).

    CUDA_VISIBLE_DEVICES=0 ~/venvs/pathoassist-torch/bin/python training/region/train_region.py
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import training.augment as A

D = ROOT / "training" / "data" / "bach"
MIDOG = ROOT / "training" / "data" / "midog"
CLASSES = ["invasive", "in_situ", "benign", "normal_other", "background"]
UM = 0.93                      # model µm/px
PATCH = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def midog_tiles(n_per_slide=60, seed=0):
    """Unlabelled 224 px tiles at 0.93 µm/px from the MIDOG++ breast slides (0.25 µm/px)."""
    out_dir = D / "midog_tiles"
    idx = out_dir / "index.json"
    if idx.exists():
        return json.loads(idx.read_text())
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    recs = []
    for f in sorted((MIDOG / "full").glob("*.jpg")):
        im = cv2.imread(str(f))
        im = cv2.resize(im, None, fx=0.25 / UM, fy=0.25 / UM, interpolation=cv2.INTER_AREA)
        H, W = im.shape[:2]
        for j in range(n_per_slide):
            y, x = rng.integers(0, H - PATCH), rng.integers(0, W - PATCH)
            t = im[y:y + PATCH, x:x + PATCH]
            if (cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[..., 1] > 20).mean() < 0.3:
                continue
            fn = f"{f.stem}_{j:03d}.jpg"
            cv2.imwrite(str(out_dir / fn), t, [cv2.IMWRITE_JPEG_QUALITY, 90])
            recs.append(dict(file=f"midog_tiles/{fn}", label=-1, split="unlabeled", slide=f"midog_{f.stem}"))
    idx.write_text(json.dumps(recs))
    return recs


def photo_tiles(photos):
    """224 px tiles (stride 112) of the BACH microscope photos at the model scale. The photos are
    microscope-camera images - the app's own domain - and a slide-only teacher transfers poorly
    to them, so their tiles are labelled training data (Normal -> normal_other)."""
    out_dir = D / "photo_tiles"
    idx = out_dir / "index.json"
    if idx.exists():
        return json.loads(idx.read_text())
    out_dir.mkdir(exist_ok=True)
    recs = []
    for k, p in enumerate(photos):
        im = cv2.imread(str(D / "photos" / p["file"]))
        im = cv2.resize(im, None, fx=1.0 / UM, fy=1.0 / UM)
        H, W = im.shape[:2]
        for a in range(0, H - PATCH + 1, PATCH // 2):
            for b in range(0, W - PATCH + 1, PATCH // 2):
                fn = f"{Path(p['file']).stem}_{a}_{b}.jpg"
                cv2.imwrite(str(out_dir / fn), im[a:a + PATCH, b:b + PATCH], [cv2.IMWRITE_JPEG_QUALITY, 92])
                recs.append(dict(file=f"photo_tiles/{fn}", label=p["label"], split=f"{p['split']}_photo", slide=f"photo{k}"))
    idx.write_text(json.dumps(recs))
    return recs


def load_records():
    index = json.loads((D / "index.json").read_text())
    pats = [dict(r, file=f"patches/{r['file']}") for r in index["patches"]]
    return pats + midog_tiles() + photo_tiles(index["photos"]), index["photos"]


def read(rel):
    return cv2.cvtColor(cv2.imread(str(D / rel)), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def embed(recs, device, batch=128):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon-v2", token=False).to(device).eval().half()
    out = np.zeros((len(recs), 1024), np.float16)
    for i in range(0, len(recs), batch):
        x = np.stack([read(r["file"]) for r in recs[i:i + batch]]).astype(np.float32) / 255
        x = torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2)).to(device).half()
        out[i:i + batch] = m(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy()
    return out


def train_teacher(E, y, tr, va, device, epochs=300):
    Et = torch.from_numpy(E.astype(np.float32)).to(device)
    mu, sd = Et[tr].mean(0), Et[tr].std(0) + 1e-6
    Et = (Et - mu) / sd
    yt = torch.from_numpy(y).to(device)
    w = torch.tensor(1.0 / np.bincount(y[tr], minlength=5).clip(1), dtype=torch.float32, device=device)
    head = nn.Linear(1024, 5).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state = -1, None
    for ep in range(epochs):
        head.train()
        loss = F.cross_entropy(head(Et[tr]), yt[tr], weight=w / w.mean(), label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 10 == 9:
            head.eval()
            p = head(Et[va]).argmax(1)
            bacc = np.mean([(p[yt[va] == c] == c).float().mean().item() for c in range(5) if (yt[va] == c).any()])
            if bacc > best:
                best, state = bacc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        logits = head(Et).cpu().numpy()
    return logits, best


class PatchSet(Dataset):
    def __init__(self, recs, soft, train):
        self.r, self.soft, self.train = recs, soft, train

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        x = read(self.r[i]["file"]).astype(np.float32) / 255
        if self.train:
            rng = np.random.default_rng()
            A.rng = rng
            if rng.random() < 0.5:
                x = x[:, ::-1]
            x = np.ascontiguousarray(np.rot90(x, rng.integers(4)))
            s = rng.uniform(0.8, 1.25)                      # magnification calibration error
            if abs(s - 1) > 0.02:
                x = cv2.resize(cv2.resize(x, None, fx=s, fy=s), (PATCH, PATCH)) if s > 1 else \
                    cv2.resize(cv2.copyMakeBorder(cv2.resize(x, None, fx=s, fy=s), 0, PATCH - round(PATCH * s), 0,
                                                  PATCH - round(PATCH * s), cv2.BORDER_REFLECT), (PATCH, PATCH))
            x = A._stain(x, np.zeros(x.shape[:2], np.uint16), p_ihc=0.0)
            x = A._optics(x)
        x = (np.clip(x, 0, 1) - MEAN) / STD
        lab = self.r[i]["label"]
        return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)), torch.from_numpy(self.soft[i]), lab


@torch.no_grad()
def predict(model, recs, device, batch=256):
    model.eval()
    P = []
    for i in range(0, len(recs), batch):
        x = np.stack([read(r["file"]) for r in recs[i:i + batch]]).astype(np.float32) / 255
        x = torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2)).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(model(x).float().softmax(1).cpu().numpy())
    return np.concatenate(P)


def patch_metrics(P, y):
    pred = P.argmax(1)
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (y, pred), 1)
    tum_t, tum_p = y <= 1, pred <= 1
    tissue = y != 4
    return dict(n=int(len(y)), accuracy=float((pred == y).mean()),
                balanced_accuracy=float(np.mean([(pred[y == c] == c).mean() for c in range(5) if (y == c).any()])),
                tumour_accuracy=float((tum_t == tum_p).mean()),
                tumour_sensitivity=float((tum_p & tum_t).sum() / max(tum_t.sum(), 1)),
                tumour_specificity=float((~tum_p & ~tum_t & tissue).sum() / max((~tum_t & tissue).sum(), 1)),
                confusion=cm.tolist())


def photo_probs(model, photos, device):
    """Per photo: tile at stride 112 -> class-probability per tile."""
    out = []
    for r in photos:
        im = read(f"photos/{r['file']}").astype(np.float32) / 255
        im = cv2.resize(im, None, fx=1.0 / UM, fy=1.0 / UM)            # photos are stored at 1 µm/px
        H, W = im.shape[:2]
        tiles = [im[y:y + PATCH, x:x + PATCH] for y in range(0, H - PATCH + 1, 112) for x in range(0, W - PATCH + 1, 112)]
        x = torch.from_numpy(((np.stack(tiles) - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().softmax(1).cpu().numpy())
    return out


def photo_rule(tile_probs, fr):
    """Image label (0 invasive, 1 in situ, 2 benign, 3 normal) from tile fractions, most severe first."""
    pred = []
    for P in tile_probs:
        c = P.argmax(1)
        frac = np.array([(c == k).mean() for k in range(3)])
        pred.append(next((k for k in range(3) if frac[k] >= fr[k]), 3))
    return np.array(pred)


def fit_photo_rule(tile_probs, y):
    grid = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    best = max(((np.mean(photo_rule(tile_probs, (a, b, c)) == y), (a, b, c)) for a in grid for b in grid for c in grid))
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--T", type=float, default=2.0)
    args = ap.parse_args()
    device = "cuda"
    recs, photos = load_records()
    y = np.array([r["label"] for r in recs])
    split = np.array([r["split"] for r in recs])
    print(f"{len(recs)} patches: " + ", ".join(f"{s} {np.sum(split == s)}" for s in ("train", "val", "test", "unlabeled", "train_photo", "test_photo")),
          "| labelled class counts", np.bincount(y[y >= 0], minlength=5).tolist(), flush=True)

    emb_path = D / "phikon_v2_cls.npy"
    if emb_path.exists() and len(np.load(emb_path, mmap_mode="r")) == len(recs):
        E = np.load(emb_path)
    else:
        t0 = time.time()
        E = embed(recs, device)
        np.save(emb_path, E)
        print(f"embedded {len(recs)} patches in {time.time() - t0:.0f}s", flush=True)

    tr = np.nonzero(np.isin(split, ["train", "val", "train_photo"]))[0]
    va, te = np.nonzero(split == "val")[0], np.nonzero(split == "test")[0]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(E[tr].astype(np.float32))
    teacher = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(sc.transform(E[tr].astype(np.float32)), y[tr])
    teacher_P = teacher.predict_proba(sc.transform(E.astype(np.float32)))
    res = {"teacher": {"test": patch_metrics(teacher_P[te], y[te])}}
    print("teacher test (held-out slides):", {k: round(v, 3) for k, v in res["teacher"]["test"].items() if k != "confusion"}, flush=True)

    logits = np.log(np.clip(teacher_P, 1e-6, 1))
    soft = torch.softmax(torch.from_numpy(logits) / args.T, 1).numpy().astype(np.float32)
    use = np.nonzero(~np.isin(split, ["test", "test_photo"]))[0]   # held-out slides / photos never seen by the student
    dl = DataLoader(PatchSet([recs[i] for i in use], soft[use], True), args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=True)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    val_recs = [recs[i] for i in va]
    best = -1
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, s, lab in dl:
            x, s, lab = x.to(device, non_blocking=True), s.to(device), lab.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x).float()
            loss = F.kl_div(F.log_softmax(out / args.T, 1), s, reduction="batchmean") * args.T ** 2
            hard = lab >= 0
            if hard.any():
                loss = loss + 0.5 * F.cross_entropy(out[hard], lab[hard])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        m = patch_metrics(predict(model, val_recs, device), y[va])
        print(f"epoch {ep + 1}: loss {tot / len(dl):.4f}  val bal.acc {m['balanced_accuracy']:.3f} "
              f"tumour acc {m['tumour_accuracy']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            torch.save(model.state_dict(), ROOT / "models" / "region_effb0.pt")

    model.load_state_dict(torch.load(ROOT / "models" / "region_effb0.pt"))
    res["student"] = {"test": patch_metrics(predict(model, [recs[i] for i in te], device), y[te])}
    print("student test:", {k: round(v, 3) for k, v in res["student"]["test"].items() if k != "confusion"}, flush=True)

    ptr = [p for p in photos if p["split"] == "train"]
    pte = [p for p in photos if p["split"] == "test"]
    if pte:
        fr, acc_tr = fit_photo_rule(photo_probs(model, ptr, device), np.array([p["label"] for p in ptr]))
        yt = np.array([p["label"] for p in pte])
        pp = photo_rule(photo_probs(model, pte, device), fr)
        cm = np.zeros((4, 4), int)
        np.add.at(cm, (yt, pp), 1)
        res["photos"] = dict(rule_fractions=fr, train_accuracy=acc_tr, test_accuracy=float((pp == yt).mean()),
                             test_cancer_vs_noncancer=float(((pp <= 1) == (yt <= 1)).mean()), confusion=cm.tolist(),
                             n_test=len(pte))
        print("photos (image level):", {k: v for k, v in res["photos"].items() if k != "confusion"}, flush=True)

    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, PATCH, PATCH), ROOT / "models" / "region_effb0.onnx",
                      input_names=["patches"], output_names=["logits"], dynamic_axes={"patches": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    (ROOT / "models" / "region_effb0.json").write_text(json.dumps(dict(
        classes=CLASSES, um_per_px=UM, patch=PATCH, photo_rule_fractions=res.get("photos", {}).get("rule_fractions"),
        source="BACH WSIs A01-A08 + photos; distilled from Phikon-v2 linear head"), indent=2))
    (ROOT / "eval" / "results" / "region_bach.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
