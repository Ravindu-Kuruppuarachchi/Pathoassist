"""Phikon-v2 embeddings of 40x tumour fields for Feature 7 (MITOS-ATYPIA-14 + Databiox).

Each 40x image is resampled to 0.5 µm/px (Phikon's 20x training scale), cut into 224 px tiles
(112 µm), tissue tiles are embedded (CLS token, 1024-d) and the mean (+ std) embedding is kept per
image. Outputs training/data/pleo_embeddings.npz: image id, source, mean / std embedding.

    CUDA_VISIBLE_DEVICES=0 ~/venvs/pathoassist-torch/bin/python training/pleomorphism/embed_tiles.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import cv2
import numpy as np
import torch

UM, TILE = 0.5, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def images():
    out = []
    mi = json.loads((ROOT / "training/data/mitos_atypia/index.json").read_text())
    for r in mi:
        for fn in r["x40_frames"]:
            out.append(("mitos", Path(fn).stem, ROOT / "training/data/mitos_atypia/x40" / fn, r["um_per_px"]))
    dx = json.loads((ROOT / "training/data/databiox/index.json").read_text())
    for r in dx:
        if r["mag"] == "40x" and r.get("um_per_px"):
            out.append(("databiox", Path(r["file"]).stem, ROOT / "training/data/databiox" / r["file"], r["um_per_px"]))
    return out


def tiles(path, um):
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    s = um / UM
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = im.shape[:2]
    out = []
    for y in range(0, H - TILE + 1, TILE):
        for x in range(0, W - TILE + 1, TILE):
            t = im[y:y + TILE, x:x + TILE]
            hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
            if ((hsv[..., 1] > 25) & (hsv[..., 2] < 230)).mean() > 0.5:      # tissue, not glass / eyepiece edge
                out.append(t)
    return out


@torch.no_grad()
def main():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon-v2", token=False).cuda().eval().half()
    ids, srcs, mu, sd, n = [], [], [], [], []
    ims = images()
    for k, (src, iid, path, um) in enumerate(ims):
        ts = tiles(path, um)
        if not ts:
            continue
        x = torch.from_numpy(((np.stack(ts).astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)).cuda().half()
        e = torch.cat([m(pixel_values=x[i:i + 128]).last_hidden_state[:, 0].float() for i in range(0, len(x), 128)]).cpu().numpy()
        ids.append(iid), srcs.append(src), mu.append(e.mean(0)), sd.append(e.std(0)), n.append(len(e))
        if k % 200 == 0:
            print(f"{k}/{len(ims)} images", flush=True)
    np.savez(ROOT / "training/data/pleo_embeddings.npz", ids=np.array(ids), source=np.array(srcs), mean=np.array(mu, np.float32),
             std=np.array(sd, np.float32), n_tiles=np.array(n))
    print(f"done: {len(ids)} images, median {int(np.median(n))} tiles each")


if __name__ == "__main__":
    main()
