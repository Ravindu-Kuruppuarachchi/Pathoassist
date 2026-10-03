"""Per-tile Phikon-v2 embeddings for Databiox 40x and 20x images (for tile-level training).

Tiles: 224 px at 0.5 µm/px (same as core.pleomorphism.embedding.field_tiles). Saves
training/data/databiox/tile_embeddings.npz: embedding (float16), patient, grade, mag, image.

    CUDA_VISIBLE_DEVICES=0 ~/venvs/pathoassist-torch/bin/python training/pleomorphism/embed_databiox_tiles.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from core.pleomorphism.embedding import MEAN, STD, field_tiles

D = ROOT / "training/data/databiox"
SCALES = json.loads((D / "scales.json").read_text())


@torch.no_grad()
def main():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon-v2", token=False).cuda().eval().half()
    recs = [r for r in json.loads((D / "index.json").read_text()) if r["mag"] in ("40x", "20x")]
    E, P, G, M, I = [], [], [], [], []
    for k, r in enumerate(recs):
        um = SCALES.get(f"{r['mag']}_{'large' if r['width'] > 2000 else 'small'}")
        im = cv2.cvtColor(cv2.imread(str(D / r["file"])), cv2.COLOR_BGR2RGB)
        ts = field_tiles(im, um)
        if not ts:
            continue
        x = torch.from_numpy(((np.stack(ts).astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)).cuda().half()
        e = torch.cat([m(pixel_values=x[i:i + 128]).last_hidden_state[:, 0].float() for i in range(0, len(x), 128)]).cpu().numpy()
        E.append(e.astype(np.float16))
        n = len(e)
        P += [r["patient"]] * n
        G += [r["grade"]] * n
        M += [r["mag"]] * n
        I += [Path(r["file"]).stem] * n
        if k % 100 == 0:
            print(f"{k}/{len(recs)}", flush=True)
    np.savez(D / "tile_embeddings.npz", emb=np.concatenate(E), patient=np.array(P), grade=np.array(G), mag=np.array(M), image=np.array(I))
    print(f"done: {sum(len(e) for e in E)} tiles from {len(E)} images", flush=True)


if __name__ == "__main__":
    main()
