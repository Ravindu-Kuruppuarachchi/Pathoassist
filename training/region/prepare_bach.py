"""Build 10x tissue-region patches from BACH (ICIAR 2018, Aresta et al.) without keeping the
12.7 GB archive: members are read with HTTP range requests; each WSI is streamed to a temp
file, sampled, then deleted.

Patches: 224x224 px at ~0.93 µm/px (480 px of the 0.467 µm/px level 0 -> 224 px), i.e. a
224 µm field - what Feature 5 sees on a 10x camera frame.

Classes (index):  0 invasive, 1 in_situ, 2 benign, 3 normal_other (non-lesional tissue:
stroma, fat, normal glands), 4 background (glass)
  - annotated WSIs A01-A10: regions from the XML (Benign / Carcinoma in situ / Carcinoma
    invasive); tissue outside every region = normal_other; no tissue = background.
    A01-A07 train, A08 val, A09-A10 test (slide-level split).
  - photos (50 per class of the 400 microscopy images, image-level label): resized to 1 µm/px, kept whole,
    used for image-level evaluation and as extra weak labels. 80/20 split.
  - unannotated WSIs 01-10: tissue patches without labels (pool for distillation).

    ~/venvs/pathoassist-torch/bin/python training/region/prepare_bach.py
"""
import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_zip import RemoteZip  # noqa: E402

URL = "https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge.zip/content"
ROOT = "ICIAR2018_BACH_Challenge/"
OUT = Path(__file__).resolve().parents[1] / "data" / "bach"
CLASSES = ["invasive", "in_situ", "benign", "normal_other", "background"]
REGION_CLASS = {"Carcinoma invasive": 0, "Carcinoma in situ": 1, "Benign": 2}
SRC = 480            # level-0 px per patch (0.467 µm/px -> 224 µm)
PATCH = 224
DS = 32              # label-mask downsample from level 0
SPLIT = {**{f"A{k:02d}": "train" for k in range(1, 8)}, "A08": "val", "A09": "test", "A10": "test"}


def region_class(label):
    """BACH XMLs name classes inconsistently ('Carcinoma invasive' / 'Invasive carcinoma',
    'Carcinoma in situ' / 'In situ carcinoma', 'Benign') - match on keywords."""
    s = label.lower()
    if "invasive" in s:
        return 0
    if "situ" in s:
        return 1
    if "benign" in s:
        return 2
    return None


def parse_regions(xml):
    """Region label is in <Attribute Value=...> for some slides and in Region Text=... for others."""
    regs = []
    for reg in re.findall(r"<Region .*?</Region>", xml, flags=re.S):
        v = re.search(r'<Attribute [^>]*Value="([^"]+)"', reg) or re.search(r'<Region [^>]*Text="([^"]+)"', reg)
        c = region_class(v.group(1)) if v else None
        pts = np.array([(float(x), float(y)) for x, y in re.findall(r'<Vertex X="([\d.]+)" Y="([\d.]+)"', reg)])
        if c is not None and len(pts) >= 3:
            regs.append((c, pts))
    return regs


def open_level0(path):
    tif = tifffile.TiffFile(path)
    store = tif.series[0].aszarr(level=0)
    return tif, zarr.open(store, mode="r")


def tissue_mask(thumb_rgb):
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] > 20) & (hsv[..., 2] < 235)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def low_res(arr, ds):
    """Thumbnail of the level-0 array by reading a strided grid of small tiles (fast enough
    at ds=32 for ~50k px slides)."""
    H, W = arr.shape[:2]
    h, w = H // ds, W // ds
    out = np.zeros((h, w, 3), np.uint8)
    step = 64                                  # read 64x64 thumbnail-px blocks = 2048 level-0 px
    for y in range(0, h, step):
        for x in range(0, w, step):
            blk = np.asarray(arr[y * ds:min(H, (y + step) * ds), x * ds:min(W, (x + step) * ds), :3])
            out[y:y + blk.shape[0] // ds, x:x + blk.shape[1] // ds] = cv2.resize(
                blk, (blk.shape[1] // ds, blk.shape[0] // ds), interpolation=cv2.INTER_AREA)
    return out


def save_patch(arr, cx, cy, path):
    x0, y0 = int(cx - SRC // 2), int(cy - SRC // 2)
    p = np.asarray(arr[y0:y0 + SRC, x0:x0 + SRC, :3])
    if p.shape[:2] != (SRC, SRC):
        return False
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(p, (PATCH, PATCH), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def sample_centres(mask, cls, n, rng, purity=0.9):
    """Patch centres (level-0 px) whose patch area in the low-res mask is >= purity class `cls`."""
    r = SRC // DS // 2
    m = (mask == cls).astype(np.float32)
    frac = cv2.blur(m, (2 * r + 1, 2 * r + 1))
    ys, xs = np.nonzero((frac >= purity) & (m > 0))
    ok = (ys > r) & (xs > r) & (ys < mask.shape[0] - r) & (xs < mask.shape[1] - r)
    ys, xs = ys[ok], xs[ok]
    if not len(ys):
        return []
    idx = rng.choice(len(ys), min(n, len(ys)), replace=False)
    return [((xs[i] + rng.uniform()) * DS, (ys[i] + rng.uniform()) * DS) for i in idx]


def do_wsi(z, name, annotated, rng, n_lesion=700, n_other=900, n_bg=80, n_unlab=1200):
    tmp = OUT / f"tmp_{name}.svs"
    z.extract_to(ROOT + f"WSI/{name}.svs", tmp)
    tif, arr = open_level0(tmp)
    H, W = arr.shape[:2]
    lv = tif.series[0].levels
    small = min((l for l in lv if l.shape[0] >= H // DS), key=lambda l: l.shape[0])   # smallest level >= 1/32
    thumb = cv2.resize(small.asarray()[..., :3], (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    tissue = tissue_mask(thumb)
    recs = []
    d = OUT / "patches"
    d.mkdir(parents=True, exist_ok=True)
    if annotated:
        mask = np.where(tissue > 0, 3, 4).astype(np.uint8)
        regs = parse_regions(z.read(ROOT + f"WSI/{name}.xml").decode("latin-1"))
        for cls in (2, 1, 0):                       # paint benign, then in situ, then invasive on top
            for c, pts in regs:
                if c == cls:
                    cv2.fillPoly(mask, [np.round(pts / DS).astype(np.int32)], int(cls))
        cv2.imwrite(str(OUT / f"mask_{name}.png"), mask)
        cv2.imwrite(str(OUT / f"thumb_{name}.jpg"), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        plan = [(0, n_lesion), (1, n_lesion), (2, n_lesion), (3, n_other), (4, n_bg)]
        for cls, n in plan:
            for j, (cx, cy) in enumerate(sample_centres(mask, cls, n, rng)):
                fn = f"{name}_{cls}_{j:04d}.jpg"
                if save_patch(arr, cx, cy, d / fn):
                    recs.append(dict(file=fn, slide=name, label=cls, split=SPLIT[name], x=round(cx), y=round(cy)))
    else:
        for j, (cx, cy) in enumerate(sample_centres(tissue.astype(np.uint8), 1, n_unlab, rng, purity=0.5)):
            fn = f"{name}_u_{j:04d}.jpg"
            if save_patch(arr, cx, cy, d / fn):
                recs.append(dict(file=fn, slide=name, label=-1, split="unlabeled", x=round(cx), y=round(cy)))
    tif.close()
    tmp.unlink()
    print(f"{name}: {len(recs)} patches, label counts {np.bincount([r['label'] + 1 for r in recs], minlength=6)[1:].tolist() if annotated else len(recs)}", flush=True)
    return recs


def do_photos(z, rng, per_class=50, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    d = OUT / "photos"
    d.mkdir(parents=True, exist_ok=True)
    lab = {"Invasive": 0, "InSitu": 1, "Benign": 2, "Normal": 3}
    names = sorted(n for n in z.namelist() if "/Photos/" in n and n.endswith(".tif"))
    names = [n for c in lab for n in sorted(x for x in names if x.split("/")[-2] == c)[:per_class]]

    def one(n):
        cls = lab[n.split("/")[-2]]
        fn = Path(n).stem + ".jpg"
        if not (d / fn).exists():
            im = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
            s = 0.42 / 1.0                              # photos are 0.42 µm/px -> 1 µm/px
            cv2.imwrite(str(d / fn), cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                                  cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return dict(file=fn, label=cls)

    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(one, names))
    by = {}
    for r in recs:
        by.setdefault(r["label"], []).append(r)
    for rs in by.values():
        rng.shuffle(rs)
        for i, r in enumerate(rs):
            r["split"] = "test" if i < len(rs) // 5 else "train"
    print(f"photos: {len(recs)}", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled", type=int, default=3, help="how many unannotated WSIs to sample")
    ap.add_argument("--photos-per-class", type=int, default=50)
    ap.add_argument("--only", default=None,
                    help="comma-separated items for this worker (WSI names and/or 'photos'); parallel workers")
    ap.add_argument("--merge", action="store_true", help="merge the workers' index_*.json into index.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        index = {"patches": [], "photos": [], "done": []}
        for f in sorted(OUT.glob("index_*.json")):
            part = json.loads(f.read_text())
            for k in index:
                index[k] += part[k]
        (OUT / "index.json").write_text(json.dumps(index))
        print({k: len(v) for k, v in index.items()})
        return
    z = RemoteZip(URL)
    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    sel = args.only.split(",") if args.only else None
    idx_path = OUT / (f"index_{'_'.join(sel)}.json" if sel else "index.json")
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"patches": [], "photos": [], "done": []}
    todo = [(f"A{k:02d}", True) for k in range(1, 11)] + [(f"{k:02d}", False) for k in range(1, args.unlabeled + 1)]
    if sel:
        todo = [t for t in todo if t[0] in sel]
    for name, ann in todo:
        if name in index["done"]:
            continue
        index["patches"] += do_wsi(z, name, ann, nrng)
        index["done"].append(name)
        idx_path.write_text(json.dumps(index))
    if (sel is None or "photos" in sel) and not index["photos"]:
        index["photos"] = do_photos(z, rng, per_class=args.photos_per_class)
        idx_path.write_text(json.dumps(index))


if __name__ == "__main__":
    main()
