"""PanNuke loading / conversion.

PanNuke: 7,901 H&E patches, 256x256 at 40x (~0.25 µm/px), 19 tissues (breast is the
largest). Raw masks are float64 (N,256,256,6): channels 0-4 are per-class instance maps
(neoplastic, inflammatory, connective, dead, epithelial), channel 5 is background.

We merge channels 0-4 into one uint16 instance map per patch and store:
    training/data/pannuke/fold{k}_images.npy   (N,256,256,3) uint8
    training/data/pannuke/fold{k}_labels.npy   (N,256,256)   uint16
    training/data/pannuke/fold{k}_types.npy    (N,)          tissue names

Splits (the standard PanNuke protocol, fold 3 is held-out test):
    train = fold1 + fold2 minus val,  val = 10% of fold2,  test = fold3
"""
import sys
import zipfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "pannuke"


def normalize(img, pmin=1.0, pmax=99.8):
    """Same per-image percentile normalisation the engine uses on a full frame."""
    lo, hi = np.percentile(img, (pmin, pmax))
    return ((img.astype(np.float32) - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def _stream_npy(z, name, chunk=64):
    """Yield chunks of an .npy inside a zip without extracting it (masks.npy is ~8 GB float64)."""
    member = next(n for n in z.namelist() if n.endswith(name))
    with z.open(member) as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        assert not fortran
        per_item = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - start)
            buf = f.read(n * per_item)
            yield np.frombuffer(buf, dtype).reshape((n,) + tuple(shape[1:]))


def _merge_instances(m):
    """(256,256,5) per-class instance maps -> one sequential instance map."""
    lab = np.zeros(m.shape[:2], np.int32)
    nxt = 1
    for c in range(m.shape[-1]):
        ch = m[..., c].astype(np.int64)
        ids = np.unique(ch)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        lut = np.zeros(ch.max() + 1, np.int32)
        lut[ids] = np.arange(nxt, nxt + len(ids))
        new = lut[ch]
        lab = np.where((lab == 0) & (new > 0), new, lab)
        nxt += len(ids)
    from stardist import fill_label_holes
    return fill_label_holes(lab)


def convert_fold(k, delete_raw=True):
    zpath = DATA / "raw" / f"fold_{k}.zip"
    with zipfile.ZipFile(zpath) as z:
        X = np.concatenate([c.astype(np.uint8) for c in _stream_npy(z, "images.npy")])
        Y = np.concatenate([np.stack([_merge_instances(m[..., :5]) for m in c]).astype(np.uint16)
                            for c in _stream_npy(z, "masks.npy")])
        with z.open(next(n for n in z.namelist() if n.endswith("types.npy"))) as f:
            types = np.load(f)
    n = len(X)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"fold{k}_images.npy", X)
    np.save(OUT / f"fold{k}_labels.npy", Y)
    np.save(OUT / f"fold{k}_types.npy", types)
    print(f"fold {k}: {n} patches, {sum(len(np.unique(y)) - 1 for y in Y)} nuclei")
    if delete_raw:
        zpath.unlink()


PANNUKE_CLASSES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]


def _class_map(m):
    """(256,256,5) per-class instance maps -> uint8 class per pixel (index into PANNUKE_CLASSES),
    first channel wins on overlap, matching _merge_instances."""
    cls = np.zeros(m.shape[:2], np.uint8)
    for c in range(m.shape[-1]):
        cls[(cls == 0) & (m[..., c] > 0)] = c + 1
    return cls


def convert_classes(k):
    """Re-download fold k and store only its class map (fold{k}_classes.npy, uint8)."""
    import subprocess
    zpath = DATA / "raw" / f"fold_{k}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{k}.zip"
    for _ in range(20):  # the Warwick server drops long transfers; resume until the zip is valid
        subprocess.run(["curl", "-sL", "-C", "-", "-o", str(zpath), url])
        if zipfile.is_zipfile(zpath):
            break
    with zipfile.ZipFile(zpath) as z:
        C = np.concatenate([np.stack([_class_map(m[..., :5]) for m in c]) for c in _stream_npy(z, "masks.npy")])
    np.save(OUT / f"fold{k}_classes.npy", C)
    print(f"fold {k}: class map {C.shape}, pixel counts {np.bincount(C.ravel(), minlength=6).tolist()}")
    zpath.unlink()


def _load_fold(k):
    return (np.load(OUT / f"fold{k}_images.npy"), np.load(OUT / f"fold{k}_labels.npy"),
            np.load(OUT / f"fold{k}_types.npy"))


def load_split(split, seed=0):
    """Returns X (N,256,256,3) uint8, Y (N,256,256) uint16, tissue types (N,)."""
    if split == "test":
        return _load_fold(3)
    X2, Y2, T2 = _load_fold(2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X2))
    n_val = len(X2) // 10
    val, rest = perm[:n_val], perm[n_val:]
    if split == "val":
        return X2[val], Y2[val], T2[val]
    if split == "train":
        X1, Y1, T1 = _load_fold(1)
        return (np.concatenate([X1, X2[rest]]), np.concatenate([Y1, Y2[rest]]),
                np.concatenate([T1, T2[rest]]))
    raise ValueError(split)


if __name__ == "__main__":
    for k in sys.argv[1:] or [1, 2, 3]:
        convert_fold(int(k))
