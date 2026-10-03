"""Feature 7 tests: per-nucleus measurements and field features on synthetic nuclei."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from core.pleomorphism.features import field_features, nucleus_table, pool_tables


def _field(radii, um=0.25, texture=0.0, seed=0, star=()):
    """Nuclei as discs (or stars) on a grid; haematoxylin OD with optional chromatin texture."""
    rng = np.random.default_rng(seed)
    n = len(radii)
    side = int(np.ceil(np.sqrt(n)))
    H = W = side * 80
    labels = np.zeros((H, W), np.int32)
    for i, r in enumerate(radii):
        cy, cx = 40 + (i // side) * 80, 40 + (i % side) * 80
        if i in star:
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            rr = np.where(np.arange(16) % 2 == 0, r, r * 0.5)
            pts = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)], 1).astype(np.int32)
            cv2.fillPoly(labels, [pts], i + 1)
        else:
            cv2.circle(labels, (cx, cy), int(r), i + 1, -1)
    h = np.where(labels > 0, 0.5, 0.05) + texture * rng.normal(0, 1, (H, W)) * (labels > 0)
    return labels, np.clip(h, 0, None).astype(np.float32)


def test_shape_measurements():
    labels, h = _field([20, 20, 20, 20], um=0.25, star=(3,))
    t = nucleus_table(labels, h, 0.25)
    area_disc = np.pi * (20 * 0.25) ** 2                       # 19.6 µm²
    assert np.allclose(t["area"][:3], area_disc, rtol=0.05)
    assert t["solidity"][3] < 0.85 < t["solidity"][0]
    assert t["roundness"][0] > 0.85 and t["roundness"][3] < t["roundness"][0]
    assert np.allclose(t["eccentricity"][:3], 0, atol=0.2)


def test_glcm_texture_and_field_features():
    rng = np.random.default_rng(1)
    smooth, hs = _field([18] * 16, texture=0.0)
    rough, hr = _field([18] * 16, texture=0.25, seed=2)
    ts, tr = nucleus_table(smooth, hs, 0.25), nucleus_table(rough, hr, 0.25)
    assert np.nanmean(tr["glcm_contrast"]) > np.nanmean(ts["glcm_contrast"]) + 1
    assert np.nanmean(tr["h_std"]) > np.nanmean(ts["h_std"])

    uni, hu = _field([18] * 30)
    var, hv = _field(list(rng.uniform(10, 30, 30)), seed=3)
    tu, tv = nucleus_table(uni, hu, 0.25), nucleus_table(var, hv, 0.25)
    tum = np.ones(30, bool)
    lym = np.zeros(30, bool)
    fu, fv = field_features(tu, tum, lym), field_features(tv, tum, lym)
    assert fv["area_cv"] > 3 * fu["area_cv"] and fv["area_iqr_rel"] > fu["area_iqr_rel"]
    assert np.isnan(fu["area_rel_lymph"])                      # no lymphocytes in the field
    lym2 = np.zeros(30, bool)
    lym2[:6] = True
    f2 = field_features(tv, ~lym2, lym2)
    assert f2["area_rel_lymph"] > 0 and f2["n_tumour"] == 24
    pooled = pool_tables([tu, tv])
    assert len(pooled["area"]) == 60 and "label" not in pooled


def test_too_few_tumour_nuclei():
    labels, h = _field([18] * 4)
    t = nucleus_table(labels, h, 0.25)
    assert field_features(t, np.ones(4, bool), np.zeros(4, bool)) == {"n_tumour": 4}


EMB_MODEL = ROOT / "models/pleomorphism_phikon.npz"
EMBEDDER = ROOT / "models/phikon_v2_int8w.xml"


@pytest.mark.skipif(not (EMB_MODEL.exists() and EMBEDDER.exists() and (ROOT / "training/data/pleo_embeddings.npz").exists()),
                    reason="embedding model / training embeddings missing")
def test_runtime_embedding_matches_training_and_predicts():
    import json
    from core.pleomorphism.embedding import EmbeddingPleoModel, TileEmbedder
    E = np.load(ROOT / "training/data/pleo_embeddings.npz")
    rec = next(r for r in json.loads((ROOT / "training/data/databiox/index.json").read_text())
               if r["mag"] == "40x" and r.get("um_per_px"))
    stem = Path(rec["file"]).stem
    ref = E["mean"][list(E["ids"]).index(stem)]
    img = cv2.cvtColor(cv2.imread(str(ROOT / "training/data/databiox" / rec["file"])), cv2.COLOR_BGR2RGB)
    emb = TileEmbedder(EMBEDDER).embed_field(img, rec["um_per_px"])
    assert float(emb @ ref / np.linalg.norm(emb) / np.linalg.norm(ref)) > 0.99      # INT8 runtime == FP16 training
    p = EmbeddingPleoModel(EMB_MODEL).predict(emb)
    assert p.shape == (3,) and abs(p.sum() - 1) < 1e-6
    assert TileEmbedder(EMBEDDER).embed_field(np.full((800, 1000, 3), 245, np.uint8), 0.25) is None   # glass only
