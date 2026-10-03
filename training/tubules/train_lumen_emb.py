"""Phikon-v2 opinion for uncertain lumen candidates (Feature 8).

Hybrid used at runtime: the hand-crafted LightGBM decides confident candidates; candidates with
p in BAND additionally get a Phikon-v2 embedding of the 112 µm around them (224 px at 0.5 µm/px)
-> PCA32 -> logistic, and the two probabilities are averaged. Case-grouped CV on BreCaHAD reports
AUC for hand / both / hybrid; the embedding model is fitted on all cases and saved to
models/tubule_lumen_phikon.npz.

    CUDA_VISIBLE_DEVICES=0 ~/venvs/pathoassist-torch/bin/python training/tubules/train_lumen_emb.py
"""
import json, sys, numpy as np, cv2, torch; sys.path.insert(0, '.')
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel
from core.tubules.lumen import candidates
import lightgbm as lgb
D = Path('training/data/brecahad'); S = 0.318
m = AutoModel.from_pretrained('owkin/phikon-v2', token=False).cuda().eval().half()
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
X, Z, y, g = [], [], [], []
for jf in sorted((D / 'groundTruth').glob('*.json')):
    ann = json.loads(jf.read_text()); im = cv2.cvtColor(cv2.imread(str(D / 'jpg' / (jf.stem + '.jpg'))), cv2.COLOR_BGR2RGB)
    c = candidates(im, S); h, w = c.labels.shape
    dist, (iy, ix) = ndi.distance_transform_edt(c.labels == 0, return_indices=True)
    im05 = cv2.resize(im, None, fx=S / 0.5, fy=S / 0.5, interpolation=cv2.INTER_AREA)
    pad = cv2.copyMakeBorder(im05, 112, 112, 112, 112, cv2.BORDER_REFLECT)
    crops, rows = [], []
    for cls, lab in (('lumen', 1), ('non_lumen', 0)):
        for p in ann.get(cls, []):
            x, yy = min(int(p['x'] * w), w - 1), min(int(p['y'] * h), h - 1)
            k = c.labels[yy, x] if c.labels[yy, x] else (c.labels[iy[yy, x], ix[yy, x]] if dist[yy, x] <= 3 else 0)
            if not k: continue
            cx, cy = c.centroids[k - 1] / c.scale * (S / 0.5)
            crops.append(pad[int(cy):int(cy) + 224, int(cx):int(cx) + 224]); rows.append((c.feats[k - 1], lab))
    if not crops: continue
    t = torch.from_numpy(((np.stack(crops).astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)).cuda().half()
    with torch.no_grad(): e = m(pixel_values=t).last_hidden_state[:, 0].float().cpu().numpy()
    for (f, lab), ee in zip(rows, e): X.append(f); Z.append(ee); y.append(lab); g.append(jf.stem.split('-')[0])
X, Z, y, g = np.array(X), np.array(Z), np.array(y), np.array(g)
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, is_unbalance=True)
BAND = (0.15, 0.85)
res = {"hand": [], "both": [], "hybrid": []}; frac = []
for r in range(5):
    Ph, Pe = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
        Ph[te] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 200).predict(X[te])
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
        pca = PCA(32, random_state=0).fit((Z[tr] - mu) / sd)
        lr = LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced').fit(pca.transform((Z[tr] - mu) / sd), y[tr])
        Pe[te] = lr.predict_proba(pca.transform((Z[te] - mu) / sd))[:, 1]
    unc = (Ph >= BAND[0]) & (Ph <= BAND[1])
    Phy = np.where(unc, (Ph + Pe) / 2, Ph)
    frac.append(unc.mean())
    for k, P in zip(res, (Ph, (Ph + Pe) / 2, Phy)): res[k].append(roc_auc_score(y, P))
out = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}
for k, v in out.items(): print(f'{k:8s} case-grouped AUC {v[0]:.3f} ± {v[1]:.3f}')
print(f'hybrid embeds {np.mean(frac):.0%} of candidates')
mu, sd = Z.mean(0), Z.std(0) + 1e-6
pca = PCA(32, random_state=0).fit((Z - mu) / sd)
lr = LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced').fit(pca.transform((Z - mu) / sd), y)
np.savez('models/tubule_lumen_phikon.npz', emb_mean=mu, emb_std=sd, pca_mean=pca.mean_, pca_components=pca.components_,
         coef=lr.coef_, intercept=lr.intercept_)
meta = json.loads(Path('models/tubule_lumen_lgbm.json').read_text())
meta.update(hybrid=dict(band=list(BAND), embedder='models/phikon_v2_int8w.xml', crop_um=112, cv_auc=out, embedded_fraction=float(np.mean(frac))))
Path('models/tubule_lumen_lgbm.json').write_text(json.dumps(meta, indent=1))
