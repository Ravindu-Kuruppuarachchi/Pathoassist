"""Can Databiox be used better? Patient-grouped repeated CV (same splits as train_final.py) over:
pooling (image-mean / tile-mean / mean+std), tile-level training, and adding 20x tiles."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from core.ihc.scoring import weighted_kappa

T = np.load(ROOT / "training/data/databiox/tile_embeddings.npz")
emb, pat, grade, mag, img = T["emb"].astype(np.float32), T["patient"], T["grade"], T["mag"], T["image"]
patients = sorted(set(pat[mag == "40x"]))
y = np.array([grade[pat == p][0] for p in patients]) - 1
rng = np.random.default_rng(0)

def pooled(p, mags, how):
    m = (pat == p) & np.isin(mag, mags)
    if how == "image_mean":
        e = np.array([emb[m & (img == i)].mean(0) for i in sorted(set(img[m]))]).mean(0)
    else:
        e = emb[m].mean(0)
    if how == "mean_std":
        e = np.concatenate([e, emb[m].std(0)])
    return e

def patient_level(mags, how, n_pca=16, C=0.05):
    X = np.array([pooled(p, mags, how) for p in patients])
    def fit_predict(tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        pca = PCA(n_pca, random_state=0).fit((X[tr] - mu) / sd)
        w = 1.0 / np.bincount(y[tr], minlength=3)[y[tr]]
        lr = LogisticRegression(C=C, max_iter=5000).fit(pca.transform((X[tr] - mu) / sd), y[tr], sample_weight=w / w.mean())
        return lr.predict_proba(pca.transform((X[te] - mu) / sd))
    return fit_predict

def tile_level(mags, n_pca=64, C=0.01, per_patient=300):
    idx_by_p = {p: np.nonzero((pat == p) & np.isin(mag, mags))[0] for p in patients}
    def fit_predict(tr, te):
        sel = np.concatenate([rng.permutation(idx_by_p[patients[i]])[:per_patient] for i in tr])
        Xt = emb[sel]; yt = np.array([grade[s] - 1 for s in sel])
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        pca = PCA(n_pca, random_state=0).fit((Xt - mu) / sd)
        w = 1.0 / np.bincount(yt, minlength=3)[yt]
        lr = LogisticRegression(C=C, max_iter=5000).fit(pca.transform((Xt - mu) / sd), yt, sample_weight=w / w.mean())
        out = []
        for i in te:
            pr = lr.predict_proba(pca.transform((emb[idx_by_p[patients[i]]] - mu) / sd))
            lp = np.log(np.clip(pr, 1e-6, 1)).mean(0)          # average log-probability over the patient's tiles
            out.append(np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum())
        return np.array(out)
    return fit_predict

def run(name, fp):
    ks, rs = [], []
    for r in range(5):
        P = np.zeros((len(y), 3))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y, patients):
            P[te] = fp(tr, te)
        ks.append(weighted_kappa(y, P.argmax(1), n=3)); rs.append(spearmanr(P @ [1, 2, 3], y)[0])
    print(f"{name:52s} kappa {np.mean(ks):.3f} ± {np.std(ks):.3f}  rho {np.mean(rs):.2f}", flush=True)
    return float(np.mean(ks))

print(f"{len(patients)} patients, grade counts {np.bincount(y).tolist()}")
res = {}
res["A image-mean 40x (current model)"] = run("A  patient = mean of image means, 40x (current)", patient_level(["40x"], "image_mean"))
res["B tile-mean 40x"] = run("B  patient = mean of all tiles, 40x", patient_level(["40x"], "tile_mean"))
res["C mean+std 40x"] = run("C  patient = mean + std of tiles, 40x", patient_level(["40x"], "mean_std"))
res["D tile-level 40x"] = run("D  tile-level training, 40x", tile_level(["40x"]))
res["E tile-level 40x+20x"] = run("E  tile-level training, 40x + 20x", tile_level(["40x", "20x"]))
res["F tile-mean 40x+20x"] = run("F  patient = mean of all tiles, 40x + 20x", patient_level(["40x", "20x"], "tile_mean"))
res["G tile-level 40x+20x, PCA128"] = run("G  tile-level, 40x + 20x, PCA 128", tile_level(["40x", "20x"], n_pca=128, C=0.01))
(ROOT / "eval/results/pleomorphism_tile_experiments.json").write_text(json.dumps(res, indent=2))
