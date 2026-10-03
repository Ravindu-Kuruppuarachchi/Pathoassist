"""Feature 7 model comparison on MITOS-ATYPIA-14 + Databiox: hand-crafted nuclear features vs
Phikon-v2 tile embeddings vs both. All evaluation is grouped (Databiox patient / MITOS slide never
split across folds); PCA and scaling are fitted inside each fold.

Units: Databiox = patient (label: overall grade, proxy), MITOS = x20 frame (label: atypia score).

    python training/pleomorphism/compare_models.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from core.ihc.scoring import weighted_kappa

E = np.load(ROOT / "training/data/pleo_embeddings.npz")
EMB = {(s, i): (m, sd) for s, i, m, sd in zip(E["source"], E["ids"], E["mean"], E["std"])}


def build_units():
    import databiox as dbm
    from train_pleomorphism import D as MD, load_units
    units = []
    # Databiox patients
    recs = json.loads((ROOT / "training/data/databiox/index.json").read_text())
    hand = {u["group"]: u for u in dbm.patient_units(recs)}
    by = {}
    for r in recs:
        if r["mag"] == "40x" and ("databiox", Path(r["file"]).stem) in EMB:
            by.setdefault(r["patient"], []).append(EMB[("databiox", Path(r["file"]).stem)][0])
    for pid, es in by.items():
        g = f"dbx_{pid}"
        if g in hand:
            units.append(dict(source="databiox", group=g, y=hand[g]["y"], emb=np.mean(es, 0), feats=hand[g]["feats"]))
    # MITOS x20 frames
    idx = json.loads((MD / "index.json").read_text())
    for u in load_units(idx):
        if u["feats"].get("n_tumour", 0) < 5:
            continue
        es = [EMB[("mitos", f"{u['x20']}{c}")][0] for c in "abcd" if ("mitos", f"{u['x20']}{c}") in EMB]
        if es:
            units.append(dict(source="mitos", group=f"mitos_{u['slide']}", y=u["y"], emb=np.mean(es, 0), feats=u["feats"]))
    return units


class Model:
    """kind: 'hand' (LightGBM on hand-crafted), 'emb' (PCA+logistic on embeddings), 'both' (logistic on PCA(emb) + hand)."""

    def __init__(self, kind, keys, n_pca=32, C=0.05):
        self.kind, self.keys, self.n_pca, self.C = kind, keys, n_pca, C

    def _hand(self, units):
        return np.array([[u["feats"].get(k, np.nan) for k in self.keys] for u in units], float)

    def fit(self, units, w):
        y = np.array([u["y"] for u in units]) - 1
        if self.kind == "hand":
            import lightgbm as lgb
            p = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7, min_data_in_leaf=10,
                     feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1)
            self.m = lgb.train(p, lgb.Dataset(self._hand(units), y, weight=w * len(y)), 150)
            return self
        Z = np.array([u["emb"] for u in units])
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        self.pca = PCA(min(self.n_pca, len(units) - 1), random_state=0).fit((Z - self.mu) / self.sd)
        X = self.pca.transform((Z - self.mu) / self.sd)
        if self.kind == "both":
            H = self._hand(units)
            self.med = np.nanmedian(H, 0)
            H = np.where(np.isnan(H), self.med, H)
            self.hmu, self.hsd = H.mean(0), H.std(0) + 1e-9
            X = np.hstack([X / X.std(0), (H - self.hmu) / self.hsd])
            self.xsd = None
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit(X, y, sample_weight=w * len(y))
        self._xs = X.std(0) if self.kind == "both" else None
        return self

    def predict(self, units):
        if self.kind == "hand":
            return self.m.predict(self._hand(units))
        Z = np.array([u["emb"] for u in units])
        X = self.pca.transform((Z - self.mu) / self.sd)
        if self.kind == "both":
            H = self._hand(units)
            H = np.where(np.isnan(H), self.med, H)
            X = np.hstack([X / self._xs[:X.shape[1]], (H - self.hmu) / self.hsd])
        return self.lr.predict_proba(X)


def weights(units):
    y = np.array([u["y"] for u in units]) - 1
    s = np.array([u["source"] for u in units])
    w = np.zeros(len(y))
    for src in set(s):
        m = s == src
        cnt = np.bincount(y[m], minlength=3).clip(1)
        w[m] = 1.0 / cnt[y[m]]
        w[m] /= w[m].sum()
    return w / w.sum()


def score(y, P):
    pred = P.argmax(1)
    return dict(kappa=float(weighted_kappa(y, pred, n=3)), spearman=float(spearmanr(P @ [1, 2, 3], y)[0]),
                accuracy=float((pred == y).mean()))


def cv(units, make, splitter, repeats=1):
    y = np.array([u["y"] for u in units]) - 1
    g = np.array([u["group"] for u in units])
    out = []
    for r in range(repeats):
        P = np.zeros((len(y), 3))
        sp = splitter(r)
        for tr, te in sp.split(np.zeros(len(y)), y, g):
            tu = [units[i] for i in tr]
            P[te] = make().fit(tu, weights(tu)).predict([units[i] for i in te])
        out.append((score(y, P), P))
    return out


def main():
    units = build_units()
    dbx = [u for u in units if u["source"] == "databiox"]
    mit = [u for u in units if u["source"] == "mitos"]
    hand_keys = sorted({k for u in units for k in u["feats"]})
    small_keys = ["major_p90", "area_median", "area_cv", "area_iqr_rel", "area_rel_lymph", "h_std_mean", "glcm_contrast_mean",
                  "dark_frac_mean", "solidity_std"]
    print(f"units: Databiox {len(dbx)} patients, MITOS {len(mit)} x20 frames ({len({u['group'] for u in mit})} slides)", flush=True)
    kinds = {"hand-crafted (LightGBM)": lambda: Model("hand", hand_keys),
             "Phikon embeddings (PCA32 + logistic)": lambda: Model("emb", hand_keys),
             "embeddings + key measurements": lambda: Model("both", small_keys)}
    res = {}
    for name, mk in kinds.items():
        d = cv(dbx, mk, lambda r: StratifiedGroupKFold(5, shuffle=True, random_state=r), repeats=5)
        dk = [x[0]["kappa"] for x in d]
        drho = [x[0]["spearman"] for x in d]
        m = cv(mit, mk, lambda r: GroupKFold(len({u['group'] for u in mit})))[0][0]
        # cross-dataset transfer (continuous ranking only - label types differ)
        a = mk().fit(mit, weights(mit)).predict(dbx)
        b = mk().fit(dbx, weights(dbx)).predict(mit)
        tr = dict(mitos_to_databiox_rho=float(spearmanr(a @ [1, 2, 3], [u["y"] for u in dbx])[0]),
                  databiox_to_mitos_rho=float(spearmanr(b @ [1, 2, 3], [u["y"] for u in mit])[0]))
        res[name] = dict(databiox_kappa=float(np.mean(dk)), databiox_kappa_sd=float(np.std(dk)), databiox_rho=float(np.mean(drho)),
                         mitos_loso=m, transfer=tr)
        print(f"{name:40s} Databiox kappa {np.mean(dk):.3f} ± {np.std(dk):.3f} (rho {np.mean(drho):.2f}) | "
              f"MITOS LOSO kappa {m['kappa']:+.3f} (rho {m['spearman']:+.2f}) | transfer rho M->D {tr['mitos_to_databiox_rho']:+.2f}, "
              f"D->M {tr['databiox_to_mitos_rho']:+.2f}", flush=True)
    (ROOT / "eval/results/pleomorphism_compare.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
