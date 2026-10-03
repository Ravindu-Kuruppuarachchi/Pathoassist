"""Feature 7 on Databiox (Bolhasani et al. 2020): evaluate, then retrain with MITOS-ATYPIA-14.

Databiox: 118 IDC patients (grade 1/2/3 = overall Nottingham grade, NOT the pleomorphism
component), microscope-camera images at 4x/10x/20x/40x from two cameras (4032 and 1276 px wide).
Scale is not documented: it is calibrated per (magnification, camera) group from lymphocyte
nuclear size (training/data/databiox/scales.json, written by --calibrate).

  1. extract : 40x images -> nuclei, cell types, per-nucleus features (cached npz)
  2. evaluate: pool tumour nuclei per patient -> current model score + measurements vs grade
  3. retrain : MITOS-ATYPIA x20 frames (label = atypia score) + Databiox patients (label = grade,
               a proxy for pleomorphism). Grouped 10-fold CV (MITOS slide / Databiox patient never
               split across folds). Compares a 1-feature linear model, a small linear model and
               LightGBM; the best by mean kappa across both sources becomes the default model.

    OMP_NUM_THREADS=4 python training/pleomorphism/databiox.py --workers 8
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

D = ROOT / "training" / "data" / "databiox"
NAME = re.compile(r"(\d+)_BC_G(\d)_(\d+)_(\d+x)_(\d+)", re.I)
_sc = None


def index():
    recs = []
    for f in sorted(D.glob("BC_IDC_Grade_*/*/*")):
        m = NAME.search(f.stem)
        if not m or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif"):
            continue
        im = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_8)    # fast header-ish read for the width
        if im is None:
            continue
        recs.append(dict(file=str(f.relative_to(D)), grade=int(m.group(2)), patient=m.group(3), mag=m.group(4).lower(),
                         width=im.shape[1] * 8))
    return recs


def group_key(r):
    return f"{r['mag']}_{'large' if r['width'] > 2000 else 'small'}"


def _init():
    global _sc
    cv2.setNumThreads(1)
    from core.ihc.celltype import CellTypeClassifier
    from core.nuclei import NucleiDetector
    from core.pleomorphism.scorer import PleomorphismScorer
    _sc = PleomorphismScorer.__new__(PleomorphismScorer)
    _sc.detector = NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml", num_threads=4)
    _sc.classifier = CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx", num_threads=2)


def _extract(job):
    rel, um = job
    out = D / "feats" / (Path(rel).stem + ".npz")
    if out.exists():
        return rel
    im = cv2.cvtColor(cv2.imread(str(D / rel)), cv2.COLOR_BGR2RGB)
    t, ct, pts = _sc.analyze(im, um)
    np.savez_compressed(out, cell_type=ct, **{k: v for k, v in t.items()})
    return rel


def patient_units(recs):
    from core.pleomorphism.features import field_features, pool_tables
    by = {}
    for r in recs:
        if r["mag"] == "40x" and (D / "feats" / (Path(r["file"]).stem + ".npz")).exists():
            by.setdefault(r["patient"], []).append(r)
    units = []
    for pid, rs in by.items():
        tabs, cts = [], []
        for r in rs:
            z = np.load(D / "feats" / (Path(r["file"]).stem + ".npz"))
            if len(z["cell_type"]):
                tabs.append({k: z[k] for k in z.files if k not in ("cell_type", "label")})
                cts.append(z["cell_type"])
        if not tabs:
            continue
        ct = np.concatenate(cts)
        f = field_features(pool_tables(tabs), ct == 0, ct == 1)
        if f.get("n_tumour", 0) >= 20:
            units.append(dict(source="databiox", group=f"dbx_{pid}", y=rs[0]["grade"], feats=f, n_images=len(rs)))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from scipy.stats import spearmanr
    from core.ihc.scoring import weighted_kappa

    recs = index()
    scales = json.loads((D / "scales.json").read_text())
    for r in recs:
        r["um_per_px"] = scales.get(group_key(r))
    (D / "index.json").write_text(json.dumps(recs, indent=1))
    jobs = [(r["file"], r["um_per_px"]) for r in recs if r["mag"] == "40x" and r["um_per_px"]]
    (D / "feats").mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        list(ex.map(_extract, jobs, chunksize=2))
    dbx = patient_units(recs)
    y = np.array([u["y"] for u in dbx])
    print(f"Databiox: {len(dbx)} patients with 40x tumour nuclei; grade 1/2/3 = {np.bincount(y, minlength=4)[1:].tolist()}", flush=True)

    # ---- 1. current model + measurements vs grade
    res = {"databiox_current_model": {}}
    lin = json.loads((ROOT / "models/pleomorphism_linear.json").read_text())

    def lin_predict(model, f):
        x = np.array([f.get(k, np.nan) for k in model["features"]], float)
        if np.isnan(x).any():
            return np.nan, 0
        z = (x - np.array(model["mean"])) / np.array(model["scale"])
        lg = np.array(model["coef"]) @ z + np.array(model["intercept"])
        p = np.exp(lg - lg.max())
        p /= p.sum()
        ev = float(p @ [1, 2, 3])
        return ev, int(np.digitize(ev, model["cuts"])) + 1

    ev, sc = zip(*[lin_predict(lin, u["feats"]) for u in dbx])
    ev, sc = np.array(ev), np.array(sc)
    ok = ~np.isnan(ev)
    res["databiox_current_model"] = dict(n=int(ok.sum()), kappa_vs_grade=float(weighted_kappa(y[ok] - 1, sc[ok] - 1, n=3)),
                                         spearman_vs_grade=float(spearmanr(ev[ok], y[ok])[0]),
                                         confusion=np.histogram2d(y[ok], sc[ok], bins=[[0.5, 1.5, 2.5, 3.5]] * 2)[0].astype(int).tolist())
    meas = {}
    for k in ("major_p90", "area_median", "area_cv", "area_iqr_rel", "area_rel_lymph", "h_std_mean", "glcm_contrast_mean",
              "solidity_mean", "dark_frac_mean"):
        x = np.array([u["feats"].get(k, np.nan) for u in dbx])
        m = ~np.isnan(x)
        meas[k] = dict(spearman_vs_grade=float(spearmanr(x[m], y[m])[0]),
                       median_by_grade=[float(np.nanmedian(x[y == g])) for g in (1, 2, 3)])
    res["databiox_measurements"] = meas
    print("current model on Databiox:", json.dumps({k: v for k, v in res["databiox_current_model"].items()}), flush=True)
    for k, v in meas.items():
        print(f"   {k:22s} rho vs grade {v['spearman_vs_grade']:+.2f}  median by grade {np.round(v['median_by_grade'], 2).tolist()}")

    # ---- 2. retrain on MITOS-ATYPIA + Databiox with grouped CV
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from train_pleomorphism import D as MD, load_units
    mitos = load_units(json.loads((MD / "index.json").read_text()))
    mitos = [dict(source="mitos", group=f"mitos_{u['slide']}", y=u["y"], feats=u["feats"]) for u in mitos if u["feats"].get("n_tumour", 0) >= 5]
    units = mitos + dbx
    feats_all = sorted({k for u in units for k in u["feats"]})
    Y = np.array([u["y"] for u in units]) - 1
    G = np.array([u["group"] for u in units])
    S = np.array([u["source"] for u in units])

    def X_of(keys):
        X = np.array([[u["feats"].get(k, np.nan) for k in keys] for u in units], float)
        return X

    # balance: each source contributes equal total weight, classes balanced within source
    W = np.zeros(len(Y))
    for s in ("mitos", "databiox"):
        m = S == s
        cnt = np.bincount(Y[m], minlength=3).clip(1)
        W[m] = 1.0 / cnt[Y[m]]
        W[m] /= W[m].sum()

    candidates = {
        "linear_length": ["major_p90"],
        "linear_size_aniso": ["major_p90", "area_median", "area_cv", "area_rel_lymph"],
        "linear_size_aniso_chromatin": ["major_p90", "area_median", "area_cv", "area_rel_lymph", "h_std_mean", "glcm_contrast_mean", "dark_frac_mean"],
        "lightgbm_all": feats_all,
    }
    gkf = GroupKFold(10)
    cv = {}
    for name, keys in candidates.items():
        X = X_of(keys)
        P = np.zeros((len(Y), 3))
        for tr, te in gkf.split(X, Y, G):
            if name.startswith("linear"):
                med = np.nanmedian(X[tr], 0)
                Xt, Xe = np.where(np.isnan(X[tr]), med, X[tr]), np.where(np.isnan(X[te]), med, X[te])
                mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
                m = LogisticRegression(C=0.1, max_iter=3000).fit((Xt - mu) / sd, Y[tr], sample_weight=W[tr] * len(tr))
                P[te] = m.predict_proba((Xe - mu) / sd)
            else:
                params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7, min_data_in_leaf=15,
                              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1)
                m = lgb.train(params, lgb.Dataset(X[tr], Y[tr], weight=W[tr] * len(tr)), num_boost_round=150)
                P[te] = m.predict(X[te])
        ev = P @ np.array([1, 2, 3])
        # cut points chosen to maximise kappa on all out-of-fold predictions per source would leak; use argmax
        pred = P.argmax(1)
        r = {}
        for s in ("mitos", "databiox"):
            m = S == s
            r[s] = dict(n=int(m.sum()), kappa=float(weighted_kappa(Y[m], pred[m], n=3)), accuracy=float((Y[m] == pred[m]).mean()),
                        spearman=float(spearmanr(ev[m], Y[m])[0]),
                        confusion=np.histogram2d(Y[m], pred[m], bins=[[-0.5, 0.5, 1.5, 2.5]] * 2)[0].astype(int).tolist())
        r["mean_kappa"] = float(np.mean([r[s]["kappa"] for s in ("mitos", "databiox")]))
        cv[name] = r
        print(f"{name:28s} Databiox kappa {r['databiox']['kappa']:+.3f} (rho {r['databiox']['spearman']:+.2f}) | "
              f"MITOS kappa {r['mitos']['kappa']:+.3f} (rho {r['mitos']['spearman']:+.2f})", flush=True)
    res["retrain_grouped_cv"] = cv
    best = max(cv, key=lambda k: cv[k]["mean_kappa"])
    res["chosen"] = best
    print("chosen:", best, flush=True)

    # ---- 3. final model on all data
    keys = candidates[best]
    X = X_of(keys)
    if best.startswith("linear"):
        med = np.nanmedian(X, 0)
        X = np.where(np.isnan(X), med, X)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = LogisticRegression(C=0.1, max_iter=3000).fit((X - mu) / sd, Y, sample_weight=W * len(Y))
        (ROOT / "models/pleomorphism_linear.json").write_text(json.dumps(dict(
            type="linear_ordinal", features=keys, mean=mu.tolist(), scale=sd.tolist(), coef=m.coef_.tolist(),
            intercept=m.intercept_.tolist(), cuts=[1.5, 2.5], impute=med.tolist(), validated=False,
            source="MITOS-ATYPIA-14 (atypia scores) + Databiox (overall grade as proxy)",
            cv=dict(databiox_kappa=cv[best]["databiox"]["kappa"], mitos_kappa=cv[best]["mitos"]["kappa"]),
            note="NOT validated against pleomorphism scores of local pathologists"), indent=1))
    else:
        params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7, min_data_in_leaf=15,
                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1)
        m = lgb.train(params, lgb.Dataset(X, Y, weight=W * len(Y)), num_boost_round=150)
        m.save_model(str(ROOT / "models/pleomorphism_lgbm.txt"))
        (ROOT / "models/pleomorphism_lgbm.json").write_text(json.dumps(dict(features=keys, classes=[1, 2, 3],
                                                                             source="MITOS-ATYPIA-14 + Databiox"), indent=1))
    (ROOT / "eval/results/pleomorphism_databiox.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
