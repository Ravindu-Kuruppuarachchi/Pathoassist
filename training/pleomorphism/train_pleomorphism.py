"""Train / validate the nuclear pleomorphism model on MITOS-ATYPIA-14.

1. extract : per x40 frame - nuclei, cell types, per-nucleus features (cached npz)
2. pool    : the 4 x40 frames of each x20 frame (the unit the atypia score was given for)
3. LightGBM multiclass (score 1/2/3), leave-one-slide-out CV over the 11 slides (A/H scans of a
   slide stay in the same fold). Reports weighted kappa vs the consensus score, next to the
   pathologist-vs-pathologist kappa from the individual rater scores.
4. final model on all frames -> models/pleomorphism_lgbm.txt (+ .json)

    OMP_NUM_THREADS=4 python training/pleomorphism/train_pleomorphism.py --workers 8
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

D = ROOT / "training" / "data" / "mitos_atypia"
_sc = None


def _init():
    global _sc
    import cv2
    cv2.setNumThreads(1)
    from core.ihc.celltype import CellTypeClassifier
    from core.nuclei import NucleiDetector
    from core.pleomorphism.scorer import PleomorphismScorer
    _sc = PleomorphismScorer.__new__(PleomorphismScorer)          # analyze() only needs detector + classifier
    _sc.detector = NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml", num_threads=4)
    _sc.classifier = CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx", num_threads=2)


def _extract(job):
    fn, um = job
    out = D / "feats" / (Path(fn).stem + ".npz")
    if out.exists():
        return fn
    import cv2
    im = cv2.cvtColor(cv2.imread(str(D / "x40" / fn)), cv2.COLOR_BGR2RGB)
    t, ct, pts = _sc.analyze(im, um)
    np.savez_compressed(out, cell_type=ct, **{k: v for k, v in t.items()})
    return fn


def load_units(index):
    """One unit per (scanner, x20 frame): pooled nuclei of its x40 frames + score."""
    from core.pleomorphism.features import field_features, pool_tables
    units = []
    for rec in index:
        by20 = {}
        for fn in rec["x40_frames"]:
            by20.setdefault(Path(fn).stem[:-1], []).append(fn)
        for k20, fns in by20.items():
            lab = rec["x20"].get(k20)
            if not lab or not lab.get("decision"):
                continue
            tabs, cts = [], []
            for fn in fns:
                z = np.load(D / "feats" / (Path(fn).stem + ".npz"))
                if len(z["cell_type"]) == 0:
                    continue
                tabs.append({k: z[k] for k in z.files if k != "cell_type"})
                cts.append(z["cell_type"])
            if not tabs:
                continue
            pooled = pool_tables(tabs)
            ct = np.concatenate(cts)
            f = field_features(pooled, ct == 0, ct == 1)
            units.append(dict(slide=rec["slide"][1:], scanner=rec["scanner"], x20=k20, y=lab["decision"],
                              raters=lab.get("raters", []), feats=f))
    return units


CRITERIA = ["nuclei_size", "anisonucleosis", "nucleoli_size", "chromatin_density"]


def load_x40_criteria(index):
    """One unit per scored x40 frame: its own nuclei + mean of the raters for each criterion."""
    from core.pleomorphism.features import field_features
    out = []
    for rec in index:
        for k, crit in rec["x40_criteria"].items():
            f = D / "feats" / f"{k}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            ct = z["cell_type"]
            if len(ct) == 0:
                continue
            ff = field_features({q: z[q] for q in z.files if q not in ("cell_type", "label")}, ct == 0, ct == 1)
            if ff.get("n_tumour", 0) < 5:
                continue
            out.append(dict(slide=rec["slide"][1:], x40=k, x20=k[:-1], scanner=rec["scanner"], feats=ff,
                            crit={c: float(np.mean(crit[c])) for c in CRITERIA if crit.get(c)}))
    return out


def two_stage_loso(units, x40, feats, params_reg):
    """Stage A: LightGBM regressors x40 features -> each criterion (raters' mean).
    Stage B: x20 frame = mean predicted criteria over its x40 frames -> linear score s;
    cut points on s fitted on the training slides (max weighted kappa). Fully nested in LOSO."""
    import lightgbm as lgb
    from core.ihc.scoring import weighted_kappa
    Xa = np.array([[u["feats"].get(k, np.nan) for k in feats] for u in x40], np.float64)
    sl_a = np.array([u["slide"] for u in x40])
    y20 = np.array([u["y"] for u in units]) - 1
    sl_20 = np.array([u["slide"] for u in units])
    key20 = [(u["slide"], u["scanner"], u["x20"]) for u in units]
    pred_cont = np.full(len(units), np.nan)

    def crit_models(train_mask):
        ms = {}
        for c in CRITERIA:
            yc = np.array([u["crit"].get(c, np.nan) for u in x40])
            ok = train_mask & ~np.isnan(yc)
            ms[c] = lgb.train(params_reg, lgb.Dataset(Xa[ok], yc[ok]), num_boost_round=150)
        return ms

    def x20_criteria(ms, slides_sel):
        """Mean predicted criteria per x20 frame (frames of the selected slides)."""
        P = {c: ms[c].predict(Xa) for c in CRITERIA}
        agg = {}
        for i, u in enumerate(x40):
            if u["slide"] in slides_sel:
                agg.setdefault((u["slide"], u["scanner"], u["x20"]), []).append([P[c][i] for c in CRITERIA])
        return {k: np.mean(v, 0) for k, v in agg.items()}

    def fit_stage_b(C, y):
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(C, y)
        s = lr.predict(C)
        qs = np.quantile(s, np.linspace(0.02, 0.98, 49))
        best = max(((weighted_kappa(y, np.digitize(s, [a, b]), n=3), (a, b)) for i, a in enumerate(qs) for b in qs[i + 1:]))
        return lr, best[1]

    for s_out in sorted(set(sl_20)):
        tr = sl_a != s_out
        ms = crit_models(tr)
        tr_slides = set(sl_20) - {s_out}
        c_tr = x20_criteria(ms, tr_slides)            # in-sample criteria for train slides (stage B fit)
        idx_tr = [i for i, k in enumerate(key20) if k in c_tr]
        lr, cuts = fit_stage_b(np.array([c_tr[key20[i]] for i in idx_tr]), y20[idx_tr])
        c_te = x20_criteria(ms, {s_out})
        for i, k in enumerate(key20):
            if k in c_te:
                pred_cont[i] = lr.predict(np.array([c_te[k]]))[0]
                units[i]["cuts"] = cuts
    ok = ~np.isnan(pred_cont)
    pred = np.array([np.digitize(pred_cont[i], units[i]["cuts"]) if ok[i] else -1 for i in range(len(units))])
    return pred_cont, pred, ok


def human_agreement(units):
    from core.ihc.scoring import weighted_kappa
    a = [(u["raters"][0], u["raters"][1]) for u in units if len(u.get("raters", [])) >= 2 and u["scanner"] == "aperio"]
    a = np.array(a) - 1
    return dict(n=len(a), weighted_kappa=float(weighted_kappa(a[:, 0], a[:, 1], n=3)), exact=float((a[:, 0] == a[:, 1]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    index = json.loads((D / "index.json").read_text())
    (D / "feats").mkdir(exist_ok=True)
    jobs = [(fn, r["um_per_px"]) for r in index for fn in r["x40_frames"]]
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        list(ex.map(_extract, jobs, chunksize=4))

    import lightgbm as lgb
    from core.ihc.scoring import weighted_kappa
    units = load_units(index)
    feats = sorted({k for u in units for k in u["feats"]})
    X = np.array([[u["feats"].get(k, np.nan) for k in feats] for u in units], np.float64)
    y = np.array([u["y"] for u in units]) - 1
    slides = np.array([u["slide"] for u in units])
    print(f"{len(units)} scored x20 frames ({len(set(slides))} slides x 2 scanners); score 1/2/3 = "
          f"{np.bincount(y, minlength=3).tolist()}", flush=True)
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7, min_data_in_leaf=10,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  class_weight=None)
    params.pop("class_weight")
    w = (1.0 / np.bincount(y, minlength=3).clip(1))[y]
    P = np.zeros((len(y), 3))
    for s in sorted(set(slides)):
        te = slides == s
        m = lgb.train(params, lgb.Dataset(X[~te], y[~te], weight=w[~te] / w[~te].mean()), num_boost_round=200)
        P[te] = m.predict(X[te])
    pred = P.argmax(1)
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (y, pred), 1)
    # slide level: pool all units of a slide (mean probability), vs majority consensus score
    sl = sorted(set(slides))
    ys = np.array([np.bincount(y[slides == s], minlength=3).argmax() for s in sl])
    ps = np.array([P[slides == s].mean(0).argmax() for s in sl])
    hum = human_agreement(units)
    from scipy.stats import spearmanr
    rater_mean = np.array([np.mean(u["raters"]) if u.get("raters") else u["y"] for u in units])
    x40 = load_x40_criteria(index)
    params_reg = dict(objective="regression", learning_rate=0.05, num_leaves=7, min_data_in_leaf=20, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
    cont, pred2, ok2 = two_stage_loso(units, x40, feats, params_reg)
    cm2 = np.zeros((3, 3), int)
    np.add.at(cm2, (y[ok2], pred2[ok2]), 1)
    two_stage = dict(n=int(ok2.sum()), loso_weighted_kappa=float(weighted_kappa(y[ok2], pred2[ok2], n=3)),
                     loso_accuracy=float((pred2[ok2] == y[ok2]).mean()), confusion=cm2.tolist(),
                     spearman_vs_rater_mean=float(spearmanr(cont[ok2], rater_mean[ok2])[0]),
                     slide_level_spearman=float(spearmanr([np.nanmean(cont[slides == s_]) for s_ in sorted(set(slides))],
                                                         [np.mean(rater_mean[slides == s_]) for s_ in sorted(set(slides))])[0]))
    direct_cont = P @ np.array([1, 2, 3])
    print("direct model: spearman vs rater mean", round(float(spearmanr(direct_cont, rater_mean)[0]), 3), flush=True)
    print("two-stage (criteria) model:", json.dumps(two_stage), flush=True)
    res = dict(n=int(len(y)), class_counts=np.bincount(y, minlength=3).tolist(), loso_weighted_kappa=float(weighted_kappa(y, pred, n=3)),
               loso_accuracy=float((pred == y).mean()), confusion=cm.tolist(),
               by_scanner={sc: float(weighted_kappa(y[[u["scanner"] == sc for u in units]], pred[[u["scanner"] == sc for u in units]], n=3))
                           for sc in ("aperio", "hamamatsu")},
               slide_level=dict(n=len(sl), accuracy=float((ys == ps).mean()), truth=(ys + 1).tolist(), pred=(ps + 1).tolist()),
               pathologist_vs_pathologist=hum, two_stage=two_stage,
               direct_spearman_vs_rater_mean=float(spearmanr(direct_cont, rater_mean)[0]))
    print(json.dumps(res, indent=1), flush=True)

    # Final default model: the simplest one that generalised in LOSO (nuclear length p90 alone;
    # richer feature sets overfit the 11 slides). Ordinal: logistic regression + cut points on its
    # expected value. LightGBM on all features is saved too, for when local labels are available.
    final_feats = ["major_p90"]
    from sklearn.linear_model import LogisticRegression
    Xf = np.array([[u["feats"].get(k, np.nan) for k in final_feats] for u in units], np.float64)
    keep = ~np.isnan(Xf).any(1)
    mu, sd = Xf[keep].mean(0), Xf[keep].std(0)
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit((Xf[keep] - mu) / sd, y[keep])
    ev = lr.predict_proba((Xf[keep] - mu) / sd) @ np.array([1, 2, 3])
    qs = np.quantile(ev, np.linspace(0.02, 0.98, 49))
    _, cuts = max(((weighted_kappa(y[keep], np.digitize(ev, [a, b]), n=3), (float(a), float(b))) for i, a in enumerate(qs) for b in qs[i + 1:]))
    (ROOT / "models" / "pleomorphism_linear.json").write_text(json.dumps(dict(
        type="linear_ordinal", features=final_feats, mean=mu.tolist(), scale=sd.tolist(), coef=lr.coef_.tolist(),
        intercept=lr.intercept_.tolist(), cuts=list(cuts), validated=False,
        note="MITOS-ATYPIA-14, 11 slides; LOSO weighted kappa ~0.17 vs pathologist-pathologist 0.67 - NOT validated; "
             "refit on Databiox + local labels"), indent=1))
    m = lgb.train(params, lgb.Dataset(X, y, weight=w / w.mean()), num_boost_round=200)
    m.save_model(str(ROOT / "models" / "pleomorphism_lgbm.txt"))
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda t: -t[1])[:12]
    (ROOT / "models" / "pleomorphism_lgbm.json").write_text(json.dumps(dict(
        features=feats, classes=[1, 2, 3], source="MITOS-ATYPIA-14 training set (11 slides, Aperio + Hamamatsu), x20 consensus scores",
        top_features=[(k, round(float(v), 1)) for k, v in imp]), indent=1))
    res["top_features"] = [k for k, _ in imp]
    (ROOT / "eval" / "results" / "pleomorphism_mitos_atypia.json").write_text(json.dumps(res, indent=2))
    print("top features:", [k for k, _ in imp])


if __name__ == "__main__":
    main()
