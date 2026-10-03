"""Final Feature 7 model: Phikon-v2 embeddings -> standardise -> PCA(16) -> logistic regression.

Chooses between training on Databiox only and Databiox + MITOS-ATYPIA-14 by repeated
patient-grouped CV on Databiox (MITOS units, when used, are always in the training folds), then
fits the chosen set and saves models/pleomorphism_phikon.npz + .json (with the CV results).

    python training/pleomorphism/train_final.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import compare_models as cm

N_PCA, C = 16, 0.05


def cv_databiox(dbx, extra, repeats=5):
    """Repeated patient-grouped 5-fold CV on Databiox; `extra` units always added to training."""
    y = np.array([u["y"] for u in dbx]) - 1
    g = np.array([u["group"] for u in dbx])
    ks, rhos, accs = [], [], []
    P_all = []
    for r in range(repeats):
        P = np.zeros((len(y), 3))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y, g):
            tu = [dbx[i] for i in tr] + extra
            P[te] = cm.Model("emb", [], n_pca=N_PCA, C=C).fit(tu, cm.weights(tu)).predict([dbx[i] for i in te])
        s = cm.score(y, P)
        ks.append(s["kappa"]), rhos.append(s["spearman"]), accs.append(s["accuracy"])
        P_all.append(P)
    P = np.mean(P_all, 0)
    cmx = np.zeros((3, 3), int)
    np.add.at(cmx, (y, P.argmax(1)), 1)
    return dict(kappa=float(np.mean(ks)), kappa_sd=float(np.std(ks)), spearman=float(np.mean(rhos)),
                accuracy=float(np.mean(accs)), confusion=cmx.tolist())


def main():
    units = cm.build_units()
    dbx = [u for u in units if u["source"] == "databiox"]
    mit = [u for u in units if u["source"] == "mitos"]
    options = {"databiox_only": [], "databiox_plus_mitos": mit}
    res = {k: cv_databiox(dbx, ex) for k, ex in options.items()}
    for k, v in res.items():
        print(f"{k:22s} Databiox kappa {v['kappa']:.3f} ± {v['kappa_sd']:.3f}, rho {v['spearman']:.2f}, acc {v['accuracy']:.2f}, "
              f"confusion {v['confusion']}", flush=True)
    best = max(res, key=lambda k: res[k]["kappa"])
    train = dbx + options[best]
    m = cm.Model("emb", [], n_pca=N_PCA, C=C).fit(train, cm.weights(train))
    np.savez(ROOT / "models/pleomorphism_phikon.npz", emb_mean=m.mu, emb_std=m.sd, pca_mean=m.pca.mean_,
             pca_components=m.pca.components_, coef=m.lr.coef_, intercept=m.lr.intercept_)
    (ROOT / "models/pleomorphism_phikon.json").write_text(json.dumps(dict(
        embedder="models/phikon_v2_int8w.xml (Phikon-v2, owkin; INT8 weights)", tile="224 px at 0.5 µm/px, field = mean tile embedding",
        model=f"standardise -> PCA({N_PCA}) -> multinomial logistic regression (C={C}), class-balanced",
        trained_on=best, n_train=len(train), cv=res[best], cv_all=res, validated=False,
        labels="Databiox: overall Nottingham grade (proxy for pleomorphism); MITOS-ATYPIA-14: nuclear atypia score",
        note="Grouped CV on public data only. Not validated against local pathologists' pleomorphism scores."), indent=1))
    print("saved models/pleomorphism_phikon.npz, trained on", best, flush=True)
    (ROOT / "eval/results/pleomorphism_final.json").write_text(json.dumps(dict(chosen=best, cv=res), indent=2))


if __name__ == "__main__":
    main()
