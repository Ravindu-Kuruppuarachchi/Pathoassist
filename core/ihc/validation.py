"""Agreement statistics for validating assisted scores against pathologist reports.

Used for the archived-case study (Feature 2, step 9) and the public-data proxy
(eval/evaluate_ihc_deepliif.py):

    python -m core.ihc.validation cases.csv
    # cases.csv columns: case_id, stain, reported_pct, assisted_pct[, reported_allred, assisted_allred]
"""
from __future__ import annotations

import sys

import numpy as np

from core.ihc.scoring import status, weighted_kappa


def icc_2_1(a, b) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout & Fleiss)."""
    Y = np.column_stack([a, b]).astype(float)
    n, k = Y.shape
    gm = Y.mean()
    ssr = k * ((Y.mean(1) - gm) ** 2).sum()
    ssc = n * ((Y.mean(0) - gm) ** 2).sum()
    sse = ((Y - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def bootstrap_ci(f, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = [f(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


def agreement(reported_pct, assisted_pct) -> dict:
    r, p = np.asarray(reported_pct, float), np.asarray(assisted_pct, float)
    cats = ["Negative", "Low positive", "Positive"]
    rc = np.array([cats.index(status(v)) for v in r])
    pc = np.array([cats.index(status(v)) for v in p])
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (rc, pc), 1)
    d = p - r
    return dict(n=len(r), icc=icc_2_1(r, p), icc_ci95=bootstrap_ci(icc_2_1, r, p),
                mae_pct=float(np.abs(d).mean()), bias_pct=float(d.mean()),
                limits_of_agreement=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                category_agreement=float((rc == pc).mean()),
                category_kappa=float(weighted_kappa(rc, pc, n=3)), category_confusion=cm.tolist(),
                category_labels=cats)


def main(path):
    import json
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for stain, g in df.groupby("stain"):
        out[stain] = agreement(g["reported_pct"], g["assisted_pct"])
        if {"reported_allred", "assisted_allred"} <= set(g.columns):
            ra, pa = g["reported_allred"].astype(int), g["assisted_allred"].astype(int)
            out[stain]["allred_exact"] = float((ra == pa).mean())
            out[stain]["allred_within_1"] = float((abs(ra - pa) <= 1).mean())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
