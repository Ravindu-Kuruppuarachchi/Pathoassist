"""Nuclear pleomorphism score (Nottingham component, 1-3) from the 40x tumour fields the
pathologist visits.

Per field: StarDist nuclei -> cell type (tumour / lymphocyte / ... from the H&E-trained
classifier) -> per-nucleus shape + chromatin features (core.pleomorphism.features).
Across fields: tumour nuclei of all accepted fields are pooled (like a pathologist forming an
overall impression), field features computed on the pool, and a LightGBM model gives
P(score 1/2/3). Field-level scores are shown too so outlier fields can be reviewed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.ihc.dedup import FieldDeduplicator
from core.ihc.stain import RUIFROK_H, StainMatrix
from core.pleomorphism.features import MAX_GLCM_NUCLEI, field_features, nucleus_table, pool_tables

EOSIN = np.array([0.07, 0.99, 0.11])
HE_STAIN = StainMatrix(RUIFROK_H, EOSIN)          # second channel = eosin for H&E
TUMOUR, LYMPHOCYTE = 0, 1
# interpretable measurements shown to the pathologist (pooled tumour nuclei). On MITOS-ATYPIA-14
# nuclear length p90 tracks the pathologists' "nuclear size" criterion (Spearman 0.70; pathologist vs
# pathologist 0.55) and area CV tracks "anisonucleosis" (0.43 vs 0.58).
MEASUREMENTS = {"area_median": "median nuclear area (µm²)", "major_p90": "nuclear length, 90th percentile (µm)",
                "area_cv": "anisonucleosis (area CV)", "area_max_rel": "largest / typical nucleus (p95 / median area)",
                "area_rel_lymph": "nuclear size vs lymphocytes in the same fields", "area_rel_ref": "nuclear size vs a 7 µm lymphocyte"}


@dataclass
class PleoField:
    index: int
    table: dict                 # per-nucleus measurements
    cell_type: np.ndarray       # per nucleus (index aligned with table)
    points: np.ndarray
    probs: np.ndarray           # field-level P(1, 2, 3)
    n_tumour: int
    emb: np.ndarray = None      # mean Phikon tile embedding (embedding model only)


@dataclass
class PleoResult:
    score: int
    probs: list
    expected: float
    n_fields: int
    n_tumour: int
    field_scores: list
    warnings: list = field(default_factory=list)
    measurements: dict = field(default_factory=dict)


class PleomorphismScorer:
    """model_path:
      models/pleomorphism_phikon.npz (default) - Phikon-v2 tile embeddings -> PCA -> logistic;
                                                 needs `embedder` (core.pleomorphism.embedding.TileEmbedder)
      models/pleomorphism_lgbm.txt             - LightGBM on the hand-crafted nuclear features
      models/pleomorphism_linear.json          - ordinal logistic on a few nuclear features
    The nuclear measurements are computed in every case and shown alongside the score."""

    def __init__(self, model_path, detector, classifier=None, embedder=None):
        model_path = Path(model_path)
        self.detector, self.classifier, self.embedder = detector, classifier, embedder
        self.emb_model = None
        if model_path.suffix == ".npz":
            from core.pleomorphism.embedding import EmbeddingPleoModel
            if embedder is None:
                raise ValueError("the embedding model needs a TileEmbedder")
            self.emb_model = EmbeddingPleoModel(model_path)
            self.lin, self.model, self.features = None, None, []
            self.validated = bool(self.emb_model.meta.get("validated", False))
        elif model_path.suffix == ".json":
            self.lin = json.loads(model_path.read_text())
            self.features, self.model = self.lin["features"], None
            self.validated = bool(self.lin.get("validated", False))
        else:
            import lightgbm as lgb
            self.model, self.lin = lgb.Booster(model_file=str(model_path)), None
            self.features = json.loads(model_path.with_suffix(".json").read_text())["features"]
            self.validated = False

    def _predict(self, f: dict) -> np.ndarray:
        x = np.array([[f.get(k, np.nan) for k in self.features]], np.float64)
        if self.lin is None:
            return self.model.predict(x)[0]
        if np.isnan(x).any():
            return np.full(3, np.nan)
        z = (x - np.array(self.lin["mean"])) / np.array(self.lin["scale"])
        logit = z @ np.array(self.lin["coef"]).T + np.array(self.lin["intercept"])
        e = np.exp(logit - logit.max())
        return (e / e.sum())[0]

    def score_from_probs(self, p) -> int:
        if np.isnan(p).any():
            return 0
        if self.lin is not None and "cuts" in self.lin:
            return int(np.digitize(float(np.dot(p, [1, 2, 3])), self.lin["cuts"])) + 1
        return int(np.argmax(p)) + 1

    def analyze(self, frame_rgb, um_per_px):
        """Per-nucleus table + cell types of one 40x frame."""
        det = self.detector.detect(frame_rgb, um_per_px=um_per_px)
        h = HE_STAIN.concentrations(frame_rgb)[..., 0]
        if self.classifier is not None and len(det):
            ct = self.classifier.predict(frame_rgb, det.points, um_per_px).argmax(1)
        else:
            ct = np.zeros(len(det), int)
        tum = np.nonzero(ct == TUMOUR)[0]
        rng = np.random.default_rng(0)
        glcm_ids = rng.choice(tum, min(len(tum), MAX_GLCM_NUCLEI), replace=False) if len(tum) else []
        t = nucleus_table(det.labels, h, um_per_px, glcm_ids)
        # regionprops returns labels in order 1..n, aligned with det.points
        return t, ct, det.points

    def field_probs(self, t, ct, emb=None):
        f = field_features({k: v for k, v in t.items() if k != "label"}, ct == TUMOUR, ct == LYMPHOCYTE)
        if self.emb_model is not None:
            return (self.emb_model.predict(emb) if emb is not None else np.full(3, np.nan)), f
        return self._predict(f), f


class PleomorphismSession:
    def __init__(self, scorer: PleomorphismScorer, min_fields=3, min_nuclei=300, max_overlap=0.5):
        self.scorer, self.min_fields, self.min_nuclei = scorer, min_fields, min_nuclei
        self.fields: list[PleoField] = []
        self.dedup = FieldDeduplicator(max_overlap)

    def add_field(self, frame_rgb, um_per_px) -> dict:
        ov = self.dedup.check(frame_rgb)
        if ov is not None:
            return dict(accepted=False, reason=f"overlaps field {self.fields[ov.index].index + 1} by {ov.fraction:.0%}")
        t, ct, pts = self.scorer.analyze(frame_rgb, um_per_px)
        if not t:
            return dict(accepted=False, reason="no nuclei detected")
        emb = self.scorer.embedder.embed_field(frame_rgb, um_per_px) if self.scorer.emb_model is not None else None
        p, _ = self.scorer.field_probs(t, ct, emb)
        f = PleoField(len(self.fields), t, ct, pts, p, int((ct == TUMOUR).sum()), emb)
        self.fields.append(f)
        self.dedup.add(frame_rgb)
        return dict(accepted=True, field=f.index, probs=p.tolist(), n_tumour=f.n_tumour)

    def set_cell_type(self, field_index, nucleus, cell_type):
        """Click-to-correct: include / exclude a nucleus (e.g. a mis-called lymphocyte)."""
        f = self.fields[field_index]
        f.cell_type[nucleus] = cell_type
        f.probs, _ = self.scorer.field_probs(f.table, f.cell_type, f.emb)

    def result(self) -> PleoResult:
        if not self.fields:
            return PleoResult(0, [0, 0, 0], 0.0, 0, 0, [], ["No fields scored."])
        pooled = pool_tables([f.table for f in self.fields])
        ct = np.concatenate([f.cell_type for f in self.fields])
        embs = [f.emb for f in self.fields if f.emb is not None]
        p, feats = self.scorer.field_probs({**pooled, "label": None}, ct, np.mean(embs, 0) if embs else None)
        n = int((ct == TUMOUR).sum())
        warn = []
        if not self.scorer.validated:
            warn.append("Score model NOT validated against local pathologists (public data only): use the measurements, not the score, "
                        "until it is refitted on local cases.")
        if len(self.fields) < self.min_fields:
            warn.append(f"Only {len(self.fields)} field(s); assess at least {self.min_fields} tumour fields at 40x.")
        if n < self.min_nuclei:
            warn.append(f"Only {n} tumour nuclei measured (< {self.min_nuclei}).")
        fs = [self.scorer.score_from_probs(f.probs) for f in self.fields]
        if len(set(fs)) > 1:
            warn.append(f"Field scores differ ({fs}); review the fields with the highest score.")
        meas = {k: round(float(feats[k]), 3) for k in MEASUREMENTS if k in feats and np.isfinite(feats[k])}
        return PleoResult(self.scorer.score_from_probs(p), [round(float(v), 3) for v in p],
                          round(float(np.dot(p, [1, 2, 3])), 2), len(self.fields), n, fs, warn, meas)
