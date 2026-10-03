"""Model hub: every runtime model loaded once, lazily, on the analysis thread (Qt-free).

Thresholds come from the lab's own calibration (<data>/calibration/*.json, written by the Annotate
page from pathologist labels) when present, else from the public-data defaults - and every result
says which one was used.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

ERPR_INTENSITY = "erpr_intensity.json"
KI67_POSITIVITY = "ki67_positivity.json"
HER2_THRESHOLDS = "her2_thresholds.json"


class ModelHub:
    def __init__(self, models_dir: Path, calibration_dir: Path, nuclei_model: str = "stardist_he_ihc_ft_int8.xml"):
        self.models, self.calib = Path(models_dir), Path(calibration_dir)
        self.nuclei_model = nuclei_model
        self._cache = {}
        self._lock = threading.RLock()        # re-entrant: a factory may load another model
        self.on_loading = None             # callback(str) while a model loads (status bar)

    def _get(self, key, factory):
        with self._lock:
            if key not in self._cache:
                if self.on_loading:
                    self.on_loading(f"Loading {key} model…")
                self._cache[key] = factory()
            return self._cache[key]

    def m(self, name) -> Path:
        return self.models / name

    def loaded(self) -> list[str]:
        return sorted(self._cache)

    # ------------------------------------------------------------------ models
    @property
    def detector(self):
        from core.nuclei import NucleiDetector
        return self._get("nuclei", lambda: NucleiDetector(self.m(self.nuclei_model)))

    @property
    def celltype(self):
        from core.ihc.celltype import CellTypeClassifier
        return self._get("cell type", lambda: CellTypeClassifier(self.m("celltype_mnv3s_ctx32.onnx")))

    @property
    def region(self):
        from core.region.tissue_map import TissueRegionModel
        return self._get("tumour region", lambda: TissueRegionModel(self.m("region_effb0.onnx")))

    def magnification(self, objectives: dict):
        from core.region.tissue_map import MagnificationEstimator
        est = self._get("magnification", lambda: MagnificationEstimator(self.m("magnification_mnv3.onnx"), objectives))
        est.objectives = objectives
        return est

    @property
    def mitosis(self):
        from core.mitosis.detector import MitosisDetector
        return self._get("mitosis", lambda: MitosisDetector(self.m("mitosis_det.onnx"), self.m("mitosis_ver.onnx")))

    @property
    def hotspot(self):
        from core.mitosis.hotspot import HotspotGuide
        return self._get("mitosis hotspot", lambda: HotspotGuide(self.m("mitosis_hotspot.onnx")))

    @property
    def her2_field(self):
        from core.her2.field_classifier import HER2FieldClassifier
        return self._get("HER2 field", lambda: HER2FieldClassifier(self.m("her2_field_effb0.onnx")))

    @property
    def lumen(self):
        from core.tubules.lumen import LumenClassifier
        return self._get("tubule lumen", lambda: LumenClassifier(self.m("tubule_lumen_lgbm.txt")))

    @property
    def pleomorphism(self):
        def make():
            from core.pleomorphism.scorer import PleomorphismScorer
            emb_path = self.m("phikon_v2_int8w.xml")
            if emb_path.is_file() and self.m("pleomorphism_phikon.npz").is_file():
                from core.pleomorphism.embedding import TileEmbedder
                return PleomorphismScorer(self.m("pleomorphism_phikon.npz"), self.detector, self.celltype,
                                          embedder=TileEmbedder(emb_path))
            return PleomorphismScorer(self.m("pleomorphism_lgbm.txt"), self.detector, self.celltype)
        return self._get("pleomorphism", make)

    # ------------------------------------------------------------------ thresholds
    def _calib(self, name):
        p = self.calib / name
        return p if p.is_file() else None

    def erpr_thresholds(self):
        from core.ihc.scoring import IntensityThresholds
        p = self._calib(ERPR_INTENSITY)
        return (IntensityThresholds.load(p), "local calibration") if p else (IntensityThresholds(), "public-data placeholders")

    def ki67_positivity(self):
        from core.ki67.scoring import PositivityThreshold
        p = self._calib(KI67_POSITIVITY)
        return (PositivityThreshold.load(p), "local calibration") if p else (PositivityThreshold(), "public-data default (BC-DeepLIIF)")

    def her2_thresholds(self):
        from core.her2.scoring import HER2Thresholds
        p = self._calib(HER2_THRESHOLDS)
        if p:
            return HER2Thresholds.load(p), "local calibration"
        return HER2Thresholds.load(self.m("her2_thresholds_bci.json")), "public-data (BCI)"

    def calibration_status(self) -> dict:
        out = {}
        for key, name in (("ER/PR intensity", ERPR_INTENSITY), ("Ki-67 positivity", KI67_POSITIVITY),
                          ("HER2 membrane", HER2_THRESHOLDS)):
            p = self._calib(name)
            if p:
                try:
                    out[key] = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    out[key] = None
            else:
                out[key] = None
        return out
