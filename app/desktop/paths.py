"""Where bundled resources and per-user data live, from source and inside the PyInstaller bundle.

Bundled (read-only): models/ and app/desktop/assets/, under sys._MEIPASS when frozen.
User data (writable, survives app updates/reinstalls): cases, microscope profiles, annotations, local
calibrations, captures and logs. Set PATHOASSIST_DATA to keep it elsewhere (e.g. a lab share or backup disk).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "PathoAssist"

DEFAULT_NUCLEI_MODEL = "stardist_he_ihc_ft_int8.xml"

# Runtime model files shipped with the app (no .pt checkpoints, train logs or val dumps).
# Single source of truth for what a bundle must contain (--self-test checks it).
RUNTIME_MODEL_FILES = [
    "stardist_he_ihc_ft_int8.xml", "stardist_he_ihc_ft_int8.bin", "stardist_he_ihc_ft_int8.json",
    "celltype_mnv3s_ctx32.onnx", "celltype_mnv3s_ctx32.json",
    "her2_field_effb0.onnx", "her2_field_effb0.json", "her2_thresholds_bci.json",
    "magnification_mnv3.onnx", "magnification_mnv3.json",
    "region_effb0.onnx", "region_effb0.json",
    "mitosis_det.onnx", "mitosis_det.json", "mitosis_ver.onnx",
    "mitosis_hotspot.onnx", "mitosis_hotspot.json",
    "pleomorphism_lgbm.txt", "pleomorphism_lgbm.json", "pleomorphism_linear.json",
    "pleomorphism_phikon.npz", "pleomorphism_phikon.json", "phikon_v2_int8w.xml", "phikon_v2_int8w.bin",
    "tubule_lumen_lgbm.txt", "tubule_lumen_lgbm.json",
]


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return resource_root() / "models"


def assets_dir() -> Path:
    return resource_root() / "app" / "desktop" / "assets"


def user_data_dir() -> Path:
    """$PATHOASSIST_DATA if set, else %LOCALAPPDATA%\\PathoAssist on Windows, $XDG_DATA_HOME/PathoAssist
    (~/.local/share) on Linux."""
    if os.environ.get("PATHOASSIST_DATA"):
        d = Path(os.environ["PATHOASSIST_DATA"]).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def captures_dir() -> Path:
    d = user_data_dir() / "captures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sub(name: str) -> Path:
    d = user_data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def cases_dir() -> Path:
    """One folder per case: case.json (patient, results, grade), audit.jsonl, frames/, modules/, reports/."""
    return _sub("cases")


def profiles_dir() -> Path:
    """Microscope profiles: objective calibration (µm/px) and flat-field blanks."""
    return _sub("profiles")


def annotations_dir() -> Path:
    """Pathologist annotations for retraining, one folder per task (de-identified: case ID only)."""
    return _sub("annotations")


def calibration_dir() -> Path:
    """Thresholds calibrated on local pathologist labels; used instead of the public-data defaults."""
    return _sub("calibration")
