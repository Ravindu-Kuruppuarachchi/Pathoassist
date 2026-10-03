"""Where bundled resources and per-user data live, from source and inside the PyInstaller bundle.

Bundled (read-only): models/ and app/desktop/assets/, under sys._MEIPASS when frozen.
User data (writable, survives app updates/reinstalls): captures, logs, later annotations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "PathoAssist"

DEFAULT_NUCLEI_MODEL = "stardist_he_ihc_ft_int8.xml"

# Runtime model files shipped with the app (no .pt checkpoints, train logs or val dumps).
# packaging/pathoassist.spec reads this list, so it is the single source of truth.
RUNTIME_MODEL_FILES = [
    "stardist_he_ihc_ft_int8.xml", "stardist_he_ihc_ft_int8.bin", "stardist_he_ihc_ft_int8.json",
    "celltype_mnv3s_ctx32.onnx", "celltype_mnv3s_ctx32.json",
    "her2_field_effb0.onnx", "her2_field_effb0.json", "her2_thresholds_bci.json",
    "magnification_mnv3.onnx", "magnification_mnv3.json",
    "region_effb0.onnx", "region_effb0.json",
    "mitosis_det.onnx", "mitosis_det.json", "mitosis_ver.onnx",
    "mitosis_hotspot.onnx", "mitosis_hotspot.json",
    "pleomorphism_lgbm.txt", "pleomorphism_lgbm.json", "pleomorphism_linear.json",
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
    """%LOCALAPPDATA%\\PathoAssist on Windows, $XDG_DATA_HOME/PathoAssist (~/.local/share) on Linux."""
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
