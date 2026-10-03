"""Entry point.

    python -m app.desktop               # launch the app (from the repo root)
    python -m app.desktop --self-test   # headless check: every bundled model loads, nuclei detection runs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

from . import __version__
from .paths import DEFAULT_NUCLEI_MODEL, RUNTIME_MODEL_FILES, assets_dir, logs_dir, models_dir


def self_test() -> int:
    """Used after packaging to prove the bundle is self-contained (run on a clean machine/container)."""
    report, ok = {"version": __version__, "frozen": bool(getattr(sys, "frozen", False))}, True
    try:
        missing = [f for f in RUNTIME_MODEL_FILES if not (models_dir() / f).is_file()]
        report["missing_models"] = missing
        ok &= not missing

        import cv2
        import onnxruntime as ort
        from core.nuclei import NucleiDetector

        onnx = {}
        for f in RUNTIME_MODEL_FILES:
            if f.endswith(".onnx"):
                s = ort.InferenceSession(str(models_dir() / f), providers=["CPUExecutionProvider"])
                onnx[f] = [i.shape for i in s.get_inputs()]
        report["onnx_models_loaded"] = len(onnx)

        rgb = cv2.cvtColor(cv2.imread(str(assets_dir() / "sample_field.jpg")), cv2.COLOR_BGR2RGB)
        det = NucleiDetector(models_dir() / DEFAULT_NUCLEI_MODEL)
        d = det.detect(rgb, um_per_px=0.25)
        report["nuclei"] = len(d)
        report["detect_seconds"] = round(d.timings["total"], 3)
        ok &= len(d) > 50

        # Qt: platform plugin loads and the main window builds (offscreen when there is no display)
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from .main_window import MainWindow
        from .modules import MODULES
        from .annotation import TASKS
        app = QApplication.instance() or QApplication([])
        from . import theme
        theme.apply(app)
        w = MainWindow()
        report["qt_platform"] = app.platformName()
        report["analysis_modules"] = [m.key for m in MODULES]
        report["annotation_tasks"] = [t.key for t in TASKS]
        w.close()
    except Exception:  # noqa: BLE001
        report["error"] = traceback.format_exc()
        ok = False
    report["ok"] = bool(ok)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def run_app() -> int:
    logging.basicConfig(filename=logs_dir() / "pathoassist.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("start %s", __version__)
    from PySide6.QtWidgets import QApplication, QMessageBox
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PathoAssist")
    from . import theme
    theme.apply(app)
    app.setDesktopFileName("pathoassist")

    def excepthook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        logging.error(msg)
        QMessageBox.critical(None, "PathoAssist error",
                             f"{value}\n\nDetails were written to\n{logs_dir() / 'pathoassist.log'}")
    sys.excepthook = excepthook

    w = MainWindow()
    w.show()
    w.ask_operator()
    return app.exec()


def main() -> int:
    ap = argparse.ArgumentParser(prog="PathoAssist")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args()
    return self_test() if args.self_test else run_app()


if __name__ == "__main__":
    sys.exit(main())
