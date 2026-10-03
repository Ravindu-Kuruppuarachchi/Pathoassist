"""Application state shared by all pages: operator, open case, microscope profile and objective, latest
camera frame, model hub and the background job runner."""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QObject, QSettings, Signal

from . import __version__
from .casestore import CaseStore
from .hub import ModelHub
from .microscope import FlatField, GateState, MicroscopeProfile
from .paths import calibration_dir, cases_dir, models_dir, profiles_dir
from .ui_common import JobRunner


class AppContext(QObject):
    case_changed = Signal(object)          # BreastCase or None
    case_updated = Signal()                # results / details of the open case changed
    frame = Signal(object, object)         # rgb, GateState (live camera)
    objective_changed = Signal(str)
    profile_changed = Signal()
    status = Signal(str)
    operator_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("PathoAssist", "PathoAssist")
        self.operator = self.settings.value("operator", "", str)
        self.store = CaseStore(cases_dir())
        self.hub = ModelHub(models_dir(), calibration_dir())
        self.hub.on_loading = lambda m: self.status.emit(m)
        self.jobs = JobRunner(self)
        self.jobs.log_error = lambda tb: logging.error(tb)
        self.flat = FlatField()
        self.case = None
        self.last_frame: np.ndarray | None = None
        self.last_state: GateState | None = None
        self.live = False
        self.app_version = __version__
        self.profile = self._load_profile()
        self.objective = self.settings.value("objective", "40x", str)
        if self.objective not in self.profile.objectives:
            self.objective = self.profile.objective_names()[-1]
        self._load_blank()

    # ------------------------------------------------------------------ microscope
    def _load_profile(self) -> MicroscopeProfile:
        name = self.settings.value("profile", "", str)
        profiles = MicroscopeProfile.list(profiles_dir())
        for p in profiles:
            if p.name == name:
                return p
        if profiles:
            return profiles[0]
        p = MicroscopeProfile.new("Microscope 1", profiles_dir())
        p.save()
        return p

    def set_profile(self, p: MicroscopeProfile):
        self.profile = p
        self.settings.setValue("profile", p.name)
        if self.objective not in p.objectives:
            self.objective = p.objective_names()[-1]
        self._load_blank()
        self.profile_changed.emit()
        self.objective_changed.emit(self.objective)

    def set_objective(self, name: str):
        if name not in self.profile.objectives:
            return
        self.objective = name
        self.settings.setValue("objective", name)
        self._load_blank()
        self.objective_changed.emit(name)

    def _load_blank(self):
        b = self.profile.load_blank(self.objective)
        if b is None:
            self.flat.clear()
        else:
            try:
                self.flat.set_blank(b)
            except ValueError:
                self.flat.clear()

    def um_per_px(self, width: int | None = None) -> float | None:
        if width is None:
            width = self.last_frame.shape[1] if self.last_frame is not None else None
        if width is None:
            o = self.profile.objectives.get(self.objective)
            return o.um_per_px if o and o.calibrated else None
        return self.profile.um_per_px(self.objective, width)

    def hpf_diameter_um(self) -> float | None:
        """Equivalent diameter of a 40x camera field (calibrated), for hotspot size and the HPF counter."""
        if self.last_frame is None:
            return None
        h, w = self.last_frame.shape[:2]
        names = [n for n in self.profile.objective_names() if n.startswith("40")]
        um = self.profile.um_per_px(names[0], w) if names else None
        if not um:
            return None
        return MicroscopeProfile.field_of_view(um, w, h)["equiv_diameter_mm"] * 1000

    def frame_meta(self) -> dict:
        st = self.last_state
        return dict(objective=self.objective, microscope=self.profile.name, flat_field=self.flat.active,
                    focus=round(st.focus, 1) if st else None, operator=self.operator)

    # ------------------------------------------------------------------ case
    def set_case(self, case):
        self.case = case
        self.case_changed.emit(case)

    def save_case(self):
        if self.case is not None:
            self.case.save()
            self.case_updated.emit()

    def set_operator(self, name: str):
        self.operator = name.strip()
        self.settings.setValue("operator", self.operator)
        self.operator_changed.emit(self.operator)
