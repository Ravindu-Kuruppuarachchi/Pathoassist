"""Analysis modules: one per feature, each run separately on frames from the microscope (Qt-free).

Every module has the same life cycle, driven by the Analysis page:
    add_frame(rgb, µm/px)  -> FrameOutcome (accepted / skipped as a duplicate, message)
    render(field)          -> overlay for the field (re-drawn after corrections)
    click(field, x, y)     -> click-to-correct (logged as a training label where the engine supports it)
    summary_html()         -> running result + warnings
    save_to_case(case)     -> result written into the patient's case (audit-logged), module PDF if any

All heavy calls run on the analysis thread. Modules keep their own run folder under
<case>/modules/<key>/<run>/ so earlier runs are never overwritten.
"""
from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.desktop.casestore import FrameLogger

THUMB_W = 220


@dataclass
class FrameOutcome:
    accepted: bool
    message: str
    field_id: int | None = None


@dataclass
class FieldRec:
    id: int
    rgb: np.ndarray = field(repr=False)
    um_per_px: float
    caption: str = ""
    thumb: np.ndarray | None = field(default=None, repr=False)


def _esc(x) -> str:
    return html.escape(str(x))


def table_html(rows, head=None) -> str:
    h = "<table cellspacing='0' cellpadding='3' width='100%'>"
    if head:
        h += "<tr>" + "".join(f"<th align='left'>{_esc(c)}</th>" for c in head) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{c if isinstance(c, Markup) else _esc(c)}</td>" for c in r) + "</tr>"
    return h + "</table>"


class Markup(str):
    """Trusted HTML inside table_html."""


def warnings_html(ws) -> str:
    ws = list(dict.fromkeys(w for w in ws if w))
    if not ws:
        return ""
    return "<p style='color:#c26a00'><b>Notes</b></p><ul>" + "".join(f"<li style='color:#c26a00'>{_esc(w)}</li>" for w in ws) + "</ul>"


def resize_w(rgb, w):
    h0, w0 = rgb.shape[:2]
    return cv2.resize(rgb, (w, max(1, round(h0 * w / w0))), interpolation=cv2.INTER_AREA)


class AnalysisModule:
    key = ""
    title = ""
    group = "H&E grading"          # module list section
    short = ""                     # one-line description in the module list
    feature = ""
    stain = "H&E"                 # stain the frames are saved under
    objective = "40x"             # recommended objective
    how_to = ""                   # HTML: how to sweep the slide for this module
    click_help = ""               # HTML: click-to-correct gestures ("" = not clickable)
    options_spec: list = []       # [(name, label, default bool)]
    saves_to = ""                 # what save_to_case writes, shown on the button tooltip

    def __init__(self, hub, case, operator: str = "", options: dict | None = None, context: dict | None = None):
        self.hub, self.case, self.operator = hub, case, operator
        self.options = {n: d for n, _, d in self.options_spec}
        self.options.update(options or {})
        self.context = context or {}
        self.run_dir = case.dir / "modules" / self.key / time.strftime("run_%Y%m%d_%H%M%S")
        self.fields: dict[int, FieldRec] = {}
        self.dirty = False           # fields or corrections not yet saved to the case
        self.saved_once = False
        self.logger = FrameLogger(self.context.get("app_version", ""))
        self.last_um = None
        self._next = 0

    def _new_id(self) -> int:
        self._next += 1
        return self._next - 1

    # ---------------------------------------------------------------- frames
    def add_frame(self, rgb: np.ndarray, um_per_px: float, meta: dict | None = None, force: bool = False) -> FrameOutcome:
        out = self._add(rgb, um_per_px, force)
        if out.accepted:
            self.dirty = True
            self.last_um = um_per_px
            rec = self.fields[out.field_id]
            rec.thumb = resize_w(self.render(out.field_id), THUMB_W)
            meta = dict(meta or {})
            if not meta.pop("already_saved", False) and self.options.get("save_frames", True):
                meta.update(case_id=self.case.case_id, stain=self.stain, module=self.key, um_per_px=um_per_px,
                            field=out.field_id + 1)
                self.logger.save(FrameLogger.case_frames_dir(self.case, self.stain), rgb, meta)
        return out

    def _add(self, rgb, um, force) -> FrameOutcome:
        raise NotImplementedError

    def render(self, fid: int) -> np.ndarray:
        return self.fields[fid].rgb

    def refresh_thumb(self, fid: int):
        if fid in self.fields:
            self.fields[fid].thumb = resize_w(self.render(fid), THUMB_W)

    def click(self, fid: int, x: float, y: float, mods: set) -> str | None:
        return None

    def remove(self, fid: int):
        self.fields.pop(fid, None)
        self.dirty = True

    def minimap(self) -> np.ndarray | None:
        return None

    def summary_html(self) -> str:
        raise NotImplementedError

    def headline(self) -> dict:
        """The one result to show large: value, caption, tone (neutral/good/warn/bad) and progress towards the
        minimum the method needs, (current, target, unit) or None."""
        return dict(value="-", caption="No fields analysed yet", tone="neutral", progress=None)

    def blurb(self) -> str:
        """One plain sentence for the module list."""
        return ""

    def can_save(self) -> tuple[bool, str]:
        return (bool(self.fields), "" if self.fields else "No fields analysed yet.")

    def save_to_case(self) -> str:
        raise NotImplementedError

    def _snapshot(self, fid: int, caption: str):
        self.case.add_snapshot(resize_w(self.render(fid), 900), caption, self.key)

    def _best_field(self, key=lambda f: 0) -> int:
        return max(self.fields.values(), key=key).id

    def _nucleus_radius_px(self, um=4.0):
        return um / (self.last_um or 0.25)


# ====================================================================================== F1 nuclei
class NucleiModule(AnalysisModule):
    key, title, feature, stain, objective = "nuclei", "Nuclei", "F1", "H&E", "40x"
    group, short = "Tools", "Outline every nucleus"
    how_to = ("<p>Counts and outlines every nucleus in the field (any stain). Use it to check that detection works on "
              "your slides and as a quick cellularity measure.</p><ol><li>Use the 40x (or 20x) objective, focus, "
              "and keep the stage still until <b>READY</b>.</li><li>Press <b>Analyse</b> [Enter].</li></ol>")
    options_spec = [("save_frames", "Save analysed frames", True)]
    saves_to = "nuclei counts + a snapshot (other findings)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.det = {}

    def _add(self, rgb, um, force):
        d = self.hub.detector.detect(rgb, um_per_px=um)
        fid = self._new_id()
        area = np.bincount(d.labels.ravel(), minlength=len(d) + 1)[1:] * um ** 2 if len(d) else np.zeros(0)
        fa = rgb.shape[0] * rgb.shape[1] * (um / 1000) ** 2
        self.det[fid] = dict(n=len(d), coords=d.coords, median_area=float(np.median(area)) if len(d) else 0.0,
                             density=len(d) / fa, secs=d.timings["total"])
        self.fields[fid] = FieldRec(fid, rgb, um, f"{len(d)} nuclei")
        return FrameOutcome(True, f"{len(d)} nuclei in {d.timings['total']:.2f} s", fid)

    def render(self, fid):
        r = self.fields[fid].rgb.copy()
        polys = [np.round(c[::-1].T).astype(np.int32) for c in self.det[fid]["coords"]]
        cv2.polylines(r, polys, True, (0, 230, 0), 2, cv2.LINE_AA)
        return r

    def remove(self, fid):
        super().remove(fid)
        self.det.pop(fid, None)

    def _totals(self):
        d = [self.det[f] for f in self.fields]
        return dict(fields=len(d), nuclei=sum(x["n"] for x in d),
                    median_area_um2=round(float(np.median([x["median_area"] for x in d])), 1) if d else 0,
                    density_per_mm2=round(float(np.mean([x["density"] for x in d])), 0) if d else 0,
                    mean_seconds=round(float(np.mean([x["secs"] for x in d])), 2) if d else 0)

    def headline(self):
        t = self._totals()
        if not t["fields"]:
            return super().headline()
        return dict(value=f"{t['nuclei']:,}", caption=f"nuclei in {t['fields']} field{'s' * (t['fields'] > 1)}, "
                    f"median area {t['median_area_um2']} µm²", tone="neutral", progress=None)

    def summary_html(self):
        t = self._totals()
        rows = [("Fields", t["fields"]), ("Nuclei", t["nuclei"]), ("Median nuclear area", f"{t['median_area_um2']} µm²"),
                ("Mean density", f"{t['density_per_mm2']:.0f} / mm²"), ("Detection time", f"{t['mean_seconds']} s / frame")]
        return table_html(rows)

    def save_to_case(self):
        t = self._totals()
        self._snapshot(self._best_field(lambda f: self.det[f.id]["n"]), "Nuclei detection (outlines)")
        s = f"{t['nuclei']} nuclei in {t['fields']} fields; median area {t['median_area_um2']} µm²"
        self.case.set_extra("Nuclei", t, s)
        return s


# ====================================================================================== F2-F4 IHC (shared)
class _IHCModule(AnalysisModule):
    objective = "40x"
    session_cls = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.session = self._make_session()
        self.analyzer = None

    def _make_session(self):
        raise NotImplementedError

    def _analyzer(self):
        raise NotImplementedError

    def _add(self, rgb, um, force):
        if self.analyzer is None:
            self.analyzer = self._analyzer()
        a = self.analyzer.analyze(rgb, um)
        r = self.session.add_field(rgb, a, force=force)
        if not r.accepted:
            return FrameOutcome(False, f"Skipped: overlaps field {r.duplicate_of + 1} by {r.overlap:.0%}"
                                       ". Move to a new field, or choose Count anyway.")
        self.fields[r.field_index] = FieldRec(r.field_index, rgb, um, self._caption(r.field_index))
        self._after_add(r.field_index, rgb, um)
        return FrameOutcome(True, f"Field {r.field_index + 1}: {self._caption(r.field_index)}", r.field_index)

    def _after_add(self, fid, rgb, um):
        pass

    def _caption(self, fid) -> str:
        return ""

    def refresh_thumb(self, fid):
        if fid in self.fields:
            self.fields[fid].caption = self._caption(fid)
        super().refresh_thumb(fid)

    def remove(self, fid):
        super().remove(fid)
        self.session.remove_field(fid)

    def _hit(self, fid, x, y):
        return self.session.nucleus_at(fid, y, x, radius=self._nucleus_radius_px(4.0))

    def _save_common(self, report_fn):
        self.session.save()
        pdf = report_fn()
        return pdf


class ERPRModule(_IHCModule):
    feature, objective = "F2", "40x"
    group = "Immunohistochemistry"
    how_to = ("<ol><li>Use the <b>40x</b> objective (20x is acceptable if calibrated).</li>"
              "<li>Visit representative fields of <b>invasive tumour</b> only - avoid DCIS, normal ducts, edges and "
              "crush artefact.</li><li>In each field wait for <b>READY</b>, then press <b>Analyse</b> [Enter], or tick "
              "<i>Auto-analyse when READY</i> and simply move from field to field.</li>"
              "<li>Continue until at least <b>500 tumour nuclei</b> are counted (the counter warns until then).</li>"
              "<li>Check each field: click a nucleus to correct it, then <b>Save result to case</b>.</li></ol>")
    click_help = ("<b>Click</b> a nucleus: cycle intensity negative → weak → moderate → strong. "
                  "<b>Shift+click</b>: tumour ↔ non-tumour (excluded from the count, grey).")
    options_spec = [("save_frames", "Save analysed frames", True),
                    ("tumour_filter", "Tumour nuclei only (experimental)", False)]

    def _make_session(self):
        from core.ihc.case import CaseSession
        return CaseSession(self.run_dir, self.case.case_id, self.stain)

    def _analyzer(self):
        from core.ihc.pipeline import ERPRAnalyzer
        th, self.th_source = self.hub.erpr_thresholds()
        clf = self.hub.celltype if self.options.get("tumour_filter") else None
        return ERPRAnalyzer(self.hub.detector, clf, thresholds=th)

    def _caption(self, fid):
        s = [f for f in self.session.field_scores()]
        i = [f.index for f in self.session.fields].index(fid)
        return f"{s[i].pct_positive:.0f}% pos, {s[i].n_tumour} nuclei"

    def render(self, fid):
        from core.ihc.overlay import draw_erpr_overlay
        f = self.session.field(fid)
        return draw_erpr_overlay(self.fields[fid].rgb, f.analysis, thickness=2, corrected=f.corrected)

    def click(self, fid, x, y, mods):
        i = self._hit(fid, x, y)
        if i is None:
            return None
        if "shift" in mods:
            self.session.toggle_tumour(fid, i, self.operator)
            msg = "tumour/non-tumour toggled"
        else:
            a = self.session.field(fid).analysis
            new = (int(a.intensity[i]) + 1) % 4
            self.session.set_intensity(fid, i, new, self.operator)
            from core.ihc.scoring import INTENSITY_NAMES
            msg = f"intensity → {INTENSITY_NAMES[new]}"
        self.dirty = True
        return msg

    def headline(self):
        if not self.session.fields:
            return super().headline()
        s = self.session.score()
        tone = {"Positive": "good", "Low positive": "warn", "Negative": "neutral"}.get(s.status, "warn")
        return dict(value=s.status, caption=f"{s.pct_positive:.1f}% positive, Allred {s.allred_total}, H-score {s.h_score:.0f}",
                    tone=tone, progress=(s.n_tumour, self.session.min_cells, "tumour nuclei"))

    def summary_html(self):
        from core.ihc.scoring import INTENSITY_NAMES
        s = self.session.score()
        th = getattr(self, "th_source", self.hub.erpr_thresholds()[1])
        rows = [("Status", Markup(f"<b>{_esc(s.status)}</b>")), ("% positive tumour nuclei", f"{s.pct_positive:.1f} %"),
                ("H-score", f"{s.h_score:.0f}"), ("Allred", f"{s.allred_total} (PS {s.allred_ps} + IS {s.allred_is})"),
                ("Tumour nuclei counted", s.n_tumour), ("Fields", len(self.session.fields))]
        rows += [(f"  {n}", f"{p:.1f} %") for n, p in zip(INTENSITY_NAMES, s.pct_by_bin)]
        notes = list(s.warnings) + [f"Intensity thresholds: {th}."]
        if not self.options.get("tumour_filter"):
            notes.append("Tumour filter off: every detected nucleus is counted - exclude non-tumour cells with Shift+click.")
        return table_html(rows) + warnings_html(notes)

    def save_to_case(self):
        from core.ihc.report import write_report
        pdf = self._save_common(lambda: write_report(self.session, self.run_dir / f"{self.stain}_{self.case.case_id}.pdf",
                                                      pathologist=self.operator))
        s = self.session.score()
        th = getattr(self, "th_source", "")
        self.case.set_biomarker(self.stain, s, validated=False,
                                method=f"DAB nuclear OD ({th}), {'tumour nuclei' if self.options.get('tumour_filter') else 'all nuclei'}")
        self._snapshot(self._best_field(lambda f: len(self.session.field(f.id).analysis)), f"{self.stain} - counted field")
        return f"{self.stain}: {s.status}, {s.pct_positive}% positive, Allred {s.allred_total}, H-score {s.h_score} (PDF {pdf.name})"


class ERModule(ERPRModule):
    key, title, stain = "ER", "ER", "ER"
    short = "Oestrogen receptor, Allred, H-score"
    saves_to = "ER result (biomarkers)"


class PRModule(ERPRModule):
    key, title, stain = "PR", "PR", "PR"
    short = "Progesterone receptor, Allred"
    saves_to = "PR result (biomarkers)"


class Ki67Module(_IHCModule):
    key, title, feature, stain, objective = "Ki67", "Ki-67", "F3", "Ki67", "40x"
    group, short = "Immunohistochemistry", "Proliferation index"
    how_to = ("<ol><li>Use the <b>40x</b> objective on invasive tumour.</li>"
              "<li><b>Global</b> (default, IKWG): visit fields across the whole tumour - hot, average and cold areas - "
              "in proportion to their extent.</li><li><b>Hotspot</b>: scan at 10x to find the area with the most "
              "positive nuclei, then count adjacent 40x fields there.</li>"
              "<li>Analyse each field when <b>READY</b> until at least <b>500 tumour nuclei</b>; correct by clicking.</li></ol>")
    click_help = ("<b>Click</b> a nucleus: positive ↔ negative. <b>Shift+click</b>: tumour ↔ non-tumour.")
    options_spec = [("save_frames", "Save analysed frames", True),
                    ("tumour_filter", "Tumour nuclei only (experimental)", False),
                    ("hotspot_primary", "Report the hotspot index", False)]
    saves_to = "Ki-67 result (biomarkers)"

    def _make_session(self):
        from core.ki67.session import Ki67Session
        return Ki67Session(self.run_dir, self.case.case_id, celltype_filter=bool(self.options.get("tumour_filter")))

    def _analyzer(self):
        from core.ki67.session import ki67_analyzer
        p, self.th_source = self.hub.ki67_positivity()
        clf = self.hub.celltype if self.options.get("tumour_filter") else None
        return ki67_analyzer(self.hub.detector, clf, positivity=p)

    def _caption(self, fid):
        i = [f.index for f in self.session.fields].index(fid)
        c = self.session.field_counts()[i]
        return f"{100 * c[1] / max(c.sum(), 1):.0f}% ({c[1]}/{c.sum()})"

    def render(self, fid):
        from core.ki67.report import draw_ki67_overlay
        f = self.session.field(fid)
        return draw_ki67_overlay(self.fields[fid].rgb, f.analysis, corrected=f.corrected)

    def click(self, fid, x, y, mods):
        i = self._hit(fid, x, y)
        if i is None:
            return None
        if "shift" in mods:
            self.session.toggle_tumour(fid, i, self.operator)
            msg = "tumour/non-tumour toggled"
        else:
            self.session.toggle_positive(fid, i, self.operator)
            msg = "positive/negative toggled"
        self.dirty = True
        return msg

    def headline(self):
        if not self.session.fields:
            return super().headline()
        hot = self.options.get("hotspot_primary")
        r = self.session.hotspot_result() if hot else self.session.global_result()
        tone = "warn" if r.band.startswith("Intermediate") else "neutral"
        return dict(value=f"{r.index_pct:.1f}%", caption=f"{'Hotspot' if hot else 'Global'} index, {r.band.lower()}",
                    tone=tone, progress=(r.n_counted, self.session.min_cells, "tumour nuclei"))

    def summary_html(self):
        g, h = self.session.global_result(), self.session.hotspot_result()
        primary = "hotspot" if self.options.get("hotspot_primary") else "global"
        main = h if primary == "hotspot" else g
        rows = [("Ki-67 index (" + primary + ")", Markup(f"<b>{main.index_pct:.1f} %</b>")), ("Range (IKWG)", main.band),
                ("Global", f"{g.index_pct:.1f} % ({g.n_positive}/{g.n_counted})"),
                ("Hotspot", f"{h.index_pct:.1f} % ({h.n_positive}/{h.n_counted}, fields {', '.join(str(i + 1) for i in h.fields_used)})"),
                ("Fields", len(self.session.fields))]
        th = getattr(self, "th_source", self.hub.ki67_positivity()[1])
        return table_html(rows) + warnings_html(g.warnings + h.warnings + [f"Positivity cut-off: {th}."])

    def save_to_case(self):
        from core.ki67.report import write_ki67_report
        primary = "hotspot" if self.options.get("hotspot_primary") else "global"
        pdf = self._save_common(lambda: write_ki67_report(self.session, self.run_dir / f"Ki67_{self.case.case_id}.pdf",
                                                          pathologist=self.operator, primary=primary))
        g, h = self.session.global_result().as_dict(), self.session.hotspot_result().as_dict()
        res = {"global": g, "hotspot": h, "primary": primary}
        self.case.set_biomarker("Ki67", res, validated=False, method=f"DAB positivity, {primary} index reported")
        self._snapshot(self._best_field(lambda f: len(self.session.field(f.id).analysis)), "Ki-67 - counted field")
        return f"Ki-67 {res[primary]['index_pct']}% ({primary}) (PDF {pdf.name})"


class HER2Module(_IHCModule):
    key, title, feature, stain, objective = "HER2", "HER2", "F4", "HER2", "20x"
    group, short = "Immunohistochemistry", "Membrane score 0 to 3+"
    how_to = ("<ol><li>Scan the whole invasive tumour at <b>10x</b> first to judge heterogeneity, then score fields at "
              "<b>20x or 40x</b> (calibrated).</li><li>Score invasive tumour only - avoid DCIS (often strongly "
              "positive), edges with edge artefact, and crushed areas.</li><li>Analyse fields when <b>READY</b> until "
              "≥ <b>500 tumour cells</b>. Watch the 0 vs 1+ boundary (any staining 10%).</li>"
              "<li>2+ means <b>reflex ISH</b>. Check flagged fields for cytoplasmic or background staining.</li></ol>")
    click_help = ("<b>Click</b> a cell: cycle membrane category none → faint/incomplete → weak-moderate complete → "
                  "strong complete. <b>Shift+click</b>: include/exclude the cell.")
    options_spec = [("save_frames", "Save analysed frames", True),
                    ("cross_check", "Cross-check with field classifier", True)]
    saves_to = "HER2 result (biomarkers)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.field_probs = {}

    def _make_session(self):
        from core.her2.session import HER2Session
        return HER2Session(self.run_dir, self.case.case_id)

    def _analyzer(self):
        from core.her2.pipeline import HER2Analyzer
        th, self.th_source = self.hub.her2_thresholds()
        return HER2Analyzer(self.hub.detector, None, thresholds=th)

    def _after_add(self, fid, rgb, um):
        if self.options.get("cross_check"):
            self.field_probs[fid] = self.hub.her2_field.predict(rgb, um)

    def remove(self, fid):
        super().remove(fid)
        self.field_probs.pop(fid, None)

    def _caption(self, fid):
        from core.her2.pipeline import field_counts
        fc, _ = field_counts(self.session.field(fid).analysis)
        n = max(fc.sum(), 1)
        return f"{fc.sum()} cells, any {100 * fc[1:].sum() / n:.0f}%, complete {100 * fc[2:].sum() / n:.0f}%"

    def render(self, fid):
        from core.her2.session import draw_her2_overlay
        f = self.session.field(fid)
        return draw_her2_overlay(self.fields[fid].rgb, f.analysis, f.corrected)

    def click(self, fid, x, y, mods):
        i = self.session.nucleus_at(fid, y, x, radius=self._nucleus_radius_px(4.0))
        if i is None:
            return None
        a = self.session.field(fid).analysis
        if "shift" in mods:
            self.session.toggle_tumour(fid, i, self.operator)
            msg = "cell included/excluded"
        else:
            from core.her2.scoring import CELL_CATEGORIES
            new = (int(a.intensity[i]) + 1) % 4
            self.session.set_intensity(fid, i, new, self.operator)
            a.measurable[i] = True           # a cell the pathologist categorised is counted
            msg = f"membrane → {CELL_CATEGORIES[new]}"
        self.dirty = True
        return msg

    def _check(self, res):
        from core.her2.field_classifier import cross_check
        return cross_check(res.score, list(self.field_probs.values()))

    def headline(self):
        if not self.session.fields:
            return super().headline()
        r = self.session.result()
        sc = r.score + (f" {r.subgroup}" if r.subgroup else "")
        tone = {"3+": "bad", "2+": "warn"}.get(r.score, "neutral")
        cap = r.her2_low + (", reflex ISH required" if r.reflex_ish else "")
        return dict(value=sc, caption=cap, tone=tone, progress=(r.n_cells, self.session.min_cells, "tumour cells"))

    def summary_html(self):
        res = self.session.result()
        cc = self._check(res)
        score = res.score + (f" ({res.subgroup})" if res.subgroup else "")
        rows = [("HER2 IHC score", Markup(f"<b>{_esc(score)}</b>")), ("Category", res.her2_low),
                ("Reflex ISH", Markup("<b style='color:#c00'>REQUIRED</b>") if res.reflex_ish else "not required"),
                ("Tumour cells", f"{res.n_cells} in {len(self.session.fields)} fields"),
                ("Any membrane staining", f"{res.pct_any_staining:.1f} %  (0/1+ cut 10%)"),
                ("Complete staining", f"{res.pct_complete:.1f} %"), ("Strong complete", f"{res.pct[3]:.1f} %")]
        if cc["classifier_score"]:
            rows.append(("Field classifier", f"{cc['classifier_score']} ({'agrees' if cc['agrees'] else 'DISAGREES'})"))
        th = getattr(self, "th_source", self.hub.her2_thresholds()[1])
        notes = list(res.flags) + ([cc["note"]] if cc["note"] else []) + [f"Membrane thresholds: {th}."]
        if not res.rules_verified:
            notes.append(f"Scoring rules ({res.rules_version}) not yet verified by a pathologist for this installation.")
        return table_html(rows) + warnings_html(notes)

    def save_to_case(self):
        from core.her2.session import write_her2_report
        pdf = self._save_common(lambda: write_her2_report(self.session, self.run_dir / f"HER2_{self.case.case_id}.pdf",
                                                          pathologist=self.operator))
        res = self.session.result()
        cc = self._check(res)
        if cc["note"]:
            res.flags.append(cc["note"])
        d = res.as_dict()
        d["field_classifier"] = cc
        self.case.set_biomarker("HER2", d, validated=False, method="membrane ring, ASCO/CAP rules")
        self._snapshot(self._best_field(lambda f: len(self.session.field(f.id).analysis)), "HER2 - counted field")
        return f"HER2 {res.score}{' (' + res.subgroup + ')' if res.subgroup else ''} (PDF {pdf.name})"


# ====================================================================================== F5 region + slide map
class RegionModule(AnalysisModule):
    key, title, feature, stain, objective = "region", "Tumour map", "F5", "H&E", "10x"
    short = "Tumour map and hotspots"
    how_to = ("<ol><li>Use the <b>10x</b> (or 4x) objective on the H&amp;E slide.</li>"
              "<li>Sweep the tissue in overlapping rows (about one third overlap). Tick <i>Auto-analyse when READY</i> "
              "and pause briefly on each field.</li><li>The live frame is coloured: <span style='color:#d00'>invasive"
              "</span>, <span style='color:#909'>in situ</span>, <span style='color:#0a0'>benign</span>. The mini-map "
              "shows covered tissue, tumour (red) and the top mitosis hotspots (▼) to visit for the 10-HPF count.</li></ol>")
    options_spec = [("save_frames", "Save analysed frames", False),
                    ("hotspots", "Find mitosis hotspots", True)]
    saves_to = "region summary + mini-map (other findings)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from core.region.mosaic import Mosaic
        from core.mitosis.hotspot import HotspotMap
        self.mosaic = Mosaic()
        self.hmap = HotspotMap()
        self.maps, self.spots, self.places, self.fracs = {}, {}, {}, {}

    def _hpf_um(self):
        return float(self.context.get("hpf_diameter_um") or 500.0)

    def _add(self, rgb, um, force):
        p = self.mosaic.add_frame(rgb, um)
        if p.status == "no_tissue":
            return FrameOutcome(False, "No tissue in this field - skipped")
        rm = self.hub.region.predict(rgb, um)
        fid = self._new_id()
        self.mosaic.put_overlay(p, rm.tumour_prob())
        spots = []
        if self.options.get("hotspots"):
            spots = self.hub.hotspot.hotspots(rgb, um, self._hpf_um(), top_k=3)
            self.hmap.add(fid, spots, um, p.um(self.mosaic.map_um))
        self.maps[fid], self.spots[fid], self.places[fid], self.fracs[fid] = rm, spots, p, rm.fractions()
        inv = self.fracs[fid].get("invasive", 0)
        self.fields[fid] = FieldRec(fid, rgb, um, f"invasive {100 * inv:.0f}% ({p.status})")
        return FrameOutcome(True, f"Field {fid + 1}: invasive {100 * inv:.0f}% of tissue, map {p.status}", fid)

    def render(self, fid):
        from core.region.tissue_map import draw_region_overlay
        r = draw_region_overlay(self.fields[fid].rgb, self.maps[fid])
        um = self.fields[fid].um_per_px
        rad = int(self._hpf_um() / 2 / um)
        for s in self.spots.get(fid, []):
            cv2.circle(r, (int(s.x), int(s.y)), rad, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(r, f"#{s.rank}", (int(s.x) - 15, int(s.y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        return r

    def remove(self, fid):
        super().remove(fid)      # the mosaic keeps the frame (it only maps where the slide was seen)

    def minimap(self, width=360):
        markers = []
        for x, y, fid, s in self.hmap.ranked(5):
            if x is not None and fid in self.places:
                markers.append((x, y, self.places[fid].island, (255, 200, 0)))
        mm = self.mosaic.minimap(1200, markers=markers)
        ys, xs = np.nonzero((np.abs(mm.astype(int) - 235).sum(-1) > 12))      # crop to what was mapped
        if len(ys):
            pad = 12
            mm = mm[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
        return resize_w(mm, width)

    def _totals(self):
        fr = [self.fracs[f] for f in self.fields]
        mean = {k: round(float(np.mean([f.get(k, 0) for f in fr])) * 100, 1) for k in (fr[0] if fr else {})}
        spots = [dict(field=fid + 1, x=round(s.x), y=round(s.y), score=round(s.expected_count, 2))
                 for _, _, fid, s in self.hmap.ranked(5)]
        return dict(fields=len(fr), coverage_mm2=round(self.mosaic.coverage_mm2(), 2), islands=self.mosaic.n_islands,
                    tissue_pct=mean, hotspots=spots)

    def headline(self):
        if not self.fields:
            return super().headline()
        t = self._totals()
        inv = t["tissue_pct"].get("invasive", 0)
        return dict(value=f"{inv:.0f}% invasive", caption=f"{t['coverage_mm2']} mm² of tissue mapped in {t['fields']} fields",
                    tone="neutral", progress=None)

    def summary_html(self):
        t = self._totals()
        rows = [("Fields", t["fields"]), ("Area covered", f"{t['coverage_mm2']} mm²"), ("Map pieces", t["islands"])]
        rows += [(f"  {k.replace('_', ' ')}", f"{v:.0f} % of tissue") for k, v in t["tissue_pct"].items()]
        if t["hotspots"]:
            rows.append(("Top hotspots", ", ".join(f"field {h['field']}" for h in t["hotspots"][:3])))
        notes = ["Region model trained on public data (BACH); check the map against the slide."]
        if not self.context.get("hpf_diameter_um"):
            notes.append("40x not calibrated: hotspot size assumes a 0.5 mm HPF.")
        return table_html(rows) + warnings_html(notes)

    def save_to_case(self):
        t = self._totals()
        mm = self.minimap(900)
        self.case.add_snapshot(mm, "Slide map: covered tissue, tumour (red), hotspots", self.key)
        best = self._best_field(lambda f: self.fracs[f.id].get("invasive", 0))
        self._snapshot(best, "Tumour region (10x)")
        inv = t["tissue_pct"].get("invasive", 0)
        s = f"{t['fields']} fields, {t['coverage_mm2']} mm² covered; invasive {inv:.0f}% of tissue seen"
        self.case.set_extra("Tumour region", t, s)
        return s


# ====================================================================================== F6 mitoses
class MitosisModule(AnalysisModule):
    key, title, feature, stain, objective = "mitoses", "Mitoses", "F6", "H&E", "40x"
    short = "Count in 10 high-power fields"
    how_to = ("<ol><li>Find the most mitotically active area at 10x (use <i>Tumour region &amp; slide map</i> "
              "hotspots, ▼).</li><li>Switch to <b>40x</b> (calibrated - the HPF area sets the score thresholds).</li>"
              "<li>Analyse <b>10 consecutive non-overlapping fields</b> at the invasive edge / hotspot. Each accepted "
              "frame is one HPF; overlapping frames are refused and a mitosis seen twice is counted once.</li>"
              "<li>Review every circle: click to reject a false call; Ctrl+click to add a missed mitosis.</li></ol>")
    click_help = ("<b>Click</b> a circle: reject ↔ accept. <b>Ctrl+click</b> (or Shift+click) on a missed mitosis: add it. "
                  "Orange = automatic, green = accepted, cyan = added, grey X = rejected.")
    options_spec = [("save_frames", "Save analysed frames", True)]
    saves_to = "mitotic score (grade component)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.counter = None

    def _add(self, rgb, um, force):
        from core.mitosis.counting import HPFCounter
        if self.counter is None:
            self.counter = HPFCounter(um, rgb.shape[:2])
        elif abs(um / self.counter.um_per_px - 1) > 0.05 or rgb.shape[:2] != self.counter.hpfs[0].shape:
            return FrameOutcome(False, "This frame has a different objective or resolution from HPF 1 - "
                                       "count all 10 HPFs with the same objective (start a new run to change).")
        cands, _ = self.hub.mitosis.detect(rgb, um)
        r = self.counter.add_frame(rgb, cands)
        if not r["accepted"]:
            return FrameOutcome(False, "Skipped: " + r["reason"])
        fid = r["hpf"]
        self.fields[fid] = FieldRec(fid, rgb, um, "")
        self.fields[fid].caption = self._caption(fid)
        dup = f", {len(r['duplicates'])} already counted" if r["duplicates"] else ""
        return FrameOutcome(True, f"HPF {fid + 1}/{self.counter.n_hpf}: {len(r['new'])} new mitoses{dup}", fid)

    def _caption(self, fid):
        n = sum(m.counted for m in self.counter.in_hpf(fid))
        return f"HPF {fid + 1}: {n} mitoses"

    def refresh_thumb(self, fid):
        if fid in self.fields:
            self.fields[fid].caption = self._caption(fid)
        super().refresh_thumb(fid)

    def render(self, fid):
        r = self.fields[fid].rgb.copy()
        rad = max(8, int(10 / self.fields[fid].um_per_px))
        for m in self.counter.in_hpf(fid):
            c = {"auto": (255, 140, 0), "accepted": (0, 200, 0), "added": (0, 220, 255), "rejected": (150, 150, 150)}[m.status]
            p = (int(m.x), int(m.y))
            cv2.circle(r, p, rad, c, 3, cv2.LINE_AA)
            if m.status == "rejected":
                d = int(rad * 0.7)
                cv2.line(r, (p[0] - d, p[1] - d), (p[0] + d, p[1] + d), c, 3)
                cv2.line(r, (p[0] - d, p[1] + d), (p[0] + d, p[1] - d), c, 3)
        return r

    def click(self, fid, x, y, mods):
        um = self.fields[fid].um_per_px
        ms = self.counter.in_hpf(fid)
        near = [(np.hypot(m.x - x, m.y - y), m) for m in ms]
        near = [t for t in near if t[0] <= 14 / um]
        if ("ctrl" in mods or "shift" in mods) and not near:
            self.counter.add_manual(fid, x, y)
            self.dirty = True
            return "mitosis added"
        if not near:
            return None
        m = min(near, key=lambda t: t[0])[1]
        self.counter.set_status(m.id, "accepted" if m.status == "rejected" else "rejected")
        self.dirty = True
        return f"mitosis {m.status}"

    def remove(self, fid):
        raise ValueError("HPFs cannot be removed one by one (the count is registered across them); "
                         "reject its mitoses or start a new run.")

    def headline(self):
        if self.counter is None or not self.counter.hpfs:
            return super().headline()
        r = self.counter.result()
        return dict(value=f"Score {r.get('score', '-')}", caption=f"{r['mitoses']} mitoses in {r['hpfs_counted']} HPF",
                    tone="neutral", progress=(r["hpfs_counted"], self.counter.n_hpf, "HPF"))

    def summary_html(self):
        if self.counter is None or not self.counter.hpfs:
            return "<p>No HPF counted yet.</p>"
        r = self.counter.result()
        rows = [("Mitotic score", Markup(f"<b>{r.get('score', '-')}</b>")),
                ("Mitoses", f"{r['mitoses']} in {r['hpfs_counted']} HPF"),
                ("Per mm²", r["per_mm2"]), ("HPF diameter", f"{r['field_diameter_mm']} mm"),
                ("Score 1 / 3 thresholds", f"≤ {r.get('score1_max')} / ≥ {r.get('score3_min')} per 10 HPF"),
                ("Rejected / added", f"{r['rejected']} / {r['added']}")]
        return table_html(rows) + warnings_html(r["warnings"] + r.get("notes", []))

    def can_save(self):
        if self.counter is None or not self.counter.hpfs:
            return False, "No HPF counted yet."
        return True, ""

    def save_to_case(self):
        from core.grading.case import store_component
        r = store_component(self.case, "mitoses", self.counter)
        best = self._best_field(lambda f: sum(m.counted for m in self.counter.in_hpf(f.id)))
        self._snapshot(best, f"Mitoses - HPF {best + 1}")
        return f"Mitotic score {r['score']} ({r['mitoses']} mitoses / {r['hpfs_counted']} HPF)"


# ====================================================================================== F7 pleomorphism
class PleomorphismModule(AnalysisModule):
    key, title, feature, stain, objective = "pleomorphism", "Pleomorphism", "F7", "H&E", "40x"
    short = "Nuclear size and shape"
    how_to = ("<ol><li>Use the <b>40x</b> objective on invasive tumour (not DCIS, not normal ducts).</li>"
              "<li>Analyse at least <b>3 representative fields</b>, including the most atypical area.</li>"
              "<li>Red circles = tumour nuclei measured, grey = other cells. Click a nucleus to include/exclude it.</li>"
              "<li>Use the measurements (size vs lymphocytes, anisonucleosis) with the score.</li></ol>")
    click_help = "<b>Click</b> a nucleus: tumour (measured) ↔ excluded."
    options_spec = [("save_frames", "Save analysed frames", True)]
    saves_to = "pleomorphism score (grade component)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.session = None

    def _pos(self, fid):
        return [f.index for f in self.session.fields].index(fid)

    def _add(self, rgb, um, force):
        from core.pleomorphism.scorer import PleomorphismSession
        if self.session is None:
            self.session = PleomorphismSession(self.hub.pleomorphism)
        r = self.session.add_field(rgb, um)
        if not r["accepted"]:
            return FrameOutcome(False, "Skipped: " + r["reason"])
        f = self.session.fields[-1]
        self.fields[f.index] = FieldRec(f.index, rgb, um, "")
        self.fields[f.index].caption = self._caption(f.index)
        return FrameOutcome(True, f"Field {f.index + 1}: {f.n_tumour} tumour nuclei", f.index)

    def _caption(self, fid):
        f = self.session.fields[self._pos(fid)]
        s = self.session.scorer.score_from_probs(f.probs)
        return f"score {s or '-'}, {int((f.cell_type == 0).sum())} tumour nuclei"

    def refresh_thumb(self, fid):
        if fid in self.fields:
            self.fields[fid].caption = self._caption(fid)
        super().refresh_thumb(fid)

    def render(self, fid):
        from core.pleomorphism.scorer import TUMOUR
        f = self.session.fields[self._pos(fid)]
        r = self.fields[fid].rgb.copy()
        rad = max(4, int(3.5 / self.fields[fid].um_per_px))
        for (y, x), c in zip(f.points.astype(int), f.cell_type):
            cv2.circle(r, (int(x), int(y)), rad, (220, 0, 0) if c == TUMOUR else (150, 150, 150), 2, cv2.LINE_AA)
        return r

    def click(self, fid, x, y, mods):
        from core.pleomorphism.scorer import LYMPHOCYTE, TUMOUR
        pos = self._pos(fid)
        f = self.session.fields[pos]
        if not len(f.points):
            return None
        d = np.hypot(f.points[:, 0] - y, f.points[:, 1] - x)
        i = int(d.argmin())
        if d[i] > 6 / self.fields[fid].um_per_px:
            return None
        new = LYMPHOCYTE if f.cell_type[i] == TUMOUR else TUMOUR
        self.session.set_cell_type(pos, i, new)
        self.dirty = True
        return "nucleus excluded" if new != TUMOUR else "nucleus included"

    def remove(self, fid):
        pos = self._pos(fid)
        self.session.fields.pop(pos)
        self.session.dedup.remove(pos)
        super().remove(fid)

    def headline(self):
        if self.session is None or not self.session.fields:
            return super().headline()
        r = self.session.result()
        return dict(value=f"Score {r.score or '-'}", caption=f"{r.n_tumour} tumour nuclei measured",
                    tone="warn" if len(set(r.field_scores)) > 1 else "neutral",
                    progress=(r.n_fields, self.session.min_fields, "fields"))

    def summary_html(self):
        if self.session is None or not self.session.fields:
            return "<p>No field analysed yet.</p>"
        from core.pleomorphism.scorer import MEASUREMENTS
        r = self.session.result()
        rows = [("Pleomorphism score", Markup(f"<b>{r.score or '-'}</b>")), ("P(1 / 2 / 3)", " / ".join(f"{p:.2f}" for p in r.probs)),
                ("Fields / tumour nuclei", f"{r.n_fields} / {r.n_tumour}"), ("Field scores", ", ".join(map(str, r.field_scores)))]
        rows += [(MEASUREMENTS[k], v) for k, v in r.measurements.items()]
        return table_html(rows) + warnings_html(r.warnings)

    def can_save(self):
        if self.session is None or not self.session.fields:
            return False, "No field analysed yet."
        if self.session.result().score not in (1, 2, 3):
            return False, "No score could be computed (no tumour nuclei measured)."
        return True, ""

    def save_to_case(self):
        from core.grading.case import store_component
        r = store_component(self.case, "pleomorphism", self.session)
        self._snapshot(self._best_field(lambda f: self.session.fields[self._pos(f.id)].n_tumour), "Nuclear pleomorphism (40x)")
        return f"Pleomorphism score {r.score} ({r.n_fields} fields)"


# ====================================================================================== F8 tubules
class TubulesModule(AnalysisModule):
    key, title, feature, stain, objective = "tubules", "Tubules", "F8", "H&E", "10x"
    short = "Gland formation"
    how_to = ("<ol><li>Use the <b>10x</b> objective and sweep the <b>whole invasive tumour</b> in non-overlapping "
              "fields (the score is accumulated over the sweep, not per field).</li>"
              "<li>Red outline = invasive tumour epithelium, green = tubules around true lumina.</li>"
              "<li>Cover at least 1 mm² of tumour. The score is <b>not validated</b> yet (see notes) - "
              "confirm or override it on the Grade &amp; Report page.</li></ol>")
    options_spec = [("save_frames", "Save analysed frames", True)]
    saves_to = "tubule score (grade component)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.session = None
        self.fmap = {}            # fid -> TubuleField

    def _add(self, rgb, um, force):
        from core.tubules.score import TubuleSession
        if self.session is None:
            self.session = TubuleSession(self.hub.lumen, self.hub.region)
        r = self.session.add_field(rgb, um)
        if not r["accepted"]:
            return FrameOutcome(False, "Skipped: " + r["reason"])
        fid = self._new_id()
        f = self.session.fields[-1]
        self.fmap[fid] = f
        pct = 100 * f.tubule_area_mm2 / f.tumour_area_mm2 if f.tumour_area_mm2 else 0
        self.fields[fid] = FieldRec(fid, rgb, um, f"tubules {pct:.0f}%, {r['lumens']} lumina")
        return FrameOutcome(True, f"Field {fid + 1}: tumour {r['tumour_mm2']:.2f} mm², tubules {pct:.0f}%", fid)

    def render(self, fid):
        from core.tubules.score import draw_tubule_overlay
        return draw_tubule_overlay(self.fields[fid].rgb, self.fmap[fid])

    def remove(self, fid):
        f = self.fmap.pop(fid)
        s = self.session
        pos = s.fields.index(f)
        s.fields.pop(pos)
        s.dedup.remove(pos)
        s.tum_area -= f.tumour_area_mm2
        s.tub_area -= f.tubule_area_mm2
        s.n_fields -= 1
        s.n_lumens -= int(f.lumens.sum())
        super().remove(fid)

    def headline(self):
        if self.session is None or not self.session.fields:
            return super().headline()
        r = self.session.result()
        return dict(value=f"Score {r.score or '-'}", caption=f"{r.tubule_pct:.0f}% of {r.tumour_area_mm2:.2f} mm² tumour forms tubules",
                    tone="warn", progress=(round(r.tumour_area_mm2, 2), self.session.min_tumour_mm2, "mm² tumour"))

    def summary_html(self):
        if self.session is None or not self.session.fields:
            return "<p>No field analysed yet.</p>"
        r = self.session.result()
        rows = [("Tubule score", Markup(f"<b>{r.score or '-'}</b>")), ("Tubule formation", f"{r.tubule_pct:.1f} % of tumour"),
                ("Invasive tumour assessed", f"{r.tumour_area_mm2:.2f} mm²"), ("Fields / lumina", f"{r.n_fields} / {r.n_lumens}")]
        return table_html(rows) + warnings_html(r.warnings)

    def can_save(self):
        if self.session is None or not self.session.fields:
            return False, "No field analysed yet."
        if self.session.result().score not in (1, 2, 3):
            return False, "No invasive tumour found in the analysed fields."
        return True, ""

    def save_to_case(self):
        from core.grading.case import store_component
        r = store_component(self.case, "tubules", self.session)
        self._snapshot(self._best_field(lambda f: self.fmap[f.id].tumour_area_mm2), "Tubule formation (10x)")
        return f"Tubule score {r.score} ({r.tubule_pct:.0f}% of {r.tumour_area_mm2:.2f} mm²)"


MODULES = [ERModule, PRModule, Ki67Module, HER2Module, RegionModule, MitosisModule, PleomorphismModule,
           TubulesModule, NucleiModule]
MODULE_BY_KEY = {m.key: m for m in MODULES}
