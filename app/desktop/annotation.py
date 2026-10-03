"""Pathologist annotation for new training data (Qt-free): tasks, documents, model pre-fill, export, and
re-calibration of the scoring thresholds from the labels.

One task per model or threshold that needs local data (the "next step" column of the README):

    nuclei           F1  polygon outline of every nucleus          -> instance mask (StarDist fine-tuning)
    celltype         F2  point per nucleus: tumour/lymph/stroma/other -> cell-type classifier
    erpr_intensity   F2  point per tumour nucleus: neg/weak/mod/strong -> ER/PR intensity cut-offs
    ki67_positivity  F3  point per tumour nucleus: negative/positive -> Ki-67 positivity cut-off
    her2_membrane    F4  point per tumour cell: membrane category     -> HER2 membrane thresholds (+ field score)
    region           F5  polygons: invasive / in situ / benign / stroma / fat -> region model patches
    mitosis          F6  point: mitotic figure / mimic (hard negative) -> detector + verifier
    pleomorphism     F7  field label: score 1/2/3                       -> pleomorphism model
    tubules          F8  polygon of each true tubule lumen (+ score)    -> lumen classifier / U-Net

Storage, per document:  <annotations>/<task>/<doc_id>/
    image.png              the frame, lossless
    annotation.json        native format (re-opened by the app)
    annotation.geojson     QuPath-compatible (File > Import objects), classification name + colour
    points.csv             point tasks: x, y, class, source
    mask.png               polygon tasks: 16-bit instance labels (nuclei) or 8-bit class index (region, tubules)
Annotations are de-identified: they carry the case ID at most, never patient details.
"""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AnnClass:
    name: str
    color: tuple


@dataclass
class Task:
    key: str
    title: str
    feature: str
    geometry: str                        # "point", "polygon" or "field"
    classes: list
    objective: str
    stain: str
    instructions: str                    # HTML
    field_labels: dict = field(default_factory=dict)    # name -> choices
    mask: str = ""                       # "instance" / "semantic" / ""
    prefill: str = ""                    # what the model pre-fill does ("" = none)
    calibrates: str = ""                 # what "Calibrate thresholds" fits ("" = none)
    min_items: int = 0                   # recommended per image

    def cls(self, name) -> AnnClass | None:
        return next((c for c in self.classes if c.name == name), None)


_COMMON_TOOLS = ("<h4>Tools</h4><ul>"
                 "<li><b>Mouse wheel</b> zoom, <b>middle-drag</b> or <b>Space+drag</b> pan, <b>F</b> fit to window.</li>"
                 "<li><b>1-9</b> choose the class; <b>right-click</b> deletes the object under the cursor; "
                 "<b>Ctrl+Z</b> undo; <b>Ctrl+S</b> save.</li>"
                 "<li>Objects proposed by the model are drawn <i>dashed</i>: they count only after you confirm them "
                 "(<b>click</b> it with the right class, or <b>Confirm all</b> once every one is checked).</li></ul>")
_POINT_TOOL = ("<p><b>Point tool:</b> click the <b>centre</b> of the cell to add a point of the current class; click an "
               "existing point to change it to the current class.</p>")
_POLY_TOOL = ("<p><b>Polygon tool:</b> click vertices along the boundary and double-click (or press Enter, or click the "
              "first vertex) to close; or <b>hold the button and trace</b> the outline, releasing to close. "
              "Backspace removes the last vertex, Esc cancels. Click inside an existing polygon with the "
              "<b>Classify</b> tool to change its class.</p>")
_QA = ("<h4>Quality</h4><ul><li>Annotate the <b>whole field</b> (or a clearly bounded region), not a selection of easy "
       "examples - unlabelled cells in an image are read as 'not this class'.</li><li>When unsure, use the "
       "uncertain class where offered, or leave a note; do not guess.</li><li>Press <b>Mark complete</b> only when the "
       "image is finished; incomplete images are excluded from training and calibration.</li></ul>")

TASKS = [
    Task("nuclei", "Nuclei outlines", "F1", "polygon", [AnnClass("nucleus", (0, 220, 0))], "40x", "H&E or IHC",
         "<h3>Nuclei outlines (Feature 1 - nuclei detection)</h3>"
         "<p><b>Goal:</b> 50-100 fields from your own slides (both H&amp;E and IHC) to fine-tune the nucleus "
         "detector on your scanner, stain and camera.</p>"
         "<h4>What to outline</h4><ul><li>Every nucleus whose centre lies inside the image - tumour, lymphocytes, "
         "stroma, endothelium, all of them.</li><li>Trace the <b>nuclear membrane</b> (the edge of the haematoxylin / "
         "DAB stained nucleus), not the cell. Touching nuclei are outlined <b>separately</b>.</li>"
         "<li>Include pale, negative nuclei on IHC - these are the ones the model misses most.</li>"
         "<li>Skip out-of-focus nuclei and debris; skip nuclei cut by the image edge by more than half.</li></ul>"
         "<h4>Fast workflow</h4><ol><li><b>Pre-fill with model</b> - the current detector proposes outlines.</li>"
         "<li>Right-click wrong outlines (debris, merged nuclei) to delete them.</li><li>Draw the missed nuclei and "
         "re-draw badly fitted ones.</li><li><b>Confirm all</b>, then <b>Mark complete</b>.</li></ol>"
         "<p>Aim for 40x fields of about 300-1000 nuclei; 20-40 minutes per field is typical.</p>" + _POLY_TOOL,
         mask="instance", prefill="nucleus outlines from the current detector", min_items=100),
    Task("celltype", "Cell types", "F2", "point",
         [AnnClass("tumour", (220, 0, 0)), AnnClass("lymphocyte", (0, 120, 255)), AnnClass("stroma", (0, 170, 0)),
          AnnClass("other", (200, 0, 200))], "40x", "ER / PR / Ki-67 IHC (and H&E)",
         "<h3>Cell types (Feature 2 - tumour-cell filter)</h3>"
         "<p><b>Goal:</b> 1-2 thousand labelled nuclei on <b>real IHC</b> slides (the current classifier was trained on "
         "H&amp;E and does not transfer well to IHC).</p><h4>Classes</h4><ul>"
         "<li><b>1 tumour</b> - invasive carcinoma cells (also in situ if you cannot tell; note it).</li>"
         "<li><b>2 lymphocyte</b> - small round dark nuclei, including plasma cells.</li>"
         "<li><b>3 stroma</b> - fibroblasts, endothelium, smooth muscle (elongated nuclei).</li>"
         "<li><b>4 other</b> - normal epithelium, myoepithelium, histiocytes, anything else.</li></ul>"
         "<h4>Fast workflow</h4><ol><li><b>Pre-fill with model</b> - every detected nucleus gets the classifier's "
         "guess (dashed).</li><li>Choose a class with 1-4 and click every nucleus whose label is wrong; click the "
         "correct ones too, or press <b>Confirm all</b> when you have checked the whole image.</li></ol>" + _POINT_TOOL,
         prefill="a point on every nucleus with the classifier's class", min_items=200),
    Task("erpr_intensity", "ER/PR nuclear intensity", "F2", "point",
         [AnnClass("negative", (40, 110, 255)), AnnClass("weak", (255, 220, 0)), AnnClass("moderate", (255, 140, 0)),
          AnnClass("strong", (220, 0, 0))], "40x", "ER or PR",
         "<h3>ER / PR nuclear intensity (Feature 2 - intensity thresholds)</h3>"
         "<p><b>Goal:</b> about <b>200 tumour nuclei</b> covering all four intensities, from several slides and "
         "staining runs. <b>Calibrate thresholds</b> then fits the negative / weak / moderate / strong cut-offs "
         "to your eye and your lab's DAB.</p><h4>Rules</h4><ul><li>Label <b>invasive tumour nuclei only</b>.</li>"
         "<li>Judge the nucleus as you would for the Allred intensity score; ignore cytoplasmic blush.</li>"
         "<li>Balance the classes - include clearly negative nuclei and the strongest ones, not only borderline "
         "cases.</li></ul><h4>Fast workflow</h4><ol><li><b>Pre-fill with model</b> (current thresholds, dashed).</li>"
         "<li>Click each nucleus you checked with the correct class (1-4). Delete pre-filled points on non-tumour "
         "cells (right-click).</li></ol>" + _POINT_TOOL,
         prefill="a point on every nucleus with its current intensity bin",
         calibrates="ER/PR intensity cut-offs (DAB OD)", min_items=50),
    Task("ki67_positivity", "Ki-67 positivity", "F3", "point",
         [AnnClass("negative", (40, 110, 255)), AnnClass("positive", (220, 0, 0))], "40x", "Ki-67",
         "<h3>Ki-67 positivity (Feature 3 - positivity cut-off)</h3>"
         "<p><b>Goal:</b> about <b>200 tumour nuclei</b> (at least 10 of each class, ideally 50+), from several "
         "slides. <b>Calibrate thresholds</b> fits the DAB cut-off that best matches your calls.</p>"
         "<h4>Rules</h4><ul><li>Invasive tumour nuclei only.</li><li><b>Any</b> definite brown nuclear staining is "
         "positive, however weak (IKWG).</li><li>Include borderline faint nuclei - they decide the cut-off.</li></ul>"
         + _POINT_TOOL, prefill="a point on every nucleus with the current positivity call",
         calibrates="Ki-67 positivity cut-off", min_items=50),
    Task("her2_membrane", "HER2 membrane categories", "F4", "point",
         [AnnClass("none", (60, 120, 255)), AnnClass("faint/incomplete", (230, 200, 0)),
          AnnClass("weak-moderate complete", (255, 130, 0)), AnnClass("strong complete", (220, 0, 0))], "40x", "HER2",
         "<h3>HER2 membrane categories (Feature 4 - membrane thresholds)</h3>"
         "<p><b>Goal:</b> 200-400 tumour cells across 0, 1+, 2+ and 3+ cases. <b>Calibrate thresholds</b> fits the "
         "membrane staining thresholds; the <b>field score</b> labels train the field classifier.</p>"
         "<h4>Classes (per cell, ASCO/CAP wording)</h4><ul><li><b>1 none</b> - no membrane staining.</li>"
         "<li><b>2 faint/incomplete</b> - faint, barely perceptible, or incomplete membrane staining.</li>"
         "<li><b>3 weak-moderate complete</b> - complete circumferential, weak to moderate.</li>"
         "<li><b>4 strong complete</b> - complete, intense (chicken-wire).</li></ul>"
         "<p>Click the <b>nucleus</b> of the cell (the point is matched to the cell drawn around it). Invasive tumour "
         "only. Also set the <b>HER2 score of this field</b> on the right.</p>" + _POINT_TOOL,
         field_labels={"her2_field_score": ["", "0", "1+", "2+", "3+"]},
         prefill="a point on every cell with its current membrane category",
         calibrates="HER2 membrane thresholds", min_items=50),
    Task("region", "Tissue regions", "F5", "polygon",
         [AnnClass("invasive", (220, 0, 0)), AnnClass("in situ", (150, 0, 200)), AnnClass("benign epithelium", (0, 170, 0)),
          AnnClass("stroma", (230, 150, 150)), AnnClass("fat / other", (240, 220, 120))], "10x", "H&E",
         "<h3>Tissue regions (Feature 5 - tumour region map)</h3>"
         "<p><b>Goal:</b> 10x camera fields from your own microscope (the model was trained on scanner images and "
         "public photos).</p><h4>What to draw</h4><ul><li>Outline each region and give it a class: <b>invasive</b> "
         "carcinoma, carcinoma <b>in situ</b>, <b>benign epithelium</b> (normal ducts, lobules, benign lesions), "
         "<b>stroma</b>, <b>fat / other</b>.</li><li>Outlines may be generous (about 50 µm), but do not let "
         "invasive and in situ overlap. Leave glass unlabelled.</li></ul><p><b>Pre-fill with model</b> draws the "
         "current map as polygons to correct.</p>" + _POLY_TOOL,
         mask="semantic", prefill="region polygons from the current region model", min_items=1),
    Task("mitosis", "Mitotic figures", "F6", "point",
         [AnnClass("mitotic figure", (255, 0, 0)), AnnClass("mimic (not mitosis)", (120, 120, 120)),
          AnnClass("uncertain", (255, 200, 0))], "40x", "H&E",
         "<h3>Mitotic figures (Feature 6 - mitosis detector and verifier)</h3>"
         "<p><b>Goal:</b> 40x fields from hotspots, ideally labelled by 2-3 pathologists separately (the export keeps "
         "the annotator, so a consensus can be formed).</p><h4>Classes</h4><ul><li><b>1 mitotic figure</b> - "
         "prophase (late) to telophase; count a telophase pair as <b>one</b> point between the two.</li>"
         "<li><b>2 mimic</b> - apoptotic bodies, pyknotic nuclei, lymphocytes, crushed or hyperchromatic nuclei "
         "that the model calls mitoses. These hard negatives are as valuable as the true figures.</li>"
         "<li><b>3 uncertain</b> - use sparingly; excluded from training.</li></ul><p>Label <b>every</b> mitotic figure "
         "in the field. <b>Pre-fill with model</b> marks its accepted calls as mitotic figure and its rejected "
         "candidates as mimic - check every one.</p>" + _POINT_TOOL,
         prefill="detector candidates (accepted = mitotic figure, rejected = mimic)", min_items=0),
    Task("pleomorphism", "Pleomorphism score", "F7", "field", [], "40x", "H&E",
         "<h3>Nuclear pleomorphism score (Feature 7)</h3>"
         "<p><b>Goal:</b> 40x fields of invasive tumour with your pleomorphism score, from many cases (all three "
         "scores).</p><ul><li><b>1</b> - small, regular, uniform nuclei.</li><li><b>2</b> - moderate increase in "
         "size and variability.</li><li><b>3</b> - marked variation, large nuclei, vesicular chromatin, prominent "
         "nucleoli.</li></ul><p>Score the field as shown (not the whole case). Use the note field for anything "
         "unusual.</p>", field_labels={"pleomorphism_score": ["", "1", "2", "3"]}),
    Task("tubules", "Tubule lumina", "F8", "polygon",
         [AnnClass("tubule lumen", (0, 170, 0)), AnnClass("not a tubule (space)", (120, 120, 120))], "10x", "H&E",
         "<h3>Tubule lumina (Feature 8 - tubule formation)</h3>"
         "<p><b>Goal:</b> 150-300 10x fields of invasive tumour. This is the data the tubule score is waiting for: "
         "the current score is not reliable.</p><h4>What to draw</h4><ul><li><b>tubule lumen</b> - each true lumen "
         "of a tumour gland: a central space enclosed by polarised tumour cells (as counted for the Nottingham "
         "tubule score).</li><li><b>not a tubule</b> - white spaces that are not lumina (retraction artefact, fat, "
         "vessels, cribriform DCIS spaces, normal ducts) - optional but helpful.</li></ul><p>Also give the "
         "<b>tubule score</b> of the field if it is representative.</p>" + _POLY_TOOL,
         field_labels={"tubule_score": ["", "1", "2", "3"]}, mask="semantic",
         prefill="lumina accepted by the current lumen classifier", min_items=0),
]
TASK_BY_KEY = {t.key: t for t in TASKS}


def instructions_html(task: Task) -> str:
    extra = ""
    if task.geometry != "field":
        extra = _COMMON_TOOLS
    return (task.instructions + extra + _QA +
            f"<p style='color:#666'>Objective: {task.objective}; stain: {task.stain}.</p>")


# --------------------------------------------------------------------------------------------- documents
@dataclass
class Item:
    kind: str                  # "point" or "polygon"
    cls: str
    pts: list                  # [[x, y], ...] image px (one point for "point")
    source: str = "annotator"  # "annotator" or "model" (model proposals not yet confirmed)

    def contains(self, x, y, tol) -> bool:
        if self.kind == "point":
            return (self.pts[0][0] - x) ** 2 + (self.pts[0][1] - y) ** 2 <= tol ** 2
        c = np.asarray(self.pts, np.float32)
        return cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0


@dataclass
class AnnotationDoc:
    id: str
    task: str
    width: int
    height: int
    um_per_px: float | None
    objective: str = ""
    stain: str = ""
    source: str = ""           # e.g. "case B-001 frames/HE/....png" or a file name
    case_id: str = ""
    annotator: str = ""
    created: str = ""
    modified: str = ""
    status: str = "in progress"    # or "complete"
    items: list = field(default_factory=list)
    field_labels: dict = field(default_factory=dict)
    notes: str = ""
    dir: Path | None = None

    @property
    def image_path(self) -> Path:
        return self.dir / "image.png"

    def counts(self) -> dict:
        out = {}
        for it in self.items:
            if it.source == "annotator":
                out[it.cls] = out.get(it.cls, 0) + 1
        return out

    def n_unconfirmed(self) -> int:
        return sum(it.source == "model" for it in self.items)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("dir")
        return d

    def item_at(self, x, y, tol) -> int | None:
        """Topmost item under (x, y): points first (within tol px), then polygons."""
        best = None
        for i, it in enumerate(self.items):
            if it.kind == "point" and it.contains(x, y, tol):
                d = (it.pts[0][0] - x) ** 2 + (it.pts[0][1] - y) ** 2
                if best is None or d < best[0]:
                    best = (d, i)
        if best is not None:
            return best[1]
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i].kind == "polygon" and self.items[i].contains(x, y, tol):
                return i
        return None


class AnnotationStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def task_dir(self, task: str) -> Path:
        d = self.root / task
        d.mkdir(parents=True, exist_ok=True)
        return d

    def new(self, task: str, rgb: np.ndarray, um_per_px: float | None, annotator: str, **meta) -> AnnotationDoc:
        did = time.strftime("%Y%m%d_%H%M%S")
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", meta.get("case_id") or "")
        did = f"{base}_{did}" if base else did
        d = self.task_dir(task) / did
        k = 1
        while d.exists():
            k += 1
            d = self.task_dir(task) / f"{did}_{k}"
        d.mkdir(parents=True)
        doc = AnnotationDoc(d.name, task, int(rgb.shape[1]), int(rgb.shape[0]), um_per_px, annotator=annotator,
                            created=time.strftime("%Y-%m-%dT%H:%M:%S"), dir=d,
                            **{k: v for k, v in meta.items() if k in ("objective", "stain", "source", "case_id")})
        cv2.imwrite(str(doc.image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.save(doc)
        return doc

    def load(self, doc_dir: Path) -> AnnotationDoc:
        d = json.loads((Path(doc_dir) / "annotation.json").read_text())
        d["items"] = [Item(**i) for i in d.get("items", [])]
        return AnnotationDoc(**d, dir=Path(doc_dir))

    def list(self, task: str) -> list[AnnotationDoc]:
        out = []
        for p in sorted(self.task_dir(task).glob("*/annotation.json"), reverse=True):
            try:
                out.append(self.load(p.parent))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return out

    def delete(self, doc: AnnotationDoc):
        import shutil
        shutil.rmtree(doc.dir)

    # -------------------------------------------------------------- save / export
    def save(self, doc: AnnotationDoc):
        doc.modified = time.strftime("%Y-%m-%dT%H:%M:%S")
        task = TASK_BY_KEY[doc.task]
        tmp = doc.dir / "annotation.json.tmp"
        tmp.write_text(json.dumps(doc.to_json(), indent=1))
        tmp.replace(doc.dir / "annotation.json")
        (doc.dir / "annotation.geojson").write_text(json.dumps(to_geojson(doc, task)))
        if task.geometry == "point":
            with open(doc.dir / "points.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "class", "source"])
                for it in doc.items:
                    w.writerow([round(it.pts[0][0], 1), round(it.pts[0][1], 1), it.cls, it.source])
        if task.mask:
            cv2.imwrite(str(doc.dir / "mask.png"), render_mask(doc, task))

    def export_manifest(self, task: str) -> Path:
        """manifest.csv of the task folder: one row per document with status, scale, counts and field labels."""
        t = TASK_BY_KEY[task]
        docs = self.list(task)
        labels = list(t.field_labels)
        cls = [c.name for c in t.classes]
        p = self.task_dir(task) / "manifest.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "status", "image", "um_per_px", "objective", "stain", "case_id", "annotator", "modified",
                        "unconfirmed_model_items"] + [f"n_{c}" for c in cls] + labels + ["notes"])
            for d in docs:
                c = d.counts()
                w.writerow([d.id, d.status, f"{d.id}/image.png", d.um_per_px, d.objective, d.stain, d.case_id, d.annotator,
                            d.modified, d.n_unconfirmed()] + [c.get(k, 0) for k in cls]
                           + [d.field_labels.get(k, "") for k in labels] + [d.notes])
        return p


def to_geojson(doc: AnnotationDoc, task: Task) -> dict:
    feats = []
    for it in doc.items:
        c = task.cls(it.cls)
        col = list(c.color) if c else [128, 128, 128]
        if it.kind == "point":
            geom = {"type": "Point", "coordinates": [float(it.pts[0][0]), float(it.pts[0][1])]}
        else:
            ring = [[float(x), float(y)] for x, y in it.pts]
            ring.append(ring[0])
            geom = {"type": "Polygon", "coordinates": [ring]}
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {"objectType": "annotation", "classification": {"name": it.cls, "color": col},
                                     "source": it.source, "isLocked": False}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"task": doc.task, "um_per_px": doc.um_per_px, "status": doc.status, **doc.field_labels}}


def render_mask(doc: AnnotationDoc, task: Task) -> np.ndarray:
    """instance: uint16, polygon k -> k (drawing order); semantic: uint8 class index + 1 (0 = unlabelled)."""
    inst = task.mask == "instance"
    m = np.zeros((doc.height, doc.width), np.uint16 if inst else np.uint8)
    names = [c.name for c in task.classes]
    k = 0
    for it in doc.items:
        if it.kind != "polygon" or it.source != "annotator":
            continue
        k += 1
        v = k if inst else names.index(it.cls) + 1 if it.cls in names else 0
        cv2.fillPoly(m, [np.round(np.asarray(it.pts)).astype(np.int32)], int(v))
    return m


# --------------------------------------------------------------------------------------------- model pre-fill
def _poly(c):
    """StarDist polygon (2, rays) y,x -> [[x, y], ...]."""
    return [[round(float(x), 1), round(float(y), 1)] for y, x in c.T]


def _contours(mask, scale=1.0, min_area_px=30, eps=1.5):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        c = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(c) >= 3:
            out.append([[round(float(x) / scale, 1), round(float(y) / scale, 1)] for x, y in c])
    return out


def prefill(task: Task, rgb: np.ndarray, um_per_px: float, hub) -> list[Item]:
    """Model proposals for a task (source="model": must be confirmed by the annotator)."""
    k = task.key
    if k == "nuclei":
        d = hub.detector.detect(rgb, um_per_px=um_per_px, return_labels=False)
        return [Item("polygon", "nucleus", _poly(c), "model") for c in d.coords]
    if k == "celltype":
        d = hub.detector.detect(rgb, um_per_px=um_per_px, return_labels=False)
        if not len(d):
            return []
        p = hub.celltype.predict(rgb, d.points, um_per_px).argmax(1)
        names = [c.name for c in task.classes]
        return [Item("point", names[int(c)], [[float(x), float(y)]], "model") for (y, x), c in zip(d.points, p)]
    if k in ("erpr_intensity", "ki67_positivity"):
        from core.ihc.pipeline import ERPRAnalyzer
        if k == "erpr_intensity":
            th = hub.erpr_thresholds()[0]
        else:
            from core.ihc.scoring import IntensityThresholds
            th = IntensityThresholds(hub.ki67_positivity()[0].cut, np.inf, np.inf)
        a = ERPRAnalyzer(hub.detector, None, thresholds=th).analyze(rgb, um_per_px)
        names = [c.name for c in task.classes]
        return [Item("point", names[min(int(v), len(names) - 1)], [[float(x), float(y)]], "model")
                for (y, x), v in zip(a.points, a.intensity)]
    if k == "her2_membrane":
        from core.her2.pipeline import HER2Analyzer
        a = HER2Analyzer(hub.detector, None, thresholds=hub.her2_thresholds()[0]).analyze(rgb, um_per_px)
        names = [c.name for c in task.classes]
        return [Item("point", names[int(v)], [[float(x), float(y)]], "model")
                for (y, x), v, m in zip(a.points, a.intensity, a.measurable) if m]
    if k == "mitosis":
        cands, _ = hub.mitosis.detect(rgb, um_per_px)
        return [Item("point", "mitotic figure" if c.accepted else "mimic (not mitosis)", [[c.x, c.y]], "model")
                for c in cands if c.accepted or c.ver_score >= 0.2]
    if k == "region":
        rm = hub.region.predict(rgb, um_per_px, overlap=True)
        full = rm.upsample(rgb.shape)
        cl = full.argmax(-1)
        # region model classes: invasive, in_situ, benign, normal_other, background
        mapping = {0: "invasive", 1: "in situ", 2: "benign epithelium", 3: "stroma"}
        s = 0.25
        out = []
        for ci, name in mapping.items():
            m = cv2.resize(((cl == ci) & (full[..., ci] >= 0.5)).astype(np.uint8), None, fx=s, fy=s,
                           interpolation=cv2.INTER_NEAREST)
            out += [Item("polygon", name, p, "model") for p in _contours(m, s, min_area_px=40, eps=2.0)]
        return out
    if k == "tubules":
        from core.tubules.lumen import candidates
        c = candidates(rgb, um_per_px)
        p = hub.lumen.predict(c)
        acc = np.nonzero(p >= hub.lumen.threshold)[0] + 1
        if not len(acc):
            return []
        m = np.isin(c.labels, acc)
        return [Item("polygon", "tubule lumen", q, "model") for q in _contours(m, c.scale, min_area_px=20, eps=1.0)]
    return []


# --------------------------------------------------------------------------------------------- calibration
def _complete(docs):
    return [d for d in docs if d.status == "complete" and d.um_per_px]


def calibrate(task_key: str, store: AnnotationStore, hub, out_dir: Path) -> dict:
    """Fit the task's thresholds to all completed documents (annotator-confirmed points only). The image is
    re-measured exactly as the runtime pipeline measures it, so the cut-offs match what the app computes.
    Writes <calibration>/<name>.json (picked up by the analysis modules) and returns a report."""
    from app.desktop.hub import ERPR_INTENSITY, HER2_THRESHOLDS, KI67_POSITIVITY
    from app.desktop.casestore import read_rgb
    task = TASK_BY_KEY[task_key]
    docs = _complete(store.list(task_key))
    if not docs:
        raise ValueError("no completed annotations with a known µm/px - mark finished images as complete")
    names = [c.name for c in task.classes]
    xs, ys = [], []
    for d in docs:
        pts = [(it.pts[0], names.index(it.cls)) for it in d.items if it.kind == "point" and it.source == "annotator"
               and it.cls in names]
        if not pts:
            continue
        rgb = read_rgb(d.image_path)
        if task_key == "her2_membrane":
            from core.her2.pipeline import HER2Analyzer
            a = HER2Analyzer(hub.detector, None, thresholds=hub.her2_thresholds()[0]).analyze(rgb, d.um_per_px)
            for (x, y), lab in pts:
                yi, xi = int(round(y)), int(round(x))
                if 0 <= yi < a.labels.shape[0] and 0 <= xi < a.labels.shape[1] and a.labels[yi, xi] > 0:
                    xs.append(a.sector_od[a.labels[yi, xi] - 1])
                    ys.append(lab)
        else:
            from core.ihc.pipeline import ERPRAnalyzer
            a = ERPRAnalyzer(hub.detector, None).analyze(rgb, d.um_per_px)
            if not len(a):
                continue
            for (x, y), lab in pts:
                dist = np.hypot(a.points[:, 0] - y, a.points[:, 1] - x)
                i = int(dist.argmin())
                if dist[i] <= 5.0 / d.um_per_px:          # matched to a detected nucleus
                    xs.append(float(a.dab_od[i]))
                    ys.append(lab)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = f"{len(docs)} images annotated locally ({time.strftime('%Y-%m-%d')})"
    if task_key == "erpr_intensity":
        from core.ihc.scoring import calibrate as fit
        th, rep = fit(xs, ys)
        path = out_dir / ERPR_INTENSITY
        th.save(path)
        rep["thresholds"] = dict(weak=th.weak, moderate=th.moderate, strong=th.strong)
    elif task_key == "ki67_positivity":
        from core.ki67.scoring import calibrate_positivity
        th, rep = calibrate_positivity(xs, np.asarray(ys) == 1, source=src)
        path = out_dir / KI67_POSITIVITY
        th.save(path)
        rep["cut"] = th.cut
    elif task_key == "her2_membrane":
        if len(ys) < 40 or len(set(ys)) < 3:
            raise ValueError(f"need at least 40 matched cells covering 3+ categories (have {len(ys)})")
        from core.her2.scoring import calibrate_cells
        th, rep = calibrate_cells(xs, ys)
        th.source = src
        path = out_dir / HER2_THRESHOLDS
        th.save(path)
        rep["thresholds"] = asdict(th)
    else:
        raise ValueError(f"task {task_key} has no threshold calibration")
    rep.update(n_labels=len(ys), n_images=len(docs), saved_to=str(path))
    return rep
