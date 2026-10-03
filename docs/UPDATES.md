# Updates log

This file records what was built, changed or decided, and why, newest first. The "why" is the most useful part for whoever comes next. Bugs go in [BUGS.md](BUGS.md); this file only links to them.

**How to add an entry.** Put it at the top, under today's date (YYYY-MM-DD):

```markdown
## YYYY-MM-DD — short title
**Area:** Feature N / core module / model name
**Change:** what changed (files, models, thresholds).
**Why:** the reason, plus the evidence (metric before → after, dataset, split).
**Impact:** what callers, models cards, docs or tests had to change. Anything still to do.
```

Retraining a model counts as an update: record the old and new held-out metrics and the data used. Changing a threshold also counts: record the old value, the new value and how it was chosen.

---

## 2026-10-04 — Desktop app redesign
**Area:** app/desktop (UI only; no change to any analysis)
**Change:** new visual system (`app/desktop/theme.py`): colours taken from the stains (haematoxylin indigo for actions, eosin only for the open-case slide label, DAB amber for warnings), a neutral dark stage behind images, IBM Plex Sans (static Regular/Medium/SemiBold cut from the variable font, SIL OFL, `app/desktop/assets/fonts/OFL.txt`), and an inline icon set. Sidebar navigation (Home, Cases, Microscope, Report, Annotate) replaces tabs and menus; the open case is drawn as a slide label (case ID on the frosted end, patient on the glass). New Home page with the microscope set-up steps, recent cases and local-data status. Microscope page: camera bar with an objective selector, module list grouped by stain with saved/unsaved state, image overlay with Ready state and a zoom-aware scale bar, headline result with progress to the minimum count (500 nuclei, 10 HPF, 3 fields, 1 mm²). Report page: grade badge, 1/2/3 control per component with a link to its analysis, biomarker tiles. Toast confirmations; plain-language messages throughout.
**Why:** the first version worked but read as a bare tool; pathologists need to see what to do next and the state of the case at a glance.
**Impact:** modules gained `headline()`, `group` and `short`. `fonttools` was used once to cut the font weights (not a runtime dependency).

## 2026-10-04 — Desktop workstation app (Feature 0 + all modules in one application)
**Area:** app/desktop, core/grading, core/her2
**Change:** `python -m app.desktop` is now the full application for a USB microscope camera, with four pages:
- **Cases**: create / search / open cases; each case keeps the patient ID and request details (`case.json`), its frames, every module run, snapshots, reports and the audit log.
- **Microscope & Analysis**: live view with flat-field correction, motion + focus READY gate (dwell detection), objective selector (keys 1-4 or the magnification model), calibrated µm/px, frame logger. Each feature runs as a separate module (F1 nuclei, F2 ER, F2 PR, F3 Ki-67, F4 HER2 + field-classifier cross-check, F5 region map + mini-map + mitosis hotspots, F6 10-HPF mitoses, F7 pleomorphism, F8 tubules) with running results, field gallery, click-to-correct and "Save result to case".
- **Grade & Report**: Nottingham components with pathologist overrides (reason logged), biomarkers, other findings, snapshots, audit viewer, draft PDF, sign & lock.
- **Annotate**: nine annotation tasks with instructions (nuclei outlines, cell types, ER/PR intensity, Ki-67 positivity, HER2 membrane, regions, mitoses, pleomorphism score, tubule lumina), model pre-fill, QuPath GeoJSON / masks / CSV export, manifest, and threshold calibration (ER/PR, Ki-67, HER2) written to `<data>/calibration/` and used by new analysis runs.
- Calibration wizard (stage micrometer, two clicks) and microscope profiles in `<data>/profiles/`.
- `core/grading/case.py`: patient details, `BreastCase.open()` (reopen a saved case), `set_extra()`, `store_component()` (shared by the wizard and the modules), IHC-only signing (`allow_no_grade`; a partial grade is still refused), atomic `case.json` writes. `core/grading/report.py`: patient block, "grade not assessed", other findings, Ki-67 primary index.
- `core/her2/field_classifier.py`: runtime for `her2_field_effb0.onnx` (plan step 4.6).
**Why:** the CLI tools needed saved frames and could not keep a patient case across modules; the plan's Feature 0 and the annotation loop for local data (O-04 … O-07) needed a UI.
**Privacy:** patient details stay in the case folder and the report. Frame sidecars and annotation exports carry the case ID only (`tests/test_desktop.py`).
**Impact:** `tests/test_desktop.py` (8 tests). `RUNTIME_MODEL_FILES` now includes Phikon-v2, pleomorphism and tubule models (O-02). Data root can be moved with `PATHOASSIST_DATA`. Still to do: Windows installer, local validation of every module.

## 2026-10-03 — Engineering docs
**Change:** added [README.md](../README.md), [docs/FEATURES.md](FEATURES.md), [docs/algorithms.html](algorithms.html), this file and [BUGS.md](BUGS.md).
**Why:** to hand the project over to future engineers. FEATURES.md explains each feature and the reasons behind its technology choices; algorithms.html explains each algorithm visually.

## 2026-10-03 — Feature 9: Nottingham grade, wizard, audit log, combined report
**Change:** added `core/grading/` (`nottingham.py`, `case.py`, `audit.py`, `report.py`), `app/breast_case.py` and `tests/test_grading.py`.
- `NottinghamGrade` holds three components. Each records its source (auto, manual, edited or missing), the automatic score, the final score and whether the score is validated. Totals map to grades as 3–5 → 1, 6–7 → 2, 8–9 → 3.
- `GradingWizard` steps through tubules → pleomorphism → mitoses → review → signed. A step can be entered manually, edited or stepped back.
- `BreastCase` holds the grade, the ER/PR/HER2/Ki-67 results and snapshots. It is locked once signed; any change after signing is refused and logged.
- `AuditLog` writes JSON lines, each containing the SHA-256 of the previous line. `verify()` detects any edited, deleted or reordered line.
- The PDF report (ReportLab) contains the grade table, biomarkers, all warnings (including nested Ki-67 warnings), snapshots, signature and the audit fingerprint. Core biopsies get a caveat that the grade may be understated.

**Why:** the plan's Feature 9. The hash chain makes later edits visible without needing a database.
**Impact:** 82 tests pass. The on-screen wizard still has to be built in the camera app. Demo output: `demo_outputs/breast_case_demo/report_DEMO-B1.pdf`.

## 2026-10-03 — Feature 8: tubule formation (public-data version)
**Change:** added `core/tubules/lumen.py`, `core/tubules/score.py`, `training/tubules/train_lumen.py`, `training/tubules/train_lumen_emb.py` and `eval/evaluate_tubules_databiox.py`. New models: `models/tubule_lumen_lgbm.txt/.json` and `models/tubule_lumen_phikon.npz`.
- **Candidates:** near-white regions (total OD < 0.22) at 1 µm/px, between 15 µm² and 0.2 mm².
- **Features:** shape, contents, and stain/nuclear density in rings at 0–6, 6–15 and 15–25 µm, plus counts of neighbouring white spaces.
- **Classifier:** LightGBM, trained on BreCaHAD (17 cases) with case-grouped CV. Lumen AUC 0.87.
- **Acceptance:** classifier p ≥ threshold, area ≥ 30 µm², and at least 60% of the inner ring is epithelium.
- **Tubule %** = epithelium within 25 µm of accepted lumens ÷ all tumour epithelium. Score: > 75% → 1, 10–75% → 2, < 10% → 3.

**Why:** the plan specifies lumen detection plus area measurement. A ring-feature model can be trained from BreCaHAD's point labels; a segmentation network would need masks that don't exist.
**Result:** on Databiox 10× images (94 patients), the tubule score **does not separate the grades** (Spearman vs grade −0.12; κ −0.10). It is reported as not validated.
**Next:** 150–300 locally annotated 10× fields, or TCGA tubule scores once the GDC API works. See BUGS.md B-016 to B-018 for the fixes made along the way.

## 2026-10-03 — Feature 7: switched pleomorphism to Phikon-v2 embeddings
**Change:**
- Added `core/pleomorphism/embedding.py`, with `TileEmbedder` (Phikon-v2 INT8-weight OpenVINO model, 224 px tiles at 0.5 µm/px) and `EmbeddingPleoModel` (PCA to 16 dimensions, then logistic regression with C 0.05).
- `scorer.py` uses this model when `models/pleomorphism_phikon.npz` is present. The hand-crafted LightGBM model is kept as a fallback.
- Training scripts are in `training/pleomorphism/`; `train_final.py` builds the final model.

**Why:** hand-crafted nuclear features reached κ 0.34 against the Databiox grade; embeddings reached **κ 0.56** with patients held out.
- Adding the MITOS-ATYPIA-14 tiles to training and retraining on Databiox variants did not improve on about 0.57, which looks like the ceiling of a proxy label on 115 patients.
- The final model is therefore trained on Databiox only.

**Impact:** about 1.4 s per 1080p field for the embedding, plus about 1 s for nuclei. The report warns that the score is not validated against local pathologists.
**Next:** TCGA-BRCA (1,040 cases with true pleomorphism scores). The labels are already downloaded (Thennavan et al., Data S2); the slides are waiting on the NCI GDC API, which returned 503.

## 2026-10-03 — Databiox dataset added
**Change:** downloaded to `training/data/databiox/` (823 MB), with `index.json`, `scales.json`, per-image features and tile embeddings.
**Why:** this is the only public breast dataset with patient-level Nottingham grade and real microscope-camera photos at 4×, 10×, 20× and 40×. It is used to train and check Features 7 and 8.
**Note:** the TCIA/NBIA manifest that was offered as a data source contains MRI series, not slides. Don't download it.

## Earlier (before 2026-10-03) — Features 1 to 6
These were built in earlier sessions. The full write-up is in FEATURES.md; the main decisions are summarised here.

### Feature 6: mitosis counting
- **Design:** a CenterNet-style centre-heatmap detector plus an EfficientNet verifier on 96 px crops. The plan named RTMDet-tiny or YOLOX-Tiny, but MIDOG labels every mitosis as a point with the same fixed box, so a box detector has nothing extra to learn. This design also avoids YOLOv8's AGPL licence.
- **Hotspot model:** a separate model guides the pathologist to the 10 HPF with the most mitoses.
- **Score table:** field diameter → mitotic score, from the Nottingham table (`core/mitosis/mitotic_score_table.json`). Diameters outside the table are scaled by field area.
- **Result:** F1 0.72 on 33 held-out MIDOG++ breast slides; the same mitotic score as the experts on 91% of slides.

### Feature 5: tumour region, auto-magnification, mosaic
- **Region model:** Phikon-v2 teacher distilled into an EfficientNet-B0 student (T = 2). H-optimus-0 was considered but needs approved access and too much disk.
- **Training data:** BACH slide patches and BACH microscope photos together. A slide-only model got just 33% of photos right; with photos added, 97.5%.
- **Magnification:** a MobileNetV3 regresses log2(µm/px) and the result snaps to the nearest calibrated objective. 85% correct overall; 20× is weak at 41%.
- **Mosaic:** phase correlation between frames, refined against the map, with relocation and island merging. Error stayed within 6 µm on a 49-frame synthetic sweep.

### Feature 4: HER2
- **Membrane analysis:** watershed cells (at most 8 µm), and the DAB peak in each of 16 sectors measured relative to the field background.
- **Scoring:** an ASCO/CAP rule engine with thresholds fitted on BCI (`models/her2_thresholds_bci.json`), plus an EfficientNet-B0 field classifier as a cross-check.
- **Result:** κ 0.72 for the rules and 0.80 for the classifier. When they disagree, the case is flagged "review carefully".

### Feature 3: Ki-67
- Global and hotspot modes; hotspot takes fields until about 500 tumour cells are counted.
- Lab-specific positivity cut-off: on another lab's slides, fields within ±5 points went from 52% to 67%.

### Feature 2: ER/PR
- Beer–Lambert colour deconvolution, with Ruifrok stain vectors by default or Macenko vectors per lab.
- DAB OD per nucleus is binned negative / weak / moderate / strong, giving % positive, H-score and Allred.
- A cell-type classifier (MobileNetV3-small, 32 µm context crop) counts tumour nuclei only. The wider 32 µm context raised validation tumour F1 from 0.79 to 0.83.
- **Nuclei model retrained on real IHC (`stardist_he_ihc_ft`):** the H&E model missed pale negative nuclei, which inflated % positive by about 11 points. It is now the default for both stains.
- De-duplication: a frame overlapping an earlier one by more than 50% is skipped.

### Feature 1: nuclei detection
- **Model:** StarDist (32 rays), pretrained `2D_versatile_he` fine-tuned on breast data (PanNuke and others). It reaches F1 0.841 on held-out breast H&E (pretrained 0.749) and 0.861 on IHC (pretrained 0.671).
- **Export:** ONNX → OpenVINO, with INT8 weight compression (NNCF). 0.75–1.0 s per 1080p frame on CPU.
- **Tiling:** 512 px tiles, with only the centre of each kept, so nuclei on seams are found once.
