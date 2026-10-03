# PathoAssist: features, technologies and how they work

PathoAssist is a decision-support tool for breast pathology that runs **live on a microscope camera** on an ordinary CPU. The pathologist moves the slide as usual. The assistant analyses the frames the pathologist rests on, which this document calls *dwell frames*. It overlays its findings, keeps running totals across fields, and produces a signed, audit-logged report.

This document explains, for each feature, **what it does for the pathologist**, **which technologies it uses and why**, **how it works step by step**, and **how well it performs today**. Results quoted here are on **public data only**. Every "done when" target in the project plan is defined on local slides and pathologists, and none has been measured yet.

A companion page, [`docs/algorithms.html`](algorithms.html), explains the algorithms with interactive diagrams.

---

## Contents

1. [System overview](#1-system-overview)
2. [Shared building blocks](#2-shared-building-blocks)
3. [Feature 1: Nuclei detection engine](#feature-1-nuclei-detection-engine)
4. [Feature 2: ER / PR scoring](#feature-2-er--pr-scoring)
5. [Feature 3: Ki-67 index](#feature-3-ki-67-index)
6. [Feature 4: HER2 scoring](#feature-4-her2-scoring)
7. [Feature 5: Tumour region, auto-magnification, mosaic](#feature-5-tumour-region-auto-magnification-mosaic)
8. [Feature 6: Mitosis counting](#feature-6-mitosis-counting-guided-10-hpf)
9. [Feature 7: Nuclear pleomorphism](#feature-7-nuclear-pleomorphism)
10. [Feature 8: Tubule formation](#feature-8-tubule-formation)
11. [Feature 9: Nottingham grade and combined report](#feature-9-nottingham-grade-and-combined-report)
12. [Status summary and what is still needed](#status-summary-and-what-is-still-needed)
13. [Repository map and how to run](#repository-map-and-how-to-run)

---

## 1. System overview

```
camera frame ──► calibration (µm/px) ──► Feature 5: magnification check, tumour map, mosaic position
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────────┐
         ▼                                  ▼                                      ▼
   IHC stains                          H&E grading                          (all features)
   F1 nuclei ─► F2 ER/PR              F8 tubules (10x)                     de-duplication,
              ─► F3 Ki-67             F7 pleomorphism (40x)                running totals,
              ─► F4 HER2              F6 mitoses (10 HPF, 40x)             click-to-correct,
                                            │                              corrections saved as labels
                                            ▼
                                F9 Nottingham grade + combined breast report + audit log
```

**Design principles that apply everywhere**

| Principle | Why |
|---|---|
| **Everything in micrometres.** Every frame is converted with the microscope's calibrated µm/px before analysis. | Microscopes, cameras and objectives differ. Nuclear size, tubule area and HPF area only mean something in physical units, and models are run at the scale they were trained on. |
| **CPU only at the microscope.** Models are exported to ONNX, then OpenVINO INT8 where it helps. | Labs have ordinary PCs. INT8 roughly halves inference time with no measurable loss in accuracy (shown below). |
| **Accumulate over fields, never per frame.** Sessions keep running totals and skip a frame that overlaps an earlier one by more than 50%. | Clinical scores are defined over many cells or fields (≥ 500 tumour cells, 10 HPF, the whole tumour for tubules). |
| **The pathologist stays in charge.** Every automatic result can be corrected, and every correction is saved as a training label. | Decision support, not autonomous diagnosis. Corrections become local training data. |
| **Explicit validation status.** Each model's metadata carries `validated: false` until it has been validated locally, and reports print this. | So nobody mistakes a public-data result for a clinically validated one. |

**Technology stack**

| Layer | Technology | Reason |
|---|---|---|
| Nuclei model | StarDist 2D (TensorFlow for training) | Best open method for crowded, touching nuclei. Its star-convex polygon output is fast and well suited to histology. |
| Other models | PyTorch + timm (training) | Mature training stack with pretrained ImageNet backbones (MobileNetV3, EfficientNet). |
| Foundation model | Phikon-v2 (Owkin, ViT-L, open weights) | A strong pathology foundation model with open weights. H-optimus-0 is gated behind an access request, and at about 4.5 GB it didn't fit the disk. |
| Inference | ONNX Runtime and OpenVINO (INT8) | Fast CPU inference on Windows and Linux with no GPU. INT8 quantisation (NNCF) shrinks models about 4× and speeds them up. |
| Classical CV | OpenCV, scikit-image, SciPy | Colour deconvolution, watershed, phase correlation, region measurements. |
| Tabular models | LightGBM, scikit-learn | Small labelled sets with interpretable features. Gradient boosting and regularised linear models suit them. |
| Reports | ReportLab (PDF) | Pure-Python PDF generation, no external services. |
| Audit | SHA-256 hash chain (stdlib) | Tamper-evident log without any infrastructure. |

---

## 2. Shared building blocks

| Block | File | What it does |
|---|---|---|
| Calibration | (input to every engine) | µm/px per objective, measured on a stage micrometer. Everything below takes `um_per_px`. |
| Nuclei engine | `core/nuclei/engine.py` | StarDist on CPU with tiling; used by Features 2, 3, 4 and 7 (see Feature 1). |
| Stain separation | `core/ihc/stain.py` | Beer–Lambert colour deconvolution into haematoxylin and DAB (or eosin). Can estimate a lab's own stain vectors (Macenko). |
| Cell-type classifier | `core/ihc/celltype.py` | MobileNetV3-Small on 64 px crops covering 32 µm around each nucleus: tumour / lymphocyte / stroma / other. |
| De-duplication | `core/ihc/dedup.py` | Phase correlation plus a correlation check between frame thumbnails. It skips a frame that overlaps an earlier one by more than 50%, and filters out slow brightness changes so camera vignetting doesn't fool it. |
| Case sessions | `core/ihc/case.py` and per-feature sessions | Running totals, click-to-correct, correction log (`corrections.jsonl`), session JSON. |
| Reports | `core/*/report.py`, `core/grading/report.py` | Per-module PDFs and the combined breast report. |

**Stain separation (used by ER/PR, Ki-67, HER2, pleomorphism, tubules).** Light passing through stained tissue follows the Beer–Lambert law. Optical density per channel, `OD = −log10(I / I₀)`, is a linear mix of each stain's colour vector times its amount. Inverting the 3 × 3 stain matrix (haematoxylin, DAB or eosin, residual) recovers the amount of each stain per pixel. Default vectors are Ruifrok & Johnston's. `estimate_hdab_matrix` fits a lab's own vectors from a few frames (Macenko method), because DAB hue and counterstain strength vary between labs.

---

## Feature 1: Nuclei detection engine

**What it's for.** Find and outline every nucleus in a frame. Everything that counts cells or measures nuclei builds on it.

**How the pathologist uses it.** It runs silently inside the other features. Nucleus outlines can be shown as an overlay (`core/nuclei/draw_overlay`). For saved images: `python app/detect_image.py image.png --um-per-px 0.25`.

**Technologies and reasons**

| Choice | Reason |
|---|---|
| **StarDist** (`2D_versatile_he`, fine-tuned) | Represents each nucleus as a star-convex polygon (32 radial distances from its centre). That separates touching nuclei better than plain segmentation and is cheap to post-process. |
| Fine-tuning on **PanNuke** (H&E, 19 tissues) plus **BC-DeepLIIF** (Ki-67 IHC) | The pretrained model missed about a third of nuclei on held-out data. IHC frames need IHC examples, since hematoxylin-counterstained negative nuclei look pale. |
| Heavy augmentation (stain shift, synthetic DAB, blur, vignetting, noise, JPEG, scale) | Microscope cameras differ from slide scanners: softer focus, uneven light, different colour. |
| **ONNX → OpenVINO INT8** | 2.5× faster than FP32 with the same accuracy (0.804 vs 0.804 detection F1 on breast). |
| Tiling (512 px tiles, 64 px margin, centre-crop stitching) | Large frames fit in memory, and nuclei on tile seams are found exactly once. |

**How it works**
1. Resize the frame to the model's scale (0.25 µm/px, a 40× view) using the calibration.
2. Cut it into overlapping 512 px tiles. For each tile the network predicts, per pixel, (a) the probability of being near a nucleus centre and (b) 32 distances to the nucleus boundary along fixed directions.
3. Keep only each tile's central part (the margin covers neighbours' context) and stitch.
4. Candidate pixels above the probability threshold become polygons. Polygon non-maximum suppression (StarDist's compiled routine) removes duplicates that overlap by IoU > 0.3.
5. Rasterise the polygons into a label map in the original frame's coordinates.

**Results (public data)**

| Test | Result |
|---|---|
| PanNuke breast (fold 3, held out), detection F1 (centres within 3 µm) | **0.841** fine-tuned (pretrained 0.749) |
| BC-DeepLIIF IHC fields (6 independent), detection F1 | **0.861** IHC-tuned (pretrained 0.671) |
| Speed, 1920×1080 frame, INT8 | 0.75–1.02 s (40 → 4 threads); 5 MP: 1.9–2.8 s |

**Status.** Built. The "done when" target (F1 ≥ 0.85 on local fields) needs 50–100 annotated local fields.

---

## Feature 2: ER / PR scoring

**What it's for.** Percentage of positive tumour nuclei, intensity, Allred score, H-score and status (Negative < 1%, Low positive 1–10%, Positive > 10%).

**How the pathologist uses it.** They rest on IHC fields. Each field adds to running totals, with a warning below 500 tumour cells. A field overlapping an earlier one by more than 50% is skipped. Clicking a nucleus flips its class or its positivity; totals update and the correction is saved. The PDF report shows scores, counts and field thumbnails. From saved frames: the `CaseSession` API, or `app/breast_case.py --er/--pr`.

**Technologies and reasons**

| Choice | Reason |
|---|---|
| Colour deconvolution, then **mean DAB optical density per nucleus** | Transparent and physically meaningful, with no black box. Intensity bins map directly to what a pathologist sees. |
| Intensity cut-offs fitted by **quadratic-weighted kappa** to pathologist labels (`calibrate()`) | The intensity bins are ordinal, so kappa penalises "weak vs strong" more than "weak vs moderate". About 200 labelled nuclei are enough. |
| **MobileNetV3-Small** cell-type classifier (64 px at 0.5 µm/px) | Tiny and fast on CPU, about 0.4 s for 1,300 nuclei. 32 µm of context separates tumour from lymphocytes and stroma. |
| Phase-correlation de-duplication | The simple version the plan asked for, before full mosaicking. It handles shifts of any size and checks matches by correlation. |

**How it works**
1. Nuclei (Feature 1) are detected, then the cell type of each nucleus is classified.
2. DAB and haematoxylin optical density is separated per pixel and averaged inside each nucleus.
3. Each nucleus is binned negative/weak/moderate/strong by DAB OD (lab-calibrated cut-offs).
4. Only tumour nuclei are counted:
   - % positive = (weak + moderate + strong) ÷ all;
   - H-score = 1×%weak + 2×%moderate + 3×%strong (0–300);
   - Allred = proportion score (0–5) + intensity score (1–3, mean intensity of positive cells);
   - status uses the ASCO/CAP cut-offs.
5. Warnings fire for under 500 cells, and near the 1% and 10% cut-offs.

**Results (public data).** Real breast IHC (BC-DeepLIIF, Ki-67 used as a stand-in, since no public labelled ER/PR exists):

| Measure | Result |
|---|---|
| Positive/negative call per nucleus | 98.8% accurate |
| % positive per field vs reference | ICC 0.90 (95% CI 0.74–0.99), 6 independent fields |
| Cell-type classifier, tumour F1 | 0.84 on H&E, 0.83 on synthetic IHC; **fails on real IHC** (called ~16% of nuclei tumour) |

**Status.** Built. The classifier is switched off for IHC (all nuclei counted) until it's retrained on labelled IHC cells. Still needed: intensity cut-offs from a pathologist, and validation on 50–100 archived cases (target ICC ≥ 0.85, category agreement ≥ 90%).

---

## Feature 3: Ki-67 index

**What it's for.** Proliferation index, with **global** and **hotspot** averaging.

**How the pathologist uses it.** Same capture flow as ER/PR. The report shows both indices and the IKWG 2021 bands: ≤ 5% low, ≥ 30% high, 6–29% intermediate with a note about poor reproducibility. Command-line tool: `python app/ki67_score.py frames/*.jpg --um-per-px 0.25`.

**Technologies and reasons.** It reuses the Feature 2 pipeline with one **binary DAB cut-off**, which `calibrate_positivity` fits by balanced accuracy on labelled nuclei. Global = all fields pooled, optionally weighted by tissue share per IKWG. Hotspot = the highest-index fields until about 500 tumour cells are counted.

**Results (public data)**

| Test | Within ±5 pts | Notes |
|---|---|---|
| BC-DeepLIIF breast (6 fields) | 5/6 fields | ICC 0.90 |
| DeepLIIF lung + bladder (598 fields, another lab) | **52%** | Cut-off fitted on breast misses about half the positives; with a lab-specific cut-off **67%** |

**Status.** Built. **Each lab must calibrate the positivity cut-off**, because it doesn't transfer between labs. The "done when" target is ±5 points on 30–50 local cases.

---

## Feature 4: HER2 scoring

**What it's for.** HER2 IHC score 0 (null / ultralow), 1+, 2+ (→ reflex ISH) or 3+, following ASCO/CAP.

**How the pathologist uses it.** Cells are coloured by membrane category and running percentages are shown. Flags appear for the 0/1+ boundary, heterogeneity, unusual staining patterns and staining artefacts. Command-line tool: `python app/her2_score.py frames/*.jpg --um-per-px 0.5`.

**Technologies and reasons**

| Choice | Reason |
|---|---|
| Cells from nuclei by **watershed on DAB**, limited to 8 µm | Membranes are DAB-rich ridges, so cell borders settle on them. The limit stops cells growing into stroma. |
| **Membrane = DAB peak per angular sector** (16 sectors) | It finds the membrane even where a cell borders stroma, which a fixed band at the cell border misses. Sectors make completeness independent of pixel counts. |
| **Background-relative** staining (minus the field's extracellular DAB) | Camera colour casts and light background staining otherwise make every cell look positive. |
| **Rule engine in plain Python** with unit tests | ASCO/CAP rules must be readable and auditable, not learned. |
| Optional **EfficientNet-B0** field cross-check | A second opinion; disagreement shows "review carefully". It's trained on the non-commercial BCI dataset, so research use only. |

**How it works**
1. Nuclei → cells → membrane DAB per sector, above background.
2. Each cell becomes **none / faint-incomplete / weak-moderate complete / strong complete**, from completeness (share of sectors stained) and intensity.
3. Rule engine over tumour cells:
   - strong complete > 10% → **3+**;
   - complete > 10% → **2+** (reflex ISH);
   - any staining > 10% → **1+**;
   - any staining > 0% → **0 ultralow**;
   - none → **0 null**.
4. Flags for near-cut-off percentages, heterogeneity, intense-but-incomplete staining, cytoplasmic or background staining, and too few cells.

**Results (public data).** BCI dataset, 693 held-out patches with pathologist-refined labels; about 10× magnification, so membranes are about 1 px wide:

| Method | Weighted kappa |
|---|---|
| Rule engine (thresholds fitted on training set) | **0.72**; 3+ recognised 91%; ISH-triage agreement 80% |
| Cross-check classifier | 0.80 against refined labels (0.89 against the original slide-level labels) |

**Status.** Built. Needed: pathologist calibration of the cell thresholds, a pathologist check of the rule text against the current ASCO/CAP guideline (`RULES_VERIFIED = False`), and validation on 100+ local cases with FISH results (target weighted kappa ≥ 0.8, with 0 vs 1+ reported separately).

---

## Feature 5: Tumour region, auto-magnification, mosaic

**What it's for.** It marks invasive tumour on the live view, recognises the objective in use, and builds a slide map (mini-map) as the pathologist sweeps.

**Technologies and reasons**

| Part | Choice | Reason |
|---|---|---|
| Tumour region | **Phikon-v2 teacher → EfficientNet-B0 student** (knowledge distillation) | The foundation model is accurate but too slow on CPU. The student learns its soft labels and runs in 0.3–0.4 s per 10× frame. |
| Training data | **BACH** slides (drawn regions) **plus BACH microscope photos** | A slide-scan-only model got just 33% of microscope photos right. Adding camera photos lifted that to 97.5%. |
| Auto-magnification | **Tiny CNN (MobileNetV3-Small 0.5×)** predicting log₂(µm/px), snapped to the nearest calibrated objective | It learns texture scale (nucleus size, fibre width) instead of fixed classes, so it works with any microscope's objectives. |
| Mosaic | **Phase correlation + correlation check** against the previous frame, **refined against the map**, relocation and **island merging** | Matching against the growing map prevents drift. Merging joins pieces split by bare glass. No feature-point tuning needed. |

**How the region model works.** Each 10× frame is cut into 224 µm patches. Each patch is classified as invasive / in situ / benign / other tissue / background, and the probabilities are drawn as a map. Overlapping patches are optional; they give a smoother map but take four times longer.

**How the mosaic works**
1. Shrink each frame to 4 µm/px and remove slow brightness variation.
2. Track it against the previous frame.
3. Refine its position against the map built so far.
4. If tracking fails, search every island; if that fails too, start a new island.
5. Whenever a frame overlaps another island, merge the two.
6. Overlays (tumour probability, hotspots) are stored in map coordinates.

**Results (public data)**

| Part | Result |
|---|---|
| Region model, held-out BACH photos (image level) | **97.5%**; cancer vs non-cancer 100% |
| Region model, held-out slide patches, tumour vs non-tumour | 80%; full sweep of a held-out slide: accuracy 0.78, sensitivity 0.58 |
| Auto-magnification | 85% overall (4× 99%, 10× 87%, 40× 87%, **20× 41%**) |
| Mosaic | 49-frame sweep: worst position error **6 µm**; a 272-frame real sweep split into 8 islands and merged back into 1 map |

**Status.** Built. Target: tumour patch accuracy ≥ 90% on **local** camera images. The 20× magnification class needs more training data.

---

## Feature 6: Mitosis counting (guided 10 HPF)

**What it's for.** Mitotic score from 10 high-power fields, using the calibrated field area.

**How the pathologist uses it.** At 10×, hotspot guidance shows where mitoses are densest. At 40× each accepted dwell frame counts as one HPF, up to 10. The same mitosis seen in two overlapping frames is counted once. The pathologist can accept, reject or add detections, and the count and score update.

**Technologies and reasons**

| Choice | Reason |
|---|---|
| **Centre-heatmap detector** (CenterNet-style, MobileNetV3-Large + light FPN) instead of RTMDet/YOLOX | MIDOG++ labels are points, with every box the same fixed size, so box regression adds nothing. This also avoids YOLOv8's AGPL licence. |
| Focal loss with extra weight on labelled **mimics** | Look-alike cells (apoptosis, lymphocytes) are the main source of false positives. |
| **EfficientNet-B0 verifier** on 96 px crops, trained with hard negatives mined from the detector | A second look at each candidate. In the end it added little, because the detector had already learned the mimics. |
| Score from **field diameter** (Nottingham table) | Thresholds depend on field area. Camera fields smaller than the table's range are scaled by area and flagged. |

**How it works**
1. The detector produces a probability heatmap at a quarter of the image resolution; local maxima are candidates.
2. The verifier rescores the candidates.
3. The HPF counter registers each frame against earlier HPFs, places mitoses in shared coordinates, and merges detections closer than 8 µm.
4. The count over 10 HPF is converted to a score via the field diameter.

**Results (MIDOG++ breast, 33 held-out slides)**

| Measure | Result |
|---|---|
| Detection F1 | **0.72** (precision 0.78, recall 0.66) |
| Mitotic score agreement | **91%**, weighted kappa 0.93 |
| 10× hotspot guidance | Chosen field holds 2.3× the average field's mitoses, but only 46% of the best field's (weak) |
| Speed | 0.5–0.95 s per 1080p 40× frame |

**Status.** Built. Needed: fine-tuning on locally annotated 40× fields, a pathologist check of the score table (`verified: false`), and a better hotspot model.

---

## Feature 7: Nuclear pleomorphism

**What it's for.** Nuclear pleomorphism score 1–3 from the 40× tumour fields the pathologist visits, plus measurements a pathologist can interpret.

**How the pathologist uses it.** They rest on at least 3 tumour fields at 40×. The assistant shows per-field and pooled scores, nuclear measurements (size, size relative to lymphocytes, anisonucleosis), and warnings: too few fields or nuclei, or fields that disagree. Command-line tool: `python app/pleomorphism_score.py fields/*.jpg --um-per-px 0.25`.

**Technologies and reasons**

| Choice | Reason |
|---|---|
| **Hand-measured nuclear features** (area and spread, eccentricity, solidity, GLCM chromatin texture, size relative to lymphocytes) | The plan's interpretable route. Nuclear size tracks pathologists' "nuclear size" judgement (correlation 0.70; pathologist vs pathologist 0.55). These are shown as measurements. |
| **Phikon-v2 tile embeddings → PCA(16) → logistic regression** for the score | Hand features plateaued at kappa 0.34. Embeddings reached **0.56** with the same patients held out. |
| **INT8 OpenVINO** Phikon-v2 (306 MB) | Same embeddings as full precision (cosine 0.9999), about 1.4 s per 1080p field. |
| Trained on **Databiox** only | Adding MITOS-ATYPIA (scanned slides, different labels) lowered kappa to 0.31. |

**How it works**
1. Each 40× field is resampled to 0.5 µm/px and cut into 112 µm tiles; glass and the eyepiece edge are dropped. Large frames are capped at 24 tiles.
2. Each tile is embedded (1,024 numbers). A field is the mean of its tiles, and a case is the mean of its fields.
3. Standardise, project onto 16 principal components, and a logistic regression gives P(1), P(2), P(3).
4. Nuclear measurements are computed from StarDist nuclei classified as tumour.

**Results (public data)**

| Model | Databiox patients held out (kappa vs grade) |
|---|---|
| Phikon embeddings (default) | **0.56 ± 0.04** (0.565 on single-camera patients, so it isn't learning the camera) |
| Hand-measured features (LightGBM) | 0.34 |
| MITOS-ATYPIA (11 slides), any model | ~0, because the 11 slides are too few |

**Status.** Built. Databiox labels are the overall grade (a proxy), so the model isn't validated against pleomorphism scores. Next data: TCGA-BRCA pleomorphism scores (1,040 cases, labels downloaded, slides waiting on the GDC catalogue) and local scores from 2–3 pathologists. Target: weighted kappa ≥ 0.6.

---

## Feature 8: Tubule formation

**What it's for.** Share of tumour forming tubules (glands with a central lumen) → score 1–3 (> 75% / 10–75% / < 10%).

**Technologies and reasons**

| Choice | Reason |
|---|---|
| **Candidate spaces + lumen classifier** (LightGBM on shape and ring features) | BreCaHAD gives *points* for 493 lumens and 610 non-lumens, so a candidate-classification design fits the labels. |
| Ring features at 0–6, 6–15 and 15–25 µm (nuclear density, eosin, spread) | A true lumen is lined by epithelium (a dense ring of nuclei). Stromal clefts, vessels and fat aren't. This is the top feature. |
| Tubule % = epithelium lining **enclosed** lumens ÷ **tumour epithelium** | Matches the Nottingham definition (share of tumour, stroma excluded). |
| White balance plus eyepiece-edge removal | Camera photos have colour casts and a black border that otherwise look like stain. |
| Accumulation in mosaic coordinates | Tubules are scored over the swept tumour, not per frame, and overlaps aren't double-counted. |

**How it works**
1. At 1 µm/px, find near-white regions (low H&E optical density) inside the field.
2. Measure each one's shape and the stain in rings around it.
3. Classify each as lumen or not.
4. Accept lumens that are mostly enclosed by epithelium and larger than 30 µm².
5. Tubular area is the epithelium within 25 µm of an accepted lumen; tumour area is all epithelium inside the invasive region.

**Results (public data)**

| Test | Result |
|---|---|
| Lumen vs non-lumen, BreCaHAD, whole cases held out | **AUC 0.87** (0.92 with Phikon added, but too slow for every candidate) |
| Tubule % vs overall grade, Databiox 10× / 20× | **No usable separation** (medians 17/18/20% and 23/22/19%) |

**Status.** Built, but **the score is not reliable yet**. The pathologist can enter the tubule score manually in the wizard. Needed: the plan's 150–300 locally annotated 10× fields, or the TCGA tubule scores once the GDC is back.

---

## Feature 9: Nottingham grade and combined report

**What it's for.** It combines the three components into the Nottingham grade and produces one breast report with grade, ER, PR, HER2 and Ki-67.

**How the pathologist uses it.** A wizard leads through Tubules (10×) → Pleomorphism (40×) → Mitoses (10 HPF) → Review → Sign. Every component can be accepted, overridden or entered manually. After signing the case is locked. The report prints the core-biopsy caveat when relevant. Command-line tool for a whole case: `python app/breast_case.py ...`.

**Technologies and reasons**

| Choice | Reason |
|---|---|
| Grade rules in plain Python: total 3–5 → 1, 6–7 → 2, 8–9 → 3 | Deterministic, testable and auditable. |
| **Hash-chained audit log** (SHA-256, append-only JSON lines) | Any edit or deletion breaks the chain and is detected. The report prints the chain fingerprint. No database or server needed. |
| Per-component provenance: assistant / accepted / edited / manual, plus validation status | The report shows exactly what was automatic and what the pathologist decided. |
| ReportLab PDF | Self-contained, works offline. |

**How it works.** `BreastCase` holds the grade, the biomarker results, snapshots and the audit log. `GradingWizard` routes frames to the Feature 8, 7 and 6 engines, stores the automatic scores, and allows edits. `sign()` requires all three components and locks the case. `write_breast_report()` renders the combined PDF.

**Status.** **Done** for the engine, report, audit and command-line tool (tests cover grade mapping, the wizard, the post-signing lock and tamper detection). The on-screen wizard is part of the camera app, which isn't built yet.

---

## Status summary and what is still needed

| Feature | Built | Public-data result | Plan's "done when" (local) | Blocked on |
|---|---|---|---|---|
| 1 Nuclei | ✅ | F1 0.84 (H&E), 0.86 (IHC); 0.75–1.0 s | F1 ≥ 0.85, < 1 s | 50–100 annotated local fields |
| 2 ER/PR | ✅ | ICC 0.90 (6 fields, Ki-67 stand-in) | ICC ≥ 0.85, ≥ 90% category | intensity calibration, IHC cell types, 50–100 cases |
| 3 Ki-67 | ✅ | 52–67% within ±5 pts on another lab's slides | ±5 pts | local cut-off, 30–50 cases |
| 4 HER2 | ✅ | kappa 0.72 (rules), 0.80 (classifier) | kappa ≥ 0.8 | threshold calibration, rule check, 100+ cases with FISH |
| 5 Region / mag / mosaic | ✅ | 97.5% photos; mosaic 6 µm | ≥ 90% local; stable mini-map | local camera patches |
| 6 Mitosis | ✅ | F1 0.72; score agreement 91% | F1 ≥ 0.7 local; human-level kappa | local annotations, score table check |
| 7 Pleomorphism | ✅ | kappa 0.56 (grade proxy) | kappa ≥ 0.6 | TCGA (GDC down), local scores |
| 8 Tubules | ✅ (score unreliable) | lumen AUC 0.87; no grade separation | kappa ≥ 0.6 | 150–300 annotated local fields or TCGA |
| 9 Grade + report | ✅ | n/a | n/a | on-screen wizard (camera app) |
| 10 Validation → product | ❌ | | | cases, pathologists, app, regulatory |

**Also still to do:** the camera capture app (Feature 0), which brings the overlays, click-to-correct screens, mini-map and wizard onto the screen; and the Feature 10 packaging pieces (installer, licence keys, minimum-spec benchmark).

---

## Repository map and how to run

```
core/        runtime engines (CPU): nuclei, ihc (ER/PR, stain, celltype, dedup, case), ki67, her2,
             region (tissue map, magnification, mosaic), mitosis, pleomorphism, tubules, grading
app/         command-line tools: detect_image, ki67_score, her2_score, pleomorphism_score, breast_case
training/    data preparation and training scripts (GPU) for every model
eval/        evaluation scripts; results in eval/results/*.json
models/      exported models (ONNX / OpenVINO / LightGBM / npz) with .json model cards
tests/       unit and integration tests (pytest; 82 tests)
docs/        this document and algorithms.html
```

**Environments.**
- Runtime and nuclei work: conda env `pathoassist` (Python 3.11, TensorFlow 2.15, StarDist, OpenVINO, LightGBM).
- PyTorch training: `~/venvs/pathoassist-torch`.
- Run tests with `~/miniconda3/envs/pathoassist/bin/python -m pytest tests/`.

**Data and model licences.** Several training sources are research-only:
- BCI states non-commercial use.
- PanNuke, MIDOG++, BACH and MITOS-ATYPIA-14 carry academic/non-commercial terms.
- BreCaHAD is CC BY 4.0.
- Databiox's Mendeley record lists no licence.
- Phikon-v2 ships with Owkin's own licence.

Verify each licence before any commercial release, and retrain on local data where a licence requires it.
