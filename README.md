# PathoAssist

PathoAssist is decision support for breast pathology. It runs on an ordinary CPU and reads images from a camera on the microscope. The pathologist sweeps the slide as usual, and the assistant:
- counts and scores nuclei for ER, PR, Ki-67 and HER2;
- counts mitoses;
- scores nuclear pleomorphism and tubule formation;
- combines the three components into a Nottingham grade;
- writes a signed PDF report backed by a tamper-evident audit log.

**Status: research prototype.** Every model has been trained and measured on public datasets only. None of the plan's local acceptance targets ("done when") has been measured, and none of the results may be used for diagnosis. See [Current status](#current-status).

| Read next | What it contains |
|---|---|
| [docs/FEATURES.md](docs/FEATURES.md) | Each feature: what it does, technologies chosen and why, how it works, results |
| [docs/algorithms.html](docs/algorithms.html) | Interactive visual explainer of every algorithm (open in a browser) |
| [docs/UPDATES.md](docs/UPDATES.md) | What was built, changed or decided, in order; add to it with every change |
| [docs/BUGS.md](docs/BUGS.md) | Bugs found and fixed (cause and fix), known open issues, and environment traps |

---

## Quick start

### 1. Environments

There are two Python environments, because TensorFlow (StarDist training) and PyTorch (all other training) pin conflicting libraries.

| Environment | Used for | Python |
|---|---|---|
| conda `pathoassist` (`~/miniconda3/envs/pathoassist`) | Runtime, tests, StarDist training/export, evaluation | 3.11 |
| venv `~/venvs/pathoassist-torch` | PyTorch training: cell type, HER2, region, magnification, mitosis, Phikon-v2 embeddings | 3.11 |

Creating the runtime environment from scratch:

```bash
conda create -n pathoassist python=3.11 -y
conda activate pathoassist
pip install -r requirements-training.txt      # includes requirements-runtime.txt
pip install lightgbm scipy scikit-image reportlab   # used by core/ but not yet listed (see docs/BUGS.md)
```

The desktop app bundle pins exact versions in [packaging/linux/requirements-app.txt](packaging/linux/requirements-app.txt).

### 2. Run the tests

```bash
~/miniconda3/envs/pathoassist/bin/python -m pytest tests/ -q     # 82 tests, about 45 s
```

### 3. Try the command-line tools

Until the live capture app is finished, each feature runs on saved camera frames. Every tool needs the **calibrated scale in µm per pixel** for the objective used. All scoring depends on it, so never guess it.

```bash
# Feature 1: nuclei outlines + per-nucleus CSV
python app/detect_image.py frame.png --um-per-px 0.25

# Features 2-3: Ki-67 (ER/PR uses the same pipeline, see core/ihc)
python app/ki67_score.py frames/*.jpg --um-per-px 0.25 --case-id K-001 --out cases/K-001 --celltype

# Feature 4: HER2
python app/her2_score.py frames/*.jpg --um-per-px 0.5 --case-id H-001 --thresholds models/her2_thresholds_bci.json

# Feature 7: pleomorphism score from 40x tumour fields
python app/pleomorphism_score.py fields/*.jpg --um-per-px 0.25

# Features 2-4 + 6-9: complete breast case -> grade + combined PDF report
python app/breast_case.py --case-id B-001 --specimen "core biopsy" \
    --tubules frames/he_10x --um10 0.97 \
    --pleomorphism frames/he_40x --hpf frames/he_hpf --um40 0.24 \
    --er frames/er --pr frames/pr --her2 frames/her2 --um-her2 0.5 --ki67 frames/ki67 --um-ihc 0.24 \
    --pathologist "Dr A"
```

Each folder in `breast_case.py` is optional. If one is left out, that module is skipped, and a missing grade component can be supplied with `--override mitoses=3`. The grade is signed only when `--pathologist` is given and all three components are present. A worked example is in [demo_outputs/breast_case_demo/](demo_outputs/breast_case_demo/).

### 4. Desktop app (early)

```bash
python -m app.desktop               # USB microscope camera: live view, freeze, capture, nuclei detection
python -m app.desktop --self-test   # headless: checks the bundled models load and detection runs
```

The desktop app is the start of the camera app (Feature 0). So far it only runs nuclei detection. Scoring, the overlays, the mini-map and the grading wizard still live in `core/` and the CLI tools. [packaging/](packaging/) holds the PyInstaller entry point and Linux AppImage files.

---

## Repository layout

```
app/              command-line tools (one per feature) and app/desktop (PySide6 camera app)
core/             runtime engines; CPU only; no training dependencies
  nuclei/         F1  StarDist detector (OpenVINO INT8 / ONNX), tiling, calibration
  ihc/            F2  stain deconvolution, ER/PR scoring, cell type, de-duplication, case, report
  ki67/           F3  Ki-67 global / hotspot session and report
  her2/           F4  membrane analysis (16 sectors), ASCO/CAP rule engine, session
  region/         F5  tumour region map, auto-magnification, mosaic
  mitosis/        F6  heatmap detector + verifier, guided 10 HPF counting, score table
  pleomorphism/   F7  Phikon-v2 tile embeddings + score model, nuclear measurements
  tubules/        F8  lumen candidates + classifier, tubule % and score
  grading/        F9  Nottingham grade, case + wizard, hash-chained audit log, PDF report
training/         data preparation and training scripts (GPU), one folder per model
  data/           public datasets (about 10 GB, not for git)
  runs/           training checkpoints
eval/             evaluation and benchmark scripts; results in eval/results/
models/           exported runtime models; each has a .json "model card" (data, metrics, thresholds)
tests/            pytest suite
docs/             engineering documentation
packaging/        desktop app bundling (PyInstaller, Linux AppImage)
logs/             logs of past training / export / evaluation runs
demo_outputs/     example overlays and reports
```

### Conventions to keep

- **Physical units everywhere.** Every function that takes an image also takes `um_per_px`. Each model works at a fixed scale (nuclei 0.25 µm/px, tubules 1.0, Phikon tiles 0.5), and frames are resized to that scale. Thresholds are in µm, µm² or mm², never pixels.
- **A model card for every model.** `models/<name>.json` records the training data, the held-out metrics, thresholds and working scale. Runtime code reads thresholds from the card, so update the card whenever you retrain.
- **`core/` is inference only.** No TensorFlow or PyTorch imports; models run through OpenVINO, ONNX Runtime or LightGBM.
- **Hold out by patient.** Cross-validation and test splits are grouped by patient, slide or case. Image-level splits inflate results, and this has happened on this project (see BUGS.md, BC-DeepLIIF).
- **Warnings, not silent defaults.** Low cell counts, values near cut-offs, unvalidated models and missing calibration all appear as warnings on screen and in the report.
- **Every user-facing result can be overridden** by the pathologist, and each override is written to the audit log.

---

## Current status

| Feature | Built | Public-data result | Next step |
|---|---|---|---|
| 1 Nuclei | ✅ | F1 0.84 (H&E), 0.86 (IHC); 0.75–1.0 s/frame | 50–100 annotated local fields |
| 2 ER/PR | ✅ | ICC 0.90 (6 fields) | intensity calibration; 50–100 local cases |
| 3 Ki-67 | ✅ | 52–67% of fields within ±5 points | local cut-off; 30–50 cases |
| 4 HER2 | ✅ | κ 0.72 rules, 0.80 classifier | threshold calibration; 100+ cases with FISH |
| 5 Region / magnification / mosaic | ✅ | 97.5% of photos; mosaic error 6 µm | local camera patches |
| 6 Mitosis | ✅ | F1 0.72; score agreement 91% | local annotations; check the score table |
| 7 Pleomorphism | ✅ | κ 0.56 (grade used as a proxy label) | TCGA-BRCA labels (waiting on GDC); local scores |
| 8 Tubules | ✅, score unreliable | lumen AUC 0.87; does not separate grades | 150–300 annotated local 10× fields |
| 9 Grade + report | ✅ | n/a | on-screen wizard in the camera app |
| 0 Camera app | started | nuclei detection only | overlays, wizard, mini-map |
| 10 Validation / product | ❌ | n/a | local cases, pathologists, regulatory work |

**Licences:** several training datasets are for research or non-commercial use only. Check them before any commercial release (see the licence section of FEATURES.md).

## Hardware used for development

- Training: 3 × Quadro RTX 5000 (16 GB). Pick a GPU with `CUDA_VISIBLE_DEVICES`.
- Runtime targets a 4-core CPU; benchmarks used 4–40 threads.
- **Disk is the main constraint.** The server's disk is about 99% full, and adding datasets has filled it before. Run `df -h ~` before downloading or unpacking anything.

## Making changes

1. Run `pytest tests/` before and after.
2. Add a line to [docs/UPDATES.md](docs/UPDATES.md) for every feature, model retrain, threshold change or design decision.
3. When you find a bug, add it to [docs/BUGS.md](docs/BUGS.md) with its symptom, cause and fix. A known bug that isn't fixed yet goes under "Open issues".
4. If a model, threshold or result changes, update its model card, `docs/FEATURES.md` and the figures in `docs/algorithms.html`.
