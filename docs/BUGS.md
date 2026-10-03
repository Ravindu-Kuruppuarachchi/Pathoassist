# Bugs, fixes and known issues

This file has three parts:
1. **Open issues:** known problems that are not fixed yet. Check here before starting work.
2. **Fixed bugs:** each with its symptom, root cause, fix and how to recognise a regression.
3. **Environment traps:** setup problems that cost time once already.

**How to add a bug.** Give it the next number (`B-0NN`). Put it under Open issues until it is fixed, then move it to Fixed bugs with the date:

```markdown
### B-0NN — one-line title  (fixed YYYY-MM-DD | open)
**Symptom:** what you saw (output, numbers, error text).
**Cause:** the actual root cause, not the first guess.
**Fix:** what changed, with file links.
**Check:** the test or command that proves it, so a regression is noticed.
```

When the fix changes a model or a threshold, also log it in [UPDATES.md](UPDATES.md).

---

## 1. Open issues

### O-01 — Runtime requirements file is incomplete  (open)
`requirements-runtime.txt` does not list `lightgbm`, `scipy`, `scikit-image` or `reportlab`, all of which `core/` imports. It also omits `PySide6`, which the desktop app needs. A fresh install from the file fails at import. Add them, pinned to the versions in the `pathoassist` conda env (`pip freeze`).

### O-02 — Desktop bundle is missing the newer models  (open)
`RUNTIME_MODEL_FILES` in [app/desktop/paths.py](../app/desktop/paths.py) predates Features 7–9. It leaves out:
- `phikon_v2_int8w.xml` and `phikon_v2_int8w.bin`
- `pleomorphism_phikon.npz` and `pleomorphism_phikon.json`
- `tubule_lumen_lgbm.txt`, `tubule_lumen_lgbm.json` and `tubule_lumen_phikon.npz`

Its comment also refers to `packaging/pathoassist.spec`, which is not in the repo. Phikon-v2 adds about 300 MB to the bundle; decide whether it ships before adding it.

### O-03 — Git repository not set up properly  (open)
- There are no commits yet.
- Compiled `__pycache__/*.pyc` files are staged, and there is no `.gitignore`.
- `models/` (about 490 MB) and logs are staged as normal files.

Before the first commit:
- add a `.gitignore` covering `__pycache__/`, `training/data/` (about 10 GB), `training/runs/` and `*.pt`;
- unstage the `.pyc` files;
- put model binaries in Git LFS or a release store.

### O-04 — IHC cell-type classifier does not transfer to real IHC  (open)
**Symptom:** the tumour/non-tumour filter for ER/PR/Ki-67 is unreliable on real IHC fields.
**Cause:** the classifier was trained on H&E recoloured to look like IHC, and that does not carry over.
**Fix needed:** labelled IHC cells, from SHIDC or about 1–2k nuclei labelled in QuPath.

### O-05 — Tubule score does not separate grades  (open, by design until local data exists)
The lumen classifier works (AUC 0.87), but on Databiox the tubule score per patient does not track grade. The Spearman correlation is −0.12, and median tubule % is 16.6, 17.6 and 19.8 for grades 1, 2 and 3. See `eval/results/tubules_databiox.json`.

Likely causes:
- the camera photos are low resolution at 10× (0.96 µm/px);
- Databiox gives only the overall grade, not a tubule score.

The report labels the score as not validated. **Needs:** 150–300 annotated local 10× fields, or TCGA tubule scores.

### O-06 — 20× auto-magnification is unreliable  (open)
Only 41% of 20× frames are recognised correctly; most are confused with 10× or 40×. Until it improves, the app should ask the user to confirm the objective whenever the estimate is 20×.

### O-07 — Thresholds are public-data placeholders  (open)
The following must be calibrated by a pathologist on local slides before any clinical use:
- ER/PR intensity cut-offs (DAB OD 0.15 / 0.35 / 0.60);
- HER2 membrane thresholds (`models/her2_thresholds_bci.json`);
- the Ki-67 positivity cut-off;
- the mitotic score table.

### O-08 — TCGA-BRCA blocked on the GDC API  (open, external)
The NCI GDC API returned 503 every time it was tried. The labels (Thennavan Data S2) are downloaded to the session scratchpad as `tcga/mmc3.xlsx`; copy them into `training/data/` before they are lost. The slides are on the AWS open-data mirror, but mapping cases to slide files needs GDC. An alternative is the NCI Imaging Data Commons (DICOM).

### O-09 — algorithms.html has not been tested in a browser  (open)
The server has no browser or JavaScript runtime, so the page was written but never run. Open it locally and check each figure.

### O-10 — 4 pytest warnings  (open, not investigated)
`pytest tests/` passes all 82 tests with 4 warnings. Check whether any are deprecations that will break with newer library versions.

---

## 2. Fixed bugs

### B-001 — `onnx` import fails with an `ml_dtypes` error  (fixed)
**Cause:** TensorFlow 2.15 pins an old `ml_dtypes`, and onnx ≥ 1.17 needs a newer one.
**Fix:** pin `onnx==1.16.2` in [requirements-training.txt](../requirements-training.txt).

### B-002 — `stardist` import fails: no module `pkg_resources`  (fixed)
**Cause:** setuptools ≥ 70 removed `pkg_resources`, which stardist 0.9.1 still imports.
**Fix:** pin `setuptools<70`.

### B-003 — ER/PR/Ki-67 % positive about 11 points too high  (fixed)
**Symptom:** % positive on real IHC fields was consistently above the reference.
**Cause:** the H&E-trained nuclei model missed pale (negative) nuclei much more often than brown ones, which inflated the positive share.
**Fix:** fine-tuned the nuclei model on real IHC as well (`models/stardist_he_ihc_ft_*`). It is now the default for both stains, with no loss on H&E.
**Check:** `eval/evaluate_ihc_deepliif.py`.

### B-004 — BC-DeepLIIF results looked better and tighter than they were  (fixed)
**Cause:** the validation set has 6 original fields; the other files are flipped, rotated and zoomed copies. All 65 files were counted as independent fields.
**Fix:** evaluate on the 6 originals only. This is why the ER/PR ICC 0.90 has a wide confidence interval.
**Lesson:** look for augmented copies in any public validation set before counting samples.

### B-005 — Mitosis verifier crashes on peaks at the image edge  (fixed)
**Cause:** heatmap peaks can land up to 2 px outside the image after rescaling, which produces an invalid verifier crop.
**Fix:** clamp peak coordinates to the image before cropping (`core/mitosis/detector.py`).

### B-006 — Reading MITOS-ATYPIA / BACH TIFFs fails in the torch venv  (fixed)
**Cause:** the installed `tifffile` needs `zarr` 3.
**Fix:** upgrade `zarr` in `~/venvs/pathoassist-torch`.

### B-007 — Phikon-v2 download fails with 401  (fixed)
**Cause:** an expired Hugging Face token is configured on the server and is sent even for public models.
**Fix:** load with `token=False`, or set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.

### B-008 — De-duplication flags different patients as 93–96% overlapping  (fixed 2026-10-03)
**Symptom:** on Databiox camera photos, unrelated fields were rejected as duplicates.
**Cause:** the dark eyepiece circle and the uneven lighting were the same in every photo and dominated the phase correlation and NCC.
**Fix:** high-pass filter the thumbnail before matching (`g - GaussianBlur(g, 8)` in [core/ihc/dedup.py](../core/ihc/dedup.py)).
**Check:** zero false matches across Databiox patients; `tests/test_ihc.py`.

### B-009 — `lightgbm` missing in the torch venv  (fixed)
Training scripts that combine Phikon embeddings with the LightGBM lumen model failed to import it. Installed it in `~/venvs/pathoassist-torch`.

### B-010 — `NottinghamGrade.set_auto` raises "got multiple values for argument 'score'"  (fixed 2026-10-03)
**Cause:** extra keyword arguments were merged into the call and collided with `score`.
**Fix:** extra values are now passed as a `details` dict ([core/grading/nottingham.py](../core/grading/nottingham.py)).

### B-011 — HER2 demo scored at the wrong scale  (fixed 2026-10-03)
**Cause:** `breast_case.py` used the ER/PR/Ki-67 scale (`--um-ihc`) for HER2 frames, which were taken at a different magnification.
**Fix:** added `--um-her2`.

### B-012 — Ki-67 warnings missing from the combined report  (fixed 2026-10-03)
**Cause:** Ki-67 keeps its warnings in a nested result object, and the report read only the top-level list.
**Fix:** [core/grading/report.py](../core/grading/report.py) now collects the nested warnings as well.

### B-013 — `pkill` in a script kills the shell running it (exit 144)  (fixed)
**Cause:** `pkill -f <pattern>` also matched the command line of the shell that ran it.
**Fix:** match the Python executable path, from a separate script.

### B-014 — Loop over file names breaks in zsh  (fixed)
**Cause:** zsh does not word-split unquoted variables the way bash does, so `for f in $FILES` sees a single item.
**Fix:** use arrays or explicit commands, or run the script with `bash`.

### B-015 — No lumens found on camera photos  (fixed 2026-10-03)
**Cause:** camera white balance made glass and lumina look slightly stained, so nothing passed the white-space threshold (OD < 0.22).
**Fix:** `white_balance()` in [core/tubules/lumen.py](../core/tubules/lumen.py) scales each channel so the 99.5th percentile reads as white.

### B-016 — Tubule % was diluted by stroma  (fixed 2026-10-03)
**Cause:** the denominator was all tissue in the tumour region, including stroma. Nottingham defines tubule formation as a share of tumour epithelium.
**Fix:** the denominator is now the epithelium mask (local nuclear density ≥ 0.30) inside the tumour region.

### B-017 — Black eyepiece border counted as dense tissue  (fixed 2026-10-03)
**Cause:** pixels outside the eyepiece circle have a very high optical density, so they counted as stained tissue.
**Fix:** `field_mask()` in `lumen.py` excludes them from both the tissue and white-space masks.

### B-018 — Hundreds of tiny false lumens per field  (fixed 2026-10-03)
**Cause:** small gaps between cells and retraction spaces passed the classifier.
**Fix:** a lumen is accepted only if it is at least 30 µm² and at least 60% of its inner (0–6 µm) ring is epithelium.
**Check:** `tests/test_tubules.py`.

---

## 3. Environment traps

- **Disk is about 99% full** (12 GB free out of 935 GB at last check). Unpacking a dataset has filled it before, which crashes training and corrupts downloads. Run `df -h ~` before any download. Delete archives after extracting them, and clear the pip cache (`pip cache purge`) when space runs low.
- **Use two environments, never one.** Keep TensorFlow (conda `pathoassist`) and PyTorch (`~/venvs/pathoassist-torch`) apart. Installing either framework into the other's environment breaks pinned versions.
- **Keep `numpy<2`.** TensorFlow 2.15 and stardist 0.9.1 break with NumPy 2.
- **Choose the GPU explicitly** with `CUDA_VISIBLE_DEVICES=0` (or 1 or 2). Jobs sometimes run in parallel on different GPUs.
- **External data sources can be down for hours or days.** NCI GDC was down for days. A TCIA/NBIA manifest is radiology (MRI), not pathology slides.
