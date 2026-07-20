# Fall Detection & Fall Prediction — Research Codebase

Two independent pipelines sharing one harmonization/data layer. See
`fall_project_implementation_blueprint.md` (kept alongside this repo,
not inside it) for the full design rationale.

## Status: Stage 3 nearly complete (see PROJECT_CHECKPOINT.md for full detail)

- [x] Stage 1 — repo scaffold, config loader (`shared/config.py`), run
      logger with wandb-or-local fallback (`shared/tracking/logger.py`)
- [x] Stage 2 — KFall reader (`shared/io/readers_kfall.py`), verified
      against real data, not just fixtures (`tests/test_kfall_reader.py`)
- [x] Stage 3, Tasks 3.1-3.11 — KFall harmonization pipeline complete,
      including the visual QA pass run against real data. Stage 3 is
      done for KFall.
- [x] Stage 4 — manifest builder: `shared/manifest.py` extended into
      the full cross-dataset manifest (upsert-safe across datasets,
      `query_detection_trials`/`query_prediction_trials` helpers)
- [x] Stage 5 — SisFall harmonization: reader real-data-verified (all
      4,505 real files); orchestration wiring run against the full real
      dataset (0 quarantined, all 38 subjects calibrated via
      auto_detected); visual QA pass against real data confirms the
      calibration assumption and fall-signal shape look correct. One
      known gap: only 1 of 7 standing-initiated activity codes was
      individually eyeballed (see PROJECT_CHECKPOINT.md).
- [ ] Stage 6+ — FallAllD -- see blueprint
- [x] Detection pipeline (XGBoost baseline): windowing, 54 handcrafted
      features, subject-aware train/val/test split, training,
      evaluation, and inference all built (`detection/`,
      `scripts/train_detection_model.py`), 183 tests passing.
      **REAL final cross-dataset baseline** (both datasets, 70 real
      subjects -- full KFall 32 subjects + full SisFall 38 subjects):
      test-set accuracy 0.836, precision 0.692, recall 0.839, ROC-AUC
      0.921 on genuinely held-out subjects. Precision and recall both
      improved over the earlier SisFall-only run. Known scope limits
      (deliberate, not oversights): coarse whole-trial labels, no
      raw-signal deep model / domain adaptation, no formal LOSO/LODO
      (see PROJECT_CHECKPOINT.md).
- [ ] Stage 5+ (remaining) — FallAllD -- see blueprint
- [ ] Stage 7 — Prediction pipeline (IN PROGRESS): windowing +
      onset/impact labeling, LOSO folds, and features all REAL-DATA
      VERIFIED (see PROJECT_CHECKPOINT.md). Models, focal loss,
      lead-time metric, training loop, and LOSO training CLI script
      all built and RUN FOR REAL (10/32 folds total across runs,
      ConvLSTM, on Anshul's RTX 3050 Ti). 268/268 tests passing.
      **KEY OPEN FINDING, UPDATED**: `pre_impact` precision stays stuck
      around 0.09-0.16 regardless of loss-weight boost (2.0/1.0/0.5)
      AND regardless of window length (1.0s vs. 0.5s, both tried for
      real) -- both were plausible structural fixes and BOTH were
      ruled out this session with real experiments, not just
      hypothesized away. Per-trial diagnosis
      (`scripts/inspect_trial_predictions.py`) shows THREE DIFFERENT
      failure shapes across three real trials (early false-alarm,
      early false-alarm at a different onset distance, and a
      late/missed detection) -- not one clean, fixable mechanism.
      Aggregate analysis across a whole held-out fold
      (`scripts/analyze_temporal_errors.py`, NEW) shows false-positive
      rate is highest (60-90%) on ordinary pre-onset activity WITHIN
      fall trials -- higher than the ADL baseline (29%) -- suggesting
      the model can't yet distinguish genuine pre-fall motion from a
      staged task's simply-more-energetic lead-in. Per-trial
      normalization/leakage was checked directly in the code and ruled
      out (no scaler exists anywhere in the prediction pipeline).
      Leading remaining theory (untested): the model lacks broader
      trial context (e.g. elapsed time since trial start) needed to
      disambiguate this. **Root-cause work is intentionally PAUSED**
      in favor of a 2-day live-demo deadline -- see below.
      NOT started yet: full 32-fold run, TinyTransformer branch (zero
      real runs so far), Euler-angle channel gap.
- [ ] **Live demo layer (NEW, in progress)** — since the `pre_impact`
      issue above won't be root-caused before a live demo, built a
      smoothing/hysteresis layer (`prediction/live_smoothing.py`) that
      turns the existing checkpoint's jittery raw per-window output
      into a stable displayed alert state (EMA + hysteresis + a
      minimum-hold latch once FALL triggers) — sanity-checked against
      a real noisy trace, collapses dozens of flickering raw
      predictions into a handful of clean state changes with no added
      detection latency. `scripts/run_live_demo.py` replays a real
      trial through model+smoother with live-paced colored terminal
      output for presenting. `scripts/curate_demo_trials.py` batch-
      scans a held-out subject's real trials to find ones that are
      actually safe to demo (no early false alarm, does catch the real
      fall) rather than picking trials by hand — first hand-picked
- [ ] **Digital-twin file bridge (NEW, in progress)** —
      `scripts/run_digital_twin_bridge.py` is a long-running background
      process (runs until you Ctrl+C it, does not exit on its own) that
      watches a folder for JSON IMU-sample files, buffers them into a
      rolling window, runs them through the same checkpoint +
      `PredictionSmoother` the live demo uses, and continuously
      rewrites a single output JSON file with the current predicted
      state. Built for the digital twin (Unity dashboard) to consume,
      since its integration pattern is file-based JSON on disk.
      Input contract: `{"acc_x":..,"acc_y":..,"acc_z":..,"gyro_x":..,"gyro_y":..,"gyro_z":..}`
      per sample (or a batch list), at 100Hz, already harmonized the
      same way training data is. Output contract:
      `{"timestamp":..,"window_ready":bool,"state":"calm"|"pre_impact"|"fall"|"buffering","probabilities":{...}}`.
      Logic dry-run-verified (buffering/batching/malformed-input
      handling, correct windowed-array shape) without a real checkpoint
      — NOT yet smoke-tested against the actual digital twin's real
      output, and the exact schema above is this session's best design
      given "file-based JSON," not yet confirmed against what the
      Unity side actually produces. See PROJECT_CHECKPOINT.md's latest
      section for the full contract, reliability caveats, and the
      concrete "not yet done" list before this is a real integration.
      **Reliability reminder for anyone integrating downstream**: this
      bridge relays whatever the model actually predicts — it does not
      fix the `pre_impact` false-positive/late-detection issues
      documented above. Feeding it arbitrary live data (vs. the two
      specifically-verified curated trials,
      `kfall_SA06/T22/R05` and `kfall_SA06/T24/R01`) will surface those
      same issues live.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -e .
```

## Verify everything still works

```bash
pytest tests/ -v
python scripts/verify_setup.py
```

Both should pass with zero setup beyond `pip install -e .` — they run
entirely against the synthetic fixtures in `tests/fixtures/kfall_mock/`,
not real data.

## Important: fixtures vs. real data

`tests/fixtures/kfall_mock/` is **synthetic data** generated to match
KFall's documented filename convention and column schema — it exists to
test the reader's *parsing logic*, not to validate anything about real
KFall signal content. The label file's column names (`Task Code`,
`Trial ID`, `Fall_onset_frame`, `Fall_impact_frame`) are a best guess
based on the KFall documentation — `read_label_file()` in
`readers_kfall.py` normalizes column names defensively (lowercase,
underscores) and `_find_label_columns()` matches by substring rather
than exact string, so minor naming differences in the real files
shouldn't break it, but **the first time you run this against real
KFall label Excel files, manually verify the onset/impact frames it
extracts for 2-3 known trials against the raw spreadsheet by eye**
before trusting it further. If real column names differ enough that the
substring heuristic picks the wrong column, tighten
`_find_label_columns()` accordingly.

Once you have real KFall data locally:
1. Drop it under `data/raw/kfall/{sensor_data,label_data}/`
2. Add a `tests/test_kfall_reader_real.py` that runs the same kind of
   checks as `test_kfall_reader.py` but against `data/raw/kfall/`
   (skipped automatically if that path doesn't exist, so the fixture
   tests keep working in CI/on a fresh clone without the real dataset)
3. Update `configs/datasets/kfall.yaml` paths if your local layout
   differs from the default

## Repo layout

```
configs/            YAML configs, composed via `defaults:` (see shared/config.py)
shared/              Code used by BOTH pipelines — readers, harmonization,
                     windowing, folds, metrics, tracking. No pipeline-specific
                     logic lives here.
detection/           Fall detection pipeline (COMPLETE: windowing, features, split, XGBoost model, inference)
prediction/          Fall prediction pipeline (not yet built)
scripts/             Thin CLI entry points
tests/               pytest suite + synthetic fixtures
data/raw/            Untouched dataset downloads (gitignored)
data/harmonized/     Persisted harmonized continuous signals + manifest (gitignored)
results/             Per-run logs/configs/checkpoints (gitignored)
```
