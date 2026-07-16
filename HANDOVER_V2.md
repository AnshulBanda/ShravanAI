# ShravanAI — Handover Doc v2 (for a fresh Claude chat)

**Paste this whole file as your first message in the new chat**, then say what you want to work on next. It gives Claude everything it needs to pick up without re-deriving context.

This supersedes the original `HANDOVER.md` (which covered up through the completed detection pipeline) -- that doc is still in the repo root for historical reference, but this one reflects everything since, through the first real prediction-pipeline training runs.

---

## What this project is

ShravanAI: an elderly fall detection/prediction system using wearable IMU (accelerometer + gyroscope) data. Two public real-world datasets — **KFall** (Korean, 32 subjects, lab-simulated falls) and **SisFall** (Colombian, 38 subjects, lab-simulated falls) — harmonized into a common format. Two downstream pipelines:
1. **Detection** (binary fall/ADL, both datasets) — **COMPLETE**, real trained XGBoost model, ROC-AUC 0.921.
2. **Prediction** (3-class pre-impact, KFall-only, pre-impact classification before a fall's actual impact) — **IN PROGRESS**, this doc's main focus.

- **GitHub**: `https://github.com/AnshulBanda/ShravanAI.git`
- **Local path**: `C:\Users\Anshul Banda\Desktop\ShravanAI\fall-project`
- **Source of truth for full detail**: `PROJECT_CHECKPOINT.md` in the repo root. This handover doc is a summary/orientation layer — `PROJECT_CHECKPOINT.md` has the exhaustive stage-by-stage history, every real bug found and fixed, and the reasoning behind every design decision, all the way through the events below. **Read it before doing anything nontrivial** — specifically, read the LATEST Stage 7 entries at the bottom; there are several, each documenting a distinct sub-stage.
- **Blueprint docs**: `fall_project_implementation_blueprint.md` and `fall_detection_prediction_pipelines.md` (kept alongside the repo, not committed to it — ask the person to paste relevant sections if a design question comes up that these would settle; don't assume you have them already).

## Current status (as of this handover)

| Stage | Status |
|---|---|
| 1–6: Detection pipeline (harmonization + XGBoost baseline) | **COMPLETE**, real trained model, ROC-AUC 0.921 |
| 7a: Prediction — windowing + onset/impact labeling | **COMPLETE, real-data verified** (348,941 real KFall windows; frame-exact spot check on SA06 T22 R01) |
| 7b: Prediction — feature engineering (rolling accel-mag/jerk/tilt aux channels) | Built, structurally tested, **NOT yet spot-checked against real fall-trial signal** |
| 7c: Prediction — LOSO folds + PyTorch data loading | **COMPLETE, real-data verified** (32 folds against real KFall, matches subject count) |
| 7d: Prediction — model architectures (ConvLSTM, TinyTransformer), focal loss, lead-time metric, training loop | Built, unit-tested, smoke-tested |
| 7e: Prediction — real training runs | **8/32 folds run for real** (ConvLSTM only, TinyTransformer untouched), plus a 2-fold hyperparameter sweep |
| 7f: Prediction — diagnostic tooling | Built (`scripts/inspect_trial_predictions.py`), **not yet run against a real checkpoint** |
| FallAllD (3rd dataset) | Not started |

## THE KEY OPEN FINDING — read this before doing anything else

Real training on real KFall data (RTX 3050 Ti GPU) converges cleanly and reproducibly: 8 different held-out LOSO subjects all show early stopping kicking in sensibly (8-16 epochs), stable val_loss (~0.055-0.062). But every single fold shows the same problem:

- `pre_impact` classification precision stuck at **0.08-0.16** even after real convergence (not an undertraining artifact).
- Mean lead time stuck at **~2.3-3.4 seconds** across every fold and every hyperparameter tried so far. The REAL KFall onset→impact gap is only **~0.6-1.0 seconds** (e.g. the project's repeatedly-used real spot-check trial, SA06/T22/R01, has onset=130, impact=208 frames = 780ms). So the model is firing its early-warning flag roughly 3-4x earlier than any real pre-fall motion could be occurring — meaning it's very likely just guessing broadly, not detecting anything real.

**A 3-point hyperparameter sweep already ruled out the loss weighting as the cause**: `pre_impact_extra_boost` values of 2.0, 1.0, and 0.5 all show basically the same lead-time number (2774ms → 2680ms → 2310ms) despite big precision/recall shifts elsewhere. Something more structural is going on.

**The concrete next step, already tooled up for but not yet executed**: `scripts/inspect_trial_predictions.py` was built specifically to investigate this — it loads one saved model checkpoint and prints every window of one real trial, in time order, with the TRUE label, the model's PREDICTED label, and full 3-class confidence, with onset/impact frames marked inline. Real checkpoints already exist on Anshul's machine at `results/prediction_model/checkpoints/convlstm_boost{2.0,1.0,0.5}_kfall_SA06.pt` (and `_kfall_SA07.pt`) from the runs already completed. **Running this script against one of those real checkpoints, on the real SA06 T22 R01 trial, is the natural first action in a new session** — it'll show directly whether `pre_impact` firing is scattered randomly across the whole trial or clustered somewhere specific but mistimed, which should explain the sweep finding above.

## Repo layout (additions since the original HANDOVER.md)

```
prediction/                    Prediction pipeline (Stage 7)
  windowing.py                 Dense window config (1.0s window / 0.1s stride)
  labelers.py                  3-class onset/impact labeling (non_fall/pre_impact/fall)
  dataset.py                   Windows manifest builder + window loader (CHANNELS = 6 raw acc/gyro)
  features.py                  Rolling auxiliary channels: accel_mag, jerk, tilt_deviation_deg
  loso.py                      Leave-one-subject-out fold construction (the real leakage-prevention mechanism)
  torch_dataset.py              PyTorch Dataset + trial-grouped batch sampler
  losses.py                    3-class focal loss + default_alpha_weights()
  lead_time.py                 The core reported metric (ms before impact model first flags pre_impact)
  training.py                   train_one_fold() -- reusable across both model branches, with early stopping
  models/
    convlstm.py                 CNN (32->64->128, kernels 5/5/3) -> unidirectional LSTM -> 3-class head
    tiny_transformer.py         ViT-tiny style, d_model=48, 3 layers/3 heads, CLS token
scripts/
  train_prediction_model.py     CLI: runs LOSO training across folds, saves checkpoints + JSON reports
  inspect_trial_predictions.py  NEW, diagnostic: per-window prediction trace for one real trial
results/prediction_model/
  checkpoints/                  Saved model weights per fold (gitignored, real files exist locally)
  *_loso_report.json            Per-run JSON reports (filename now encodes model/boost/epochs/folds)
```

`shared/windowing.py` also now exists (the boundary-generation logic moved here from `detection/windowing.py` so `prediction/windowing.py` could reuse it — `detection/windowing.py` is now a thin re-export, zero behavior change, verified against the full pre-existing test suite).

## Key things worth knowing before touching anything (Stage 7 additions)

1. **Known channel gap**: harmonization drops KFall's Euler angles for every trial (by design, for the detection pipeline) — the prediction pipeline currently only has the same 6 acc/gyro channels as detection, not Euler, despite the blueprint's aspiration otherwise. Flagged, not fixed — fixing it means touching the already-verified harmonization pipeline (Stage 3), which hasn't happened.
2. **Leakage prevention is `prediction/loso.py`'s job specifically** — kept conceptually separate from `torch_dataset.py`'s trial-grouped batch sampler, which only affects training dynamics within an already-clean split, not leakage prevention itself. Don't conflate the two if extending either.
3. **`pip install -e .` needs to be current** — it silently never registered the `prediction` package for a while (editable-install finder was generated before that package existed on disk), invisible under `pytest` but real when running scripts directly. Already fixed once; if `ModuleNotFoundError: No module named 'prediction'` ever reappears when running a script directly, re-run `pip install -e .`.
4. **Training on this project's GPU (RTX 3050 Ti, 4GB) runs ~4-9 min/epoch** — a full 32-fold run at up to 20-50 epochs each is realistically many hours. Running in an open foreground terminal (not `nohup`/backgrounded) proved most reliable on this Windows/MINGW64 setup after real background-job reliability issues. The training script now prints live per-epoch progress + a running ETA specifically because of this.
5. **Report/checkpoint filenames encode run config** (`{model}_boost{X}_ep{Y}_{fold_tag}...`) — this was a real near-miss fix; a fixed filename per model almost caused one sweep run to silently overwrite another's results before they'd been reviewed.

## How Claude and I have been working (please keep doing this)

Same as the original handover doc's section — still fully accurate:
- Claude works in its own sandbox (no access to the person's local files/terminal), builds + tests against fixtures/synthetic data, packages changed files into a **zip with folder structure preserved** (the person specifically asked for this — always zip with real subdirectories, not flat files), and the person copies them into their local repo.
- **Real-data verification matters a lot.** Claude writes code against fixtures, but the real proof is the person running it locally and pasting back real output. Several real bugs (a missing `pandas` import, a stale editable-install, a test's flawed assumption about BatchNorm buffers under `lr=0`, an output-filename collision) were only caught this way — don't treat fixture-passing as "done."
- **Git commit messages: single line only.**
- **`PROJECT_CHECKPOINT.md` gets updated after every real milestone.** Keep doing this.
- The person pastes terminal output directly and expects Claude to interpret it, flag anything surprising, and give exact next commands to run.
- **Give exact copy-pasteable commands** after every response that involves the person running something locally — this was explicitly requested partway through the last session.
- The person is relatively new to some of the operational side (background processes, GPU setup, buffering) — be ready to explain those plainly alongside the ML content, not just the ML content itself.

## Natural next steps (pick one, or something else)

1. **Run `scripts/inspect_trial_predictions.py` against a real checkpoint** (see "THE KEY OPEN FINDING" above) — the concrete, already-tooled-up next action.
2. Based on what that shows, likely directions: window length/stride reconsideration, real-data spot-check of the auxiliary tilt/jerk channels (never actually verified against a real fall trial's signal, only synthetic vectors), or revisiting the 3-class boundary itself.
3. Finish the full 32-fold run (any boost value) — only 8/32 plus a 2-fold sweep done so far.
4. Run TinyTransformer for real — zero real training runs on it yet, only ConvLSTM.
5. The Euler-angle channel gap — still an open, undecided item.
6. FallAllD (3rd dataset) — not started, same as before.

---

*Generated as a handover doc at the end of a long chat session covering Stage 7's prediction pipeline build-out and first real training runs. If anything here seems inconsistent with `PROJECT_CHECKPOINT.md`, trust the checkpoint file — it's the more detailed, more authoritative source, and has several dated Stage 7 sub-sections documenting this session's work in full.*
