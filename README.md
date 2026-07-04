<<<<<<< HEAD
# Fall Detection & Fall Prediction — Research Codebase

Two independent pipelines sharing one harmonization/data layer. See
`fall_project_implementation_blueprint.md` (kept alongside this repo,
not inside it) for the full design rationale.

## Status: Stage 1 + 2 complete

- [x] Stage 1 — repo scaffold, config loader (`shared/config.py`), run
      logger with wandb-or-local fallback (`shared/tracking/logger.py`)
- [x] Stage 2 — KFall reader (`shared/io/readers_kfall.py`), verified
      against synthetic fixtures matching KFall's documented schema
      (`tests/test_kfall_reader.py`)
- [ ] Stage 3 — harmonization on KFall only (unit conversion, axis
      alignment, resample, filter)
- [ ] Stage 4 — manifest builder for KFall
- [ ] Stage 5+ — see blueprint

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
detection/           Fall detection pipeline (not yet built)
prediction/          Fall prediction pipeline (not yet built)
scripts/             Thin CLI entry points
tests/               pytest suite + synthetic fixtures
data/raw/            Untouched dataset downloads (gitignored)
data/harmonized/     Persisted harmonized continuous signals + manifest (gitignored)
results/             Per-run logs/configs/checkpoints (gitignored)
```
=======
# ShravanAI
>>>>>>> b9692449f3ecc668e812bca5f1dfaea4fe95a2ca
