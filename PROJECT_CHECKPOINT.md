# Project Checkpoint — Fall Detection & Prediction Pipelines

Last updated: after Stage 3, Task 3.6 (group-average calibration fallback)

**Purpose of this file:** a durable, factual record of decisions and
verified-against-real-data findings, kept in the repo itself
(`PROJECT_CHECKPOINT.md` at repo root) rather than relied on from
conversation memory. Paste or upload this file at the start of a new
session with Claude to re-establish context accurately instead of
risking Claude reconstructing it from a compressed/incomplete memory of
past conversations. Update this file at the end of each task.

---

## Project structure

Two independent pipelines sharing one harmonization layer:
- **Detection**: binary fall/ADL, trained across KFall + SisFall + FallAllD
- **Prediction**: pre-impact fall prediction, KFall-only (frame-level
  onset/impact labels required; explicitly NOT fabricated for the other
  two datasets)

Repo layout: `shared/` (harmonization, readers, tracking — used by both
pipelines), `detection/` and `prediction/` (not yet built), `configs/`,
`data/`, `scripts/`, `tests/`. Full rationale in the original blueprint
document (kept alongside this repo, not inside it — if you don't have
it anymore, ask Claude to regenerate the "implementation blueprint" and
"Stage 3 sprint plan" documents from earlier in this project's history).

Dev workflow: code in GitHub, developed locally in VS Code, Kaggle for
GPU training, Colab as overflow. Real datasets (KFall, SisFall,
FallAllD) are NOT stored in git — `data/raw/` is gitignored.

---

## Progress so far

### Stage 1 — COMPLETE
- `shared/config.py`: Hydra-lite YAML config loader with `defaults:` composition (OmegaConf-based)
- `shared/tracking/logger.py`: experiment logger, tries W&B, falls back to local `results/<project>/<run>/{config_resolved.yaml,metrics.json}` if wandb unavailable
- `configs/base.yaml`, `configs/datasets/kfall.yaml`

### Stage 2 — COMPLETE (verified against real data, not just fixtures)
- `shared/io/readers_kfall.py`: parses KFall sensor CSVs + label Excel files
- `tests/test_kfall_reader.py`
- Real-data verification done end-to-end (see "Verified facts about real KFall data" below) — this took three rounds of fixing real mismatches, all now handled and tested.

### Stage 3 — IN PROGRESS
Frozen design (do not redesign further, per explicit instruction):
- Unit conversion: common interface, KFall is a no-op
- Target sampling rate: **100 Hz**
- Filter: 4th-order Butterworth **band-pass, 0.5–20 Hz** (chosen over an
  originally-proposed 5 Hz low-pass, because 5 Hz risks attenuating the
  actual fall-impact transient — see literature citation in conversation
  history: a sacral-IMU study found 5 Hz cutoff only weakly correlated
  with true peak impact force vs. moderate correlation at 10 Hz)
- Axis alignment: T01 preferred calibration source → auto-detected
  stationary-segment fallback → group-average last resort
- Order of operations: unit conversion → resample → axis alignment → filter
  (this order was explicitly checked: axis alignment is a static
  per-subject rotation, and rotation mathematically commutes with both
  resampling and filtering since those operate independently over time
  per-channel while rotation only mixes channels at a fixed instant — so
  this order is a valid implementation choice, not the only mathematically
  possible one)
- Continuous harmonized signals persisted to disk (`data/harmonized/<dataset>/<trial_id>.parquet`);
  **windowing is always dynamic at dataset-construction time, never persisted**
- Every harmonized trial passes automated validation + provenance
  logging before being accepted (planned: Task 3.8)

**Sprint plan**: 11 tasks (3.1–3.11), each independently
completable/testable/committable. Full task-by-task spec (objective,
files, API, tests, validation, commit message) was generated earlier in
this project's history — ask Claude to regenerate the "Stage 3 sprint
plan" document if you no longer have it.

#### Task 3.1 — COMPLETE
`shared/harmonize/units.py`: `UnitConverter` protocol, `KFallUnitConverter`
(verified no-op), `get_unit_converter(dataset)` registry.
Tests: `tests/test_units.py` (7 tests). Commit: "Add unit conversion
interface with KFall no-op implementation"

#### Task 3.2 — COMPLETE
`shared/harmonize/resample.py`: `resample_signal(signal, native_rate_hz,
target_rate_hz, time_col="time_s")`, using `scipy.signal.resample_poly`
(built-in anti-aliasing). Downsampling only (raises `ValueError` on
upsample requests). Handles non-integer rate ratios (tested against a
synthetic ~238 Hz case, standing in for FallAllD's eventual real rate).
Tests: `tests/test_resample.py` (9 tests, including an explicit
anti-aliasing correctness check on an 80 Hz signal). Commit: "Add
resampling module with anti-alias filtering"

#### Task 3.3 — COMPLETE
`shared/harmonize/filtering.py`: `apply_bandpass_filter(signal, columns,
sample_rate_hz, low_hz=0.5, high_hz=20.0, order=4)`, 4th-order
Butterworth band-pass via `scipy.signal.butter` + `filtfilt` (zero-phase,
no time-shift -- important since onset/impact frame indices must stay
aligned to the original samples). Raises `ValueError` if `high_hz` is at
or above the Nyquist frequency for the given sample rate.

**Concrete empirical result validating the 0.5-20 Hz choice over the
originally-proposed 5 Hz low-pass** (synthetic fall-impact-like pulse,
100ms half-sine, amplitude 5, on a noisy background): the 0.5-20 Hz
band-pass retained **95.6%** of the impact peak amplitude; a straight
5 Hz low-pass (the original proposal) retained only **54.0%**. This is
the strongest evidence yet that the cutoff correction was the right
call -- worth citing if this decision is ever questioned later.

Tests: `tests/test_filtering.py` (7 tests: composite drift/movement/noise
signal, impact-spike retention, DC attenuation, zero-phase/no-time-shift,
passthrough of non-filtered columns, Nyquist guard, no-mutation).
Commit: "Add Butterworth band-pass filtering (0.5-20 Hz) module"

**Current total test count: 42 passed** -- run `pytest tests/ -v` to
confirm current count matches before trusting this number blindly; it
should only grow from here.

#### Task 3.4 — COMPLETE
`shared/harmonize/stationarity.py`: `detect_stationary_segment(signal,
sample_rate_hz, min_duration_s=2.0, accel_var_threshold=0.01,
gyro_mag_threshold=5.0)`. Finds the longest contiguous window where
rolling acceleration variance and gyro magnitude both stay below
threshold for at least `min_duration_s`; returns `(start_idx, end_idx)`
or `None` if nothing qualifies. Picks the LONGEST qualifying window when
multiple exist (tested explicitly). Off-by-one-or-two-sample edge fuzz
on detected boundaries is expected/normal rolling-window behavior, not a
bug -- confirmed via manual check (true segment `[100,400)`, detected
`(99,401)`).
Tests: `tests/test_stationarity.py` (6 tests). Commit: "Add generic
stationary-segment detector"

#### Task 3.5 — COMPLETE
`shared/harmonize/axis_alignment.py`: `compute_gravity_rotation(accel_segment)`
(Rodrigues-formula rotation aligning mean gravity direction to canonical
vertical `[0,0,1]`, handles near-parallel/antiparallel degenerate cases),
`apply_rotation(signal, rotation)` (rotates BOTH acc_* and gyro_* columns
consistently, since both are vectors in the same sensor frame),
`calibrate_subject(trials, standing_initiated_task_ids=..., ...)`
(T01-preferred calibration; only trusts T01 if the Task 3.4 detector
confirms stillness for >= `t01_min_coverage_fraction` [default 0.5] of
the trial, else falls through to auto-detect on a standing-initiated
trial: T02, T06-T09, T20-T21 -- sit/lie-down tasks like T11 are
deliberately excluded from the fallback set, since seated/reclined tilt
differs from standing tilt). Returns `None` if nothing usable (Task
3.6's job to fill that gap). Module is dataset-agnostic (only needs
`.signal`/`.metadata.task_id`, doesn't import KFall's reader directly)
so this same logic applies unchanged to SisFall/FallAllD later.

**Validated against the real SA06 finding** (fact #3 below): a
synthetic signal reproducing gravity-on-acc_y-at-(-1.0) was correctly
rotated so gravity moved to acc_z ≈ 0.999, acc_x/acc_y ≈ 0 -- confirms
the calibration does what it's designed to do, not just that it passes
synthetic unit tests.

Tests: `tests/test_axis_alignment.py` (12 tests, including the
exact-antiparallel degenerate case and the "T01 exists but subject was
fidgety, correctly falls through to auto-detect" case). Commit: "Add
per-subject gravity-alignment calibration with T01/auto-detect fallback"

#### Task 3.6 — COMPLETE
Extended `shared/harmonize/axis_alignment.py`:
`resolve_group_fallback(per_subject: dict[str, Optional[CalibrationResult]])`
-- fills in `None` entries using the average of successfully-calibrated
subjects' NORMALIZED gravity directions, re-running
`compute_gravity_rotation` on that average; tags result `source="group_fallback"`.
Raises `ValueError` if every subject in the batch is `None` (nothing to
average from). Also added `summarize_calibration_sources(resolved)` --
one-line count-per-source helper for the human sanity check ("is
group_fallback rare, as it should be").

**Bug caught in my own test, not the code**: an early test assumed the
group-averaged rotation should map the averaged gravity vector to
EXACTLY magnitude 1.0 -- this failed, correctly, because averaging two
non-identical unit vectors mathematically produces a vector shorter
than 1 (basic vector geometry). Fixed the test to check alignment
(x/y components vanish) rather than an incorrect magnitude assumption.
Worth remembering if a similar "slightly under 1.0" number shows up
later in real data -- not automatically a bug.

Tests: `tests/test_axis_alignment_group_fallback.py` (6 tests).
Commit: "Add group-average calibration fallback for subjects with no
usable stationary segment"

**Current total test count: 66 passed** -- run `pytest tests/ -v` to
confirm current count matches before trusting this number blindly; it
should only grow from here.

#### Next up: Task 3.7 — Harmonization orchestrator
Not yet started as of this checkpoint. Composes unit conversion →
resample → axis alignment → filter into one per-trial function
(`harmonize_trial(trial, calibration, config)`), using the four modules
already built in Tasks 3.1-3.3 and 3.5/3.6.

#### Remaining after that: 3.8 (validation checks), 3.9 (provenance
writer), 3.10 (end-to-end KFall harmonization script -- this is where
Tasks 3.4-3.6's calibration logic finally gets run against REAL KFall
subjects for the first time, not just synthetic fixtures), 3.11 (visual QA).

---

## Verified facts about real KFall data (Kaggle mirror: usmanabbasi2002/kfall-dataset)

These are NOT assumptions — each was confirmed against the actual
downloaded files during Stage 2/Task 3.1 real-data verification. If a
future session (or a different Claude instance) suggests something
that contradicts these, trust this checkpoint over a fresh guess:

1. **Folder vs. filename naming mismatch**: parent folders are named
   `SA06`, `SA07`, etc. (full form), but the **sensor CSV filenames
   themselves** drop the "A": `S06T01R01.csv`, not `SA06T01R01.csv`.
   Label files (`SA06_label.xlsx`) keep the full "SA" form. The reader
   (`parse_trial_filename`, `discover_trials`) handles both variants and
   always normalizes output to canonical `SAxx` form.

2. **Real sensor CSV column headers** (confirmed via direct inspection):
   ```
   TimeStamp(s), FrameCounter, AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ
   ```
   Note: `TimeStamp(s)` has a units suffix (not just `TimeStamp`), and
   gyro columns are `GyrX/Y/Z` (not `GyroX/Y/Z`). Already fixed in
   `EXPECTED_SENSOR_COLUMNS` / `_RENAME_MAP`.

3. **Real sensor mounting orientation**: for subject SA06's T01
   (stand-still) trial, gravity (~-1.0g) appears on **`acc_y`**, not
   `acc_z`. Raw `acc_z` was measured near 0. This is real, not a bug —
   it's direct evidence that this specific sensor's mounting orientation
   doesn't align "vertical" with the Z-axis by default, which is
   exactly why the Stage 3 axis-alignment step is necessary and not
   just a theoretical concern. Don't assume `acc_z ≈ 1g` as a validation
   check on RAW (pre-alignment) real KFall data — that check is only
   valid AFTER axis alignment has been applied.

4. **Label file structure is significantly different from the official
   KFall documentation's apparent implication.** Real `SA06_label.xlsx`
   structure:
   - Task codes are written as `F01 (20)` through `F15 (34)` — NOT
     `T22`–`T36` as the sensor filenames and official docs use.
   - **Verified mapping** (checked against all 15 fall-type
     descriptions, not assumed): `canonical_task_id = parenthetical_number + 2`.
     This holds the invariant `F_number + 19 == parenthetical_number`
     across all 15 rows on the real SA06 file.
   - Task-code and description cells are **Excel-merged** across each
     task's repeated trial rows — pandas reads this as the value only on
     the first row of each block, `NaN` on the rest. Must be
     forward-filled (`.ffill()`) before use.
   - `Trial ID` column is a **plain integer** (1, 2, 3...), not an
     `"R01"`-style string.
   - Some tasks have fewer than 5 trial rows (e.g. one observed block
     had trials 1, 2, 4, 5 with trial 3 missing) — consistent with the
     official documentation's note about occasional dropped
     repetitions due to Bluetooth/sync issues. The reader handles this
     as a normal "no match found" case (returns `None, None`), not an error.
   - Implemented in `_resolve_official_task_id()` and the updated
     `read_label_file()` / `_label_lookup()` in `readers_kfall.py`.
   - **End-to-end confirmed on real data**: SA06 T22 R01 resolves to
     `fall_onset_frame=130, fall_impact_frame=208`, matching the real
     spreadsheet's row exactly.

5. **Kaggle API access pattern that worked**: individual files CAN be
   pulled without downloading the full dataset, via:
   ```bash
   kaggle datasets download -d usmanabbasi2002/kfall-dataset \
     -f "KFall Dataset/KFall Dataset/sensor_data/SA06/S06T01R01.csv" \
     -p /tmp/kfall_sample --unzip --force
   ```
   Note the doubled `KFall Dataset/KFall Dataset/` path prefix specific
   to this Kaggle mirror — not part of the official dataset structure.
   File listing (to find exact paths) requires pagination — the
   Kaggle CLI's `datasets files` command only shows one page; use the
   Python API (`KaggleApi().dataset_list_files(..., page_token=...)`)
   looped until no more pages, as done in `scripts/list_kfall_files.py`.

6. **Not yet verified on real data**: Tasks 3.4-3.6 (stationary
   detection, per-subject calibration, group-average fallback) are all
   implemented and pass synthetic tests, but have NOT been run against
   real KFall subjects yet -- that first happens in Task 3.10's
   end-to-end script. Don't assume the `t01_min_coverage_fraction=0.5`
   threshold or the `STANDING_INITIATED_TASK_IDS` set are well-tuned
   for real subjects until that run happens. The full harmonization
   orchestration (Tasks 3.7-3.9) also hasn't touched real data yet.

---

## Known open items / things to double check later

- SisFall and FallAllD real file structures haven't been inspected yet
  at all — expect similar mismatches to what KFall had; don't assume
  their documented structure is accurate either.
- FallAllD's exact native sampling rate needs confirming from its real
  documentation/files, not assumed (resampling code handles non-integer
  ratios already, so this shouldn't require a code change, just a
  config value).
- The "F-code + 19 invariant" was verified on SA06 only. Worth spot-checking
  at least one more subject's label file once more real data is pulled,
  in case some subjects' sheets differ.
- The `t01_min_coverage_fraction` (0.5) and `STANDING_INITIATED_TASK_IDS`
  (`{2,6,7,8,9,20,21}`) constants in `axis_alignment.py` are judgment
  calls, not derived from real data. Worth revisiting once Task 3.10 runs
  calibration against all 42 real KFall subjects and
  `summarize_calibration_sources` shows how often each fallback tier
  actually gets used in practice.
- `resolve_group_fallback`'s assumption (sensor mounted the same way
  across subjects in a study, so averaging others' gravity direction is
  a reasonable stand-in) hasn't been checked against real data either --
  worth a sanity look once Task 3.10 runs, if `group_fallback` ever
  triggers on a real subject (expected to be rare, e.g. SA34 given its
  documented full-data-loss issue).
