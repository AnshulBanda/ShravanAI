# Project Checkpoint — Fall Detection & Prediction Pipelines

Last updated: after Stage 3, Task 3.11 (visual QA script, run against
real SA06 data by a human -- Stage 3 is now complete for KFall)

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

### Stage 3 — COMPLETE (for KFall)
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

#### Task 3.7 — COMPLETE
`shared/harmonize/pipeline.py`: `HarmonizationConfig` dataclass
(target_rate_hz=100.0, filter_low_hz=0.5, filter_high_hz=20.0,
filter_order=4) and `harmonize_trial(trial, calibration, config)`,
composing unit conversion -> resample -> axis alignment -> filter in
that order. Output is channel-restricted to exactly `time_s` +
acc_*/gyro_* (drops KFall's Euler columns -- callers wanting those for
a KFall-only experiment should read `trial.signal` directly before
harmonization, not the harmonized output).

**Critical, non-obvious finding from this task**: because gravity is a
0 Hz/DC signal, and the filter runs AFTER alignment, the final
harmonized output does NOT retain a persistent ~1g bias on any axis --
the band-pass filter removes it by design (same as it removes postural
drift). This means the "calibration sanity" check envisioned in the
original sprint plan (checking ~1g/~0g on the final signal) is not
physically checkable post-filter -- Task 3.8 checks the calibration
object's own rotation-applied-to-its-recorded-gravity-vector instead,
which is filter-independent. **Don't be alarmed by an absent gravity
bias in harmonized output -- that's correct, not a bug.**

Tests: `tests/test_pipeline.py` (5 tests, including a monkeypatch-based
call-order regression test, and a test proving alignment's effect
survives filtering by tracking a movement-frequency component rather
than the (removed) DC gravity bias). Commit: "Add harmonization
orchestrator composing unit conversion, resampling, alignment, and filtering"

#### Task 3.8 — COMPLETE
`shared/harmonize/validation.py`: `validate_harmonized_trial(signal,
metadata, calibration, expected_rate_hz, expected_duration_range_s=None)
-> list[str]`. Checks: schema, NaN/Inf, timing integrity, physical
plausibility (max ~20g bound + flatline/std-near-zero detection),
calibration sanity (via the calibration object itself, per the Task 3.7
finding above -- NOT the final signal), duration-vs-protocol (optional,
skipped if no range given -- kept dataset-agnostic rather than
hardcoding KFall's per-task duration table into a shared module), and
fall-trial label consistency (onset < impact <= signal length).

**Deviation from sprint plan's literal API**: takes the full
`calibration: CalibrationResult` object, not just a `calibration_source`
string, specifically so the calibration-sanity check can be computed
correctly (see Task 3.7 finding).

Tests: `tests/test_validation.py` (15 tests, one per failure mode).
Commit: "Add harmonized-trial validation checks"

#### Task 3.9 — COMPLETE
`shared/harmonize/writer.py`: `write_harmonized_trial(signal, metadata,
calibration_source, issues, harmonized_root, quarantine_root,
provenance_extra=None) -> Path`. Writes parquet + a sidecar JSON
(chosen over parquet's internal metadata API for simplicity/inspectability)
containing dataset/subject/activity/trial IDs, label, onset/impact
frames, calibration_source, `accepted` bool, and the issues list.
Routes to `harmonized_root/<dataset>/` if no issues, else
`quarantine_root/<dataset>/`. Filename stem:
`<subject_id>_<activity_code>_<trial_id>`.

Tests: `tests/test_writer.py` (6 tests, including a round-trip value
check and a provenance-content check). Commit: "Add provenance-aware
harmonized trial writer with quarantine routing"

#### Task 3.10 — COMPLETE
`shared/manifest.py` (new, minimal): `ManifestRow` dataclass +
`write_manifest`/`load_manifest` (parquet-backed). `shared/harmonize/orchestration.py`
(new): `HarmonizationSummary` dataclass + `run_harmonization(dataset,
sensor_root, label_root, harmonized_root, quarantine_root,
harmonization_config, manifest_path=None) -> HarmonizationSummary` --
loads all trials for a dataset, does two-pass calibration (per-subject
via `calibrate_subject`, then `resolve_group_fallback` for gaps),
harmonizes/validates/writes every trial, returns a summary. `scripts/harmonize_dataset.py`
(new): thin CLI wrapper (`python scripts/harmonize_dataset.py --dataset kfall`),
resolves paths from `configs/datasets/kfall.yaml` + `configs/base.yaml`,
calls `run_harmonization`, prints the summary.

**Deviation from sprint plan's literal file placement**: the plan put
`run_harmonization` directly in `scripts/harmonize_dataset.py`. Moved
the actual logic to `shared/harmonize/orchestration.py` instead, keeping
the script as a thin wrapper -- consistent with this repo's own
"scripts hold no logic" convention, which the plan's literal wording
would have violated.

Test fixtures extended: added a real `SA06T01R01.csv` fixture (the
original Stage 2 fixtures never actually included one -- caught before
it silently made the end-to-end test exercise `group_fallback` instead
of the intended T01 path) and a `S07T02R01.csv` fixture (standing-
initiated, quiet start) so SA07's auto-detect fallback is genuinely
exercised end-to-end, not just in isolation.

Tests: `tests/test_manifest.py` (2), `tests/test_orchestration.py` (6,
including one confirming BOTH the T01 and auto_detected tiers fire
together in one run, with `group_fallback` correctly absent). Also
updated `tests/test_kfall_reader.py` for the two new fixture files (5
trials total now, was 3). Commit: "Add end-to-end KFall harmonization
script with summary reporting"

**Current total test count: 100 passed** -- run `pytest tests/ -v` to
confirm current count matches before trusting this number blindly; it
should only grow from here.

---

## REAL-DATA MILESTONE: Task 3.10 run against actual KFall files (SA06, T01 + T22)

This is the first time the full harmonization pipeline (Stages 2+3
combined) has been run against real KFall data, not synthetic fixtures.
Ran `python scripts/harmonize_dataset.py --dataset kfall` against
SA06's real T01 and T22 R01 trials (the only two downloaded so far).

**Result: fully successful, verified in detail, not just "it ran without
erroring":**
- Both trials: `accepted=True`, `calibration_source=T01`, 0 quarantined.
- Harmonized `acc_z` mean ~0 (confirms the Task 3.7 finding holds on
  real data too -- gravity DC correctly removed, not a bug).
- **The real fall's impact signature was traced precisely**: T22 R01's
  real labeled `fall_onset_frame=130, fall_impact_frame=208`. The
  harmonized signal's largest `acc_z` swing (-1.307g) occurs at frame
  202 -- 6 frames before the labeled impact, well inside the onset-
  impact window, with the signal visibly transitioning from calm
  (~130-185) to violently oscillating (~187 onward) before the sharp
  transient. `acc_x`'s peak (2.16g) was actually larger than `acc_z`'s
  in this window -- consistent with T22 being a FORWARD fall (dominant
  deceleration is horizontal, not vertical), not a red flag.
- This is the strongest evidence yet, on real data, that the full
  chain (real onset/impact labels -> real axis alignment correcting
  the actual sensor mounting tilt -> real band-pass filtering) works
  as designed.

**Not yet tested on real data**: `auto_detected` and `group_fallback`
calibration tiers (only `T01` has fired on real data so far, since only
SA06 is downloaded and it has a working T01). Re-run
`harmonize_dataset.py` as more real subjects are downloaded and watch
`calibration_source_counts` in the printed summary -- `group_fallback`
should stay rare, per the design intent.

#### Task 3.11 — COMPLETE (run against real data, script since improved)
`notebooks/stage3_visual_qa.py`: one-off exploratory script (not
imported by anything, no unit tests, by design). Reuses Task 3.10's
two-pass calibration logic directly (rather than importing
`run_harmonization`, since that writes to `data/harmonized/` and this
script should never touch that), then for a handful of trials per
subject: harmonizes, plots raw-vs-harmonized overlays to
`results/stage3_visual_qa/<subject>_<activity>_<trial>.png`, prints and
saves a calibration-source count table, and for every fall trial in the
QA set, compares the labeled impact frame to detected peak frames near
it. Degrades gracefully to whatever subjects are present --
`--subjects`/`--max-trials-per-subject` flags exist but aren't required.

**Real-data run (SA06, T01 + T22)**:
- T01 (calibration trial, ADL): raw shows gravity on `acc_y` at ~-1.0g
  (matches the documented real-mounting quirk); harmonized shows no
  bias on any axis, just low-amplitude noise (~+/-0.02g) -- exactly the
  expected shape of a correctly-calibrated, genuinely-still trial.
- T22 (real fall, labeled `fall_onset_frame=130, fall_impact_frame=208`):
  the harmonized signal's COMBINED 3-axis magnitude peaks at frame 192
  (offset -16 from the labeled impact) -- notably different from the
  frame-202 (offset -6) found by the earlier Task 3.10 manual check,
  which specifically tracked `acc_z`. Investigated by eye against the
  saved plot: this is NOT a bug. `acc_x` (horizontal deceleration) has
  the largest overall peak (~+2.15g at frame ~192), while `acc_z`
  (vertical, ground-contact) has its own separate, smaller peak
  (~-1.2g) closer to frame ~202-205, right near the labeled impact.
  This matches the already-documented finding that T22 is a FORWARD
  fall with horizontal deceleration dominant over vertical -- the
  combined-magnitude peak simply locks onto the earlier, larger
  horizontal transient rather than the later vertical one. Both are
  real physical events within the same fall, a few hundred ms apart.

**Script improved as a result of this finding**: `_impact_frame_check`
now reports the peak frame/offset/value for EACH axis individually, not
just the combined-magnitude peak -- so a human can see at a glance
which axis is driving a given peak and whether a different axis lines
up more closely with the labeled frame, instead of one number silently
conflating two different physical events. Also removed a static
"matches the earlier check" message that had been asserting agreement
without actually checking it — that message would have been actively
wrong on real T22 data and was itself a checkpoint-writing mistake
worth remembering, not just a code bug.

**Smoke-tested against fixtures too**: run against the repo's own
synthetic `tests/fixtures/kfall_mock/` data (temporarily copied to
`data/raw/kfall/`, then removed -- not committed) to confirm the script
runs end-to-end without crashing, both before and after the per-axis
change. Caught and fixed one real bug this way: the first draft's trial
de-duplication used `t not in picks`, which raises `ValueError: The
truth value of a DataFrame is ambiguous` because `ParsedTrial` holds a
DataFrame field -- fixed to compare by `id()` instead.

**Still open**: only SA06 has been downloaded so far, so
`auto_detected`/`group_fallback` calibration tiers and the impact-frame
check on other subjects/fall types remain untested on real data --
re-run this script as more real subjects are downloaded.

This is the last Stage 3 item before Stage 4 (manifest builder --
`shared/manifest.py` already exists in minimal form from Task 3.10;
Stage 4 extends it into the FULL cross-dataset manifest all pipelines
will query against) and before extending harmonization to SisFall
(Stage 5).

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

6. **Now verified on real data (as of Task 3.10)**: the full
   harmonization pipeline (Stages 2+3 combined) has been run against
   real SA06 T01 + T22 R01 files and confirmed correct in detail -- see
   the "REAL-DATA MILESTONE" section above. Tasks 3.4-3.9's calibration/
   harmonization/validation logic all fired correctly on real data, not
   just synthetic fixtures. Still NOT yet tested on real data: the
   `auto_detected` and `group_fallback` calibration tiers (only `T01`
   has had a chance to fire, since only one real subject with a working
   T01 is downloaded so far), and anything involving more than one real
   subject at once.

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
  calls, not derived from real data. Worth revisiting once more real
  subjects are downloaded and `calibration_source_counts` shows how
  often each fallback tier actually gets used in practice.
- `resolve_group_fallback`'s assumption (sensor mounted the same way
  across subjects in a study, so averaging others' gravity direction is
  a reasonable stand-in) hasn't been checked against real data either --
  worth a sanity look once `group_fallback` triggers on a real subject
  (expected to be rare, e.g. SA34 given its documented full-data-loss issue).
- As more real KFall subjects get downloaded, rerun
  `python scripts/harmonize_dataset.py --dataset kfall` periodically and
  watch the `calibration_source_counts` breakdown -- this is the
  ongoing real-world check that Tasks 3.4-3.6's fallback logic is
  behaving as designed, not just working for SA06.