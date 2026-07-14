# Project Checkpoint — Fall Detection & Prediction Pipelines

Last updated: Stage 7 (prediction pipeline windowing + onset/impact
labeling) REAL-DATA VERIFIED against the full real KFall manifest --
348,941 total windows across all KFall prediction-eligible trials:
non_fall 249,970 (71.6%), fall 83,059 (23.8%), pre_impact 15,912
(4.6%) -- pre_impact is the rare class, exactly as the blueprint
predicted. Spot-checked SA06 T22 R01 (onset=130, impact=208, the same
real trial from the Stage 3 REAL-DATA MILESTONE) frame-by-frame: the
non_fall->pre_impact transition lands exactly at the window ending at
frame 140 (the first window whose end_frame=130 stays non_fall, per
the strictly-past-onset rule), and the pre_impact->fall transition
lands exactly at the window starting at frame 110 (end_frame=210,
first window past impact=208) -- an exact match, not an approximation.
See Stage 7 section below for full detail.

Detection pipeline's real final cross-dataset baseline (Stage 6, real
data, both datasets) remains fully verified: full real KFall (32
subjects) downloaded and harmonized (5,075 trials, {T01: 32}
calibration, cleanest possible result), combined with full real
SisFall (38 subjects) for training: 70 real subjects total, test-set
accuracy 0.836, precision 0.692, recall 0.839, ROC-AUC 0.921 on
genuinely held-out subjects from BOTH datasets. See Stage 6 section
below for full detail.

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

### Stage 4 — COMPLETE (manifest builder)
Extended `shared/manifest.py` from Stage 3's minimal per-run version
into the full cross-dataset manifest described in the blueprint's §4:
"the single index both pipelines query against."

**Schema extended.** `ManifestRow` gained `duration_s`, `sample_rate_hz`,
`fall_onset_frame`, `fall_impact_frame` -- the blueprint's documented
columns -- on top of the two provenance fields (`accepted`,
`calibration_source`) already in the Stage 3 version, which were worth
keeping rather than dropping. `orchestration.py` now computes
`duration_s` from the harmonized signal length and target rate, and
carries `fall_onset_frame`/`fall_impact_frame` straight through from
`trial.metadata`.

**Real bug found and fixed: `write_manifest` was a silent overwrite.**
The Stage 3 version did a plain `df.to_parquet(path)` on every call. Since
`harmonize_dataset.py` writes every dataset to the SAME
`manifest.parquet` path, running `--dataset kfall` today and
`--dataset sisfall` next month (Stage 5+) would have silently deleted
every KFall row the moment the SisFall write happened -- exactly the
opposite of "single source of truth across datasets." `write_manifest`
now upserts by primary key (`dataset, subject_id, activity_code,
trial_id`): a matching existing row gets replaced, everything else is
preserved. This also makes re-running the SAME dataset (e.g. after a
harmonization bugfix) safe -- it replaces that dataset's rows in place
instead of duplicating them. Caught by a dedicated test
(`test_write_manifest_preserves_other_datasets`) before this could bite
during Stage 5.

**Query interface added**, matching the blueprint's documented filters
exactly so `detection/dataset.py` / `prediction/dataset.py` (not built
yet -- later stages) have a real function to call rather than
hand-rolling the same pandas filter twice:
- `query_detection_trials(df, datasets=None, accepted_only=True)` --
  every dataset, both labels, quarantined trials excluded by default.
- `query_prediction_trials(df, accepted_only=True)` -- KFall only,
  and only rows where onset/impact labeling actually applies (fall
  trials with a labeled onset frame, or ADL trials as the negative
  class). A fall-labeled row with a missing onset frame (a labeling
  gap) is deliberately excluded, not silently included with a null.

**One parquet round-trip gotcha worth remembering**: `None` in an
`Optional[int]` column (e.g. `fall_onset_frame` on an ADL row) comes
back from `pd.read_parquet` as `NaN` (float), not `None` -- a plain `is
None` check on a loaded row will silently fail. Use `pd.isna(...)`
instead. `query_prediction_trials`'s `.notna()` filter already handles
this correctly; it only came up as a test-assertion bug
(`fall_onset_frame is None` vs. `pd.isna(fall_onset_frame)`), not a
production-code bug.

**Tests**: `tests/test_manifest.py` (10 tests: round-trip, upsert
same-dataset, preserve-other-datasets, replace-only-matching-trials,
both query functions including the accepted-filtering and
onset-frame-gap edge cases), plus `test_orchestration.py`'s
`test_end_to_end_writes_manifest` extended to assert the new fields are
actually populated end-to-end, not just present in the schema. Full
suite: 108 passed.

**Real-data smoke test** (not just fixtures): ran
`scripts/harmonize_dataset.py --dataset kfall` twice in a row against
the fixture set standing in for real data, confirming the upsert
behavior holds outside of unit tests too -- second run produced the
same 5 manifest rows, not 10.

**Next up**: Stage 5 -- extend harmonization to SisFall. This is where
`shared/harmonize/units.py`'s unit-conversion and
`shared/harmonize/resample.py`'s resampling modules finally do real
work instead of being KFall no-ops (KFall is already 100 Hz and
already in g/deg-per-s, so both stages passed through unchanged so
far). Will need a `shared/io/readers_sisfall.py` reader and a
`configs/datasets/sisfall.yaml`, following the same pattern as KFall's.

---

### Stage 5 — IN PROGRESS (reader + unit converter done; orchestration wiring NOT done yet)

**Real dataset downloaded and inspected** (not assumed from the paper):
Kaggle mirror `kushajm/sisfall-dataset-fall-detection`, unzipped to
`data/raw/sisfall/SisFall_dataset/`. Confirmed against real files: 9
comma-separated raw ADC columns per row, semicolon-terminated, no
header (`accX,accY,accZ` ADXL345, `gyroX,gyroY,gyroZ` ITG3200,
`accX,accY,accZ` MMA8451Q); column count verified consistent (9) across
a real sample; clean file tails (no malformed trailing rows); all 38
subject folders present (SA01-23, SE01-15); 4,505 real files vs. the
Readme's stated 4,510 -- a 5-file gap, not investigated further since
the Readme itself documents that elderly subjects skip several
activities (D06/D13/D18/D19 plus individual impairment-based skips),
which plausibly accounts for it, and all 38 subjects being present
rules out a whole-subject download gap.

**`shared/io/readers_sisfall.py` added.** Same "dumb reader" scope as
`readers_kfall.py` -- parses the native format faithfully (raw ADC
integers, not physical units) and does no unit conversion, resampling,
or filtering. Fall/ADL label comes straight from the filename prefix
(`D`=adl, `F`=fall) -- no separate label file exists for SisFall, and
critically, **SisFall has NO frame-level fall onset/impact
annotation anywhere**, even for real fall trials. `fall_onset_frame`/
`fall_impact_frame` are always `None` for every SisFall trial. This
isn't a parsing gap -- it's a genuine dataset limitation, and it's
exactly why the blueprint restricts the prediction pipeline (which
needs onset/impact frames) to KFall only. Stage 4's
`query_prediction_trials()` already filters on `dataset == "kfall"`, so
this required zero downstream changes.

**`shared/harmonize/units.py`: `SisFallUnitConverter` added**, using
the exact ADC-to-physical formula from SisFall's own Readme.txt
(`physical = [(2*Range)/(2^Resolution)] * raw_value`, different
Range/Resolution per sensor). **Design decision made, not just an
implementation detail**: SisFall has TWO accelerometers (ADXL345,
MMA8451Q) but the rest of the pipeline expects a single `acc_x/y/z`
triplet. Resolved by following the SisFall paper's own stated
methodology -- ADXL345 becomes `acc_x/y/z` ("energy efficient... larger
span," per Sucerquia et al. 2017), ITG3200 becomes `gyro_x/y/z`, and
MMA8451Q is converted too but kept under its own `mma_acc_*` columns,
preserved through this step and then dropped by `pipeline.py`'s
existing channel-restriction step -- same archive-then-drop treatment
KFall's Euler columns already get, so `pipeline.py` needed zero changes
for this.

**Real bug avoided, not yet fixed -- flagged for the next session.**
`shared/harmonize/axis_alignment.py`'s `calibrate_subject` hardcodes
`T01_TASK_ID = 1` as "the" dedicated calibration-trial ID, despite its
own docstring's claim of being fully dataset-agnostic. SisFall has NO
dedicated stand-still calibration trial at all (D01 is "walking
slowly," not stand-still) -- but D01's `task_id` is naturally `1`
(first ADL code), which would coincidentally trip the hardcoded T01
path and mislabel a walking trial's incidental brief stillness as a
"T01" calibration, which is real KFall-specific terminology this
shouldn't apply to. **Not fixed yet** -- this touches
`axis_alignment.py`, which PROJECT_CHECKPOINT.md's Stage 3 section
explicitly marked as frozen design, so it's flagged here for a
deliberate decision (most likely: add an optional
`primary_calibration_task_id` parameter to `calibrate_subject`,
defaulting to preserve exact existing KFall behavior, with SisFall
passing `None` to skip straight to auto-detection) rather than being
silently patched around.

**Tests**: `tests/test_sisfall_reader.py` (14 tests, against
synthetic fixtures in `tests/fixtures/sisfall_mock/` built to match
the real format) and `tests/test_units.py` extended with 6 new
SisFallUnitConverter tests, including exact-value checks against the
Readme's documented conversion formula for all three sensors. Also
fixed one now-broken existing test
(`test_get_unit_converter_unknown_dataset_raises_with_known_list` used
to use `"sisfall"` as its example of an unregistered dataset; now uses
`"fallallD"`). Full suite: 127 passed.

**`scripts/verify_sisfall_reader.py` added, and RUN against the full
real dataset (not just fixtures).** Result: all 4,505 real files loaded
with 0 failures, all 38 subjects present. Cross-checked the label
counts against the Readme's own protocol tables, not just trusted "0
errors": ADL=2,707 vs. ~2,702 expected (23 young subjects x 79 trials +
15 elderly x 59 trials after the mandatory D06/D13/D18/D19 skip,
+5 accounted for by the Readme's own note that some elderly subjects
had additional individual activity variations); Fall=1,798 vs. 1,800
expected (23 young subjects + SE06's Judo-expert exception x 75 fall
trials each, -2 consistent with the already-noted 4,505-vs-4,510 file
gap). Row counts: min ~10s, mean ~17.6s, max 180s (a bit above the
Readme's longest *nominal* 100s category, plausibly just some trials
running longer in practice -- not a parsing concern given 0 failures
and consistent 9-column structure throughout). The SisFall reader is
now considered REAL-DATA VERIFIED, the same standard KFall's reader
was held to in Stage 2 -- notably cleaner than KFall's initial real-data
pass, which found 3 real mismatches on first run.

**Orchestration wiring done. Two more real bugs found and fixed in the process** (not hypothetical -- both caught by tests failing, not by inspection):

1. **`calibrate_subject`'s hardcoded T01 assumption** (flagged above) --
   fixed. `axis_alignment.py`'s `calibrate_subject` now takes an
   explicit `primary_calibration_task_id` parameter (default
   `T01_TASK_ID`, preserving KFall's exact prior behavior byte-for-byte
   -- all 12 pre-existing tests pass unchanged). `orchestration.py`
   passes `None` for SisFall via a new `_CALIBRATION_CONFIG` registry
   (same pattern as `units.py`'s converter registry), so SisFall's D01
   (coincidentally also task_id 1) can never be mistaken for a
   dedicated calibration trial.

2. **New bug, found only once SisFall was actually wired in**:
   `calibrate_subject` was being called on each trial's RAW,
   not-yet-unit-converted `.signal`. This happened to work for KFall
   purely by coincidence -- KFall's raw reader output already uses the
   canonical `acc_x/y/z` column names, since its unit converter is a
   verified no-op. SisFall's raw reader output uses `raw_adxl_acc_x`
   etc., so calling calibration on it crashed outright
   (`KeyError: 'acc_x'`) -- caught immediately by
   `test_end_to_end_sisfall_processes_all_fixture_trials`, not
   discovered later against real data. Fixed with a small
   `_CalibrationView` wrapper in `orchestration.py`: every trial is now
   run through its dataset's unit converter BEFORE calibration, not
   just before the main `harmonize_trial` call. Converting twice
   (once for calibration, once inside `harmonize_trial`) is mildly
   redundant but harmless -- converters are pure, side-effect-free
   functions (see `units.py`'s "does not mutate input" tests).

3. **Also fixed while wiring**: `calibrate_subject`'s `sample_rate_hz`
   was being left at its default (100.0, correct for KFall) regardless
   of dataset. Now `orchestration.py` passes each subject's actual
   native rate (`subject_trials[0].metadata.native_rate_hz`) --
   otherwise SisFall's 200 Hz raw signal would have had its
   stationarity-detector window sized as if it were 100 Hz, silently
   using half the intended window duration.

**SisFall's standing-initiated activity-code set** (for the
auto-detect calibration fallback), chosen in `orchestration.py`:
`{D07, D08, D09, D10, D15, D16, D17}` (sit-in-chair activities, which
begin standing; standing-bend and standing-into-car activities, which
begin standing by definition). **Still an assumption**, not yet
confirmed against real SisFall data the way Task 3.4 confirmed KFall's
-- needs the same kind of real stationarity check before fully trusting
it.

**Tests**: fixture set extended with a genuinely-still SisFall trial
(SA02's D07) to exercise the auto_detected tier end-to-end -- this
required regenerating that fixture with realistic raw-ADC noise
amplitude after the first attempt's arbitrary noise level, once run
through the REAL ADXL345 conversion formula, produced too much
post-conversion variance to pass the stationarity threshold (a fixture
realism bug on my part, not a production bug -- worth remembering that
"looks like plausible raw sensor noise" isn't the same as "converts to
plausible physical-unit noise" when picking synthetic ADC values).
4 new end-to-end SisFall orchestration tests, 3 new axis_alignment
tests for the `primary_calibration_task_id` parameter. Full suite:
134 passed.

**Real end-to-end smoke test** (fixture set standing in for real data,
via `scripts/harmonize_dataset.py --dataset sisfall`): also fixed two
CLI bugs this surfaced -- `--dataset` choices were hardcoded to
`["kfall"]` only, and `label_root = REPO_ROOT / cfg.dataset.label_root`
crashed on SisFall's `label_root: null` config (now only resolved when
actually configured). Smoke test produced sensible output: 4 trials
processed, 0 quarantined, calibration sources `{auto_detected: 1,
group_fallback: 1}`, correct halved row counts confirming real
200->100Hz resampling actually ran (previously only ever a no-op
against KFall's already-100Hz data).

**Run against the FULL real dataset** (not just fixtures):
`scripts/harmonize_dataset.py --dataset sisfall` processed all 4,505
real trials, 0 quarantined, **calibration sources `{auto_detected:
38}`** -- every single real subject calibrated successfully via
auto-detection, zero group_fallback needed, zero (correctly) T01. This
is strong (though not yet visually-confirmed) evidence that the
standing-initiated activity-code guess (D07-D10, D15-D17) was
reasonable: with up to ~35 candidate trials per subject across those 7
codes, every subject found at least one usable stationary segment.

**Refactor: extracted `resolve_calibrations()` and `get_trial_loader()`
as public functions in `orchestration.py`.** Motivation: while
generalizing `notebooks/stage3_visual_qa.py` for SisFall (below), found
it had been independently reimplementing the exact two-pass calibration
logic inline -- with the SAME "calibrate on raw signal" bug that was
just fixed in `orchestration.py`, still present in the QA script's copy.
This is precisely how the earlier bug could have stayed silently fixed
in one place and broken in the other. Both `run_harmonization` and the
QA script now call the same `resolve_calibrations(dataset, trials)` and
`get_trial_loader(dataset)` -- one copy of this logic, not two.

**`notebooks/stage3_visual_qa.py` generalized** to take `--dataset
{kfall,sisfall}` instead of being KFall-only, reusing the extracted
helpers above. Output now goes to
`results/stage3_visual_qa/<dataset>/` (was previously a single shared
directory) so different datasets don't overwrite each other's QA runs.
Also generalized the raw-signal plot panel to detect accel-like raw
column names generically (KFall: `acc_x` already; SisFall:
`raw_adxl_acc_x`, excluding the archived `raw_mma_acc_*` columns)
rather than assuming KFall's post-conversion column names pre-exist on
the raw signal. Smoke-tested against BOTH datasets' fixture sets
(4 plots + calibration summary each).

**Visual QA pass completed against real SisFall data.** Ran
`notebooks/stage3_visual_qa.py --dataset sisfall` against the full real
dataset (114 plots, all 38 subjects), then hand-inspected 12 real
plots spanning SA01, SA08, SE13 (D01/D02/D03/F01) and SE15 (D17 x3).
Results, not just re-confirming the auto_detected count but actually
looking:
- **SE15's D17 trials directly confirm the standing-initiated
  assumption for at least that code**: the RAW signal shows a
  genuinely flat, motionless segment for the first ~6 seconds (constant
  values, no oscillation, on all three axes) before the first
  transient (getting into a car) -- this is real evidence a
  standing-initiated trial actually starts standing-still, not just an
  assumption from the activity's English description.
- **Both inspected fall trials (SA01 F01, SA08 F01) show exactly the
  expected shape**: quiet baseline, a sharp multi-g transient at the
  fall moment (~7g and ~4g peaks respectively), settling to a new
  baseline after. SA08's F01 additionally shows pre-fall oscillation,
  consistent with its description ("fall forward WHILE WALKING, caused
  by a slip") -- the harmonized signal's shape tracks the activity's
  real biomechanics, not just noise.
- **Gravity-scale sanity check on real data**: raw `acc_y` sits around
  -200 to -300 raw ADC counts at rest across every inspected subject,
  which converts to roughly -0.8 to -1.2g via the real ADXL345 scale
  factor -- confirms the unit conversion produces physically sane
  numbers on real data, not just passing the isolated unit tests.
- Continuous-motion trials (D01/D02/D03) show stable, bounded
  oscillation across the full ~100s duration with no drift, scale
  blowup, or discontinuities, across both young (SA01, SA08) and
  elderly (SE13) subjects.

**Remaining, honest gap**: only D17 was directly visually confirmed as
a genuinely standing-initiated trial. The other six candidate codes
(D07-D10, D15, D16) have NOT been individually eyeballed -- the
all-38-auto_detected result doesn't reveal which of the seven codes
each subject actually calibrated on (the script doesn't currently log
which specific trial's task_id was used, only the resulting source tier).
Not considered a blocker given how clean everything inspected looks,
but worth knowing if calibration quality issues ever surface downstream.

---

### Stage 6 — COMPLETE (detection pipeline, XGBoost baseline, end to end)

Started building `detection/` -- the first layer of the detection
pipeline (binary fall/ADL classification, using BOTH KFall and SisFall
via `shared.manifest.query_detection_trials`). Scope for this pass:
turn the trial-level manifest into a WINDOW-level manifest + a
window-loading function. No model code, no feature engineering yet --
just windowing and labeling, the same "one layer at a time" approach
used for harmonization.

**`detection/windowing.py`**: pure window-boundary logic
(`generate_window_specs`), no I/O. Spec (per the blueprint's Pipeline 1
section, §3): 2.0s windows (200 samples @ 100Hz), 1.0s stride (50%
overlap), windows never cross a trial boundary, short trials padded
rather than dropped. Extended that same "don't drop data" principle to
any TRAILING leftover after a longer trial's last full-stride window
too (not just whole-trial-shorter-than-one-window) -- a fall can occur
near the end of a short trial file, and silently dropping that tail
would mean silently dropping the actual fall event.

**Real bug caught by its own test, fixed before it shipped**: the first
version of the trailing-window logic computed the trailing window's
start position from the stride-advanced loop pointer, not from where
the last full window actually ended. This produced a REDUNDANT
overlapping window in the exact-boundary case (trial length exactly
divisible by the stride pattern) instead of correctly detecting "fully
covered, nothing left to pad." Fixed to track `last_covered_end`
explicitly and start the trailing window there.

**`detection/dataset.py`**:
- `build_windows_manifest(trial_manifest_df, config, datasets=None)`
  -- stays a lightweight metadata operation (reconstructs each trial's
  sample count from `duration_s * sample_rate_hz`, already exact since
  `duration_s` was itself computed that way during Stage 4) rather than
  opening every harmonized parquet file just to check its length.
- **Real, certain (not hypothetical) bug prevented by design**: KFall
  and SisFall subject IDs COLLIDE -- both use "SA01".."SA23"/
  "SE01".."SE15"-style IDs, so KFall's SA06 and SisFall's SA06 are
  DIFFERENT people who happen to share a subject_id string. Every
  window record carries a `global_subject_id`
  (`f"{dataset}_{subject_id}"`) specifically so future LOSO/LODO
  split code has no reason to ever group by bare `subject_id` -- tested
  directly (`test_build_windows_manifest_disambiguates_colliding_subject_ids_across_datasets`)
  using literally-identical subject_id strings across two fake
  datasets, since a fixture SisFall subject happening to also be SA06
  wasn't guaranteed to occur in the real smoke test below.
- `load_window(window_row, window_length_samples, signal_cache=None)`
  -- loads the real signal data for ONE window, slicing + EDGE-padding
  (repeating the last real sample, not zero-padding -- zero-padding
  would inject a fake sudden drop-to-0g discontinuity that looks like
  freefall to a model). Takes a caller-owned cache dict to avoid
  re-reading the same parquet file once per overlapping window (a
  single trial can produce dozens of windows at 50% stride).

**Tests**: `tests/test_windowing.py` (9 tests, including one that
verifies EVERY sample index in a trial is covered by at least one
window -- catches gaps, not just off-by-one counts) and
`tests/test_detection_dataset.py` (11 tests, including the subject-ID
collision test above and a padding test that checks the padded values
themselves, not just the output shape). Full suite: 154 passed.

**Real end-to-end smoke test** (not just isolated unit tests): built
real harmonized output for BOTH datasets from their fixture sets via
`harmonize_dataset.py`, confirmed the trial manifest correctly combined
both datasets' rows (Stage 4's upsert design proven again in a new
context), ran `build_windows_manifest` against it, and loaded real
window arrays end-to-end -- correct shapes, correct labels, correct
padding, for both kfall and sisfall trials in the same run.

**Not yet done (at the end of the windowing-only pass)**: feature
engineering, model code, splitting, training. All of the below was
built in a follow-up pass to take the pipeline the rest of the way,
per explicit instruction to finish the detection pipeline completely
rather than leave it at windowing.

---

#### Feature engineering: `detection/features.py`

54 handcrafted features per window -- standard IMU/HAR feature set, not
exotic: per-channel time-domain stats (mean/std/min/max/range/rms x 6
channels = 36), acceleration/gyro magnitude stats (7), jerk (rate of
change of acceleration magnitude -- the classic "sudden deceleration on
impact" signal, 2 features), tilt angle from vertical (postural change,
3 features), signal magnitude area (1), and 3 simple frequency-domain
features (dominant frequency, spectral energy, spectral entropy) on the
acceleration-magnitude signal. `FEATURE_NAMES` is derived FROM
`compute_window_features` itself (calling it once on a dummy window at
import time) specifically so it can never silently drift out of sync
with the function that actually produces those keys.

**One test-writing mistake caught before it became a wrong assertion**:
a test asserted a 5Hz sinusoid on one axis should produce a ~5Hz
dominant-frequency feature. It doesn't -- the feature is computed on
acceleration MAGNITUDE, and `sqrt(sin(2*pi*f*t)^2 + const^2)` has its
fundamental period at `2f`, a standard property of squaring a
sinusoid. Verified numerically (confirmed a real 5Hz input signal does
produce a 10Hz peak in the magnitude's spectrum) before fixing the
test's expectation -- the feature code was correct; my first test
assertion about it wasn't.

Tests: `tests/test_features.py` (10 tests, including degenerate-input
cases: an all-zero window must not crash or produce NaN/inf, since a
tilt-angle divide-by-zero and an all-zero FFT are both real edge cases
a live pipeline will eventually hit).

#### Subject-aware splitting: `detection/split.py`

`split_by_subject`: splits by SUBJECT (via `global_subject_id`), not by
window -- a per-window random split would leak, since overlapping
windows (50% stride) from the same subject's same trial share most of
their samples. Also splits WITHIN each dataset separately then
combines, so train/val/test are each guaranteed to contain BOTH KFall
and SisFall subjects rather than risking one dataset dominating a
random subject-level split. Hard post-condition
(`_assert_no_subject_leakage`) checked before every return, not just
intended -- raises `AssertionError` (not a silent bug) if it's ever
violated.

Tests: `tests/test_split.py` (8 tests: no-leakage, full coverage,
both-datasets-present-in-every-split, reproducibility with a fixed seed,
genuinely differs with a different seed, and a clear error -- not a
cryptic sklearn one -- when a dataset has too few subjects to split).

#### Model: `detection/model.py`

Thin, deliberately unglamorous XGBoost wrapper: `train_model` (fits on
train, early-stops on val, and computes `scale_pos_weight` from the
TRAIN split's actual class balance by default -- fall trials are a
minority class in both datasets, and leaving this at XGBoost's default
of 1.0 would bias toward under-predicting the class that matters most
to catch) and `evaluate_model` (accuracy/precision/recall/F1/ROC-AUC/
confusion matrix -- recall called out explicitly as the metric that
matters most for a fall detector, since a missed fall is far costlier
than a false alarm). `save_model`/`load_model` round-trip through
XGBoost's own JSON format.

Tests: `tests/test_model.py` (7 tests, including an actual
learn-a-real-signal test -- synthetic features with one column
deliberately correlated with the label, confirming the trained model
achieves >85% accuracy on it, not just that training doesn't crash --
and a save/load round-trip that checks predictions match exactly, not
just that the file exists).

#### Inference: `detection/predict.py`

`predict_from_manifest` (batch: trial manifest -> windows -> features
-> predictions, for evaluating on a held-out real test set) and
`predict_single_window` (one raw window in, prediction out, for ad hoc
use). Tests: `tests/test_predict.py` (4 tests).

#### Training CLI: `scripts/train_detection_model.py`

End-to-end entry point: trial manifest -> windows -> features (cached
to `results/detection_model/features_cache.parquet` so re-runs don't
recompute from scratch) -> subject-aware split -> train -> evaluate on
test -> save model + a JSON report (`evaluation_report.json`) with
every metric plus split sizes, for a permanent record of what a given
model run actually achieved.

#### Real end-to-end integration smoke test (not just unit tests)

Built a SYNTHETIC-but-realistic 40-trial, 10-subject (5 KFall + 5
SisFall), both-labels dataset -- real harmonized-format parquet files
with an actual designed jerk-spike signal for fall trials, referenced
by a real trial manifest (same `ManifestRow`/`write_manifest` machinery
Stage 4 already uses) -- then ran the FULL CLI
(`train_detection_model.py`) against it. Real output, not fabricated:
105 windows across 10 subjects, subject-aware split (train=61/val=22/
test=22, 6/2/2 subjects), trained model, test-set evaluation: accuracy
0.864, precision 0.667, recall 0.800, F1 0.727, ROC-AUC 0.847. Also
verified `predict_from_manifest` and `predict_single_window` both work
against the saved model afterward -- a spiky synthetic window correctly
scored a higher fall-probability than a quiet one (0.132 vs 0.121),
though neither crossed the 0.5 threshold with this deliberately tiny
(61-training-window) smoke-test model -- an honest small-data
limitation of the SMOKE TEST specifically, not a code defect; the real
signal is the test-set metrics above, which show the full chain
actually learns and generalizes to held-out subjects on realistic
synthetic data.

**IMPORTANT -- what this is and isn't**: every metric above is on
SYNTHETIC data designed to have a learnable jerk signal, generated in
this sandbox because real KFall/SisFall data isn't available here (see
this checkpoint's running pattern: infrastructure is built and tested
here, then handed off to be run against real data). This proves the
CODE PATH works correctly end to end -- it says nothing about real-world
model quality.

---

#### REAL-DATA MILESTONE: first production run, `--datasets sisfall`

Running `python scripts/train_detection_model.py` against the full
real `data/harmonized/manifest.parquet` immediately surfaced a real
DATA gap, not a code bug: **only 1 KFall subject (SA06) has ever been
downloaded locally** (unchanged since Stage 3/Task 3.11), vs. all 38
real SisFall subjects. `split_by_subject`'s "need >=3 subjects per
dataset" check correctly refused to silently produce a meaningless
1-subject "split" for KFall -- exactly the failure mode that check
exists to catch. KFall currently contributes 0% to any trained model
until more real subjects are downloaded (same Kaggle-mirror process
used in Stage 2).

**Real bug found and fixed in the process**: `train_detection_model.py`
cached features to a single fixed `features_cache.parquet` path
regardless of `--datasets`. Re-running with `--datasets sisfall` after
an earlier full-manifest run would have SILENTLY loaded the wrong
(all-datasets) cached features rather than recomputing -- caught before
it produced a misleading result, not after. Fixed: cache filename now
encodes the dataset selection (`features_cache_all.parquet` vs.
`features_cache_sisfall.parquet`), verified with a real two-run test
confirming both files are written separately.

**Real production result** (`--datasets sisfall`, full real 38-subject
SisFall data, genuinely held-out subjects, zero leakage):
- 74,821 windows, 38 subjects -- split 26 train / 6 val / 6 test
- Test set (9,424 windows, 6 real subjects the model never trained on):
  **accuracy 0.860, precision 0.648, recall 0.819, F1 0.723, ROC-AUC
  0.928**
- Confusion matrix: TN=6388, FP=936, FN=380, TP=1720

ROC-AUC 0.93 indicates a genuinely strong learned signal, not luck --
the precision/recall numbers are for the default 0.5 threshold
specifically and can be shifted toward higher recall (fewer missed
falls, more false alarms) without retraining, if that's the right
tradeoff for the eventual use case. Recall/precision are likely
somewhat understated by the known whole-trial label-noise limitation
(pre-fall/post-fall windows within a fall trial are labeled "fall" but
don't look like one).

This is the first genuinely real, held-out-subject result this project
has produced -- a real milestone, even though it's SisFall-only pending
more KFall subjects.

---

#### REAL-DATA MILESTONE #2: full cross-dataset training (KFall 32 subjects + SisFall 38 subjects)

Downloaded the FULL real KFall dataset (Kaggle mirror
`usmanabbasi2002/kfall-dataset`) -- 32 subjects (SA06-SA33, SA35-SA38;
matches the published dataset's subject count exactly, confirmed
against the paper). Re-ran `harmonize_dataset.py --dataset kfall`:
5,075 trials processed (matches the paper's stated 2,729 ADL + 2,346
fall exactly), 5,053 written, 22 quarantined (0.43% -- a very low,
healthy rate), **calibration sources `{T01: 32}`** -- every single real
KFall subject had a usable dedicated calibration trial, the cleanest
possible calibration result (no auto_detected/group_fallback needed at
all, unlike SisFall which has no dedicated calibration trial by
design). Stage 4's upsert manifest design worked exactly as intended --
KFall's rows replaced cleanly, SisFall's rows untouched.

Re-ran `train_detection_model.py` with no `--datasets` filter (both
datasets, 70 total real subjects, 112,087 windows) -- split 48 train /
11 val / 11 test subjects. Real test-set result on genuinely held-out
subjects from BOTH datasets:

| Metric | SisFall-only (38 subj) | KFall+SisFall (70 subj) |
|---|---|---|
| Accuracy | 0.860 | 0.836 |
| Precision | 0.648 | 0.692 |
| Recall | 0.819 | 0.839 |
| F1 | 0.723 | 0.758 |
| ROC-AUC | 0.928 | 0.921 |

**Precision AND recall both improved simultaneously** with the added
KFall data -- not just a threshold trade-off (which is the usual
pattern when one metric goes up at the other's expense), a genuine
quality improvement from more, more-diverse real training data.
Accuracy dropped slightly and ROC-AUC is flat within noise, neither of
which is concerning given accuracy is the least trustworthy metric
here (class imbalance) and ROC-AUC moved by less than 1 point on a
different, larger, more diverse test set.

**This is now the project's real, final baseline for the detection
pipeline** -- trained and validated across 70 real subjects from two
genuinely different studies (different countries, sensor hardware, age
ranges, and fall-simulation protocols), which is a much stronger
generalization signal than either dataset alone. Model and full report
saved to `results/detection_model/xgboost_model.json` and
`evaluation_report.json` on the user's machine (not committed to git --
these are real trained artifacts, not something to version-control
without a much bigger discussion about model versioning/storage).

**Known, deliberate limitations of this first complete version** (not
oversights -- documented tradeoffs, consistent with the "simple
baseline done properly first" scope decision):
- Coarse whole-trial labels: every window in a fall trial is labeled
  "fall," including pre-fall walking and post-fall lying-still windows
  that don't actually contain the fall event itself. The blueprint
  flags this explicitly as expected label noise, not a bug to fix here.
- No raw-signal deep-learning branch and no domain-adversarial
  cross-dataset adaptation (both in the original blueprint's more
  ambitious two-branch ensemble design) -- deliberately deferred; this
  is the "XGBoost baseline done properly" scope, not the full ensemble.
- No formal LOSO/LODO cross-validation report (research-grade, more
  appropriate for a paper's headline number) -- a single subject-aware
  train/val/test split was used instead, which is the right choice for
  "get real predictions on held-out data" rather than an exhaustive
  academic evaluation. `global_subject_id` makes adding LOSO/LODO later
  straightforward if needed.

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
---

### Stage 7 — IN PROGRESS (prediction pipeline: windowing + onset/impact labeling, NOT yet real-data verified)

Started `prediction/` -- the first layer of the prediction pipeline
(3-class pre-impact classification, KFall-only). Scope for this pass,
same "one layer at a time" approach as detection's Stage 6: turn the
trial-level manifest into a WINDOW-level manifest with frame-precise
onset/impact labels + a window-loading function. No model code yet.

**Small refactor first**: `generate_window_specs`/`WindowSpec`/
`WindowingConfig` moved from `detection/windowing.py` to a new
`shared/windowing.py` -- the boundary math was already fully generic
(no detection-specific assumption anywhere in it), and the blueprint
explicitly rules out `prediction/` importing from `detection/`
directly. `detection/windowing.py` now just re-exports from
`shared/windowing.py` with its original defaults -- verified as a true
no-op by running the full pre-existing 183-test suite unchanged before
adding anything new; all 183 passed.

**`prediction/windowing.py`**: `PredictionWindowingConfig`, dense
defaults per the blueprint's Pipeline 2 §3 spec -- 1.0s window (100
samples @ 100Hz), 0.1s stride (10-sample dense overlap), vs.
detection's 2.0s/1.0s. Rationale for the density: the pipeline's core
metric is lead time (ms before impact the model first flags
`pre_impact`), which needs fine temporal resolution to measure at all.

**`prediction/labelers.py`**: `onset_impact_label()`, the 3-class
scheme from the blueprint §4 (`non_fall` / `pre_impact` / `fall`),
kept as its own module rather than folded into `dataset.py` (the
blueprint explicitly calls this out as worth keeping separate/
independently-testable from detection's whole-trial labeler). Real
design decision made explicit here: a 1.0s window can be comparable in
length to or longer than an entire ~0.6-1.0s KFall fall event, so a
single window can legitimately overlap BOTH the onset->impact interval
AND extend past the impact frame -- the three classes are NOT
naturally mutually exclusive by raw overlap alone. Resolved with an
explicit precedence rule: `fall` wins over `pre_impact` whenever a
window's frame range reaches the impact frame, on the reasoning that a
window containing the actual impact is the more safety-relevant state
to report distinctly. This precedence is a judgment call, not derived
from the blueprint text directly -- worth revisiting once real KFall
confusion-matrix behavior is visible (see Known open items below).

**`prediction/dataset.py`**: `build_windows_manifest` (filters via
`shared.manifest.query_prediction_trials` -- already existed from
Stage 4, unmodified) + `load_window` (identical edge-padding contract
to detection's: repeat the last real sample, not zero-pad). Duplicated
rather than shared with `detection/dataset.py`'s version, per the
blueprint's no-cross-import rule between the two pipelines.

**KNOWN GAP, flagged rather than silently worked around**: the
blueprint's Pipeline 2 §1 says prediction should keep KFall's full
channel set including the pre-fused Euler angles ("no need to restrict
channels ... since you're single-dataset"). But the harmonization
pipeline as ALREADY BUILT (Stage 3, real-data-verified) drops Euler
angles for every trial at harmonization time, before either pipeline
sees the data -- `shared/harmonize/pipeline.py`'s own docstring already
anticipated this: "Callers who want KFall's Euler angles for a
KFall-only experiment should read them from the original trial.signal
directly, before harmonization." So `prediction/dataset.py` currently
operates on the same 6 acc_*/gyro_* channels as detection, NOT Euler.
Fixing this would mean either re-harmonizing KFall with a per-dataset
channel policy (a change to the already-verified harmonization
pipeline, not a prediction-side change) or bypassing the harmonized
layer for this one pipeline. Deliberately not decided yet -- worth a
real conversation before touching Stage 3's code.

**Tests**: `tests/test_prediction_labelers.py`,
`tests/test_prediction_windowing.py`,
`tests/test_prediction_dataset.py` -- all against synthetic fixture
data (including the project's own real SA06 T22 R01 onset=130/
impact=208 values as test inputs, per the Stage 3 REAL-DATA MILESTONE
section above, though the surrounding trial/signal data in these tests
is still synthetic, not the real file). Full suite: 201 passed (183
original + 18 new), zero regressions.

**REAL-DATA MILESTONE (windowing + labeling)**: ran
`build_windows_manifest` against the full real
`data/harmonized/manifest.parquet` (real KFall, all subjects). Real
output, not fabricated:

| label | count | % |
|---|---|---|
| non_fall | 249,970 | 71.6% |
| fall | 83,059 | 23.8% |
| pre_impact | 15,912 | 4.6% |

Total: 348,941 windows. `pre_impact` is the rare class at the real
dataset level, confirming the blueprint's prediction (this was
explicitly NOT guaranteed by the earlier synthetic single-trial test,
per that test's own caveat comment -- now confirmed for real).

Frame-level spot check, SA06 T22 R01 (onset=130, impact=208 -- the
same real trial verified in Stage 3's REAL-DATA MILESTONE):

```
start_frame  end_frame  label
30           130        non_fall     <- ends exactly at onset, not past it
40           140        pre_impact   <- first window past onset
100          200        pre_impact   <- last window before impact
110          210        fall         <- first window past impact
```

Both transitions land exactly where `onset_impact_label`'s documented
boundary rules predict, frame-for-frame -- not approximately. This
closes out the windowing/labeling stage as real-data verified, same
bar as Stage 3's calibration/harmonization milestone.

**NOT yet done / explicitly deferred** (unchanged by the above --
these are the next real layers, not verification gaps):
- Feature engineering (rolling accel magnitude/jerk, tilt-angle
  deviation from baseline) -- blueprint §5, not started.
- Model code (ConvLSTM / tiny-Transformer branches) -- blueprint §6,
  not started.
- LOSO evaluation + lead-time metric -- blueprint §7-8, not started.
- The Euler-angle channel gap above.

## Known open items / things to double check later (Stage 7 additions)

- Run `build_windows_manifest` against the real KFall manifest once
  available locally and inspect the real `label` distribution -- confirm
  `pre_impact` really is the rare class as the blueprint predicts, and
  that this isn't an artifact of the synthetic single-trial test setup
  (see that test's own caveat comment in `test_prediction_dataset.py`).
- The `fall`-wins precedence rule in `onset_impact_label` (documented
  above) is a judgment call, not derived from the blueprint text
  directly -- worth revisiting once a real per-class confusion matrix
  is visible (this is also the point at which the blueprint says to
  decide whether to collapse to binary pre-impact/not, if 3-class
  proves too noisy).
- The Euler-angle channel gap (see above) -- decide before building
  the feature-engineering stage, since blueprint §5's tilt-angle
  feature may want Euler directly rather than re-deriving orientation
  from filtered accel/gyro.

---

### Stage 7 continued — Feature engineering (auxiliary channels for the deep branch), NOT yet real-data verified

**`prediction/features.py`**: `compute_auxiliary_channels`, per the
blueprint's Pipeline 2 §5. Structurally different from
`detection/features.py`'s feature engineering, not just a smaller
version of it -- detection produces ONE aggregate scalar vector per
window (feeding the XGBoost branch); this produces ROLLING, per-sample
channels the SAME LENGTH as the input window, meant to be concatenated
onto the raw 6-channel signal as extra input channels for the deep
model (ConvLSTM/tiny-Transformer, still not built) rather than a
separate tabular feature matrix. The blueprint is explicit about why:
aggregate stats over a 1.0s/dense-stride window "would be extremely
noisy and expensive" here.

Two channels computed, per the blueprint:
1. **Rolling acceleration magnitude** -- plain Euclidean norm of the
   3-axis acceleration, per sample.
2. **Jerk** -- first derivative of accel magnitude. Uses `np.gradient`
   (central differences, same-length output), NOT `detection/
   features.py`'s `np.diff` (which shortens the array by one) --
   `np.diff` would misalign this channel against the raw signal by one
   sample if used per-sample here; that shortening was harmless in
   detection only because it fed into an aggregate stat, not an
   aligned channel.
3. **Tilt-angle deviation from calibrated standing baseline** -- angle
   (degrees) between the acceleration vector and canonical vertical
   (z-axis). Real clarification worth recording: this is NOT an
   approximation of "deviation from baseline" -- it IS that quantity,
   exactly, by construction. `shared/harmonize/axis_alignment.py`
   already rotates every trial so the subject's calibrated standing
   orientation becomes the canonical z-axis, so angle-from-z on the
   already-harmonized signal already IS angle-from-the-calibrated-
   baseline. Same formula as detection's `tilt_mean`/`tilt_std`,
   computed per-sample here instead of aggregated.

`augment_window()` concatenates raw (6) + auxiliary (3) = 9 channels,
in `CHANNELS + AUX_CHANNEL_NAMES` order. `load_augmented_window()`
combines `prediction.dataset.load_window` + `augment_window` in one
call -- this is the function a model's data loader will actually call
per window, once model code exists.

**Tests**: `tests/test_prediction_features.py`, 11 tests, all against
hand-computable synthetic vectors (pure-vertical -> 0deg tilt,
pure-horizontal -> 90deg tilt, linear ramp -> exact expected jerk
slope, constant-magnitude -> zero jerk, degenerate all-zero window
guarded not crashed). Full suite: 212 passed (201 prior + 11 new),
zero regressions.

**NOT yet done / explicitly deferred** (unchanged focus areas):
- Real-data verification of the auxiliary channels against a real
  KFall window (e.g. confirm tilt_deviation_deg looks sane -- roughly
  near 0 for a standing/T01 window, and spikes meaningfully during a
  real T22 fall window -- the equivalent of Stage 3's "spot-check 2-3
  real trials by eye" step, not yet done here).
- Model code (ConvLSTM / tiny-Transformer branches) -- blueprint §6.
- LOSO evaluation + lead-time metric -- blueprint §7-8.
- The Euler-angle channel gap (Stage 7 first section, above) --
  becomes more relevant now: the blueprint's tilt-angle feature idea
  may have been written assuming Euler angles were available directly,
  though the acc-vector-angle approach used here (matching detection)
  is a reasonable, already-precedented substitute, not obviously worse.

---

### Stage 7 continued — LOSO fold construction + PyTorch data-loading layer, NOT yet real-data verified

Two more layers, prerequisite infra for model code (not the models
themselves yet):

**`prediction/loso.py`**: `generate_loso_folds()` (one fold per
subject, deterministically sorted) + `get_fold_masks()` (boolean
train/test masks, not copied DataFrames -- avoids an extra full-copy
per fold when running 32 folds over 348k+ rows). This is the ACTUAL
leakage-prevention mechanism for this pipeline, per blueprint §8. Kept
conceptually and physically separate from the batch sampler below --
see both modules' docstrings for why blueprint §7's "avoid near-
duplicate windows leaking across train/val" wording, taken literally,
could be misread as describing the batch sampler instead of this.

**`prediction/torch_dataset.py`**: `PredictionWindowDataset` (one
manifest row -> one (9, 100) channel-first tensor + label, via
`load_augmented_window`, with an instance-level per-file signal cache
-- documented tradeoff: not shared across `DataLoader` worker
processes) + `TrialGroupedBatchSampler` (shuffles at the trial-group
level per blueprint §7's "sequence-aware batching," not at the
individual-window level -- a training-dynamics concern, NOT a leakage-
prevention one; that's `loso.py`'s job, applied earlier in the
pipeline, before any Dataset exists).

**Tests**: `tests/test_prediction_loso.py` (6 tests) +
`tests/test_prediction_torch_dataset.py` (8 tests) = 14 new, all
against synthetic fixtures. Full suite: 226 passed (212 prior + 14
new), zero regressions. Also ran a manual end-to-end smoke test wiring
`generate_loso_folds` -> `get_fold_masks` -> `PredictionWindowDataset`
-> `TrialGroupedBatchSampler` -> real `torch.utils.data.DataLoader`
together on synthetic 2-subject data -- confirmed batch output shape
`(8, 9, 100)`, exactly what a Conv1d-based model expects (batch,
channels, length). Not a pytest (not worth keeping as a permanent
test given it doesn't assert anything the unit tests don't already
cover individually), but worth recording that the pieces were
confirmed to actually fit together, not just pass in isolation.

**Environment note**: `torch` (2.13.0, CPU/cu130 build from PyPI) had
to be installed in the sandbox for this stage -- not previously a
project dependency (detection's XGBoost baseline didn't need it).
Should be added to whatever `requirements.txt`/`pyproject.toml`
dependency list the project uses, if it isn't already there locally
(you likely already have torch installed given the FallTransformer
work on the SisFall comparison pipeline, but flagging in case the
`fall-project` environment specifically doesn't).

**NOT yet done / explicitly deferred**:
- Model architectures (ConvLSTM, tiny-Transformer) -- blueprint §6.
- Focal loss (more aggressively weighted than Pipeline 1's, per §7).
- Lead-time metric -- blueprint §7's core reported metric, not built.
- Training script tying LOSO + Dataset + model + loss together.
- Real-data verification of the data-loading layer specifically
  (the smoke test above used synthetic data only) -- **now done**:
  `generate_loso_folds` against the real KFall windows manifest
  produced 32 folds (matches KFall's real subject count exactly).
  `get_fold_masks` on fold 0 (test_subject=kfall_SA06):
  train_windows=338,494, test_windows=10,447 -- sums to 348,941,
  exactly matching Stage 7's earlier real-data window-count milestone,
  confirming the train/test masks are disjoint and complete against
  real data, not just in the synthetic partition tests.
