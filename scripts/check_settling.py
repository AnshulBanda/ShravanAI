"""
Checks whether the "high motion variance at trial start" pattern in fall
trials (vs. stillness in ADL trials) generalizes across multiple subjects,
or was specific to SA06.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from shared.io.readers_kfall import read_sensor_csv

MANIFEST_PATH = Path("data/harmonized/manifest.parquet")
SENSOR_ROOT = Path("data/raw/kfall/sensor_data")
N_SUBJECTS = 8  # spread across the subject range, not just SA06

def raw_path(subject_id: str, activity_code: str, trial_id: str) -> Path:
    # subject_id looks like "SA06"; local mirror drops the "A" -> "S06"
    fname_subject = "S" + subject_id[2:]
    return SENSOR_ROOT / subject_id / f"{fname_subject}{activity_code}{trial_id}.csv"

def first_second_std(path: Path) -> float | None:
    if not path.exists():
        return None
    df = read_sensor_csv(path)
    first100 = df.iloc[:100]
    accel_mag = np.sqrt(first100["acc_x"]**2 + first100["acc_y"]**2 + first100["acc_z"]**2)
    return float(accel_mag.std())

def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    kfall = manifest[manifest["dataset"] == "kfall"]

    subjects = sorted(kfall["subject_id"].unique())
    # spread across the range rather than clustering, take every Nth
    step = max(1, len(subjects) // N_SUBJECTS)
    sample_subjects = subjects[::step][:N_SUBJECTS]

    print(f"{'subject':<8} {'adl_trial':<10} {'adl_std':>10}   {'fall_trial':<11} {'fall_std':>10}   ratio")
    print("-" * 70)

    for subject_id in sample_subjects:
        sub_df = kfall[kfall["subject_id"] == subject_id]

        adl_rows = sub_df[sub_df["label"] == "adl"]
        fall_rows = sub_df[sub_df["label"] == "fall"]
        if adl_rows.empty or fall_rows.empty:
            print(f"{subject_id:<8} -- missing adl or fall rows in manifest, skipping")
            continue

        adl_row = adl_rows.iloc[0]
        fall_row = fall_rows.iloc[0]

        adl_path = raw_path(subject_id, adl_row["activity_code"], adl_row["trial_id"])
        fall_path = raw_path(subject_id, fall_row["activity_code"], fall_row["trial_id"])

        adl_std = first_second_std(adl_path)
        fall_std = first_second_std(fall_path)

        if adl_std is None or fall_std is None:
            print(f"{subject_id:<8} file not found: adl={adl_path.exists()} fall={fall_path.exists()}")
            continue

        ratio = fall_std / adl_std if adl_std > 0 else float("inf")
        print(f"{subject_id:<8} {adl_row['activity_code']+adl_row['trial_id']:<10} {adl_std:>10.4f}   "
              f"{fall_row['activity_code']+fall_row['trial_id']:<11} {fall_std:>10.4f}   {ratio:>5.1f}x")

if __name__ == "__main__":
    main()
