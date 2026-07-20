"""
File-based bridge between the trained prediction checkpoint and an
external system (e.g. a digital twin) that communicates via JSON files
on disk. Run this as a long-lived background process; it polls an
"incoming" directory for new IMU sample files, maintains a rolling
window, runs the model + the same PredictionSmoother the live demo
uses, and writes the current predicted state to a single "outgoing"
JSON file that the digital twin can read at its own pace.

============================================================================
IMPORTANT -- READ BEFORE WIRING THIS TO ANYTHING LIVE:
This bridge feeds the RAW model output through the smoother, on
WHATEVER data it's given. That's different from the live-demo scripts
(`run_live_demo.py`, `curate_demo_trials.py`), which specifically use
pre-verified, curated real trials known to behave well. Fed arbitrary
live/new data, this checkpoint's real, measured reliability is:
  - Only ~3% of real held-out fall trials (2/67 for kfall_SA06) meet a
    reasonably strict "not too early, not too late" timing bar.
  - ~29% false-positive rate on ordinary ADL activity (no fall at all).
  - Failure mode is unpredictable per-sequence: sometimes early false
    alarm, sometimes multi-second-late miss, sometimes both.
See PROJECT_CHECKPOINT.md's latest Stage 7 section for the full
diagnostic writeup. Treat this bridge as DEMO/PROTOTYPE-STAGE
integration plumbing, not a validated real-time safety system.
============================================================================

INPUT CONTRACT (what the digital twin should write):
Drop one JSON file per sample (or a batch) into --incoming-dir. Each
file is either a single sample object or a list of sample objects:
    {"acc_x": 0.05, "acc_y": -1.01, "acc_z": 0.14,
     "gyro_x": 1.2, "gyro_y": -0.3, "gyro_z": 0.8}
Filenames don't matter (processed in sorted/lexicographic order, then
deleted) -- use an incrementing or timestamp-based name so ordering is
correct, e.g. sample_00001.json, sample_00002.json, ...
Samples MUST already be at 100Hz (matching the checkpoint's training
rate) and in real physical units matching what the model was trained
on (same as raw KFall sensor units -- g's for accel, deg/s for gyro,
per shared/harmonize/'s pipeline) -- this bridge does NOT resample or
re-scale, it assumes the caller has already harmonized the signal the
same way training data was.

OUTPUT CONTRACT (what the digital twin should read):
--outgoing-file is REWRITTEN (not appended) after every processed
sample/batch, always containing the CURRENT state:
    {
      "timestamp": 1737...,          // bridge's wall-clock time, seconds
      "window_ready": true,          // false until enough samples buffered
      "state": "calm" | "pre_impact" | "fall" | "buffering",
      "probabilities": {"non_fall": 0.7, "pre_impact": 0.2, "fall": 0.1}
                                      // null while buffering
    }

Usage:
    python scripts/run_digital_twin_bridge.py \
        --checkpoint results/prediction_model/checkpoints/convlstm_boost2.0_kfall_SA06.pt \
        --model convlstm \
        --incoming-dir digital_twin_bridge/incoming \
        --outgoing-file digital_twin_bridge/prediction_state.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from prediction.features import augment_window
from prediction.live_smoothing import PredictionSmoother, SmootherConfig
from prediction.models.convlstm import ConvLSTM
from prediction.models.tiny_transformer import TinyTransformer

REPO_ROOT = Path(__file__).parent.parent
MODEL_BUILDERS = {"convlstm": ConvLSTM, "tiny_transformer": TinyTransformer}
# Must match prediction/dataset.py's CHANNELS order exactly -- this is
# the order augment_window (and therefore the trained model) expects.
CHANNEL_KEYS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--window-length-s", type=float, default=1.0,
                         help="MUST match the checkpoint's training window_length_s (see its win{X} filename segment).")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--incoming-dir", default=str(REPO_ROOT / "digital_twin_bridge" / "incoming"))
    parser.add_argument("--outgoing-file", default=str(REPO_ROOT / "digital_twin_bridge" / "prediction_state.json"))
    parser.add_argument("--poll-interval-s", type=float, default=0.05,
                         help="How often to check --incoming-dir for new files. Should be well under the real sample interval (0.01s at 100Hz).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window_length_samples = round(args.window_length_s * args.sample_rate_hz)

    incoming_dir = Path(args.incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    outgoing_path = Path(args.outgoing_file)
    outgoing_path.parent.mkdir(parents=True, exist_ok=True)

    model = MODEL_BUILDERS[args.model]()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    smoother = PredictionSmoother(SmootherConfig())
    raw_buffer: deque = deque(maxlen=window_length_samples)

    print(f"ShravanAI digital-twin bridge running.")
    print(f"  watching incoming dir: {incoming_dir}")
    print(f"  writing state to:      {outgoing_path}")
    print(f"  window: {window_length_samples} samples @ {args.sample_rate_hz}Hz  device={device}")
    print(f"  (Ctrl+C to stop)\n")

    try:
        while True:
            incoming_files = sorted(incoming_dir.glob("*.json"))
            for f in incoming_files:
                try:
                    payload = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [skip] unreadable file {f.name}: {e}")
                    f.unlink(missing_ok=True)
                    continue

                samples = payload if isinstance(payload, list) else [payload]
                for s in samples:
                    try:
                        raw_buffer.append([float(s[k]) for k in CHANNEL_KEYS])
                    except (KeyError, TypeError, ValueError) as e:
                        print(f"  [skip] malformed sample in {f.name}: {e} -- expected keys {CHANNEL_KEYS}")
                f.unlink(missing_ok=True)

            result = {"timestamp": time.time(), "window_ready": len(raw_buffer) == window_length_samples}

            if len(raw_buffer) == window_length_samples:
                raw_window = np.array(raw_buffer, dtype=np.float64)  # (n, 6)
                augmented = augment_window(raw_window, sample_rate_hz=args.sample_rate_hz)  # (n, 9)
                x = torch.from_numpy(np.ascontiguousarray(augmented.T)).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
                state = smoother.update(probs)
                result["state"] = state.value
                result["probabilities"] = {
                    "non_fall": float(probs[0]), "pre_impact": float(probs[1]), "fall": float(probs[2]),
                }
            else:
                result["state"] = "buffering"
                result["probabilities"] = None
                result["buffered_samples"] = len(raw_buffer)
                result["needed_samples"] = window_length_samples

            outgoing_path.write_text(json.dumps(result, indent=2))
            time.sleep(args.poll_interval_s)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
