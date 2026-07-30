"""Replay the rehearsal data through the analysis, and check the ellipticity it returns.

    python tests/rehearsal/make_data.py         # record the synthetic scans first
    python tests/rehearsal/replay_check.py      # both, or ellipticity / laser_angle

The ellipticity half is a real test, not just a smoke run: `make_data.py` wrote down the
ellipticity it put into every pump angle, and this compares that against what the
sinusoidal fit reads back out of the files. It exits non-zero if any harmonic is off by
more than TOLERANCE.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from experiment_config import ExperimentConfig  # noqa: E402
import run_experiment  # noqa: E402

REHEARSAL = REPO / "tests" / "rehearsal"
MODES = {1: "H3T", 2: "H3R", 3: "H4T", 4: "H4R"}
TOLERANCE = 0.05  # the fit is read off two 1 s chunks per point, so it is noisy


def _config(folder: str, **fields) -> ExperimentConfig:
    return ExperimentConfig(
        MATERIAL="Rehearsal",
        EXPERIENCE_TYPE="Synthetic",
        DATE="30072026",
        DATA_DIR=REHEARSAL / folder / "data",
        RESULTS_DIR=REHEARSAL / folder / "replay",
        MODES=MODES,
        FREQUENCY=18.66e6,
        INTEGRATION_WINDOW=8,
        INTEGRATION_WINDOWS_SWEEP=np.arange(1, 11, 1),
        ANALYZE_ONLY=True,
        **fields,
    )


def ellipticity() -> bool:
    cfg = _config("ellipticity",
                  ELLIPTICITY_LASER_ANGLES=[0.0, 45.0, 90.0],
                  ELLIPTICITY_ANALYZER_ANGLES=np.arange(0.0, 180.0, 15.0))
    run_experiment.main(cfg)
    return _check_against_truth(cfg)


def laser_angle() -> bool:
    cfg = _config("laser_angle",
                  POWER_LEVEL="50mW",
                  LASER_ANGLE_SCAN=np.arange(0.0, 180.0, 22.5))
    run_experiment.main(cfg)
    return True


def _check_against_truth(cfg: ExperimentConfig) -> bool:
    """Every fitted ellipticity against the one that was injected at that pump angle."""
    truth_path = Path(cfg.DATA_DIR) / "truth.json"
    if not truth_path.is_file():
        print(f"\nNo {truth_path.name}: run make_data.py to record it. Check skipped.")
        return True
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    print("\n================ ellipticity: fitted vs injected ================")
    print(f"{'power':>8} {'laser angle':>12} {'harmonic':>9} {'fitted':>8} "
          f"{'injected':>9} {'error':>8}")
    worst = 0.0
    for path in sorted(Path(cfg.ellipticity_root).glob("*/summary/"
                                                       "ellipticity_vs_laser_angle.csv")):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                expected = truth.get(row["laser_angle_deg"])
                if expected is None:
                    continue
                error = abs(float(row["ellipticity"]) - expected)
                worst = max(worst, error)
                flag = "" if error <= TOLERANCE else "  <-- off"
                print(f"{row['power_label']:>8} {row['laser_angle_deg']:>12} "
                      f"{row['harmonic']:>9} {float(row['ellipticity']):>8.4f} "
                      f"{expected:>9.4f} {error:>8.4f}{flag}")

    ok = worst <= TOLERANCE
    print(f"\nworst error {worst:.4f} (tolerance {TOLERANCE:g}): "
          f"{'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    passed = True
    if which in ("both", "ellipticity"):
        passed &= ellipticity()
    if which in ("both", "laser_angle"):
        passed &= laser_angle()
    sys.exit(0 if passed else 1)
