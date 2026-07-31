"""Replay the rehearsal folders through the analysis, with no hardware and no tagger.

    python tests/rehearsal/replay_check.py

Reads the `.pkl` files under `tests/rehearsal/{ellipticity,laser_angle}/data/`, writes
every figure into the sibling `replay/` folder, and checks the ellipticity the fits give
against `truth.json` - the value the synthetic data was built with. It is the quickest
way to see that a change to the analysis still produces the numbers it should.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REHEARSAL = Path(__file__).resolve().parent
sys.path.insert(0, str(REHEARSAL.parent.parent / "src"))

from experiment_config import ExperimentConfig  # noqa: E402
from pkl_json_analyze import (plot_ellipticity_campaign,  # noqa: E402
                              plot_laser_angle_campaign)

MODES = {1: "H3T", 2: "H3R", 3: "H4T", 4: "H4R"}
TOLERANCE = 0.05


def ellipticity_config(analyzer_kind: str = "hwp") -> ExperimentConfig:
    root = REHEARSAL / "ellipticity"
    return ExperimentConfig(
        MODES=MODES,
        DATA_DIR=root / "data",
        RESULTS_DIR=root / "replay",
        ELLIPTICITY_LASER_ANGLES=[0.0, 45.0, 90.0],
        ELLIPTICITY_ANALYZER_ANGLES=np.arange(0.0, 180.0, 15.0),
        ELLIPTICITY_ANALYZER_KIND=analyzer_kind,
        INTEGRATION_WINDOW=8,
        ANALYZE_ONLY=True,
    )


def laser_angle_config() -> ExperimentConfig:
    root = REHEARSAL / "laser_angle"
    return ExperimentConfig(
        MODES=MODES,
        DATA_DIR=root / "data",
        RESULTS_DIR=root / "replay",
        LASER_ANGLE_SCAN=np.arange(0.0, 180.0, 22.5),
        INTEGRATION_WINDOW=8,
        ANALYZE_ONLY=True,
    )


def check_ellipticity(by_power: dict) -> int:
    """Compare every fitted ellipticity to the value the data was built with."""
    truth = json.loads((REHEARSAL / "ellipticity" / "data" / "truth.json").read_text())
    failures = 0
    print("\n================ ellipticity against truth.json ================")
    for power, campaign in sorted(by_power.items()):
        for sweep in campaign["sweeps"]:
            expected = truth.get(f"{sweep['laser_angle_deg']:g}")
            for harmonic, fit in sorted(sweep["fits"].items()):
                measured = fit["ellipticity"]
                ok = expected is not None and abs(measured - expected) <= TOLERANCE
                failures += not ok
                print(f"  {power} laser {sweep['laser_angle_deg']:>5g}° {harmonic}: "
                      f"ε {measured:.4f} (expected {expected}) "
                      f"{'ok' if ok else 'MISMATCH'}")
    return failures


def main() -> int:
    print("================ laser angle ================")
    plot_laser_angle_campaign(laser_angle_config())

    print("\n================ ellipticity ================")
    by_power = plot_ellipticity_campaign(ellipticity_config())
    failures = check_ellipticity(by_power)
    print("\nall ellipticities within tolerance." if not failures
          else f"\n{failures} ellipticity value(s) off by more than {TOLERANCE}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
