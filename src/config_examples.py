"""The `CONFIG` blocks to copy into `src/run_experiment.py`, one per kind of run.

These are real configurations, aimed at the real `data/` folder and the real hardware
serial numbers - the only thing missing is the light. Running this file builds every
one of them, so a typo or an impossible combination shows up here rather than at the
start of a six-hour scan:

    python src/config_examples.py

Copy the body of the example you want between `CONFIG = ExperimentConfig(` and `)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_config import ExperimentConfig  # noqa: E402
from hardware import RotationStageConfig  # noqa: E402

# The four channels of the David setup: transmitted and reflected arm of each harmonic.
MODES = {1: "H3T", 5: "H3R", 10: "H5T", 14: "H5R"}


# ----------------------------------------------------------------------------
# 1. One power level - the everyday run
# ----------------------------------------------------------------------------
# Ten minutes cut into ten chunks, a snapshot every minute for the stability curves,
# then the analysis. This is the shape to start from.

SINGLE_POWER = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="65mW",

    FREQUENCY=18.66e6,
    MODES=MODES,
    P1={"present": True, "angle_deg": 0.0},
    HWP_ANGLE_DEG=22.5,
    FILTERS={"H3": {"separation": "per-channel", "filter": "700-40"},
             "H5": {"separation": "per-channel", "filter": "400-20"}},

    TRIGGER_LEVEL_V=0.5,
    INPUT_DELAYS_PS={1: 0, 5: 0, 10: 0, 14: 0},
    DEADTIMES_PS={},
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=600.0,
    CHUNK_DURATION_S=60.0,
    SAVE_MERGED=True,
    EXPORT_FORMAT="pkl",

    STABILITY_ENABLED=True,
    STABILITY_SNAPSHOT_INTERVAL_S=60.0,
    STABILITY_INTEGRATION_WINDOW_NS=4.0,
    STABILITY_PEAKS=15,
    STABILITY_ROLLING_WINDOW_SNAPS=10,

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
    ANALYZE_ONLY=False,
)


# ----------------------------------------------------------------------------
# 2. A power scan - one wave-plate angle per power
# ----------------------------------------------------------------------------
# The (power, angle) pairs are measured by hand with a power meter beforehand: nothing
# in the code computes one from the other, and the power strings are only labels for
# the file names and the plots. Drop ROTATION_STAGE_DRY_RUN once the mount is wired.

POWER_SCAN = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",

    MODES=MODES,
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=600.0,
    CHUNK_DURATION_S=60.0,

    POWER_SCAN=[
        {"power": "25mW", "angle_deg": 8.0},
        {"power": "45mW", "angle_deg": 14.2},
        {"power": "65mW", "angle_deg": 21.0},
    ],
    POWER_SCAN_STOP_ON_ERROR=True,
    ROTATION_STAGE_ENABLED=True,
    ROTATION_STAGE=RotationStageConfig(
        serial_number="27260001",   # the number printed on the controller
        clockwise=True,             # must match the mount the angles were measured on
        home_on_connect=False,
    ),
    ROTATION_STAGE_DRY_RUN=True,    # rehearse first: the moves are logged, not made

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
)


# ----------------------------------------------------------------------------
# 3. A polarization scan - one angle per acquisition, one power throughout
# ----------------------------------------------------------------------------
# The polarizer turns to the requested angle and the half-wave plate to whatever angle
# the calibration table says delivers POLARIZATION_POWER there, so the power stays put
# while the polarization turns. The table in polarization_calibration/ is reused as is;
# recording a new one takes hours and is only needed after the beam path changes.
# Results land under:
#   results/.../polarization/{POWER}/summary/   ← the three circle plots
#   results/.../polarization/{POWER}/angles/    ← per-angle analysis trees

POLARIZATION_SCAN = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="50mW",                     # the label all the angles share

    MODES=MODES,
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=300.0,           # x 24 angles: budget the whole scan
    CHUNK_DURATION_S=60.0,

    POLARIZATION_SCAN=np.arange(0.0, 360.0, 15.0),
    POLARIZATION_POWER=50.0,                # mW: the table's unit, not the label
    POLARIZATION_CALIBRATION=None,          # None -> linear_polarization_lookup_latest.npz
    POLARIZATION_CALIBRATION_DIR=None,      # None -> polarization_calibration/
    POLARIZATION_STAGE_ENABLED=True,
    POLARIZATION_P1=RotationStageConfig(serial_number="27260002", clockwise=True),
    POLARIZATION_HWP=RotationStageConfig(serial_number="27260003", clockwise=True),
    POLARIZATION_SETTLE_TIME_S=0.2,
    POLARIZATION_STAGE_DRY_RUN=True,        # rehearse first, as above

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
)


# ----------------------------------------------------------------------------
# 4. Multi-power polarization overlay (analyze only)
# ----------------------------------------------------------------------------
# Record 25mW, 45mW, 65mW polarization scans in the SAME DATE folder (change
# POWER_LEVEL + POLARIZATION_POWER between runs). Then rebuild every summary and
# the butterfly overlay with all powers on each panel:
#   results/.../polarization/25mW/summary/
#   results/.../polarization/45mW/summary/
#   results/.../polarization/65mW/summary/
#   results/.../polarization/overlay/          ← intensity / g2 / harmonics

POLARIZATION_OVERLAY = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="65mW",
    POLARIZATION_SCAN=np.arange(0.0, 360.0, 15.0),
    POLARIZATION_POWERS=["25mW", "45mW", "65mW"],
    MODES=MODES,
    FREQUENCY=18.66e6,
    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    ANALYZE_ONLY=True,
)


# ----------------------------------------------------------------------------
# 5. Analysis only - no tagger, no mounts
# ----------------------------------------------------------------------------
# Re-reads what is already in DATA_DIR. Listing several powers in POWER_LEVELS is what
# produces power_sweep_summary/, so this is also how to compare powers that were
# recorded one at a time by hand.

ANALYZE_ONLY = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVELS=["25mW", "45mW", "65mW"],
    MODES=MODES,
    FREQUENCY=18.66e6,
    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    ANALYZE_ONLY=True,
)


EXAMPLES = {
    "single power": SINGLE_POWER,
    "power scan": POWER_SCAN,
    "polarization scan": POLARIZATION_SCAN,
    "polarization overlay": POLARIZATION_OVERLAY,
    "analysis only": ANALYZE_ONLY,
}


def describe(name: str, cfg: ExperimentConfig) -> None:
    print(f"\n{name}")
    print(f"  data     : {cfg.DATA_DIR}")
    print(f"  results  : {cfg.RESULTS_DIR}")
    print(f"  powers   : {cfg.POWER_LEVELS if len(cfg.POWER_LEVELS) < 5 else str(cfg.POWER_LEVELS[:3]) + f' ... ({len(cfg.POWER_LEVELS)} labels)'}")
    print(f"  pairs    : {len(cfg.correlation_pairs())} correlations over channels {cfg.active_channels()}")
    if not cfg.ANALYZE_ONLY:
        print(f"  per point: {cfg.n_chunks} chunk(s) of {cfg.CHUNK_DURATION_S:g}s")
        points = len(cfg.POWER_LEVELS)
        print(f"  measuring: {points * cfg.ACQUISITION_DURATION_S / 3600:.2f} h of acquisition in total")


if __name__ == "__main__":
    print("Every example below was accepted by ExperimentConfig's validation.")
    for name, cfg in EXAMPLES.items():
        describe(name, cfg)
