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
from hardware import ELLStageConfig, RotationStageConfig  # noqa: E402

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
# 3. A laser angle scan - one pump angle per acquisition, one power throughout
# ----------------------------------------------------------------------------
# The pump polarizer turns to the requested angle and the pump half-wave plate to
# whatever angle the calibration table says delivers PUMP_POWER there, so the power
# stays put while the polarization turns. The table in polarization_calibration/ is
# reused as is; recording a new one takes hours and is only needed after the beam path
# changes. Results land under:
#   results/.../laser_angle/{POWER}/summary/   ← the four vs-angle plots
#   results/.../laser_angle/{POWER}/angles/    ← per-angle analysis trees

LASER_ANGLE_SCAN = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="50mW",                     # the label all the angles share

    MODES=MODES,
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=300.0,           # x 24 angles: budget the whole scan
    CHUNK_DURATION_S=60.0,

    LASER_ANGLE_SCAN=np.arange(0.0, 360.0, 15.0),
    PUMP_POWER=50.0,                        # mW: the table's unit, not the label
    PUMP_CALIBRATION=None,                  # None -> linear_polarization_lookup_latest.npz
    PUMP_CALIBRATION_DIR=None,              # None -> polarization_calibration/
    PUMP_STAGE_ENABLED=True,
    PUMP_P1=RotationStageConfig(serial_number="27260002", clockwise=True),
    PUMP_HWP=RotationStageConfig(serial_number="27260003", clockwise=True),
    PUMP_SETTLE_TIME_S=0.2,
    PUMP_STAGE_DRY_RUN=True,                # rehearse first, as above

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
)


# ----------------------------------------------------------------------------
# 4. An ellipticity scan - an analyzer sweep at each pump angle
# ----------------------------------------------------------------------------
# The pump mounts hold one laser angle while the analyzer AFTER the crystal turns
# through its sweep. The intensity it transmits traces a sinusoid whose depth gives the
# ellipticity of the harmonics: a curve reaching zero is a linear polarization, a flat
# one is circular.
#
# This is the HWP version: a half-wave plate turns in front of a polarizer that never
# moves. The plate turns the polarization by twice its own angle, so the curve repeats
# every 90 deg. Example 4b below is the same scan with the polarizer itself on the
# mount and no plate at all.
#
# Three mounts move here, so all three must be addressed:
#   PUMP_P1 / PUMP_HWP  before the crystal, ELLIPTICITY_ANALYZER after it.
# Results land under:
#   results/.../ellipticity/{POWER}/las000p0/  ← that sweep and its fit
#   results/.../ellipticity/{POWER}/summary/   ← ellipticity vs laser angle

ELLIPTICITY_SCAN = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="50mW",                     # the label every point shares

    MODES=MODES,
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=120.0,           # x 4 laser angles x 12 analyzer angles
    CHUNK_DURATION_S=30.0,

    ELLIPTICITY_LASER_ANGLES=[0.0, 45.0, 90.0, 135.0],
    ELLIPTICITY_ANALYZER_ANGLES=np.arange(0.0, 180.0, 15.0),
    ELLIPTICITY_ANALYZER_KIND="hwp",        # plate in front of the fixed polarizer
    ELLIPTICITY_ANALYZER_ENABLED=True,
    ELLIPTICITY_ANALYZER=RotationStageConfig(serial_number="27260004", clockwise=True),
    ELLIPTICITY_ANALYZER_DRY_RUN=True,      # rehearse first, as above
    ELLIPTICITY_FIT_PERIOD_DEG=None,        # None -> 90 deg, the plate's own period
    ELLIPTICITY_FIXED_POLARIZER_DEG=0.0,    # metadata: 0 = vertical

    PUMP_POWER=50.0,
    PUMP_STAGE_ENABLED=True,
    PUMP_P1=RotationStageConfig(serial_number="27260002", clockwise=True),
    PUMP_HWP=RotationStageConfig(serial_number="27260003", clockwise=True),
    PUMP_STAGE_DRY_RUN=True,

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
)


# ----------------------------------------------------------------------------
# 4b. The same scan with the polarizer on the mount, and no half-wave plate
# ----------------------------------------------------------------------------
# The plate comes out of the beam and the polarizer itself is turned, on a Thorlabs
# Elliptec mount: a serial port rather than a Kinesis serial number, which is the only
# thing ELLStageConfig changes. Everything downstream is identical - the same file
# names, the same fits, the same figures - except that the transmitted curve now
# repeats every 180 deg instead of 90, which ELLIPTICITY_ANALYZER_KIND takes care of.
#
# The kind is written to experiment_config.txt and to ellipticity_scan.csv, and named on
# every figure, because the two setups are otherwise indistinguishable on disk.

ELLIPTICITY_SCAN_POLARIZER = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="50mW",

    MODES=MODES,
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=120.0,
    CHUNK_DURATION_S=30.0,

    ELLIPTICITY_LASER_ANGLES=[0.0, 45.0, 90.0, 135.0],
    # 0 to 180 inclusive: the mirrored half is drawn from it, so the polar figures come
    # out as full circles rather than semicircles.
    ELLIPTICITY_ANALYZER_ANGLES=np.arange(0.0, 181.0, 15.0),
    ELLIPTICITY_ANALYZER_KIND="polarizer",  # the polarizer itself, no plate in the beam
    ELLIPTICITY_ANALYZER_ENABLED=True,
    ELLIPTICITY_ANALYZER=ELLStageConfig(
        port="COM6",                        # the port the Elliptec mount enumerated as
        address="0",                        # its bus address, '0' unless several share it
        clockwise=True,
        home_on_connect=False,
    ),
    ELLIPTICITY_ANALYZER_DRY_RUN=True,      # rehearse first, as above
    ELLIPTICITY_FIT_PERIOD_DEG=None,        # None -> 180 deg, a polarizer's own period

    PUMP_POWER=50.0,
    PUMP_STAGE_ENABLED=True,
    PUMP_P1=RotationStageConfig(serial_number="27260002", clockwise=True),
    PUMP_HWP=RotationStageConfig(serial_number="27260003", clockwise=True),
    PUMP_STAGE_DRY_RUN=True,

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
)


# ----------------------------------------------------------------------------
# 5. Multi-power laser angle overlay (analyze only)
# ----------------------------------------------------------------------------
# Record 25mW, 45mW, 65mW laser angle scans in the SAME DATE folder (change
# POWER_LEVEL + PUMP_POWER between runs). Then rebuild every summary and the butterfly
# overlay with all powers on each panel:
#   results/.../laser_angle/25mW/summary/
#   results/.../laser_angle/45mW/summary/
#   results/.../laser_angle/65mW/summary/
#   results/.../laser_angle/overlay/          ← intensity / g2 / R / harmonics

LASER_ANGLE_OVERLAY = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="65mW",
    LASER_ANGLE_SCAN=np.arange(0.0, 360.0, 15.0),
    PUMP_POWERS=["25mW", "45mW", "65mW"],
    MODES=MODES,
    FREQUENCY=18.66e6,
    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    ANALYZE_ONLY=True,
)


# ----------------------------------------------------------------------------
# 6. Analysis only - no tagger, no mounts
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
    "laser angle scan": LASER_ANGLE_SCAN,
    "ellipticity scan (HWP)": ELLIPTICITY_SCAN,
    "ellipticity scan (polarizer)": ELLIPTICITY_SCAN_POLARIZER,
    "laser angle overlay": LASER_ANGLE_OVERLAY,
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
