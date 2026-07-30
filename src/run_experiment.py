"""The entry point: edit CONFIG below, then run this file.

    python src/run_experiment.py        # or python -m src.run_experiment

One command covers every mode, all selected from CONFIG: a single power level, an
automated power scan, a polarization scan, and the analysis-only replays of any of
them. See README section 3 for the field-by-field description.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

try:
    import hardware
    from experiment_config import ExperimentConfig, record_run, write_configuration
    from hardware import RotationStage, RotationStageConfig
    from pkl_json_analyze import (discover_polarization_powers, plot_polarization_campaign,
                                  plot_polarization_overlay, plot_polarization_summary,
                                  plot_stability, run_analysis)
except ImportError:
    from src import hardware
    from src.experiment_config import ExperimentConfig, record_run, write_configuration
    from src.hardware import RotationStage, RotationStageConfig
    from src.pkl_json_analyze import (discover_polarization_powers, plot_polarization_campaign,
                                      plot_polarization_overlay, plot_polarization_summary,
                                      plot_stability, run_analysis)


# ==========================================
# MAIN CONFIGURATION - edit this block
# ==========================================

CONFIG = ExperimentConfig(
    # ----------------- identity -----------------
    MATERIAL="CdTe110",
    EXPERIENCE_TYPE="Polarization_Scan",
    DATE="30072026",
    POWER_LEVEL="50mW",

    # ----------------- optics / metadata -----------------
    FREQUENCY=18.66e6,
    MODES={1: "H3T", 2: "H3R", 3: "H4T", 4: "H4R", 5: "H5T", 6: "H5R"},
    P1={"present": False, "angle_deg": 0.0},
    P2={"present": False, "angle_deg": 0.0},
    P3={"present": False, "angle_deg": 0.0},
    HWP_ANGLE_DEG=0.0,
    COVER={"present": True, "description": "between channels"},
    FILTERS={
        "H3": {"separation": "per-channel", "filter": "700-40"},
        "H4": {"separation": "per-channel", "filter": "520=40"},
        "H5": {"separation": "per-channel", "filter": "400-20"},
    },

    # ----------------- Time Tagger -----------------
    TAGGER_SERIAL="",
    TRIGGER_LEVEL_V=0.5,
    INPUT_DELAYS_PS={},
    DEADTIMES_PS={},
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=40,
    CHUNK_DURATION_S=8,
    SAVE_MERGED=True,
    EXPORT_FORMAT="pkl",
    SAVE_RAW_TTBIN=False,

    # ----------------- stability -----------------
    STABILITY_ENABLED=False,
    STABILITY_SNAPSHOT_INTERVAL_S=60.0,
    STABILITY_INTEGRATION_WINDOW_NS=8,
    STABILITY_PEAKS=15,
    STABILITY_ROLLING_WINDOW_SNAPS=10,

    # ----------------- rotation stage / power scan -----------------
    ROTATION_STAGE_ENABLED=False,
    ROTATION_STAGE_DRY_RUN=False,
    POWER_SCAN=None,
    # POWER_SCAN=[
    #     {"power": "25mW", "angle_deg": 8.0},
    #     {"power": "35mW", "angle_deg": 14.2},
    #     {"power": "45mW", "angle_deg": 21.0},
    # ],
    POWER_SCAN_STOP_ON_ERROR=True,

    # ----------------- polarization scan -----------------
    POLARIZATION_SCAN=np.arange(0.0, 180.0, 4.0),
    # POLARIZATION_POWER=90,                  # ignored while POLARIZATION_POWER_SCAN is set
    POLARIZATION_POWER_SCAN=[                 # a full angle scan at each power, one run
        {"power": "90mW", "requested_power": 90.0},
        {"power": "70mW", "requested_power": 70.0},
        {"power": "50mW", "requested_power": 50.0},
        {"power": "30mW", "requested_power": 30.0},
        {"power": "20mW", "requested_power": 20.0},
    ],
    POLARIZATION_CALIBRATION=None,          # None -> linear_polarization_lookup_latest.npz
    POLARIZATION_CALIBRATION_DIR=None,      # None -> polarization_calibration/
    POLARIZATION_STAGE_ENABLED=True,
    POLARIZATION_P1=RotationStageConfig(serial_number="27270550", clockwise=True),
    POLARIZATION_HWP=RotationStageConfig(serial_number="27270567", clockwise=True),
    POLARIZATION_SETTLE_TIME_S=0.2,
    POLARIZATION_STAGE_DRY_RUN=False,        # rehearse first; set False to move the mounts
    POLARIZATION_PLOT_ANGLES=False,         # summaries only; True to also analyse each angle

    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=True,
    ANALYZE_ONLY=False,
)


Run = Tuple[ExperimentConfig, Any]


def main(cfg: ExperimentConfig = CONFIG):
    runs: List[Run] = []

    # Analyze-only replays the files already on disk, always producing the summaries.
    if cfg.ANALYZE_ONLY:
        print(f"Configuration file: {describe_configuration(cfg)}")
        if cfg.is_polarization_scan() or cfg.POLARIZATION_POWERS:
            analyze_polarization_scan(cfg)
        else:
            run_analysis(cfg)
        print_summary(cfg, runs)
        return

    runs = acquire(cfg)

    if cfg.RUN_ANALYSIS_AFTER_ACQUIRE:
        if cfg.is_polarization_scan():
            # acquire() already drew each power's vs-angle summary as it finished and
            # the comparison overlay once every power was recorded; only the optional
            # (heavy) per-angle HBT trees remain.
            if cfg.POLARIZATION_PLOT_ANGLES:
                powers = discover_polarization_powers(cfg.DATA_DIR) or [cfg.POWER_LEVEL]
                plot_polarization_angle_trees(cfg, powers)
        else:
            run_analysis(cfg)

    print_summary(cfg, runs)


def describe_configuration(cfg: ExperimentConfig) -> Path:
    """The folder's configuration file, written only if it does not exist yet.

    A replay config usually carries the analysis fields and not the optics or the
    tagger settings, so rewriting the file would replace the record of what was
    actually measured by this config's defaults. Reprocessing must not do that.
    """
    path = cfg.config_file_path()
    return path if path.is_file() else write_configuration(cfg)


def acquire(cfg: ExperimentConfig) -> List[Run]:
    """Record a polarization scan, every point of a power scan, or a single power."""
    if cfg.is_polarization_scan():
        return acquire_polarization_scan(cfg)

    points = cfg.power_scan_points() or [(cfg.POWER_LEVEL, cfg.ROTATION_ANGLE_DEG)]
    stage = open_stage(cfg)
    runs: List[Run] = []

    try:
        for number, (power, angle) in enumerate(points, start=1):
            if len(points) > 1:
                print(f"\n===== power point {number}/{len(points)}: {power} =====")
            step = cfg.for_power(power, angle)

            if stage is not None and angle is not None:
                stage.move_to_angle_deg(angle)

            try:
                runs.append((step, acquire_one(step)))
            except Exception as exc:
                record_run(step, status="failed")
                print(f"Power point {power} failed: {exc}")
                if cfg.POWER_SCAN_STOP_ON_ERROR:
                    raise
    finally:
        if stage is not None:
            stage.disconnect()

    return runs


def acquire_one(cfg: ExperimentConfig, log_run: bool = True) -> Any:
    if log_run:
        print(f"Configuration file: {record_run(cfg)}")

    try:
        from acquisition import run_acquisition
    except ImportError:
        from src.acquisition import run_acquisition

    result = run_acquisition(cfg)
    if log_run:
        record_run(cfg, result)

    if cfg.STABILITY_ENABLED and result.snapshots:
        plot_stability(result.snapshots, cfg)
    return result


def acquire_polarization_scan(cfg: ExperimentConfig) -> List[Run]:
    """Record a polarization scan at one power, or at each power of a multi-power run.

    A multi-power run (POLARIZATION_POWER_SCAN) opens the two mounts once and drives a
    full angle scan at every power in turn. `with` releases both mounts even when a
    preflight or configuration-file write raises before the first acquisition.
    """
    power_points = cfg.polarization_power_points()
    log = hardware.ScanLog(cfg.polarization_log_path())
    runs: List[Run] = []
    series_by_power: dict = {}

    with open_polarizer(cfg) as controller:
        for number, (label, requested) in enumerate(power_points, start=1):
            if len(power_points) > 1:
                print(f"\n########## power {number}/{len(power_points)}: "
                      f"{label} ({requested:g} {controller.lookup.unit}) ##########")
            power_cfg = cfg.for_polarization_scan_power(label, requested)
            runs.extend(acquire_polarization_power(power_cfg, controller, log))

            # Draw this power's vs-angle summary the moment its scan finishes, so a long
            # multi-power campaign shows results as it goes - and a completed power keeps
            # its plots even if a later power is interrupted.
            if cfg.RUN_ANALYSIS_AFTER_ACQUIRE:
                series = plot_polarization_summary(power_cfg)
                if series is not None:
                    series_by_power[label] = series

    # Every power on disk: overlay them for the comparison (needs at least two).
    if cfg.RUN_ANALYSIS_AFTER_ACQUIRE and len(series_by_power) > 1:
        plot_polarization_overlay(series_by_power, cfg.polarization_overlay_dir)

    return runs


def acquire_polarization_power(cfg: ExperimentConfig, controller: "hardware.PolarizationController",
                               log: "hardware.ScanLog") -> List[Run]:
    """One full angle scan at a single power, on an already-open controller."""
    points = cfg.polarization_scan_points()
    runs: List[Run] = []

    try:
        print(f"Configuration file: {write_configuration(cfg)}")
        plan = hardware.preflight(controller.lookup, [angle for angle, _ in points],
                                  cfg.POLARIZATION_POWER)

        for number, ((angle, label), setting) in enumerate(zip(points, plan), start=1):
            print(f"\n===== angle {number}/{len(points)}: {angle:g} deg ({label}) =====")
            if not setting.reachable:
                log.record(label, setting, status="unreachable")
                print(f"Skipped: {setting.describe()}")
                continue

            setting = controller.set_polarization(angle, cfg.POLARIZATION_POWER)
            step = cfg.for_polarization(angle)
            log.record(label, setting, status="running")

            try:
                result = acquire_one(step, log_run=False)
                runs.append((step, result))
                log.record(label, setting, status="complete", result=result)
            except Exception as exc:
                log.record(label, setting, status="failed", note=str(exc))
                print(f"Angle {angle:g} deg failed: {exc}")
                if cfg.POWER_SCAN_STOP_ON_ERROR:
                    raise
    finally:
        print(f"Per-angle log: {log.write()}")

    return runs


def open_polarizer(cfg: ExperimentConfig) -> hardware.PolarizationController:
    """The polarizer pair, not connected yet: use it as a context manager."""
    path = cfg.POLARIZATION_CALIBRATION or hardware.find_calibration(cfg.POLARIZATION_CALIBRATION_DIR)
    lookup = hardware.load_lookup(path)
    dry_run = cfg.POLARIZATION_STAGE_DRY_RUN or not cfg.POLARIZATION_STAGE_ENABLED
    return hardware.PolarizationController(
        lookup,
        p1=cfg.POLARIZATION_P1 or RotationStageConfig(),
        hwp=cfg.POLARIZATION_HWP or RotationStageConfig(),
        dry_run=dry_run,
        settle_time_s=cfg.POLARIZATION_SETTLE_TIME_S,
    )


def analyze_polarization_scan(cfg: ExperimentConfig) -> None:
    """Write the vs-angle summaries first, then the per-angle trees only if asked.

    The circle/butterfly plots versus angle are the point of a polarization scan, so
    they are drawn first and always. The per-angle analysis trees are heavy and rarely
    the first thing looked at, so they are only produced when POLARIZATION_PLOT_ANGLES
    is True. Powers come from `POLARIZATION_POWERS` when set (analyze-only multi-power),
    else from the files already in DATA_DIR, else from the single `POWER_LEVEL`.
    """
    powers = list(cfg.POLARIZATION_POWERS) if cfg.POLARIZATION_POWERS else discover_polarization_powers(cfg.DATA_DIR)
    if not powers:
        powers = [cfg.POWER_LEVEL]

    # One summary per power, then the multi-power butterfly overlay when >= 2 powers.
    plot_polarization_campaign(
        replace(cfg, POLARIZATION_POWERS=powers) if cfg.POLARIZATION_POWERS is None else cfg
    )

    if not cfg.POLARIZATION_PLOT_ANGLES:
        print("\nPer-angle analysis skipped (set POLARIZATION_PLOT_ANGLES=True to draw them).")
        return

    plot_polarization_angle_trees(cfg, powers)


def plot_polarization_angle_trees(cfg: ExperimentConfig, powers: List[str]) -> None:
    """The heavy per-angle HBT trees under results/.../polarization/{power}/angles/.

    Drawn only when POLARIZATION_PLOT_ANGLES is True, whether analyzing on disk or
    right after an acquisition; the vs-angle summaries are handled elsewhere.
    """
    for power in powers:
        power_cfg = cfg.for_polarization_power(power)
        if cfg.POLARIZATION_SCAN is not None:
            angles = list(cfg.POLARIZATION_SCAN)
        else:
            angles = [angle for angle, _ in hardware.ScanLog(cfg.polarization_log_path()).recorded_labels(power=power)]
        if not angles:
            print(f"\n[{power}] No polarization angles found. Skipping.")
            continue
        power_cfg = replace(power_cfg, POLARIZATION_SCAN=angles)

        for angle, label in power_cfg.polarization_scan_points():
            step = power_cfg.for_polarization(angle)
            if not any(cfg.DATA_DIR.glob(f"*{label}*")):
                print(f"\n[{label}] No files found. Skipping.")
                continue
            print(f"\n================ {label} ({angle:g} deg) ================")
            run_analysis(step)


def open_stage(cfg: ExperimentConfig) -> Optional[RotationStage]:
    """The connected rotation stage, or None when this run moves nothing."""
    if not (cfg.ROTATION_STAGE_ENABLED or cfg.ROTATION_STAGE_DRY_RUN):
        return None
    stage = RotationStage(cfg.ROTATION_STAGE or RotationStageConfig(),
                          dry_run=cfg.ROTATION_STAGE_DRY_RUN)
    stage.connect()
    return stage


def print_summary(cfg: ExperimentConfig, runs: List[Run]) -> None:
    print("\n================ SUMMARY ================")
    print(f"data    : {cfg.DATA_DIR}")
    print(f"config  : {cfg.config_file_path().name}")
    print(f"results : {cfg.RESULTS_DIR}")
    if cfg.is_polarization_scan() or cfg.POLARIZATION_POWERS:
        print(f"pol root: {cfg.polarization_root}")
        print(f"log     : {cfg.polarization_log_path().name}")
        if runs:
            planned = len(cfg.polarization_scan_points()) * len(cfg.polarization_power_points())
            print(f"angles  : {len(runs)}/{planned} recorded")
        else:
            print("mode    : analyze only (no acquisition)")
        return

    if not runs:
        print("mode    : analyze only (no acquisition)")
        return

    for step, result in runs:
        angle = "" if step.ROTATION_ANGLE_DEG is None else f" at {step.ROTATION_ANGLE_DEG:g} deg"
        print(f"\n{step.POWER_LEVEL}{angle}")
        print(f"  chunks  : {len(result.chunk_paths)} file(s), {result.duration_s:g}s recorded")
        if result.merged_path:
            print(f"  merged  : {result.merged_path}")
        if result.snapshots_path:
            print(f"  snapshots: {result.snapshots_path}")
            print(f"  stability plots: {step.stability_dir}")


if __name__ == "__main__":
    main()
