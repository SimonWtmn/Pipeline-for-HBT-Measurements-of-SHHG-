"""The entry point: edit CONFIG below, then run this file.

    python src/run_experiment.py        # or python -m src.run_experiment

One command covers every mode, all selected from CONFIG: a single power level, an
automated power scan, a laser angle scan (the pump's polarization turned before the
crystal), an ellipticity scan (the analyzer turned after it), and the analysis-only
replays of any of them. See README section 3 for the field-by-field description.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

try:
    import hardware
    from experiment_config import ExperimentConfig, record_run, write_configuration
    from hardware import ELLStageConfig, RotationStage, RotationStageConfig
    from pkl_json_analyze import (discover_ellipticity_powers, discover_laser_angle_powers,
                                  plot_ellipticity_campaign, plot_ellipticity_overlay,
                                  plot_ellipticity_power, plot_laser_angle_campaign,
                                  plot_laser_angle_overlay, plot_laser_angle_summary,
                                  plot_stability, run_analysis)
except ImportError:
    from src import hardware
    from src.experiment_config import ExperimentConfig, record_run, write_configuration
    from src.hardware import ELLStageConfig, RotationStage, RotationStageConfig
    from src.pkl_json_analyze import (discover_ellipticity_powers, discover_laser_angle_powers,
                                      plot_ellipticity_campaign, plot_ellipticity_overlay,
                                      plot_ellipticity_power, plot_laser_angle_campaign,
                                      plot_laser_angle_overlay, plot_laser_angle_summary,
                                      plot_stability, run_analysis)


# Today, as the DATE folders are named (ddmmyyyy). Replace it by the literal string of
# the folder to reprocess when replaying a run recorded on another day.
TODAY = datetime.now().strftime("%d%m%Y")


# ==========================================
# MAIN CONFIGURATION - edit this block
# ==========================================

CONFIG = ExperimentConfig(
    # ----------------- identity -----------------
    MATERIAL="CdTe110",
    EXPERIENCE_TYPE="Ellipticity_Polarizer_Scan",
    DATE=TODAY,
    POWER_LEVEL="30mW",

    # ----------------- optics / metadata -----------------
    FREQUENCY=18.66e6,
    MODES={1: "H3T", 2: "H3R", 3: "H4T", 4: "H4R", 5: "H5T", 6: "H5R"},
    P1={"present": True, "angle_deg": "unknown"},
    P2={"present": False, "angle_deg": 0.0},
    P3={"present": False, "angle_deg": 0.0},
    HWP_ANGLE_DEG=0.0,
    COVER={"present": True, "description": "between channels"},
    FILTERS={
        "H3": {"separation": "per-channel", "filter": "700-40"},
        "H4": {"separation": "per-channel", "filter": "520-40"},
        "H5": {"separation": "per-channel", "filter": "400-20"},
    },

    # ----------------- Time Tagger -----------------
    TAGGER_SERIAL="",
    TRIGGER_LEVEL_V=0.5,
    INPUT_DELAYS_PS={},
    DEADTIMES_PS={},
    CORR_BINWIDTH_PS=300,
    CORR_N_BINS=2009,
    ACQUISITION_DURATION_S=2,
    CHUNK_DURATION_S=2,
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

    # ----------------- the pump mounts (both angle scans) -----------------
    PUMP_POWER=30,
    # PUMP_POWER_SCAN=[                       # the whole scan at each power, one run
    #     {"power": "90mW", "requested_power": 90.0},
    #     {"power": "70mW", "requested_power": 70.0},
    #     {"power": "50mW", "requested_power": 50.0},
    #     {"power": "30mW", "requested_power": 30.0},
    #     {"power": "20mW", "requested_power": 20.0},
    # ],
    PUMP_CALIBRATION=None,                  # None -> linear_polarization_lookup_latest.npz
    PUMP_CALIBRATION_DIR=None,              # None -> polarization_calibration/
    PUMP_STAGE_ENABLED=True,
    PUMP_P1=RotationStageConfig(serial_number="27270550", clockwise=True),
    PUMP_HWP=RotationStageConfig(serial_number="27270567", clockwise=True),
    PUMP_SETTLE_TIME_S=0.2,
    PUMP_STAGE_DRY_RUN=False,               # rehearse first; set False to move the mounts

    # ----------------- laser angle scan -----------------
    LASER_ANGLE_SCAN=None,
    # LASER_ANGLE_SCAN=np.arange(0.0, 360.0, 15.0),
    # LASER_ANGLE_PLOT_ANGLES=False,          # summaries only; True to also analyse each angle

    # ----------------- ellipticity scan -----------------
    # ELLIPTICITY_LASER_ANGLES=None,
    ELLIPTICITY_LASER_ANGLES=np.union1d(np.arange(0.0, 181.0, 4.0), [18.0]), 
    ELLIPTICITY_ANALYZER_ANGLES=np.arange(0.0, 181.0, 5.0),   # the inner loop
    ELLIPTICITY_ANALYZER_KIND="polarizer",
    ELLIPTICITY_ANALYZER=ELLStageConfig(port="COM4", address="3", clockwise=True),
    # ELLIPTICITY_ANALYZER_KIND="hwp",
    # ELLIPTICITY_ANALYZER=RotationStageConfig(serial_number="27264707", clockwise=True),
    ELLIPTICITY_ANALYZER_ENABLED=True,
    ELLIPTICITY_ANALYZER_DRY_RUN=False,     # rehearse first if True
    ELLIPTICITY_ANALYZER_SETTLE_TIME_S=0.2,
    ELLIPTICITY_FIT_PERIOD_DEG=None,        # None: 90 deg for the HWP, 180 for the polarizer
    ELLIPTICITY_FIXED_POLARIZER_DEG=0.0,    # metadata, HWP setup only: 0 = vertical
    ELLIPTICITY_PLOT_POINTS=False,          # True to also analyse every single point

    # ----------------- analysis -----------------
    INTEGRATION_WINDOW=8,
    INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    RUN_ANALYSIS_AFTER_ACQUIRE=False,
    ANALYZE_ONLY=True,
)


Run = Tuple[ExperimentConfig, Any]


def main(cfg: ExperimentConfig = CONFIG):
    runs: List[Run] = []

    # Analyze-only replays the files already on disk, always producing the summaries.
    if cfg.ANALYZE_ONLY:
        print(f"Configuration file: {describe_configuration(cfg)}")
        if cfg.is_ellipticity_scan():
            analyze_ellipticity_scan(cfg)
        elif cfg.is_laser_angle_scan() or cfg.PUMP_POWERS:
            analyze_laser_angle_scan(cfg)
        else:
            run_analysis(cfg)
        print_summary(cfg, runs)
        return

    runs = acquire(cfg)

    if cfg.RUN_ANALYSIS_AFTER_ACQUIRE:
        # acquire() already drew the summaries of every scan point as it finished, and
        # the comparison overlay once every power was recorded; only the optional
        # (heavy) per-point HBT trees remain.
        if cfg.is_ellipticity_scan():
            if cfg.ELLIPTICITY_PLOT_POINTS:
                powers = discover_ellipticity_powers(cfg.DATA_DIR) or [cfg.POWER_LEVEL]
                plot_ellipticity_point_trees(cfg, powers)
        elif cfg.is_laser_angle_scan():
            if cfg.LASER_ANGLE_PLOT_ANGLES:
                powers = discover_laser_angle_powers(cfg.DATA_DIR) or [cfg.POWER_LEVEL]
                plot_laser_angle_trees(cfg, powers)
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
    """Record an angle scan, every point of a power scan, or a single power."""
    if cfg.is_ellipticity_scan():
        return acquire_ellipticity_scan(cfg)
    if cfg.is_laser_angle_scan():
        return acquire_laser_angle_scan(cfg)

    try:
        from acquisition import open_tagger
    except ImportError:
        from src.acquisition import open_tagger

    points = cfg.power_scan_points() or [(cfg.POWER_LEVEL, cfg.ROTATION_ANGLE_DEG)]
    stage = open_stage(cfg)
    runs: List[Run] = []

    try:
        # One tagger connection for the whole power scan (or the single point).
        with open_tagger(cfg) as tagger:
            for number, (power, angle) in enumerate(points, start=1):
                if len(points) > 1:
                    print(f"\n===== power point {number}/{len(points)}: {power} =====")
                step = cfg.for_power(power, angle)

                if stage is not None and angle is not None:
                    stage.move_to_angle_deg(angle)

                try:
                    runs.append((step, acquire_one(step, tagger=tagger)))
                except Exception as exc:
                    record_run(step, status="failed")
                    print(f"Power point {power} failed: {exc}")
                    if cfg.POWER_SCAN_STOP_ON_ERROR:
                        raise
    finally:
        if stage is not None:
            stage.disconnect()

    return runs


def acquire_one(cfg: ExperimentConfig, log_run: bool = True, tagger: Any = None) -> Any:
    if log_run:
        print(f"Configuration file: {record_run(cfg)}")

    try:
        from acquisition import run_acquisition
    except ImportError:
        from src.acquisition import run_acquisition

    result = run_acquisition(cfg, session=tagger)
    if log_run:
        record_run(cfg, result)

    if cfg.STABILITY_ENABLED and result.snapshots:
        plot_stability(result.snapshots, cfg)
    return result


def acquire_laser_angle_scan(cfg: ExperimentConfig) -> List[Run]:
    """Record a laser angle scan at one power, or at each power of a multi-power run.

    A multi-power run (PUMP_POWER_SCAN) opens the two pump mounts and the Time Tagger
    once, then drives a full angle scan at every power in turn. `with` releases the
    mounts and the tagger even when a preflight or configuration-file write raises
    before the first acquisition.
    """
    try:
        from acquisition import open_tagger
    except ImportError:
        from src.acquisition import open_tagger

    power_points = cfg.pump_power_points()
    log = hardware.LaserAngleLog(cfg.laser_angle_log_path())
    runs: List[Run] = []
    series_by_power: dict = {}

    with open_pump(cfg) as pump, open_tagger(cfg) as tagger:
        for number, (label, requested) in enumerate(power_points, start=1):
            if len(power_points) > 1:
                print(f"\n########## power {number}/{len(power_points)}: "
                      f"{label} ({requested:g} {pump.lookup.unit}) ##########")
            power_cfg = cfg.for_pump_scan_power(label, requested)
            runs.extend(acquire_laser_angle_power(power_cfg, pump, log, tagger=tagger))

            # Draw this power's vs-angle summary the moment its scan finishes, so a long
            # multi-power campaign shows results as it goes - and a completed power keeps
            # its plots even if a later power is interrupted.
            if cfg.RUN_ANALYSIS_AFTER_ACQUIRE:
                series = plot_laser_angle_summary(power_cfg)
                if series is not None:
                    series_by_power[label] = series

    # Every power on disk: overlay them for the comparison (needs at least two).
    if cfg.RUN_ANALYSIS_AFTER_ACQUIRE and len(series_by_power) > 1:
        plot_laser_angle_overlay(series_by_power, cfg.laser_angle_overlay_dir)

    return runs


def acquire_laser_angle_power(cfg: ExperimentConfig, pump: "hardware.PumpController",
                              log: "hardware.LaserAngleLog",
                              tagger: Any = None) -> List[Run]:
    """One full angle scan at a single power, on an already-open pump and tagger."""
    points = cfg.laser_angle_scan_points()
    runs: List[Run] = []

    try:
        print(f"Configuration file: {write_configuration(cfg)}")
        plan = hardware.preflight(pump.lookup, [angle for angle, _ in points], cfg.PUMP_POWER)

        for number, ((angle, label), setting) in enumerate(zip(points, plan), start=1):
            print(f"\n===== laser angle {number}/{len(points)}: {angle:g} deg ({label}) =====")
            if not setting.reachable:
                log.record(label, setting, status="unreachable")
                print(f"Skipped: {setting.describe()}")
                continue

            setting = pump.set_polarization(angle, cfg.PUMP_POWER)
            step = cfg.for_laser_angle(angle)
            log.record(label, setting, status="running")

            try:
                result = acquire_one(step, log_run=False, tagger=tagger)
                runs.append((step, result))
                log.record(label, setting, status="complete", result=result)
            except Exception as exc:
                log.record(label, setting, status="failed", note=str(exc))
                print(f"Laser angle {angle:g} deg failed: {exc}")
                if cfg.POWER_SCAN_STOP_ON_ERROR:
                    raise
    finally:
        print(f"Per-angle log: {log.write()}")

    return runs


def acquire_ellipticity_scan(cfg: ExperimentConfig) -> List[Run]:
    """Record an analyzer sweep at every laser angle, at one power or at several.

    The pump mounts, the analyzer plate, and the Time Tagger are opened once for the
    whole run: three mounts and one tagger stay connected while the nested loops
    (power, then laser angle, then analyzer angle) drive them. `with` releases all
    four whatever raises.
    """
    try:
        from acquisition import open_tagger
    except ImportError:
        from src.acquisition import open_tagger

    power_points = cfg.pump_power_points()
    log = hardware.EllipticityLog(cfg.ellipticity_log_path(), cfg.ELLIPTICITY_ANALYZER_KIND)
    runs: List[Run] = []
    by_power: dict = {}

    with open_pump(cfg) as pump, open_analyzer(cfg) as analyzer, open_tagger(cfg) as tagger:
        for number, (label, requested) in enumerate(power_points, start=1):
            if len(power_points) > 1:
                print(f"\n########## power {number}/{len(power_points)}: "
                      f"{label} ({requested:g} {pump.lookup.unit}) ##########")
            power_cfg = cfg.for_pump_scan_power(label, requested)
            runs.extend(acquire_ellipticity_power(power_cfg, pump, analyzer, log, tagger=tagger))

            if cfg.RUN_ANALYSIS_AFTER_ACQUIRE:
                campaign = plot_ellipticity_power(power_cfg, label)
                if campaign is not None:
                    by_power[label] = campaign

    if cfg.RUN_ANALYSIS_AFTER_ACQUIRE and len(by_power) > 1:
        plot_ellipticity_overlay(by_power, cfg.ellipticity_overlay_dir)

    return runs


def acquire_ellipticity_power(cfg: ExperimentConfig, pump: "hardware.PumpController",
                              analyzer: "hardware.AnalyzerController",
                              log: "hardware.EllipticityLog",
                              tagger: Any = None) -> List[Run]:
    """Every (laser angle, analyzer angle) pair of one power, in run order.

    The pump is set once per laser angle and the analyzer turns through its whole sweep
    there: the pump mounts move a few times, the analyzer many, which is both the
    quickest order and the one that keeps a sweep internally consistent. The Time Tagger
    stays connected for the whole power (opened by the caller).
    """
    laser_angles = cfg.ellipticity_laser_angles()
    runs: List[Run] = []

    try:
        print(f"Configuration file: {write_configuration(cfg)}")
        plan = hardware.preflight(pump.lookup, laser_angles, cfg.PUMP_POWER)

        for number, (laser_angle, setting) in enumerate(zip(laser_angles, plan), start=1):
            print(f"\n########## laser angle {number}/{len(laser_angles)}: "
                  f"{laser_angle:g} deg ##########")
            points = cfg.ellipticity_analyzer_points(laser_angle)
            if not setting.reachable:
                # One row per skipped point, so the log says what was not recorded.
                for analyzer_angle, label in points:
                    log.record(label, laser_angle, analyzer_angle, setting, status="unreachable")
                print(f"Skipped: {setting.describe()}")
                continue

            setting = pump.set_polarization(laser_angle, cfg.PUMP_POWER)

            # An Elliptec analyzer must be re-homed before each laser angle or the
            # following sweep does not move reliably; the Kinesis HWP does not need it
            # (and homing it every angle would only slow the scan down).
            if analyzer.is_ell_stage:
                analyzer.home()

            for index, (analyzer_angle, label) in enumerate(points, start=1):
                print(f"\n===== analyzer {index}/{len(points)}: "
                      f"{analyzer_angle:g} deg ({label}) =====")
                analyzer.set_angle(analyzer_angle)
                step = cfg.for_ellipticity_point(laser_angle, analyzer_angle)
                log.record(label, laser_angle, analyzer_angle, setting, status="running")

                try:
                    result = acquire_one(step, log_run=False, tagger=tagger)
                    runs.append((step, result))
                    log.record(label, laser_angle, analyzer_angle, setting,
                               status="complete", result=result)
                except Exception as exc:
                    log.record(label, laser_angle, analyzer_angle, setting,
                               status="failed", note=str(exc))
                    print(f"Analyzer {analyzer_angle:g} deg at laser {laser_angle:g} deg "
                          f"failed: {exc}")
                    if cfg.POWER_SCAN_STOP_ON_ERROR:
                        raise
    finally:
        print(f"Per-point log: {log.write()}")

    return runs


def open_pump(cfg: ExperimentConfig) -> hardware.PumpController:
    """The pump's polarizer pair, not connected yet: use it as a context manager."""
    path = cfg.PUMP_CALIBRATION or hardware.find_calibration(cfg.PUMP_CALIBRATION_DIR)
    lookup = hardware.load_lookup(path)
    dry_run = cfg.PUMP_STAGE_DRY_RUN or not cfg.PUMP_STAGE_ENABLED
    return hardware.PumpController(
        lookup,
        p1=cfg.PUMP_P1 or RotationStageConfig(),
        hwp=cfg.PUMP_HWP or RotationStageConfig(),
        dry_run=dry_run,
        settle_time_s=cfg.PUMP_SETTLE_TIME_S,
    )


def open_analyzer(cfg: ExperimentConfig) -> hardware.AnalyzerController:
    """The analyzer after the crystal, not connected yet: use as a context manager.

    Whichever mount ELLIPTICITY_ANALYZER describes - a Kinesis controller turning the
    half-wave plate, or an Elliptec one turning the polarizer itself - is picked from
    the configuration's own type.
    """
    dry_run = cfg.ELLIPTICITY_ANALYZER_DRY_RUN or not cfg.ELLIPTICITY_ANALYZER_ENABLED
    return hardware.AnalyzerController(
        cfg.ELLIPTICITY_ANALYZER or RotationStageConfig(),
        dry_run=dry_run,
        settle_time_s=cfg.ELLIPTICITY_ANALYZER_SETTLE_TIME_S,
        element=cfg.ellipticity_analyzer_element,
    )


def analyze_laser_angle_scan(cfg: ExperimentConfig) -> None:
    """Write the vs-angle summaries first, then the per-angle trees only if asked.

    The circle/butterfly plots versus angle are the point of a laser angle scan, so they
    are drawn first and always. The per-angle analysis trees are heavy and rarely the
    first thing looked at, so they are only produced when LASER_ANGLE_PLOT_ANGLES is
    True. Powers come from `PUMP_POWERS` when set (analyze-only multi-power), else from
    the files already in DATA_DIR, else from the single `POWER_LEVEL`.
    """
    powers = list(cfg.PUMP_POWERS) if cfg.PUMP_POWERS else discover_laser_angle_powers(cfg.DATA_DIR)
    if not powers:
        powers = [cfg.POWER_LEVEL]

    # One summary per power, then the multi-power butterfly overlay when >= 2 powers.
    plot_laser_angle_campaign(
        replace(cfg, PUMP_POWERS=powers) if cfg.PUMP_POWERS is None else cfg
    )

    if not cfg.LASER_ANGLE_PLOT_ANGLES:
        print("\nPer-angle analysis skipped (set LASER_ANGLE_PLOT_ANGLES=True to draw them).")
        return

    plot_laser_angle_trees(cfg, powers)


def analyze_ellipticity_scan(cfg: ExperimentConfig) -> None:
    """Fit every analyzer sweep on disk, then the ellipticity versus laser angle."""
    powers = list(cfg.PUMP_POWERS) if cfg.PUMP_POWERS else discover_ellipticity_powers(cfg.DATA_DIR)
    if not powers:
        powers = [cfg.POWER_LEVEL]

    plot_ellipticity_campaign(
        replace(cfg, PUMP_POWERS=powers) if cfg.PUMP_POWERS is None else cfg
    )

    if not cfg.ELLIPTICITY_PLOT_POINTS:
        print("\nPer-point analysis skipped (set ELLIPTICITY_PLOT_POINTS=True to draw it).")
        return

    plot_ellipticity_point_trees(cfg, powers)


def plot_laser_angle_trees(cfg: ExperimentConfig, powers: List[str]) -> None:
    """The heavy per-angle HBT trees under results/.../laser_angle/{power}/angles/.

    Drawn only when LASER_ANGLE_PLOT_ANGLES is True, whether analyzing on disk or right
    after an acquisition; the vs-angle summaries are handled elsewhere.
    """
    for power in powers:
        power_cfg = cfg.for_pump_power(power)
        if cfg.LASER_ANGLE_SCAN is not None:
            angles = list(cfg.LASER_ANGLE_SCAN)
        else:
            angles = [angle for angle, _ in
                      hardware.LaserAngleLog(cfg.laser_angle_log_path()).recorded_labels(power=power)]
        if not angles:
            print(f"\n[{power}] No laser angles found. Skipping.")
            continue
        power_cfg = replace(power_cfg, LASER_ANGLE_SCAN=angles)

        for angle, label in power_cfg.laser_angle_scan_points():
            step = power_cfg.for_laser_angle(angle)
            if not any(cfg.DATA_DIR.glob(f"{label}_num*")):
                print(f"\n[{label}] No files found. Skipping.")
                continue
            print(f"\n================ {label} ({angle:g} deg) ================")
            run_analysis(step)


def plot_ellipticity_point_trees(cfg: ExperimentConfig, powers: List[str]) -> None:
    """The heavy per-point HBT trees of an ellipticity scan, one per analyzer angle.

    Drawn only when ELLIPTICITY_PLOT_POINTS is True: there is one tree per (laser angle,
    analyzer angle) pair, so a 3 x 12 scan produces 36 of them.
    """
    log = hardware.EllipticityLog(cfg.ellipticity_log_path())
    for power in powers:
        power_cfg = cfg.for_pump_power(power)
        recorded = log.recorded_points(power=power)
        points = ([(laser, analyzer) for laser, analyzer, _label in recorded] or
                  [(laser, analyzer)
                   for laser in cfg.ellipticity_laser_angles()
                   for analyzer in cfg.ELLIPTICITY_ANALYZER_ANGLES])
        if not points:
            print(f"\n[{power}] No ellipticity points found. Skipping.")
            continue

        for laser_angle, analyzer_angle in points:
            step = power_cfg.for_ellipticity_point(laser_angle, analyzer_angle)
            if not any(cfg.DATA_DIR.glob(f"{step.POWER_LEVEL}_num*")):
                print(f"\n[{step.POWER_LEVEL}] No files found. Skipping.")
                continue
            print(f"\n================ {step.POWER_LEVEL} "
                  f"(laser {laser_angle:g} deg, analyzer {analyzer_angle:g} deg) ================")
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
    if cfg.is_ellipticity_scan():
        print(f"ell root: {cfg.ellipticity_root}")
        print(f"log     : {cfg.ellipticity_log_path().name}")
        print(f"analyzer: {cfg.ellipticity_analyzer_note}, "
              f"fit period {cfg.ELLIPTICITY_FIT_PERIOD_DEG:g} deg")
        if runs:
            planned = len(cfg.ellipticity_scan_points()) * len(cfg.pump_power_points())
            print(f"points  : {len(runs)}/{planned} recorded "
                  f"({len(cfg.ellipticity_laser_angles())} laser angles x "
                  f"{len(cfg.ELLIPTICITY_ANALYZER_ANGLES)} analyzer angles)")
        else:
            print("mode    : analyze only (no acquisition)")
        return

    if cfg.is_laser_angle_scan() or cfg.PUMP_POWERS:
        print(f"las root: {cfg.laser_angle_root}")
        print(f"log     : {cfg.laser_angle_log_path().name}")
        if runs:
            planned = len(cfg.laser_angle_scan_points()) * len(cfg.pump_power_points())
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
