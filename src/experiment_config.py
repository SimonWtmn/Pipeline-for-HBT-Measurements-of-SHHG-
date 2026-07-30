"""The configuration of a run, and how it is written down.

`ExperimentConfig` is the single source of truth: every knob the user edits, plus the
paths, chunk names and correlation pairs derived from them. The second half of the
file writes that configuration to `experiment_config.txt` next to the data, so the
metadata on disk cannot drift from the parameters the run actually used.
"""

from __future__ import annotations

import configparser
import itertools
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from hardware import RotationStageConfig, wrap_360
except ImportError:  # pragma: no cover - import style depends on the entry point
    from src.hardware import RotationStageConfig, wrap_360

REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# PART 1 - the configuration object
# ============================================================================

@dataclass
class ExperimentConfig:

    # ---------- Identity / paths ----------
    MATERIAL: str = "ZnO100"
    EXPERIENCE_TYPE: str = "David_Setup"
    DATE: str = "11142023"
    POWER_LEVEL: str = "65mW"
    POWER_LEVELS: Optional[List[str]] = None
    FILE_PREFIX: str = ""
    CHUNK_INDEX_DIGITS: int = 1
    DATA_DIR: Optional[Path] = None
    RESULTS_DIR: Optional[Path] = None

    # ---------- Optics / metadata (written to experiment_config.txt) ----------
    FREQUENCY: float = 18.66e6  # laser repetition rate (Hz)
    MODES: Dict[int, str] = field(default_factory=lambda: {1: "H3T", 5: "H3R", 10: "H5T", 14: "H5R"})
    P1: Dict[str, Any] = field(default_factory=lambda: {"present": False, "angle_deg": 0.0})
    P2: Dict[str, Any] = field(default_factory=lambda: {"present": False, "angle_deg": 0.0})
    P3: Dict[str, Any] = field(default_factory=lambda: {"present": False, "angle_deg": 0.0})
    HWP_ANGLE_DEG: float = 0.0
    COVER: Dict[str, Any] = field(default_factory=lambda: {"present": False, "description": "between harmonics"})
    FILTERS: Dict[str, Any] = field(
        default_factory=lambda: {
            "H3": {"separation": "per-channel", "filter": "700-40"},
            "H5": {"separation": "per-channel", "filter": "400-20"},
        }
    )

    # ---------- Hardware / Time Tagger ----------
    REGISTERED_CHANNELS: Optional[List[int]] = None
    TRIGGER_LEVEL_V: float = 0.5
    INPUT_DELAYS_PS: Dict[int, int] = field(default_factory=dict)
    DEADTIMES_PS: Dict[int, int] = field(default_factory=dict)
    TAGGER_SERIAL: str = ""
    CORR_BINWIDTH_PS: int = 300
    CORR_N_BINS: int = 2009
    ACQUISITION_DURATION_S: float = 600.0
    CHUNK_DURATION_S: float = 60.0
    SAVE_MERGED: bool = True
    EXPORT_FORMAT: str = "pkl"
    SAVE_RAW_TTBIN: bool = False

    # ---------- Stability (live only) ----------
    STABILITY_ENABLED: bool = False
    STABILITY_SNAPSHOT_INTERVAL_S: float = 60.0
    STABILITY_INTEGRATION_WINDOW_NS: float = 4.0
    STABILITY_PEAKS: int = 15
    STABILITY_ROLLING_WINDOW_SNAPS: int = 10

    # ---------- Rotation stage / power scan ----------
    # POWER_SCAN drives one acquisition per entry, moving the stage to "angle_deg"
    # first. The power strings are LABELS used in file names and plots: nothing
    # computes them from the angle, the (power, angle) pairs are calibrated by hand.
    #   POWER_SCAN = [{"power": "25mW", "angle_deg": 8.0},
    #                 {"power": "35mW", "angle_deg": 14.2}]
    POWER_SCAN: Optional[List[Dict[str, Any]]] = None
    POWER_SCAN_STOP_ON_ERROR: bool = True
    ROTATION_STAGE_ENABLED: bool = False
    ROTATION_STAGE_DRY_RUN: bool = False  # log the moves, touch no hardware
    ROTATION_STAGE: Optional[RotationStageConfig] = None
    ROTATION_ANGLE_DEG: Optional[float] = None  # angle of THIS run; set per scan step

    # ---------- Polarization scan (optional) ----------
    # One acquisition per angle, at a single power. The HWP angle that delivers
    # POLARIZATION_POWER at each angle comes from the lookup table measured by the
    # lab's calibration script; angles the table cannot reach are skipped.
    #   POLARIZATION_SCAN = np.arange(0.0, 360.0, 10.0)
    POLARIZATION_SCAN: Optional[Sequence[float]] = None
    POLARIZATION_POWER: float = 0.0  # in the calibration's unit (mW or Hz), not a label
    # Several power labels already recorded as polarization scans in the same DATE
    # folder (e.g. ["25mW", "45mW", "65mW"]). Used by analyze-only to rebuild each
    # per-power summary and the multi-power butterfly overlay.
    POLARIZATION_POWERS: Optional[List[str]] = None
    # A calibration is reused run after run: recording one takes hours, and the table
    # only goes stale when the beam path changes. None picks
    # polarization_calibration/linear_polarization_lookup_latest.npz.
    POLARIZATION_CALIBRATION: Optional[Path] = None
    POLARIZATION_CALIBRATION_DIR: Optional[Path] = None
    POLARIZATION_STAGE_ENABLED: bool = False
    POLARIZATION_STAGE_DRY_RUN: bool = False
    POLARIZATION_P1: Optional[RotationStageConfig] = None
    POLARIZATION_HWP: Optional[RotationStageConfig] = None
    POLARIZATION_SETTLE_TIME_S: float = 0.2
    POLARIZATION_ANGLE_DEG: Optional[float] = None  # angle of THIS run; set per scan step
    # Base power label of THIS acquisition (kept when POWER_LEVEL becomes the
    # per-angle file label "50mW_pol000p0").
    POLARIZATION_BASE_POWER: Optional[str] = None
    # Acquire a full angle scan at each of several powers in one run. Each entry pairs
    # a file/label with the power (calibration unit) every angle is set to:
    #   POLARIZATION_POWER_SCAN = [{"power": "50mW", "requested_power": 50.0},
    #                              {"power": "60mW", "requested_power": 60.0}]
    # Results land under results/.../polarization/{power}/ exactly like a single power.
    POLARIZATION_POWER_SCAN: Optional[List[Dict[str, Any]]] = None
    # Per-angle analysis trees (results/.../polarization/{power}/angles/) are heavy and
    # rarely the first thing you look at. Off by default: the vs-angle summaries are
    # always drawn; set this True to also analyse every single angle.
    POLARIZATION_PLOT_ANGLES: bool = False

    # ---------- Analysis ----------
    INTEGRATION_WINDOW: int = 8
    INTEGRATION_WINDOWS_SWEEP: Sequence[int] = field(default_factory=lambda: np.arange(1, 31, 1))
    RUN_ANALYSIS_AFTER_ACQUIRE: bool = True
    ANALYZE_ONLY: bool = False

    def __post_init__(self) -> None:
        if self.REGISTERED_CHANNELS is None:
            self.REGISTERED_CHANNELS = sorted(self.MODES)
        self._validate_power_scan()
        self._validate_polarization_scan()
        if self.POLARIZATION_POWERS:
            _reject_nested_labels(list(self.POLARIZATION_POWERS), "POLARIZATION_POWERS")
        if self.POWER_SCAN:
            # The scan is what was actually recorded, so it defines what to analyse.
            self.POWER_LEVELS = [power for power, _angle in self.power_scan_points()]
        elif self.POLARIZATION_SCAN is not None:
            self.POWER_LEVELS = [label for _angle, label in self.polarization_scan_points()]
        if self.POWER_LEVELS is None:
            self.POWER_LEVELS = [self.POWER_LEVEL]
        if self.POLARIZATION_CALIBRATION_DIR is None:
            self.POLARIZATION_CALIBRATION_DIR = REPO_ROOT / "polarization_calibration"
        if self.DATA_DIR is None:
            self.DATA_DIR = REPO_ROOT / "data" / self.MATERIAL / self.EXPERIENCE_TYPE / self.DATE
        if self.RESULTS_DIR is None:
            self.RESULTS_DIR = REPO_ROOT / "results" / self.MATERIAL / self.EXPERIENCE_TYPE / self.DATE
        self.DATA_DIR = Path(self.DATA_DIR)
        self.RESULTS_DIR = Path(self.RESULTS_DIR)

        if self.EXPORT_FORMAT not in ("pkl", "json"):
            raise ValueError(f"EXPORT_FORMAT must be 'pkl' or 'json', got {self.EXPORT_FORMAT!r}")
        if self.CHUNK_DURATION_S <= 0 or self.ACQUISITION_DURATION_S <= 0:
            raise ValueError("ACQUISITION_DURATION_S and CHUNK_DURATION_S must be > 0")

        self._validate_stage()

    def _validate_power_scan(self) -> None:
        if not self.POWER_SCAN:
            self.POWER_SCAN = None
            return

        shape = ('POWER_SCAN entries must look like {"power": "25mW", "angle_deg": 8.0}')
        seen = set()
        for entry in self.POWER_SCAN:
            if not isinstance(entry, dict) or "power" not in entry or "angle_deg" not in entry:
                raise ValueError(f"{shape}, got {entry!r}")
            power = entry["power"]
            if not isinstance(power, str) or not power.strip():
                raise ValueError(f"{shape}: 'power' must be a non-empty label, got {power!r}")
            try:
                float(entry["angle_deg"])
            except (TypeError, ValueError):
                raise ValueError(f"{shape}: 'angle_deg' must be a number, got {entry['angle_deg']!r}") from None
            if power in seen:
                # Two points sharing a label would write over each other's chunk files.
                raise ValueError(f"POWER_SCAN lists the power label {power!r} twice")
            seen.add(power)

        _reject_nested_labels(sorted(seen), "POWER_SCAN")

    def _validate_polarization_scan(self) -> None:
        if self.POLARIZATION_SCAN is None:
            return

        angles = [float(angle) for angle in self.POLARIZATION_SCAN]
        if not angles:
            self.POLARIZATION_SCAN = None
            return
        if self.POWER_SCAN:
            raise ValueError(
                "POWER_SCAN and POLARIZATION_SCAN are both set: that would multiply into "
                "one acquisition per (power, angle) pair. Run one scan at a time."
            )

        # Labels resolve 0.1 deg, so two angles closer than that would share their
        # files. Checking the labels catches both a repeated and a too-close angle.
        labels = [angle_tag(angle) for angle in angles]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                f"POLARIZATION_SCAN maps several angles onto the same file label {duplicates}: "
                "angles must differ by at least 0.1 deg once wrapped into [0, 360)."
            )
        self.POLARIZATION_SCAN = angles

        if self.ANALYZE_ONLY:
            return
        if self.POLARIZATION_POWER_SCAN:
            self._validate_polarization_power_scan()
        elif self.POLARIZATION_POWER <= 0:
            raise ValueError(
                "POLARIZATION_SCAN needs POLARIZATION_POWER: the power every angle is set "
                "to, in the unit of the calibration table (mW for a power-meter table). "
                "For several powers in one run, use POLARIZATION_POWER_SCAN instead."
            )
        if self.POLARIZATION_STAGE_DRY_RUN:
            return
        if not self.POLARIZATION_STAGE_ENABLED:
            raise ValueError(
                "POLARIZATION_SCAN is set but POLARIZATION_STAGE_ENABLED is False: every "
                "angle would be recorded at whatever polarization the mounts happen to "
                "hold. Enable the mounts, or set POLARIZATION_STAGE_DRY_RUN=True."
            )
        for name, stage in (("POLARIZATION_P1", self.POLARIZATION_P1),
                            ("POLARIZATION_HWP", self.POLARIZATION_HWP)):
            if stage is None or not stage.serial_number:
                raise ValueError(
                    f"POLARIZATION_STAGE_ENABLED is True but {name} has no serial number: "
                    f"pass {name}=RotationStageConfig(serial_number=\"...\")."
                )

    def _validate_polarization_power_scan(self) -> None:
        shape = ('POLARIZATION_POWER_SCAN entries must look like '
                 '{"power": "50mW", "requested_power": 50.0}')
        seen = set()
        for entry in self.POLARIZATION_POWER_SCAN:
            if not isinstance(entry, dict) or "power" not in entry or "requested_power" not in entry:
                raise ValueError(f"{shape}, got {entry!r}")
            label = entry["power"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{shape}: 'power' must be a non-empty label, got {label!r}")
            try:
                value = float(entry["requested_power"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{shape}: 'requested_power' must be a number, got {entry['requested_power']!r}"
                ) from None
            if value <= 0:
                raise ValueError(f"{shape}: 'requested_power' must be > 0, got {value!r}")
            if label in seen:
                # Two powers sharing a label would write over each other's files.
                raise ValueError(f"POLARIZATION_POWER_SCAN lists the power label {label!r} twice")
            seen.add(label)
        _reject_nested_labels(sorted(seen), "POLARIZATION_POWER_SCAN")

    def _validate_stage(self) -> None:
        if self.ANALYZE_ONLY or self.ROTATION_STAGE_DRY_RUN:
            return

        if self.POWER_SCAN and not self.ROTATION_STAGE_ENABLED:
            raise ValueError(
                "POWER_SCAN is set but ROTATION_STAGE_ENABLED is False: the scan would "
                "record every power level at the same stage angle. Enable the stage, or "
                "set ROTATION_STAGE_DRY_RUN=True to rehearse without hardware."
            )
        if self.ROTATION_STAGE_ENABLED and not (self.ROTATION_STAGE and self.ROTATION_STAGE.serial_number):
            raise ValueError(
                "ROTATION_STAGE_ENABLED is True but no serial number is configured: pass "
                "ROTATION_STAGE=RotationStageConfig(serial_number=\"...\"), or set "
                "ROTATION_STAGE_DRY_RUN=True."
            )


    # ---------- Derived paths ----------

    @property
    def merged_dir(self) -> Path:
        return self.DATA_DIR / "merged"

    @property
    def power_results_dir(self) -> Path:
        return self.RESULTS_DIR / self.POWER_LEVEL

    @property
    def stability_dir(self) -> Path:
        return self.power_results_dir / "stability"

    def chunk_path(self, index: int) -> Path:
        counter = f"{index:0{self.CHUNK_INDEX_DIGITS}d}"
        return self.DATA_DIR / f"{self.FILE_PREFIX}{self.POWER_LEVEL}_num{counter}.{self.EXPORT_FORMAT}"

    def merged_path(self) -> Path:
        return self.merged_dir / f"{self.FILE_PREFIX}{self.POWER_LEVEL}_merged.{self.EXPORT_FORMAT}"

    def config_file_path(self) -> Path:
        return self.DATA_DIR / "experiment_config.txt"


    # ---------- Acquisition plan ----------

    @property
    def n_chunks(self) -> int:
        return max(1, int(np.ceil(self.ACQUISITION_DURATION_S / self.CHUNK_DURATION_S)))

    def chunk_durations(self) -> List[float]:
        full, remainder = divmod(self.ACQUISITION_DURATION_S, self.CHUNK_DURATION_S)
        durations = [self.CHUNK_DURATION_S] * int(full)
        if remainder > 1e-9:
            durations.append(remainder)
        return durations or [self.ACQUISITION_DURATION_S]

    def active_harmonics(self) -> Dict[str, Tuple[int, ...]]:
        groups: Dict[str, List[int]] = {}
        for ch in self.REGISTERED_CHANNELS:
            label = self.MODES.get(ch)
            if label is None:
                continue
            groups.setdefault(harmonic_of(label), []).append(ch)
        return {name: tuple(sorted(chs)) for name, chs in groups.items() if len(chs) >= 2}

    def active_channels(self) -> List[int]:
        harmonics = self.active_harmonics()
        if not harmonics:
            return sorted(self.REGISTERED_CHANNELS)
        return sorted(ch for chs in harmonics.values() for ch in chs)

    def correlation_pairs(self) -> List[Tuple[int, int]]:
        return list(itertools.combinations(self.active_channels(), 2))

    # ---------- Power scan ----------

    def is_power_scan(self) -> bool:
        return bool(self.POWER_SCAN)

    def power_scan_points(self) -> List[Tuple[str, float]]:
        """[(power label, angle_deg)] in the configured order; empty without a scan."""
        if not self.POWER_SCAN:
            return []
        return [(str(entry["power"]), float(entry["angle_deg"])) for entry in self.POWER_SCAN]

    def for_power(self, power_level: str, angle_deg: Optional[float] = None) -> "ExperimentConfig":
        """A copy of this config describing one acquisition of a scan."""
        return replace(self, POWER_LEVEL=power_level, ROTATION_ANGLE_DEG=angle_deg)

    # ---------- Polarization scan ----------

    def is_polarization_scan(self) -> bool:
        return self.POLARIZATION_SCAN is not None

    def polarization_scan_points(self) -> List[Tuple[float, str]]:
        """[(angle, file label)] in the configured order; empty without a scan."""
        if self.POLARIZATION_SCAN is None:
            return []
        return [(float(angle), f"{self.POWER_LEVEL}_{angle_tag(angle)}")
                for angle in self.POLARIZATION_SCAN]

    def for_polarization(self, angle_deg: float) -> "ExperimentConfig":
        """A copy of this config describing the acquisition at one angle.

        The angle's file label becomes POWER_LEVEL (so chunk names stay
        `{power}_polXXX_numN`), while the analysis tree lands under
        `results/.../polarization/{power}/angles/`.
        """
        base = self.POLARIZATION_BASE_POWER or self.POWER_LEVEL
        for angle, label in self.polarization_scan_points():
            if angle_tag(angle) == angle_tag(angle_deg):
                # POWER_LEVELS=None so it is re-derived as [label]: without that, the
                # step would carry every angle of the scan and analyse all of them.
                return replace(
                    self,
                    POWER_LEVEL=label,
                    POWER_LEVELS=None,
                    POLARIZATION_SCAN=None,
                    POLARIZATION_ANGLE_DEG=angle,
                    POLARIZATION_BASE_POWER=base,
                    RESULTS_DIR=self.RESULTS_DIR / "polarization" / base / "angles",
                )
        raise ValueError(f"{angle_deg} deg is not one of the POLARIZATION_SCAN angles")

    def polarization_power_points(self) -> List[Tuple[str, float]]:
        """[(power label, requested power)] for the acquisition.

        A multi-power run lists them in POLARIZATION_POWER_SCAN; otherwise the single
        (POWER_LEVEL, POLARIZATION_POWER) pair is the whole scan.
        """
        if self.POLARIZATION_POWER_SCAN:
            return [(str(entry["power"]), float(entry["requested_power"]))
                    for entry in self.POLARIZATION_POWER_SCAN]
        return [(self.POWER_LEVEL, float(self.POLARIZATION_POWER))]

    def for_polarization_scan_power(self, power: str, requested_power: float) -> "ExperimentConfig":
        """A copy that records the whole angle scan at one power of a multi-power run."""
        return replace(
            self,
            POWER_LEVEL=power,
            POWER_LEVELS=None,
            POLARIZATION_POWER=float(requested_power),
            POLARIZATION_BASE_POWER=power,
            POLARIZATION_POWER_SCAN=None,
        )

    def for_polarization_power(self, power: str) -> "ExperimentConfig":
        """A copy aimed at one power of a multi-power polarization campaign."""
        return replace(
            self,
            POWER_LEVEL=power,
            POWER_LEVELS=None,
            POLARIZATION_BASE_POWER=power,
            POLARIZATION_POWERS=None,
        )

    def polarization_log_path(self) -> Path:
        return self.DATA_DIR / "polarization_scan.csv"

    @property
    def polarization_root(self) -> Path:
        return self.RESULTS_DIR / "polarization"

    def polarization_summary_dir(self, power: Optional[str] = None) -> Path:
        """`results/.../polarization/{power}/summary/` — the three circle-plot grids."""
        return self.polarization_root / (power or self.POLARIZATION_BASE_POWER or self.POWER_LEVEL) / "summary"

    @property
    def polarization_overlay_dir(self) -> Path:
        """`results/.../polarization/overlay/` — butterflies with every power overlaid."""
        return self.polarization_root / "overlay"

    # ---------- Metadata ----------

    def file_parameters(self, chunk: Any, duration_s: float) -> Dict[str, Any]:
        params = {"power_level": self.POWER_LEVEL, "chunk": chunk, "duration_s": duration_s}
        if self.ROTATION_ANGLE_DEG is not None:
            params["rotation_angle_deg"] = self.ROTATION_ANGLE_DEG
        if self.POLARIZATION_ANGLE_DEG is not None:
            params["polarization_angle_deg"] = self.POLARIZATION_ANGLE_DEG
            params["requested_power"] = self.POLARIZATION_POWER
        return params


# ============================================================================
# PART 2 - the folder configuration file
# ============================================================================
#
# `data/{MATERIAL}/{TYPE}/{DATE}/experiment_config.txt` holds everything that does not
# change from one data file to the next, so the data files themselves carry no metadata.
# One file per folder, plus one `[run <power>]` section per acquisition.
#
# INI so it stays readable by hand and re-parseable by `configparser`, which is how a
# second run appends itself without losing the previous ones. A polarization scan logs
# its angles to a CSV of its own (`hardware.ScanLog`) rather than dozens of sections.

CONFIG_FILENAME = "experiment_config.txt"

Section = Dict[str, str]

# A scan plan describes a procedure, not the setup: extending it or rehearsing it as
# a dry run is legitimate and must not be reported as a configuration change.
DRIFT_EXEMPT_SECTIONS = {"power_scan", "polarization_scan"}

HEADER = """# HBT experiment configuration
# Written by src/run_experiment.py - shared by every power level of this folder.
# The data files themselves carry no metadata: everything about the setup is here.
"""


def path_for(cfg: ExperimentConfig) -> Path:
    return cfg.DATA_DIR / CONFIG_FILENAME


def write_configuration(cfg: ExperimentConfig) -> Path:
    """Write the shared configuration, keeping the runs already listed."""
    return _write(cfg, _existing_runs(path_for(cfg)))


def record_run(cfg: ExperimentConfig, result: Any = None, status: Optional[str] = None) -> Path:
    """Write the shared configuration and add (or update) this power's run section.

    Called once before the acquisition with `result=None` so the file exists even
    if the run is interrupted, then again afterwards with the AcquisitionResult to
    replace the planned numbers by what was actually recorded. `status="failed"`
    marks a power point of a scan that raised instead of finishing.
    """
    path = path_for(cfg)
    runs = _existing_runs(path)
    runs[cfg.POWER_LEVEL] = _run_section(cfg, result, previous=runs.get(cfg.POWER_LEVEL, {}), status=status)
    return _write(cfg, runs)


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------

def _shared_sections(cfg: ExperimentConfig) -> Dict[str, Section]:
    """Everything that defines the configuration, i.e. everything but the power."""
    hardware: Section = {
        "tagger_serial": cfg.TAGGER_SERIAL or "(first device)",
        "trigger_level_v": f"{cfg.TRIGGER_LEVEL_V:g}",
        "corr_binwidth_ps": str(cfg.CORR_BINWIDTH_PS),
        "corr_n_bins": str(cfg.CORR_N_BINS),
        "input_delays_ps": _mapping(cfg.INPUT_DELAYS_PS),
        "deadtimes_ps": _mapping(cfg.DEADTIMES_PS),
        "correlation_pairs": ", ".join(f"({a}, {b})" for a, b in cfg.correlation_pairs()),
    }

    optics: Section = {
        "laser_frequency": f"{cfg.FREQUENCY / 1e6:g} MHz",
        "hwp_angle_deg": f"{cfg.HWP_ANGLE_DEG:g}",
        "cover": _cover(cfg.COVER),
        "polarizer_p1": _polarizer(cfg.P1),
        "polarizer_p2": _polarizer(cfg.P2),
        "polarizer_p3": _polarizer(cfg.P3),
    }
    for name, value in cfg.FILTERS.items():
        optics[f"filter_{name.lower()}"] = _filter(value)

    sections: Dict[str, Section] = {
        "identity": {
            "material": cfg.MATERIAL,
            "experience_type": cfg.EXPERIENCE_TYPE,
            "date": cfg.DATE,
        },
        "optics": optics,
        "channels": {str(ch): cfg.MODES.get(ch, "unused") for ch in cfg.REGISTERED_CHANNELS},
        "hardware": hardware,
        "analysis": {
            "integration_window_bins": str(int(cfg.INTEGRATION_WINDOW)),
            "integration_windows_sweep": _sweep(cfg.INTEGRATION_WINDOWS_SWEEP),
        },
    }
    if cfg.is_power_scan():
        sections["power_scan"] = _power_scan_section(cfg)
    if cfg.is_polarization_scan():
        sections["polarization_scan"] = _polarization_scan_section(cfg)
    return sections


def _power_scan_section(cfg: ExperimentConfig) -> Section:
    """The planned scan: which power label was recorded at which stage angle."""
    stage = cfg.ROTATION_STAGE
    return {
        "enabled": _yes_no(True),
        "stage_serial": (stage.serial_number if stage else "") or "(none)",
        "stage_motion": _stage_motion(cfg),
        "stop_on_error": _yes_no(cfg.POWER_SCAN_STOP_ON_ERROR),
        "points": ", ".join(f"{power} @ {angle:g} deg" for power, angle in cfg.power_scan_points()),
    }


def _polarization_scan_section(cfg: ExperimentConfig) -> Section:
    """The planned polarization scan. Its per-angle record is the CSV log, not the
    dozens of `[run ...]` sections a scan of 36 angles would otherwise add here."""
    angles = [angle for angle, _label in cfg.polarization_scan_points()]
    section: Section = {
        "enabled": _yes_no(True),
        "power_label": cfg.POWER_LEVEL,
        "requested_power": f"{cfg.POLARIZATION_POWER:g} (calibration unit)",
        "angles_deg": _angles(angles),
        "stage_motion": _polarizer_motion(cfg),
        "p1_serial": (cfg.POLARIZATION_P1.serial_number if cfg.POLARIZATION_P1 else "") or "(none)",
        "hwp_serial": (cfg.POLARIZATION_HWP.serial_number if cfg.POLARIZATION_HWP else "") or "(none)",
        "per_angle_log": cfg.polarization_log_path().name,
    }
    if cfg.POLARIZATION_CALIBRATION is not None:
        section["calibration"] = str(cfg.POLARIZATION_CALIBRATION)
    return section


def _run_section(cfg: ExperimentConfig, result: Any, previous: Section,
                 status: Optional[str] = None) -> Section:
    """One acquisition of one power level."""
    finished = result is not None
    status = status or ("complete" if finished else "running")
    section: Section = {
        "status": status,
        "started": previous.get("started", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "duration_s": f"{result.duration_s:g}" if finished else f"{cfg.ACQUISITION_DURATION_S:g}",
        "chunk_duration_s": f"{cfg.CHUNK_DURATION_S:g}",
        "chunks": str(len(result.chunk_paths) if finished else cfg.n_chunks),
        "export_format": cfg.EXPORT_FORMAT,
        "merged": _yes_no(cfg.SAVE_MERGED),
        "raw_ttbin": _yes_no(cfg.SAVE_RAW_TTBIN),
        "stability": _stability(cfg),
    }
    if cfg.ROTATION_ANGLE_DEG is not None:
        section["rotation_angle_deg"] = f"{cfg.ROTATION_ANGLE_DEG:g}"
    if status != "running":
        section["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return section


# ----------------------------------------------------------------------------
# Value formatting
# ----------------------------------------------------------------------------

def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _polarizer(polarizer: Dict[str, Any]) -> str:
    present = polarizer.get("present", polarizer.get("here", False))
    return f"present at {polarizer.get('angle_deg', 0):g} deg" if present else "absent"


def _cover(cover: Dict[str, Any]) -> str:
    state = "present" if cover.get("present", cover.get("here", False)) else "absent"
    description = cover.get("description") or cover.get("separation")
    return f"{state} ({description})" if description else state


def _filter(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value.get('separation', '?')}, {value.get('filter', '?')}"
    return str(value)


def _mapping(mapping: Dict[int, Any]) -> str:
    return ", ".join(f"{ch}: {value}" for ch, value in sorted(mapping.items())) or "none"


def _sweep(windows) -> str:
    values = [int(w) for w in windows]
    if not values:
        return "none"
    steps = {b - a for a, b in zip(values, values[1:])}
    if len(steps) == 1:
        return f"{values[0]} to {values[-1]} step {steps.pop()}"
    return ", ".join(str(v) for v in values)


def _angles(angles: Any) -> str:
    values = [float(a) for a in angles]
    if not values:
        return "none"
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    if len(steps) == 1:
        return (f"{values[0]:g} to {values[-1]:g} step {steps.pop():g} "
                f"({len(values)} points)")
    return ", ".join(f"{v:g}" for v in values)


def _polarizer_motion(cfg: ExperimentConfig) -> str:
    if cfg.POLARIZATION_STAGE_DRY_RUN:
        return "dry run (angles logged, nothing moved)"
    if cfg.POLARIZATION_STAGE_ENABLED:
        return "yes"
    return "no (polarization not set)"


def _stage_motion(cfg: ExperimentConfig) -> str:
    if cfg.ROTATION_STAGE_DRY_RUN:
        return "dry run (angles logged, nothing moved)"
    if cfg.ROTATION_STAGE_ENABLED:
        return "yes"
    return "no (angles not applied)"


def _stability(cfg: ExperimentConfig) -> str:
    if not cfg.STABILITY_ENABLED:
        return "off"
    return (f"every {cfg.STABILITY_SNAPSHOT_INTERVAL_S:g} s, "
            f"{cfg.STABILITY_INTEGRATION_WINDOW_NS:g} ns window, "
            f"{cfg.STABILITY_PEAKS} peaks, rolling {cfg.STABILITY_ROLLING_WINDOW_SNAPS} snapshots")


# ----------------------------------------------------------------------------
# Read / write
# ----------------------------------------------------------------------------

def _parser() -> configparser.ConfigParser:
    # interpolation=None: a filter name or path may contain a '%'.
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # keep channel numbers and key case as written
    return parser


def _read(path: Path) -> Optional[configparser.ConfigParser]:
    if not path.is_file():
        return None
    parser = _parser()
    parser.read(path, encoding="utf-8")
    return parser


def _existing_runs(path: Path) -> Dict[str, Section]:
    """`{power_level: section}` for every `[run <power>]` already in the file."""
    parser = _read(path)
    if parser is None:
        return {}
    return {
        name[len("run "):]: dict(parser[name])
        for name in parser.sections()
        if name.startswith("run ")
    }


def _write(cfg: ExperimentConfig, runs: Dict[str, Section]) -> Path:
    path = path_for(cfg)
    shared = _shared_sections(cfg)
    _warn_on_drift(path, shared)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(shared, runs), encoding="utf-8")
    return path


def _render(shared: Dict[str, Section], runs: Dict[str, Section]) -> str:
    blocks = [HEADER]
    for name, section in shared.items():
        blocks.append(_render_section(name, section))
    for power in sorted(runs, key=_power_value):
        blocks.append(_render_section(f"run {power}", runs[power]))
    return "\n".join(blocks)


def _render_section(name: str, section: Section) -> str:
    width = max((len(key) for key in section), default=0)
    lines = [f"[{name}]"] + [f"{key:<{width}} = {value}" for key, value in section.items()]
    return "\n".join(lines) + "\n"


def _power_value(power: str) -> float:
    digits = "".join(c for c in power if c.isdigit() or c == ".")
    return float(digits) if digits else 0.0


def _warn_on_drift(path: Path, shared: Dict[str, Section]) -> None:
    """Tell the user when the folder already describes a different configuration:
    the runs listed side by side would then not be comparable."""
    parser = _read(path)
    if parser is None:
        return

    changes = []
    for name, section in shared.items():
        if name in DRIFT_EXEMPT_SECTIONS:
            continue
        old = dict(parser[name]) if parser.has_section(name) else {}
        for key, value in section.items():
            if key in old and old[key] != value:
                changes.append(f"{name}.{key}: {old[key]!r} -> {value!r}")

    if changes:
        print(f"WARNING: {path.name} already describes a different configuration:")
        for change in changes:
            print(f"  {change}")
        print("  The runs listed in this folder are no longer comparable. "
              "Consider using another DATE / EXPERIENCE_TYPE folder.")

# ============================================================================
# PART 3 - naming helpers, shared with the analysis
# ============================================================================

def angle_tag(angle_deg: float) -> str:
    """0 -> "pol000p0", 45.5 -> "pol045p5".

    Fixed width, so labels sort by angle and none is a prefix of another - which
    matters because the analyzer finds a run's files by globbing `*{label}*`.
    """
    wrapped = wrap_360(angle_deg)
    whole = int(wrapped)
    tenths = int(round((wrapped - whole) * 10))
    if tenths == 10:
        whole, tenths = whole + 1, 0
    return f"pol{whole % 360:03d}p{tenths}"


def harmonic_of(label: str) -> str:
    """"H3T" -> "H3". Same rule as `pkl_json_analyze.is_cross_harmonic`."""
    return "".join(c for c in label if c.isalnum())[:2]


def _reject_nested_labels(labels: Sequence[str], field_name: str) -> None:
    """Refuse a set of power labels where one is contained in another.

    The analyzer finds a run's files by globbing `*{label}*`, so "5mW" would also
    pick up every "45mW" file and average two powers into one point - silently, and
    only visible as suspiciously smooth data. Padding the labels ("05mW", "45mW")
    or naming them differently ("low", "high") fixes it.
    """
    nested = sorted(
        f"{short!r} inside {long!r}"
        for short, long in itertools.permutations(labels, 2)
        if short in long
    )
    if nested:
        raise ValueError(
            f"{field_name} mixes power labels that contain one another ({', '.join(nested)}): "
            "the analyzer's *{label}* glob cannot tell their files apart. Pad them to the "
            "same width, e.g. '05mW' instead of '5mW'."
        )
