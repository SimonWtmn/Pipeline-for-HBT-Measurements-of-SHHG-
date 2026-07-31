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
    from hardware import (ELLIPTICITY_LOG_NAME, SCAN_LOG_NAME, RotationStageConfig,
                          StageConfig, describe_stage, stage_is_addressed, wrap_360)
except ImportError:  # pragma: no cover - import style depends on the entry point
    from src.hardware import (ELLIPTICITY_LOG_NAME, SCAN_LOG_NAME, RotationStageConfig,
                              StageConfig, describe_stage, stage_is_addressed, wrap_360)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two ways of building the analyzer of an ellipticity scan. They are the same
# measurement - one mount turning in the harmonic beam, the shape of the transmitted
# intensity - and differ only in what is on the mount, and therefore in how often the
# transmitted curve repeats: a half-wave plate turns the polarization by twice its own
# angle, so its Malus curve comes round every 90 deg, while a polarizer turned directly
# gives the usual 180 deg. Everything else (the file names, the fits, the figures) is
# shared, so the kind is carried as metadata and named on the figures.
ANALYZER_KINDS: Dict[str, Dict[str, Any]] = {
    "hwp": {
        "element": "half-wave plate",
        "angle_name": "HWP angle",
        "note": "HWP + fixed polarizer",
        "period_deg": 90.0,
    },
    "polarizer": {
        "element": "polarizer",
        "angle_name": "polarizer angle",
        "note": "rotating polarizer",
        "period_deg": 180.0,
    },
}


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

    # ---------- The pump mounts (shared by the laser-angle and ellipticity scans) ----------
    # P1 and the HWP sit BEFORE the crystal and set the pump's linear polarization:
    # P1 turns to the requested laser angle, and the HWP to whatever angle the measured
    # lookup table says delivers PUMP_POWER there. Angles the table cannot reach are
    # skipped.
    PUMP_POWER: float = 0.0  # in the calibration's unit (mW or Hz), not a label
    # Several power labels already recorded in the same DATE folder (e.g. ["25mW",
    # "45mW", "65mW"]). Used by analyze-only to rebuild each per-power summary and the
    # multi-power overlay.
    PUMP_POWERS: Optional[List[str]] = None
    # Acquire the whole scan at each of several powers in one run. Each entry pairs a
    # file label with the power (calibration unit) every angle is set to:
    #   PUMP_POWER_SCAN = [{"power": "50mW", "requested_power": 50.0},
    #                      {"power": "60mW", "requested_power": 60.0}]
    PUMP_POWER_SCAN: Optional[List[Dict[str, Any]]] = None
    # A calibration is reused run after run: recording one takes hours, and the table
    # only goes stale when the beam path changes. None picks
    # polarization_calibration/linear_polarization_lookup_latest.npz.
    PUMP_CALIBRATION: Optional[Path] = None
    PUMP_CALIBRATION_DIR: Optional[Path] = None
    PUMP_STAGE_ENABLED: bool = False
    PUMP_STAGE_DRY_RUN: bool = False
    PUMP_P1: Optional[RotationStageConfig] = None
    PUMP_HWP: Optional[RotationStageConfig] = None
    PUMP_SETTLE_TIME_S: float = 0.2
    # Base power label of THIS acquisition (kept when POWER_LEVEL becomes the
    # per-point file label "50mW_las000p0").
    PUMP_BASE_POWER: Optional[str] = None

    # ---------- Laser angle scan (optional) ----------
    # One acquisition per pump polarization angle, at a single power: what the crystal
    # emits as its axes are turned relative to the pump.
    #   LASER_ANGLE_SCAN = np.arange(0.0, 360.0, 10.0)
    LASER_ANGLE_SCAN: Optional[Sequence[float]] = None
    LASER_ANGLE_DEG: Optional[float] = None  # pump angle of THIS run; set per scan step
    # Per-angle analysis trees (results/.../laser_angle/{power}/angles/) are heavy and
    # rarely the first thing you look at. Off by default: the vs-angle summaries are
    # always drawn; set this True to also analyse every single angle.
    LASER_ANGLE_PLOT_ANGLES: bool = False

    # ---------- Ellipticity scan (optional) ----------
    # The polarization of the HARMONICS, measured AFTER the crystal: one mount is turned
    # at each pump angle, and the modulation of the intensity it transmits says how
    # elliptical the emission is (a flat curve is circular, one reaching zero is linear).
    #   ELLIPTICITY_LASER_ANGLES = [0.0, 45.0, 90.0]        # the outer loop
    #   ELLIPTICITY_ANALYZER_ANGLES = np.arange(0.0, 180.0, 15.0)   # the inner loop
    ELLIPTICITY_LASER_ANGLES: Optional[Sequence[float]] = None
    ELLIPTICITY_ANALYZER_ANGLES: Sequence[float] = field(
        default_factory=lambda: np.arange(0.0, 180.0, 15.0))
    # What is on that mount, one of ANALYZER_KINDS: "hwp" turns a half-wave plate in
    # front of a fixed polarizer, "polarizer" turns the polarizer itself with no plate
    # at all. It sets the period of the fitted curve, names the swept angle on every
    # figure, and is written to experiment_config.txt and the scan log.
    ELLIPTICITY_ANALYZER_KIND: str = "hwp"
    # RotationStageConfig for a Kinesis mount, ELLStageConfig for an Elliptec one.
    ELLIPTICITY_ANALYZER: Optional[StageConfig] = None
    ELLIPTICITY_ANALYZER_ENABLED: bool = False
    ELLIPTICITY_ANALYZER_DRY_RUN: bool = False
    ELLIPTICITY_ANALYZER_SETTLE_TIME_S: float = 0.2
    # The period of the fitted Malus curve. None takes it from ELLIPTICITY_ANALYZER_KIND
    # (90 deg for the half-wave plate, 180 deg for the polarizer), which is what it
    # should be: it is a property of the optic, not a fitting preference.
    ELLIPTICITY_FIT_PERIOD_DEG: Optional[float] = None
    #: Metadata only, and only with the HWP: the fixed polarizer after the plate
    #: (0 = vertical). With "polarizer" the swept mount IS the polarizer.
    ELLIPTICITY_FIXED_POLARIZER_DEG: float = 0.0
    ANALYZER_ANGLE_DEG: Optional[float] = None  # analyzer angle of THIS run
    # Per-point HBT trees (results/.../ellipticity/{power}/{laser angle}/points/) are
    # heavy: one per (laser angle, analyzer angle) pair. Off by default.
    ELLIPTICITY_PLOT_POINTS: bool = False

    # ---------- Analysis ----------
    INTEGRATION_WINDOW: int = 8
    INTEGRATION_WINDOWS_SWEEP: Sequence[int] = field(default_factory=lambda: np.arange(1, 31, 1))
    RUN_ANALYSIS_AFTER_ACQUIRE: bool = True
    ANALYZE_ONLY: bool = False

    def __post_init__(self) -> None:
        if self.REGISTERED_CHANNELS is None:
            self.REGISTERED_CHANNELS = sorted(self.MODES)
        self._validate_power_scan()
        self._validate_laser_angle_scan()
        self._validate_ellipticity_scan()
        if self.PUMP_POWERS:
            _reject_nested_labels(list(self.PUMP_POWERS), "PUMP_POWERS")
        if self.POWER_SCAN:
            # The scan is what was actually recorded, so it defines what to analyse.
            self.POWER_LEVELS = [power for power, _angle in self.power_scan_points()]
        elif self.LASER_ANGLE_SCAN is not None:
            self.POWER_LEVELS = [label for _angle, label in self.laser_angle_scan_points()]
        elif self.ELLIPTICITY_LASER_ANGLES is not None:
            self.POWER_LEVELS = [label for _la, _aa, label in self.ellipticity_scan_points()]
        if self.POWER_LEVELS is None:
            self.POWER_LEVELS = [self.POWER_LEVEL]
        if self.PUMP_CALIBRATION_DIR is None:
            self.PUMP_CALIBRATION_DIR = REPO_ROOT / "polarization_calibration"
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

    def _validate_laser_angle_scan(self) -> None:
        if self.LASER_ANGLE_SCAN is None:
            return

        angles = [float(angle) for angle in self.LASER_ANGLE_SCAN]
        if not angles:
            self.LASER_ANGLE_SCAN = None
            return
        if self.POWER_SCAN:
            raise ValueError(
                "POWER_SCAN and LASER_ANGLE_SCAN are both set: that would multiply into "
                "one acquisition per (power, angle) pair. Run one scan at a time."
            )
        if self.ELLIPTICITY_LASER_ANGLES is not None:
            raise ValueError(
                "LASER_ANGLE_SCAN and ELLIPTICITY_LASER_ANGLES are both set: the first "
                "records one acquisition per pump angle, the second a whole analyzer "
                "sweep at each pump angle. Run one scan at a time."
            )

        _reject_duplicate_angle_labels(angles, "LASER_ANGLE_SCAN", laser_angle_tag)
        self.LASER_ANGLE_SCAN = angles

        if self.ANALYZE_ONLY:
            return
        self._validate_pump_power("LASER_ANGLE_SCAN")
        self._validate_pump_stages("LASER_ANGLE_SCAN")

    def _validate_ellipticity_scan(self) -> None:
        self._resolve_analyzer_kind()
        if self.ELLIPTICITY_LASER_ANGLES is None:
            return

        laser_angles = [float(angle) for angle in self.ELLIPTICITY_LASER_ANGLES]
        if not laser_angles:
            self.ELLIPTICITY_LASER_ANGLES = None
            return
        if self.POWER_SCAN:
            raise ValueError(
                "POWER_SCAN and ELLIPTICITY_LASER_ANGLES are both set: that would "
                "multiply into one acquisition per (power, pump angle, analyzer angle). "
                "Run one scan at a time."
            )

        analyzer_angles = [float(angle) for angle in self.ELLIPTICITY_ANALYZER_ANGLES]
        if len(analyzer_angles) < 4:
            raise ValueError(
                "ELLIPTICITY_ANALYZER_ANGLES needs at least 4 angles: the Malus curve "
                "fitted to the analyzer sweep has three free parameters (amplitude, "
                f"angle, floor), and got {len(analyzer_angles)} point(s). A typical sweep "
                "is np.arange(0.0, 180.0, 15.0)."
            )
        if self.ELLIPTICITY_FIT_PERIOD_DEG <= 0:
            raise ValueError(
                "ELLIPTICITY_FIT_PERIOD_DEG must be > 0 (90 deg for a half-wave plate in "
                f"front of a fixed polarizer), got {self.ELLIPTICITY_FIT_PERIOD_DEG!r}"
            )

        _reject_duplicate_angle_labels(laser_angles, "ELLIPTICITY_LASER_ANGLES", laser_angle_tag)
        _reject_duplicate_angle_labels(analyzer_angles, "ELLIPTICITY_ANALYZER_ANGLES",
                                       analyzer_angle_tag)
        self.ELLIPTICITY_LASER_ANGLES = laser_angles
        self.ELLIPTICITY_ANALYZER_ANGLES = analyzer_angles

        if self.ANALYZE_ONLY:
            return
        self._validate_pump_power("ELLIPTICITY_LASER_ANGLES")
        self._validate_pump_stages("ELLIPTICITY_LASER_ANGLES")
        if self.ELLIPTICITY_ANALYZER_DRY_RUN:
            return
        if not self.ELLIPTICITY_ANALYZER_ENABLED:
            raise ValueError(
                "An ellipticity scan is configured but ELLIPTICITY_ANALYZER_ENABLED is "
                f"False: every analyzer angle would be recorded with the "
                f"{self.ellipticity_analyzer_element} standing still, so the sinusoid "
                "would be flat for want of motion rather than because the emission is "
                "circular. Enable the analyzer mount, or set "
                "ELLIPTICITY_ANALYZER_DRY_RUN=True to rehearse."
            )
        if not stage_is_addressed(self.ELLIPTICITY_ANALYZER):
            raise ValueError(
                "ELLIPTICITY_ANALYZER_ENABLED is True but ELLIPTICITY_ANALYZER does not "
                "name a mount: pass "
                "ELLIPTICITY_ANALYZER=RotationStageConfig(serial_number=\"...\") for a "
                "Kinesis controller, or ELLIPTICITY_ANALYZER=ELLStageConfig(port=\"...\") "
                "for an Elliptec one."
            )

    def _resolve_analyzer_kind(self) -> None:
        """Normalise the analyzer kind, and take the fit period from it when unset."""
        kind = str(self.ELLIPTICITY_ANALYZER_KIND or "hwp").strip().lower()
        if kind not in ANALYZER_KINDS:
            raise ValueError(
                f"ELLIPTICITY_ANALYZER_KIND must be one of {sorted(ANALYZER_KINDS)} "
                f"(\"hwp\" turns a half-wave plate in front of a fixed polarizer, "
                f"\"polarizer\" turns the polarizer itself), got "
                f"{self.ELLIPTICITY_ANALYZER_KIND!r}"
            )
        self.ELLIPTICITY_ANALYZER_KIND = kind
        if self.ELLIPTICITY_FIT_PERIOD_DEG is None:
            self.ELLIPTICITY_FIT_PERIOD_DEG = float(ANALYZER_KINDS[kind]["period_deg"])

    def _validate_pump_power(self, scan_field: str) -> None:
        if self.PUMP_POWER_SCAN:
            self._validate_pump_power_scan()
        elif self.PUMP_POWER <= 0:
            raise ValueError(
                f"{scan_field} needs PUMP_POWER: the power every angle is set to, in the "
                "unit of the calibration table (mW for a power-meter table). For several "
                "powers in one run, use PUMP_POWER_SCAN instead."
            )
        else:
            _check_power_label_matches(self.POWER_LEVEL, self.PUMP_POWER, "POWER_LEVEL")

    def _validate_pump_stages(self, scan_field: str) -> None:
        if self.PUMP_STAGE_DRY_RUN:
            return
        if not self.PUMP_STAGE_ENABLED:
            raise ValueError(
                f"{scan_field} is set but PUMP_STAGE_ENABLED is False: every angle would "
                "be recorded at whatever polarization the pump mounts happen to hold. "
                "Enable the mounts, or set PUMP_STAGE_DRY_RUN=True."
            )
        for name, stage in (("PUMP_P1", self.PUMP_P1), ("PUMP_HWP", self.PUMP_HWP)):
            if stage is None or not stage.serial_number:
                raise ValueError(
                    f"PUMP_STAGE_ENABLED is True but {name} has no serial number: "
                    f"pass {name}=RotationStageConfig(serial_number=\"...\")."
                )

    def _validate_pump_power_scan(self) -> None:
        shape = ('PUMP_POWER_SCAN entries must look like '
                 '{"power": "50mW", "requested_power": 50.0}')
        seen = set()
        for entry in self.PUMP_POWER_SCAN:
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
            _check_power_label_matches(label, value, "PUMP_POWER_SCAN 'power'")
            if label in seen:
                # Two powers sharing a label would write over each other's files.
                raise ValueError(f"PUMP_POWER_SCAN lists the power label {label!r} twice")
            seen.add(label)
        _reject_nested_labels(sorted(seen), "PUMP_POWER_SCAN")

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

    # ---------- The pump power, shared by both angle scans ----------

    def pump_power_points(self) -> List[Tuple[str, float]]:
        """[(power label, requested power)] for the acquisition.

        A multi-power run lists them in PUMP_POWER_SCAN; otherwise the single
        (POWER_LEVEL, PUMP_POWER) pair is the whole scan.
        """
        if self.PUMP_POWER_SCAN:
            return [(str(entry["power"]), float(entry["requested_power"]))
                    for entry in self.PUMP_POWER_SCAN]
        return [(self.POWER_LEVEL, float(self.PUMP_POWER))]

    def for_pump_scan_power(self, power: str, requested_power: float) -> "ExperimentConfig":
        """A copy that records the whole scan at one power of a multi-power run."""
        return replace(
            self,
            POWER_LEVEL=power,
            POWER_LEVELS=None,
            PUMP_POWER=float(requested_power),
            PUMP_BASE_POWER=power,
            PUMP_POWER_SCAN=None,
        )

    def for_pump_power(self, power: str) -> "ExperimentConfig":
        """A copy aimed at one power of a multi-power campaign."""
        return replace(
            self,
            POWER_LEVEL=power,
            POWER_LEVELS=None,
            PUMP_BASE_POWER=power,
            PUMP_POWERS=None,
        )

    @property
    def base_power(self) -> str:
        """The power label of the campaign, whatever POWER_LEVEL was rewritten to."""
        return self.PUMP_BASE_POWER or self.POWER_LEVEL

    # ---------- Laser angle scan ----------

    def is_laser_angle_scan(self) -> bool:
        return self.LASER_ANGLE_SCAN is not None

    def laser_angle_scan_points(self) -> List[Tuple[float, str]]:
        """[(angle, file label)] in the configured order; empty without a scan."""
        if self.LASER_ANGLE_SCAN is None:
            return []
        return [(float(angle), f"{self.POWER_LEVEL}_{laser_angle_tag(angle)}")
                for angle in self.LASER_ANGLE_SCAN]

    def for_laser_angle(self, angle_deg: float) -> "ExperimentConfig":
        """A copy of this config describing the acquisition at one pump angle.

        The angle's file label becomes POWER_LEVEL (so chunk names stay
        `{power}_lasXXX_numN`), while the analysis tree lands under
        `results/.../laser_angle/{power}/angles/`.
        """
        base = self.base_power
        for angle, label in self.laser_angle_scan_points():
            if laser_angle_tag(angle) == laser_angle_tag(angle_deg):
                # POWER_LEVELS=None so it is re-derived as [label]: without that, the
                # step would carry every angle of the scan and analyse all of them.
                return replace(
                    self,
                    POWER_LEVEL=label,
                    POWER_LEVELS=None,
                    LASER_ANGLE_SCAN=None,
                    LASER_ANGLE_DEG=angle,
                    PUMP_BASE_POWER=base,
                    RESULTS_DIR=self.RESULTS_DIR / "laser_angle" / base / "angles",
                )
        raise ValueError(f"{angle_deg} deg is not one of the LASER_ANGLE_SCAN angles")

    def laser_angle_log_path(self) -> Path:
        return self.DATA_DIR / SCAN_LOG_NAME

    @property
    def laser_angle_root(self) -> Path:
        return self.RESULTS_DIR / "laser_angle"

    def laser_angle_summary_dir(self, power: Optional[str] = None) -> Path:
        """`results/.../laser_angle/{power}/summary/` — the four vs-angle grids."""
        return self.laser_angle_root / (power or self.base_power) / "summary"

    @property
    def laser_angle_overlay_dir(self) -> Path:
        """`results/.../laser_angle/overlay/` — butterflies with every power overlaid."""
        return self.laser_angle_root / "overlay"

    # ---------- Ellipticity scan ----------

    def is_ellipticity_scan(self) -> bool:
        return self.ELLIPTICITY_LASER_ANGLES is not None

    @property
    def _analyzer_kind(self) -> Dict[str, Any]:
        return ANALYZER_KINDS[str(self.ELLIPTICITY_ANALYZER_KIND).strip().lower()]

    @property
    def ellipticity_analyzer_element(self) -> str:
        """What is on the analyzer mount: "half-wave plate" or "polarizer"."""
        return str(self._analyzer_kind["element"])

    @property
    def ellipticity_angle_name(self) -> str:
        """How the swept angle is named on the figures: "HWP angle", "polarizer angle"."""
        return str(self._analyzer_kind["angle_name"])

    @property
    def ellipticity_analyzer_note(self) -> str:
        """The short line the figures carry so a sweep says how it was measured."""
        return str(self._analyzer_kind["note"])

    def ellipticity_laser_angles(self) -> List[float]:
        return [float(angle) for angle in (self.ELLIPTICITY_LASER_ANGLES or [])]

    def ellipticity_analyzer_points(self, laser_angle_deg: float) -> List[Tuple[float, str]]:
        """[(analyzer angle, file label)] of the sweep recorded at one pump angle."""
        return [(float(analyzer),
                 f"{self.POWER_LEVEL}_{laser_angle_tag(laser_angle_deg)}"
                 f"_{analyzer_angle_tag(analyzer)}")
                for analyzer in self.ELLIPTICITY_ANALYZER_ANGLES]

    def ellipticity_scan_points(self) -> List[Tuple[float, float, str]]:
        """[(laser angle, analyzer angle, file label)] of the whole scan, in run order."""
        return [(laser, analyzer, label)
                for laser in self.ellipticity_laser_angles()
                for analyzer, label in self.ellipticity_analyzer_points(laser)]

    def for_ellipticity_point(self, laser_angle_deg: float,
                              analyzer_angle_deg: float) -> "ExperimentConfig":
        """A copy describing the acquisition at one (pump angle, analyzer angle) pair.

        The pair's file label becomes POWER_LEVEL (chunk names
        `{power}_lasXXX_hwpYYY_numN`), and the optional per-point analysis tree lands
        under `results/.../ellipticity/{power}/{laser angle}/points/`.
        """
        base = self.base_power
        return replace(
            self,
            POWER_LEVEL=f"{base}_{laser_angle_tag(laser_angle_deg)}"
                        f"_{analyzer_angle_tag(analyzer_angle_deg)}",
            POWER_LEVELS=None,
            ELLIPTICITY_LASER_ANGLES=None,
            LASER_ANGLE_DEG=float(laser_angle_deg),
            ANALYZER_ANGLE_DEG=float(analyzer_angle_deg),
            PUMP_BASE_POWER=base,
            RESULTS_DIR=self.ellipticity_laser_dir(base, laser_angle_deg) / "points",
        )

    def ellipticity_log_path(self) -> Path:
        return self.DATA_DIR / ELLIPTICITY_LOG_NAME

    @property
    def ellipticity_root(self) -> Path:
        return self.RESULTS_DIR / "ellipticity"

    def ellipticity_laser_dir(self, power: Optional[str], laser_angle_deg: float) -> Path:
        """`results/.../ellipticity/{power}/lasXXX/` — one analyzer sweep."""
        return (self.ellipticity_root / (power or self.base_power)
                / laser_angle_tag(laser_angle_deg))

    def ellipticity_summary_dir(self, power: Optional[str] = None) -> Path:
        """`results/.../ellipticity/{power}/summary/` — ellipticity vs laser angle."""
        return self.ellipticity_root / (power or self.base_power) / "summary"

    @property
    def ellipticity_overlay_dir(self) -> Path:
        """`results/.../ellipticity/overlay/` — every power on the same axes."""
        return self.ellipticity_root / "overlay"

    # ---------- Metadata ----------

    def file_parameters(self, chunk: Any, duration_s: float) -> Dict[str, Any]:
        params = {"power_level": self.POWER_LEVEL, "chunk": chunk, "duration_s": duration_s}
        if self.ROTATION_ANGLE_DEG is not None:
            params["rotation_angle_deg"] = self.ROTATION_ANGLE_DEG
        if self.LASER_ANGLE_DEG is not None:
            params["laser_angle_deg"] = self.LASER_ANGLE_DEG
            params["requested_power"] = self.PUMP_POWER
        if self.ANALYZER_ANGLE_DEG is not None:
            params["analyzer_angle_deg"] = self.ANALYZER_ANGLE_DEG
            params["analyzer_kind"] = self.ELLIPTICITY_ANALYZER_KIND
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
# second run appends itself without losing the previous ones. An angle scan logs its
# angles to a CSV of its own (`hardware.LaserAngleLog`, `hardware.EllipticityLog`)
# rather than the dozens of sections one point per angle would add here.

CONFIG_FILENAME = "experiment_config.txt"

Section = Dict[str, str]

# A scan plan describes a procedure, not the setup: extending it or rehearsing it as
# a dry run is legitimate and must not be reported as a configuration change.
DRIFT_EXEMPT_SECTIONS = {"power_scan", "laser_angle_scan", "ellipticity_scan"}

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
    if cfg.is_laser_angle_scan():
        sections["laser_angle_scan"] = _laser_angle_scan_section(cfg)
    if cfg.is_ellipticity_scan():
        sections["ellipticity_scan"] = _ellipticity_scan_section(cfg)
    return sections


def _power_scan_section(cfg: ExperimentConfig) -> Section:
    """The planned scan: which power label was recorded at which stage angle."""
    return {
        "enabled": _yes_no(True),
        "stage_serial": describe_stage(cfg.ROTATION_STAGE),
        "stage_motion": _stage_motion(cfg),
        "stop_on_error": _yes_no(cfg.POWER_SCAN_STOP_ON_ERROR),
        "points": ", ".join(f"{power} @ {angle:g} deg" for power, angle in cfg.power_scan_points()),
    }


def _laser_angle_scan_section(cfg: ExperimentConfig) -> Section:
    """The planned laser angle scan. Its per-angle record is the CSV log, not the
    dozens of `[run ...]` sections a scan of 36 angles would otherwise add here."""
    angles = [angle for angle, _label in cfg.laser_angle_scan_points()]
    section: Section = {
        "enabled": _yes_no(True),
        "power_label": cfg.POWER_LEVEL,
        "requested_power": f"{cfg.PUMP_POWER:g} (calibration unit)",
        "laser_angles_deg": _angles(angles),
        "per_angle_log": cfg.laser_angle_log_path().name,
    }
    section.update(_pump_section(cfg))
    return section


def _ellipticity_scan_section(cfg: ExperimentConfig) -> Section:
    """The planned ellipticity scan: the pump angles, and the analyzer sweep at each.

    `analyzer_kind` is the one line that says which of the two setups was on the bench,
    since the file names of the two are identical.
    """
    section: Section = {
        "enabled": _yes_no(True),
        "power_label": cfg.POWER_LEVEL,
        "requested_power": f"{cfg.PUMP_POWER:g} (calibration unit)",
        "laser_angles_deg": _angles(cfg.ellipticity_laser_angles()),
        "analyzer_kind": f"{cfg.ELLIPTICITY_ANALYZER_KIND} ({cfg.ellipticity_analyzer_note})",
        "analyzer_angles_deg": _angles(cfg.ELLIPTICITY_ANALYZER_ANGLES),
        "analyzer_stage": describe_stage(cfg.ELLIPTICITY_ANALYZER),
        "analyzer_motion": _motion(cfg.ELLIPTICITY_ANALYZER_ENABLED,
                                   cfg.ELLIPTICITY_ANALYZER_DRY_RUN,
                                   "no (analyzer not turned)"),
        "fit_period_deg": f"{cfg.ELLIPTICITY_FIT_PERIOD_DEG:g}",
        "per_angle_log": cfg.ellipticity_log_path().name,
    }
    if cfg.ELLIPTICITY_ANALYZER_KIND == "hwp":
        # With the polarizer on the mount there is no second, fixed one to describe.
        section["fixed_polarizer_deg"] = (
            f"{cfg.ELLIPTICITY_FIXED_POLARIZER_DEG:g} (0 = vertical)")
    section.update(_pump_section(cfg))
    return section


def _pump_section(cfg: ExperimentConfig) -> Section:
    """How the pump polarization was set: shared by both angle scans."""
    section: Section = {
        "pump_motion": _motion(cfg.PUMP_STAGE_ENABLED, cfg.PUMP_STAGE_DRY_RUN,
                               "no (polarization not set)"),
        "pump_p1_serial": describe_stage(cfg.PUMP_P1),
        "pump_hwp_serial": describe_stage(cfg.PUMP_HWP),
    }
    if cfg.PUMP_CALIBRATION is not None:
        section["pump_calibration"] = str(cfg.PUMP_CALIBRATION)
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
    if cfg.LASER_ANGLE_DEG is not None:
        section["laser_angle_deg"] = f"{cfg.LASER_ANGLE_DEG:g}"
    if cfg.ANALYZER_ANGLE_DEG is not None:
        section["analyzer_angle_deg"] = f"{cfg.ANALYZER_ANGLE_DEG:g}"
        section["analyzer_kind"] = cfg.ELLIPTICITY_ANALYZER_KIND
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
    if not present:
        return "absent"
    angle = polarizer.get("angle_deg", 0)
    try:
        return f"present at {float(angle):g} deg"
    except (TypeError, ValueError):
        return f"present at {angle}"  # e.g. "unknown"


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


def _motion(enabled: bool, dry_run: bool, idle: str) -> str:
    if dry_run:
        return "dry run (angles logged, nothing moved)"
    return "yes" if enabled else idle


def _stage_motion(cfg: ExperimentConfig) -> str:
    return _motion(cfg.ROTATION_STAGE_ENABLED, cfg.ROTATION_STAGE_DRY_RUN,
                   "no (angles not applied)")


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

def angle_tag(angle_deg: float, prefix: str) -> str:
    """0 -> "las000p0", 45.5 -> "las045p5" with prefix "las".

    Fixed width, so labels sort by angle and none is a prefix of another - which
    matters because the analyzer finds a run's files by globbing `*{label}_num*`.
    """
    wrapped = wrap_360(angle_deg)
    whole = int(wrapped)
    tenths = int(round((wrapped - whole) * 10))
    if tenths == 10:
        whole, tenths = whole + 1, 0
    return f"{prefix}{whole % 360:03d}p{tenths}"


def laser_angle_tag(angle_deg: float) -> str:
    """The pump polarization angle, as it appears in a file name: 45 -> "las045p0"."""
    return angle_tag(angle_deg, "las")


def analyzer_angle_tag(angle_deg: float) -> str:
    """The analyzer angle after the crystal: 15 -> "hwp015p0".

    The "hwp" prefix is historical and names the analyzer angle whichever optic carries
    it, so a scan turning the polarizer itself writes the same file names: the two are
    told apart by `analyzer_kind` in `experiment_config.txt` and in the scan log, not by
    the file names. Keeping one prefix is what lets both be found by the same glob.
    """
    return angle_tag(angle_deg, "hwp")


def harmonic_of(label: str) -> str:
    """"H3T" -> "H3". Same rule as `pkl_json_analyze.is_cross_harmonic`."""
    return "".join(c for c in label if c.isalnum())[:2]


def _reject_duplicate_angle_labels(angles: Sequence[float], field_name: str, tagger) -> None:
    """Refuse angles that would share a file label.

    Labels resolve 0.1 deg, so two angles closer than that would write over each
    other's files. Checking the labels catches both a repeated and a too-close angle.
    """
    labels = [tagger(angle) for angle in angles]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"{field_name} maps several angles onto the same file label {duplicates}: "
            "angles must differ by at least 0.1 deg once wrapped into [0, 360)."
        )


def _check_power_label_matches(label: str, requested_power: float, field_name: str) -> None:
    """Refuse a power label whose number contradicts the requested power.

    The file label (`POWER_LEVEL`) and the beam power (`PUMP_POWER`) are two
    independent fields: the first only names the files, the second is what the mounts
    actually deliver. Leaving the label at its default while asking for another power
    silently saves e.g. 90 mW data as "65mW". So whenever the label carries a number,
    it must equal the requested power. Non-numeric labels ("low"/"high") are exempt:
    there is nothing to cross-check, and they are a deliberate way to opt out.
    """
    digits = "".join(c for c in label if c.isdigit() or c == ".")
    try:
        labelled = float(digits)
    except ValueError:
        return  # non-numeric label (e.g. "low"): nothing to verify against the power
    if abs(labelled - float(requested_power)) > 1e-9:
        raise ValueError(
            f"{field_name} label {label!r} does not match the requested power "
            f"{requested_power:g} (calibration unit): the files would be saved as {label!r} "
            f"while the beam is set to {requested_power:g}. Rename the label to match, e.g. "
            f"\"{requested_power:g}mW\", or make it non-numeric (e.g. 'low') to opt out."
        )


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
