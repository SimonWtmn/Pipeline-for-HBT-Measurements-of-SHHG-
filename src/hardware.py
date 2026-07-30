"""Everything that moves: the Thorlabs rotation mounts, and the linear polarization
built on top of two of them.

* `RotationStage` drives a single Thorlabs mount through Kinesis. A power scan uses
  one of these on its own, to turn a wave plate to a calibrated angle per power.
* `PolarizationController` drives the half-wave plate and the polarizer together,
  reading the angles it needs from a P1/HWP lookup table recorded beforehand.
  `LookupTable`, `ScanLog` and `preflight` belong to that half.

No hardware library is imported until something actually connects, which is what lets
every dry run, analysis and replot happen on a laptop.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# PART 1 - a single Thorlabs rotation mount
# ============================================================================

# Kinesis is a Windows installer, and its .NET assemblies are not on sys.path by
# default: pythonnet needs the install folder before clr.AddReference works.
KINESIS_DEFAULT_PATH = r"C:\Program Files\Thorlabs\Kinesis"

# One .NET class per controller family; the factory method is always
# "Create<ClassName>". Add an entry here and set RotationStageConfig.device_type to
# drive a controller that is not listed yet.
DEVICE_CLASSES = {
    "K10CR1": ("Thorlabs.MotionControl.IntegratedStepperMotorsCLI", "CageRotator"),
    "KDC101": ("Thorlabs.MotionControl.KCube.DCServoCLI", "KCubeDCServo"),
    "TDC001": ("Thorlabs.MotionControl.TCube.DCServoCLI", "TCubeDCServo"),
    "KST101": ("Thorlabs.MotionControl.KCube.StepperMotorCLI", "KCubeStepperMotor"),
    "BSC201": ("Thorlabs.MotionControl.Benchtop.StepperMotorCLI", "BenchtopStepperMotor"),
}

# Thorlabs encodes the controller family in the first two digits of the serial
# number. Only a hint: set device_type explicitly if the guess is wrong.
SERIAL_PREFIXES = {
    "55": "K10CR1",   # integrated cage rotator, rotation built in
    "27": "KDC101",   # K-cube DC servo, e.g. driving a PRM1Z8 rotation mount
    "83": "TDC001",   # T-cube DC servo, older PRM1Z8 setups
    "26": "KST101",   # K-cube stepper
    "40": "BSC201",   # benchtop stepper
}

_PYTHONNET_HINT = (
    "Reaching the Thorlabs stage needs pythonnet (`pip install pythonnet`), which "
    "bridges Python and the Kinesis .NET API. Kinesis itself is a Windows installer, "
    "not a pip package: https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm"
    "?Code=Motion_Control. Set ROTATION_STAGE_DRY_RUN=True to rehearse a scan with no "
    "hardware at all."
)

_KINESIS_HINT = (
    "Could not load the Kinesis assembly {assembly} from {path}. Install Thorlabs "
    "Kinesis, or point RotationStageConfig.kinesis_path at its install folder. If the "
    "controller family was guessed wrong from the serial number, set "
    "RotationStageConfig.device_type to one of {known}."
)


@dataclass
class RotationStageConfig:
    serial_number: str = ""
    #: None guesses the controller family from the serial prefix (SERIAL_PREFIXES).
    device_type: Optional[str] = None
    #: Benchtop controllers only: which axis of the controller carries the stage.
    channel: int = 1
    #: False negates every requested angle, i.e. the mount reads anticlockwise. Same
    #: convention as the lab's polarization calibration script, whose lookup tables
    #: only mean anything if this matches the stage they were recorded with.
    clockwise: bool = True
    #: Leave None. Kinesis reports degrees once the motor configuration is loaded;
    #: set this only for a controller that reports raw encoder counts instead.
    scale_deg_per_count: Optional[float] = None
    home_on_connect: bool = False
    move_timeout_s: float = 120.0
    settle_time_s: float = 0.5
    #: How long to wait for a commanded move to actually start before giving up on
    #: seeing motion and proceeding to the arrival check. Guards against MoveTo
    #: returning (and briefly reporting the target position) before the stage moves.
    motion_start_timeout_s: float = 3.0
    angle_tolerance_deg: float = 0.05
    max_velocity_deg_s: Optional[float] = None
    acceleration_deg_s2: Optional[float] = None
    kinesis_path: str = KINESIS_DEFAULT_PATH
    poll_interval_s: float = 0.2


def wrap_360(angle_deg: float) -> float:
    """A mount angle in [0, 360): 375 deg and -345 deg are both 15 deg."""
    return float(angle_deg) % 360.0


def angular_difference(a_deg: float, b_deg: float) -> float:
    """Signed a - b in (-180, 180], so 359 deg and 1 deg are 2 deg apart."""
    return ((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0


def resolve_device_class(serial: str, device_type: Optional[str] = None) -> Tuple[str, str, str]:
    """(device_type, assembly, class_name) for a serial number or an explicit family."""
    name = device_type or SERIAL_PREFIXES.get(serial[:2])
    if name is None:
        raise ValueError(
            f"Cannot guess the controller family from serial {serial!r}. Set "
            f"RotationStageConfig.device_type to one of {sorted(DEVICE_CLASSES)}."
        )
    if name not in DEVICE_CLASSES:
        raise ValueError(f"Unknown device_type {name!r}. Known: {sorted(DEVICE_CLASSES)}.")
    assembly, class_name = DEVICE_CLASSES[name]
    return name, assembly, class_name


class RotationStage:
    """A Thorlabs rotation mount driven through Kinesis.

    With dry_run=True nothing is imported and nothing moves: the requested angles are
    printed and remembered, which is what lets a power scan be rehearsed on a machine
    without Kinesis (or without the stage plugged in).
    """

    def __init__(self, config: RotationStageConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._device = None      # the controller
        self._axis = None        # the moving channel (identical for single-axis controllers)
        self._decimal = None     # System.Decimal, the unit Kinesis takes positions in
        self._simulated_angle = 0.0

    # ---------- lifecycle ----------

    def __enter__(self) -> "RotationStage":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.disconnect()
        return False

    def connect(self) -> None:
        if self.dry_run:
            print(f"[stage] dry run: not connecting (serial {self.config.serial_number or 'unset'})")
            return

        serial = self.config.serial_number
        if not serial:
            raise ValueError("RotationStageConfig.serial_number is empty: it is printed on the controller.")

        device_type, device_class, class_name, decimal_type, device_manager = self._load_kinesis(serial)
        self._decimal = decimal_type

        device_manager.BuildDeviceList()
        device = getattr(device_class, f"Create{class_name}")(serial)
        device.Connect(serial)

        # Kinesis allows one client per device, so anything failing below must close the
        # connection again or the next attempt fails until the device is replugged.
        try:
            # Benchtop controllers expose one channel per axis; K-cubes and the
            # K10CR1 are single-axis and are their own channel.
            axis = device.GetChannel(self.config.channel) if hasattr(device, "GetChannel") else device

            if not axis.IsSettingsInitialized():
                axis.WaitForSettingsInitialized(10_000)
            axis.StartPolling(250)
            time.sleep(0.3)      # let the first status messages arrive before enabling
            axis.EnableDevice()
            time.sleep(0.3)

            # Loads the stage's calibration, which is what makes positions degrees
            # instead of raw device counts.
            axis.LoadMotorConfiguration(str(getattr(axis, "DeviceID", serial) or serial))

            self._device, self._axis = device, axis
            self._apply_velocity()
            print(f"[stage] connected to {device_type} {serial}, now at {self.get_angle_deg():.3f} deg")

            if self.config.home_on_connect:
                self.home()
        except Exception:
            self._device, self._axis = device, None
            self.disconnect()
            raise

    def disconnect(self) -> None:
        if self.dry_run or self._device is None:
            self._device = self._axis = None
            return
        try:
            if self._axis is not None:
                self._axis.StopPolling()
        except Exception as exc:  # a failed StopPolling must not skip the disconnect
            print(f"[stage] warning: StopPolling failed ({exc})")
        finally:
            try:
                self._device.Disconnect(True)
                print("[stage] disconnected")
            finally:
                self._device = self._axis = None

    # ---------- motion ----------

    def get_angle_deg(self) -> float:
        if self.dry_run:
            return self._simulated_angle
        self._require_connection()
        # System.Decimal has no direct float conversion under pythonnet.
        angle = float(str(self._axis.Position))
        if self.config.scale_deg_per_count is not None:
            angle *= self.config.scale_deg_per_count
        return angle

    def raw_angle(self, angle_deg: float) -> float:
        """The requested angle in the mount's own frame: signed, then wrapped."""
        sign = 1.0 if self.config.clockwise else -1.0
        return wrap_360(sign * float(angle_deg))

    def move_to_angle_deg(self, angle_deg: float, *, wait: bool = True) -> None:
        raw = self.raw_angle(angle_deg)
        raw_note = "" if raw == float(angle_deg) else f", raw {raw:g}"

        if self.dry_run:
            print(f"[stage] dry run: would move to {angle_deg:g} deg{raw_note}")
            self._simulated_angle = raw
            return

        self._require_connection()
        scale = self.config.scale_deg_per_count
        target = raw / scale if scale else raw
        print(f"[stage] moving to {angle_deg:g} deg{raw_note} (from {self.get_angle_deg():.3f} deg)")

        # Never command a new move while the previous one is still running: the stage
        # keeps a multi-turn absolute count, so a full-turn slew can look "arrived" a
        # lap early, and Kinesis then rejects the next MoveTo with a
        # DeviceMovingException ("Device already moving").
        self._wait_until_stopped()

        # A zero timeout starts the move and returns immediately, so the tolerance
        # below is what decides when the move is over, not the controller's own idea.
        try:
            self._axis.MoveTo(self._decimal(float(target)), 0)
        except Exception as exc:
            if "already moving" not in str(exc).lower():
                raise
            # A residual motion slipped past the guard above; let it finish and retry.
            self._wait_until_stopped(force=True)
            self._axis.MoveTo(self._decimal(float(target)), 0)
        if wait:
            self.wait_until_angle(angle_deg)

    def wait_until_angle(self, angle_deg: float) -> float:
        """Poll until the stage is within angle_tolerance_deg of the target and stopped."""
        if self.dry_run:
            return self._simulated_angle

        raw = self.raw_angle(angle_deg)
        deadline = time.monotonic() + self.config.move_timeout_s
        start = self.get_angle_deg()

        # Wait for the move to actually begin before checking for arrival. MoveTo(..., 0)
        # returns before the controller raises its motion flag and can briefly report the
        # *demanded* (target) position, so without this guard the very first poll below
        # satisfies the tolerance check and ends the wait a whole slew early - which is
        # what made a 180->0 deg jump "arrive" one degree from where it started, only for
        # the stage to keep slewing during the settle sleep.
        if abs(angular_difference(start, raw)) > self.config.angle_tolerance_deg:
            self._wait_until_moving(start, deadline)

        while True:
            current = self.get_angle_deg()
            # Compared the short way round, so a move from 359 to 1 deg can arrive.
            within = abs(angular_difference(current, raw)) <= self.config.angle_tolerance_deg
            # The mount keeps a multi-turn absolute count (it can read 371 or -360 deg),
            # so a mod-360 position match alone can be satisfied a full turn early, while
            # the stage is still slewing. Require the controller to also confirm it has
            # stopped whenever it can report that; fall back to position-only otherwise.
            if within and self._is_moving() is not True:
                if self.config.settle_time_s > 0:
                    time.sleep(self.config.settle_time_s)
                # Re-check after settling: if the stage was still coasting when the
                # tolerance was first met it will have drifted, so keep waiting instead
                # of reporting whatever angle it happened to reach mid-slew.
                current = self.get_angle_deg()
                if (abs(angular_difference(current, raw)) <= self.config.angle_tolerance_deg
                        and self._is_moving() is not True):
                    break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Rotation stage did not reach {angle_deg:g} deg within "
                    f"{self.config.move_timeout_s:g} s (stopped at {current:.3f} deg). "
                    "Check for an obstruction, or that the stage was homed."
                )
            time.sleep(self.config.poll_interval_s)

        print(f"[stage] arrived at {current:.3f} deg")
        return current

    def _wait_until_moving(self, start_deg: float, deadline: float) -> None:
        """Block until the stage is observed to move, so the pre-move state is never
        mistaken for arrival.

        Returns as soon as the controller flags motion or the position has left the
        starting angle by more than the tolerance. Gives up after
        motion_start_timeout_s (a move too quick to catch, or one the controller
        refuses) so the arrival check still runs rather than blocking here forever.
        """
        grace = min(time.monotonic() + self.config.motion_start_timeout_s, deadline)
        while time.monotonic() < grace:
            if self._is_moving() is True:
                return
            if abs(angular_difference(self.get_angle_deg(), start_deg)) > self.config.angle_tolerance_deg:
                return
            time.sleep(self.config.poll_interval_s)

    def home(self) -> None:
        if self.dry_run:
            print("[stage] dry run: would home")
            self._simulated_angle = 0.0
            return
        self._require_connection()
        print("[stage] homing...")
        self._axis.Home(int(self.config.move_timeout_s * 1000))
        print(f"[stage] homed, now at {self.get_angle_deg():.3f} deg")

    # ---------- internals ----------

    def _require_connection(self) -> None:
        if self._axis is None:
            raise RuntimeError("Rotation stage is not connected: call connect() first.")

    def _is_moving(self) -> Optional[bool]:
        """Whether the controller reports the stage is still in motion.

        Returns None when Kinesis cannot report a motion flag, so callers fall back to
        the position-only check rather than block forever on an unknown status.
        """
        if self.dry_run or self._axis is None:
            return False
        try:
            status = self._axis.Status
        except Exception:
            return None
        known = False
        for attr in ("IsMoving", "IsInMotion", "IsJogging"):
            value = getattr(status, attr, None)
            if value is None:
                continue
            known = True
            if bool(value):
                return True
        return False if known else None

    def _wait_until_stopped(self, *, force: bool = False) -> None:
        """Block until the controller reports the stage has stopped moving.

        With force=True a final StopImmediate is issued if the stage is still moving at
        the timeout, so a stuck motion cannot wedge every following move.
        """
        if self.dry_run or self._axis is None:
            return
        deadline = time.monotonic() + self.config.move_timeout_s
        while self._is_moving() is True:
            if time.monotonic() > deadline:
                if force:
                    try:
                        self._axis.StopImmediate()
                    except Exception as exc:
                        print(f"[stage] warning: StopImmediate failed ({exc})")
                break
            time.sleep(self.config.poll_interval_s)

    def _apply_velocity(self) -> None:
        if self.config.max_velocity_deg_s is None and self.config.acceleration_deg_s2 is None:
            return
        params = self._axis.GetVelocityParams()
        if self.config.max_velocity_deg_s is not None:
            params.MaxVelocity = self._decimal(float(self.config.max_velocity_deg_s))
        if self.config.acceleration_deg_s2 is not None:
            params.Acceleration = self._decimal(float(self.config.acceleration_deg_s2))
        self._axis.SetVelocityParams(params)
        print(f"[stage] velocity {params.MaxVelocity} deg/s, acceleration {params.Acceleration} deg/s^2")

    def _load_kinesis(self, serial: str):
        """Import the Kinesis assemblies for this controller family (lazily, so the
        rest of the pipeline runs on machines without Kinesis)."""
        kinesis_path = Path(self.config.kinesis_path)
        if kinesis_path.is_dir() and str(kinesis_path) not in sys.path:
            sys.path.append(str(kinesis_path))

        try:
            import clr
        except ImportError as exc:
            raise ImportError(_PYTHONNET_HINT) from exc

        device_type, assembly, class_name = resolve_device_class(serial, self.config.device_type)
        try:
            clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
            clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
            clr.AddReference(assembly)
            from System import Decimal
            from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI

            device_class = getattr(importlib.import_module(assembly), class_name)
        except Exception as exc:
            raise ImportError(
                _KINESIS_HINT.format(assembly=assembly, path=kinesis_path, known=sorted(DEVICE_CLASSES))
            ) from exc

        return device_type, device_class, class_name, Decimal, DeviceManagerCLI


# ============================================================================
# PART 2 - linear polarization from a measured lookup table
# ============================================================================
#
# The table lives in `polarization_calibration/` and is REUSED across runs: recording
# one takes hours, and it only goes stale when the beam path changes. Record a new one
# with `Polarization_calibration_v20_linear_only.py` (menu option `c`) only then.
#
# No polarization model is involved. To get a linear polarization at a given angle and
# power, P1 turns to that angle and the HWP angle is read off the measured table.
#
# Three conventions must match the calibration script, or the table means nothing:
#
# * P1 = 0 deg is vertical, and the table's P1 axis is the requested polarization angle;
# * angles are raw mount angles, `(sign * angle) % 360`, with the sign set by
#   `RotationStageConfig.clockwise` (the calibration script uses clockwise for both);
# * the table covers P1 in [0, 180) because a linear polarization repeats every
#   180 deg, so the HWP setting for 190 deg is the one measured at 10 deg.

LATEST_CALIBRATION_NAME = "linear_polarization_lookup_latest.npz"
CALIBRATION_GLOB = "linear_polarization_lookup_*.npz"
SCAN_LOG_NAME = "polarization_scan.csv"

SCAN_LOG_COLUMNS = [
    "polarization_angle_deg",
    "power_label",
    "requested_power",
    "unit",
    "p1_angle_deg",
    "hwp_angle_deg",
    "reachable_min",
    "reachable_max",
    "status",
    "chunks",
    "duration_s",
    "recorded_at",
    "note",
]


def wrap_180(angle_deg: float) -> float:
    """A polarization direction in [0, 180): 190 deg and 10 deg are the same state."""
    return float(angle_deg) % 180.0


# ----------------------------------------------------------------------------
# The calibration table
# ----------------------------------------------------------------------------

@dataclass
class LookupTable:
    """A measured power map: `power[i, j]` at P1 `p1_angles[i]`, HWP `hwp_angles[j]`."""

    p1_angles: np.ndarray
    hwp_angles: np.ndarray
    power: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def unit(self) -> str:
        return str(self.metadata.get("unit", "a.u."))

    def describe(self) -> str:
        return (
            f"{self.path.name if self.path else 'lookup table'}: "
            f"{len(self.p1_angles)} P1 angles ({self.p1_angles[0]:g} to {self.p1_angles[-1]:g} deg) "
            f"x {len(self.hwp_angles)} HWP angles ({self.hwp_angles[0]:g} to {self.hwp_angles[-1]:g} deg), "
            f"{np.nanmin(self.power):g} to {np.nanmax(self.power):g} {self.unit}"
            + (f", recorded {self.metadata['timestamp']}" if "timestamp" in self.metadata else "")
        )

    def power_curve(self, angle_deg: float) -> np.ndarray:
        """Power versus HWP angle at one polarization angle, interpolated along P1.

        The P1 axis is 180 deg periodic, so the first row is reused as the row at
        +180 deg to interpolate across the seam.
        """
        target = wrap_180(angle_deg)
        order = np.argsort(self.p1_angles)
        p1 = self.p1_angles[order]
        grid = self.power[order, :]

        inside = p1 < 180.0 - 1e-9  # a row at exactly 180 deg duplicates the one at 0
        if np.any(inside):
            p1, grid = p1[inside], grid[inside, :]

        p1_periodic = np.concatenate([p1, [p1[0] + 180.0]])
        grid_periodic = np.vstack([grid, grid[0:1, :]])

        curve = np.full(grid_periodic.shape[1], np.nan)
        for j in range(grid_periodic.shape[1]):
            column = grid_periodic[:, j]
            finite = np.isfinite(column)
            if finite.sum() >= 2:
                curve[j] = np.interp(target, p1_periodic[finite], column[finite])
        return curve

    def hwp_angle_for(self, angle_deg: float, power: float,
                      tolerance: float = 1e-9) -> Tuple[Optional[float], float, float]:
        """(HWP angle, reachable min, reachable max) for a polarization and a power.

        The HWP angle is the *smallest* one whose interpolated power equals the
        request, matching the calibration script's choice. It is None when the
        request lies outside what the table reaches at this polarization angle.
        """
        curve = self.power_curve(angle_deg)
        finite = np.isfinite(self.hwp_angles) & np.isfinite(curve)
        hwp, values = self.hwp_angles[finite], curve[finite]
        order = np.argsort(hwp)
        hwp, values = hwp[order], values[order]

        if len(values) < 2:
            return None, float("nan"), float("nan")

        low, high = float(np.min(values)), float(np.max(values))
        if power < low - tolerance or power > high + tolerance:
            return None, low, high

        crossings = [float(a) for a in hwp[np.isclose(values, power, rtol=0.0, atol=tolerance)]]
        for j in range(len(hwp) - 1):
            before, after = values[j] - power, values[j + 1] - power
            if before == 0 or after == 0:
                continue
            if before * after < 0:  # the curve crosses the requested power in between
                fraction = (power - values[j]) / (values[j + 1] - values[j])
                crossings.append(float(hwp[j] + fraction * (hwp[j + 1] - hwp[j])))

        if not crossings:  # the request equals the min or the max only numerically
            nearest = int(np.nanargmin(np.abs(values - power)))
            if abs(values[nearest] - power) <= tolerance:
                crossings.append(float(hwp[nearest]))

        if not crossings:
            return None, low, high
        return min(crossings), low, high


def load_lookup(path: Path) -> LookupTable:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No polarization lookup table at {path}. Reuse one of the tables in "
            "polarization_calibration/, or record a new one with "
            "Polarization_calibration_v20_linear_only.py (option 'c')."
        )
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
    return LookupTable(
        p1_angles=np.asarray(data["p1_angles"], dtype=float),
        hwp_angles=np.asarray(data["hwp_angles"], dtype=float),
        power=np.asarray(data["power_grid"], dtype=float),
        metadata=metadata,
        path=path,
    )


def find_calibration(directory: Path) -> Path:
    """The `_latest` table of a calibration folder, or its most recent file."""
    directory = Path(directory)
    latest = directory / LATEST_CALIBRATION_NAME
    if latest.is_file():
        return latest
    candidates = sorted(directory.glob(CALIBRATION_GLOB), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No {CALIBRATION_GLOB} in {directory}. Point POLARIZATION_CALIBRATION at an "
            "existing table, or record a new one with the calibration script."
        )
    return candidates[-1]


# ----------------------------------------------------------------------------
# Setting a polarization
# ----------------------------------------------------------------------------

@dataclass
class Setting:
    """What was asked for, and what the mounts were set to."""

    angle_deg: float
    requested_power: float
    unit: str
    p1_angle_deg: Optional[float]
    hwp_angle_deg: Optional[float]
    reachable_min: float
    reachable_max: float

    @property
    def reachable(self) -> bool:
        return self.hwp_angle_deg is not None

    def describe(self) -> str:
        if not self.reachable:
            return (f"{self.angle_deg:g} deg: {self.requested_power:g} {self.unit} out of reach "
                    f"({self.reachable_min:g} to {self.reachable_max:g} {self.unit})")
        return (f"{self.angle_deg:g} deg: P1 {self.p1_angle_deg:g} deg, "
                f"HWP {self.hwp_angle_deg:g} deg for {self.requested_power:g} {self.unit}")


class PolarizationController:
    """The half-wave plate and the polarizer, driven from the lookup table.

    `dry_run=True` connects to nothing and moves nothing, but still reads the table
    and reports the angles it would have used - which is what makes a scan
    rehearsable, and the reachability of every angle checkable, without hardware.
    """

    def __init__(self, lookup: LookupTable, p1: RotationStageConfig, hwp: RotationStageConfig,
                 dry_run: bool = False, settle_time_s: float = 0.2) -> None:
        self.lookup = lookup
        self.settle_time_s = settle_time_s
        self.p1 = RotationStage(p1, dry_run=dry_run)
        self.hwp = RotationStage(hwp, dry_run=dry_run)
        self.dry_run = dry_run

    def __enter__(self) -> "PolarizationController":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.disconnect()
        return False

    def connect(self) -> None:
        print(f"[pol] {self.lookup.describe()}")
        self.p1.connect()
        try:
            self.hwp.connect()
        except Exception:
            self.p1.disconnect()  # never leave one mount of the pair open
            raise

    def disconnect(self) -> None:
        try:
            self.p1.disconnect()
        finally:
            self.hwp.disconnect()

    def plan(self, angle_deg: float, power: float) -> Setting:
        """Which mount angles a polarization would need. Moves nothing."""
        hwp_angle, low, high = self.lookup.hwp_angle_for(angle_deg, power)
        return Setting(
            angle_deg=float(angle_deg),
            requested_power=float(power),
            unit=self.lookup.unit,
            # P1 carries the requested angle over the full turn: 190 deg is the same
            # polarization as 10 deg but a different mount position, which is what makes
            # a 0-360 scan a repeatability check rather than a copy.
            p1_angle_deg=wrap_360(angle_deg) if hwp_angle is not None else None,
            hwp_angle_deg=hwp_angle,
            reachable_min=low,
            reachable_max=high,
        )

    def set_polarization(self, angle_deg: float, power: float) -> Setting:
        """Move P1 then the HWP. An unreachable power moves nothing."""
        setting = self.plan(angle_deg, power)
        if not setting.reachable:
            print(f"[pol] {setting.describe()} - nothing moved")
            return setting

        print(f"[pol] {setting.describe()}")
        # P1 first: it defines the polarization angle, the HWP then sets the power.
        self.p1.move_to_angle_deg(setting.p1_angle_deg)
        self.hwp.move_to_angle_deg(setting.hwp_angle_deg)
        if self.settle_time_s > 0 and not self.dry_run:
            time.sleep(self.settle_time_s)
        return setting


def preflight(lookup: LookupTable, angles_deg: Sequence[float], power: float) -> List[Setting]:
    """Check every angle of a scan against the table before the run starts."""
    settings = [
        Setting(
            angle_deg=float(angle),
            requested_power=float(power),
            unit=lookup.unit,
            p1_angle_deg=wrap_360(angle),
            hwp_angle_deg=None,
            reachable_min=float("nan"),
            reachable_max=float("nan"),
        )
        for angle in angles_deg
    ]
    for setting in settings:
        hwp_angle, low, high = lookup.hwp_angle_for(setting.angle_deg, power)
        setting.hwp_angle_deg = hwp_angle
        setting.reachable_min, setting.reachable_max = low, high
        if hwp_angle is None:
            setting.p1_angle_deg = None

    unreachable = [s for s in settings if not s.reachable]
    print(f"[pol] preflight: {len(settings) - len(unreachable)}/{len(settings)} angles can deliver "
          f"{power:g} {lookup.unit}")
    for setting in unreachable:
        print(f"  skipped - {setting.describe()}")
    if unreachable:
        print("  Lower the requested power to keep those angles, or accept the gaps.")
    return settings


# ----------------------------------------------------------------------------
# The per-angle log
# ----------------------------------------------------------------------------

class ScanLog:
    """`polarization_scan.csv`: one row per angle of the scan.

    A polarization scan is dozens of acquisitions, so the per-angle record is a table
    rather than dozens of sections in `experiment_config.txt` (which keeps holding the
    setup, shared by the whole folder). Re-running an angle replaces its row.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            with self.path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.rows[row.get("power_label", "")] = row

    def record(self, label: str, setting: Setting, status: str,
               result: Any = None, note: str = "") -> Path:
        self.rows[label] = {
            "polarization_angle_deg": f"{setting.angle_deg:g}",
            "power_label": label,
            "requested_power": f"{setting.requested_power:g}",
            "unit": setting.unit,
            "p1_angle_deg": "" if setting.p1_angle_deg is None else f"{setting.p1_angle_deg:g}",
            "hwp_angle_deg": "" if setting.hwp_angle_deg is None else f"{setting.hwp_angle_deg:g}",
            "reachable_min": f"{setting.reachable_min:g}",
            "reachable_max": f"{setting.reachable_max:g}",
            "status": status,
            "chunks": str(len(result.chunk_paths)) if result is not None else "",
            "duration_s": f"{result.duration_s:g}" if result is not None else "",
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": note,
        }
        return self.write()

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.rows.values(), key=lambda row: float(row["polarization_angle_deg"]))
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCAN_LOG_COLUMNS)
            writer.writeheader()
            for row in ordered:
                writer.writerow({key: row.get(key, "") for key in SCAN_LOG_COLUMNS})
        return self.path

    def recorded_labels(self, power: Optional[str] = None) -> List[Tuple[float, str]]:
        """(angle, label) of the angles that actually hold data, in angular order.

        `power` restricts to one campaign (labels starting with `{power}_pol`).
        """
        rows = []
        for row in self.rows.values():
            if row.get("status") != "complete":
                continue
            label = row.get("power_label", "")
            if power is not None and not label.startswith(f"{power}_pol"):
                continue
            rows.append(row)
        return [
            (float(row["polarization_angle_deg"]), row["power_label"])
            for row in sorted(rows, key=lambda r: float(r["polarization_angle_deg"]))
        ]
