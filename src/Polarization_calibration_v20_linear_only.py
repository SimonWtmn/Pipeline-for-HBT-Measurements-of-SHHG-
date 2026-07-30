# -*- coding: utf-8 -*-
"""
Linear-polarization lookup calibration/control

Setup assumed:
    initial almost-linear polarization -> non-ideal HWP -> rotating polarizer P1 -> optional P2 -> powermeter / timetagger

Main idea:
    No Jones/Malus model is used for setting the polarization.
    A full measured 2D lookup table P(P1_angle, HWP_angle) is recorded.
    To create a requested linear polarization:
        1. P1 is rotated to the requested linear-polarization angle.
           Convention: P1 = 0 deg is vertical in the P1 stage reference frame.
        2. The HWP angle is found by interpolation of the measured 2D calibration table.
        3. If several HWP angles give the requested power, the minimal HWP angle is chosen.
        4. If the requested power is outside the measured range for this P1 angle, no motion is made.

Menu:
    c   - Full 2D lookup calibration: scan P1 and HWP, save data
    s   - Scan Stage (Interactive); asks powermeter [default] or timetagger [type t]
    r   - Rotate Stage (Interactive)
    v   - Set Vertical Polarization
    p   - Set Custom Polarization (Interactive): asks angle + power
    b   - Measure Background
    max - Measure max power
    h   - Home All Stages
    he  - Home Elliptec stages only, i.e. only stages with integer IDs
    q   - Quit
"""

import atexit
import csv
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Hardware drivers used in the original script ---
import thorlabs_rotation_stage_v2 as trs
import thorlabs_powermeter as tpm

# Tektronix is intentionally not used here. The requested scan choices are
# powermeter or timetagger, but the import can be restored if needed later.


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# The folder the pipeline reads its lookup tables from, so a new calibration is
# reusable straight away. Absolute, so it does not depend on the launch directory.
DATA_DIR = Path(__file__).resolve().parent.parent / "polarization_calibration"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CALIBRATION_FILE = DATA_DIR / "linear_polarization_lookup_latest.npz"
LATEST_METADATA_FILE = DATA_DIR / "linear_polarization_lookup_latest_metadata.json"
BACKGROUND_FILE = DATA_DIR / "background_levels.json"
MAX_POWER_FILE = DATA_DIR / "max_power_measurements.csv"

# Default full 2D lookup scan. Change these if a 1-degree full scan is too slow.
P1_SCAN_START_DEG = 0.0
P1_SCAN_STOP_DEG = 180.0
P1_SCAN_STEP_DEG = 5.0
HWP_SCAN_START_DEG = 0.0
HWP_SCAN_STOP_DEG = 90.0
HWP_SCAN_STEP_DEG = 3.0

# Stage metadata. Angles in this script are RAW stage-frame angles, not optical
# zero-offset-corrected angles. This is deliberate: the lookup table absorbs
# non-ideal HWP behavior and the unknown HWP reference frame.
STAGE_METADATA = {
    "HWP": {
        "id": "27270567",      # Kinesis/Thorlabs serial: not Elliptec
        "desc": "Half-wave plate",
        "clockwise": True,
    },
    "P1": {
        "id": "27270550",      # Kinesis/Thorlabs serial: not Elliptec
        "desc": "Rotating output polarizer; 0 deg = vertical",
        "clockwise": True,
    },
    "P2": {
        "id": 0,               # Elliptec ID: integer
        "desc": "Optional analyzer / kept for compatibility",
        "clockwise": True,
    },
}

HARDWARE_CONFIG = {
    "powermeter": {
        "wavelength_nm": 2121,
        "unit": "mW",
        "n_averages": 4,
        "stabilization_delay":5.20,
    },
    "timetagger": {
        "channel": 4,
        "integration_time_s": 0.5,
        "n_averages": 1,
        "trigger_level_v": 0.5,
        "stabilization_delay": 0.01,
    },
}

CURRENT_DEVICE = "powermeter"       # "powermeter" or "timetagger"
_timetagger_instance = None
_background_levels: Dict[str, float] = {}


# =============================================================================
# 2. SMALL UTILITIES
# =============================================================================

def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def wrap_180(angle_deg: float) -> float:
    """Wrap an optical/polarizer angle into [0, 180)."""
    return float(angle_deg) % 180.0


def stage_names() -> Tuple[str, ...]:
    return tuple(STAGE_METADATA.keys())


def resolve_stage(stage_identifier: Union[str, int]) -> Tuple[str, Union[str, int]]:
    """Return (stage_name, stage_id) from a stage name or hardware ID."""
    s = str(stage_identifier).strip()
    for name, meta in STAGE_METADATA.items():
        if s.upper() == name.upper() or s == str(meta["id"]):
            return name, meta["id"]
    raise ValueError(f"Unknown stage {stage_identifier!r}. Available stages: {list(STAGE_METADATA)}")


def angle_array(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive angle array, robust against floating-point roundoff."""
    if step <= 0:
        raise ValueError("Step must be positive.")
    n = int(np.floor((stop - start) / step + 0.5)) + 1
    arr = start + step * np.arange(n, dtype=float)
    arr = arr[arr <= stop + 1e-9]
    if len(arr) == 0 or abs(arr[-1] - stop) > 1e-9:
        arr = np.append(arr, stop)
    return arr.astype(float)


def get_unit() -> str:
    if CURRENT_DEVICE == "powermeter":
        return HARDWARE_CONFIG["powermeter"]["unit"]
    if CURRENT_DEVICE == "timetagger":
        return "Hz"
    return "a.u."


def load_background_levels() -> Dict[str, float]:
    if BACKGROUND_FILE.exists():
        try:
            with open(BACKGROUND_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {str(k): float(v) for k, v in data.items()}
        except Exception as exc:
            print(f"⚠️ Could not load {BACKGROUND_FILE}: {exc}")
    return {"powermeter": 0.0, "timetagger": 0.0}


def save_background_levels() -> None:
    with open(BACKGROUND_FILE, "w", encoding="utf-8") as f:
        json.dump(_background_levels, f, indent=2)


def safe_float_input(prompt: str, default: Optional[float] = None) -> Optional[float]:
    suffix = "" if default is None else f" [{default:g}]"
    text = input(prompt + suffix + ": ").strip()
    if not text and default is not None:
        return float(default)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        print("❌ Invalid number.")
        return None


# =============================================================================
# 3. HARDWARE CONNECTION / CLEANUP
# =============================================================================

def connect_stages() -> None:
    ids = [meta["id"] for meta in STAGE_METADATA.values()]
    print(f"Connecting rotation stages: {ids}")
    trs.connect_rotation_stages(ids)


def connect_measurement_device(device: Optional[str] = None) -> bool:
    """Connect selected measurement device. Returns True if successful."""
    global CURRENT_DEVICE, _timetagger_instance
    if device is not None:
        CURRENT_DEVICE = device.lower().strip()

    if CURRENT_DEVICE == "powermeter":
        try:
            print("Connecting powermeter...")
            tpm.connect_powermeter(wavelength=HARDWARE_CONFIG["powermeter"]["wavelength_nm"])
            print("✓ Powermeter connected")
            return True
        except Exception as exc:
            print(f"⚠️ Powermeter connection failed: {exc}")
            return False

    if CURRENT_DEVICE == "timetagger":
        try:
            import TimeTagger as tt
            print("Connecting TimeTagger...")
            if _timetagger_instance is None:
                _timetagger_instance = tt.createTimeTagger()
            ch = HARDWARE_CONFIG["timetagger"]["channel"]
            lvl = HARDWARE_CONFIG["timetagger"]["trigger_level_v"]
            _timetagger_instance.setTriggerLevel(ch, lvl)
            print(f"✓ TimeTagger connected: channel {ch}, trigger {lvl:.3f} V")
            return True
        except Exception as exc:
            print(f"⚠️ TimeTagger connection failed: {exc}")
            return False

    print(f"❌ Unknown measurement device: {CURRENT_DEVICE}")
    return False


def connect_all() -> None:
    print("\n" + "=" * 70)
    print("CONNECTING HARDWARE")
    print("=" * 70)
    try:
        connect_stages()
    except Exception as exc:
        print(f"⚠️ Stage connection warning: {exc}")
    connect_measurement_device(CURRENT_DEVICE)
    print("=" * 70)


def cleanup_all_devices() -> None:
    print("\nCleaning up hardware connections...")
    try:
        trs.disconnect_all()
        print("✓ Rotation stages disconnected")
    except Exception as exc:
        print(f"Stage disconnect warning: {exc}")

    try:
        if CURRENT_DEVICE == "powermeter":
            tpm.disconnect()
            print("✓ Powermeter disconnected")
        elif CURRENT_DEVICE == "timetagger":
            global _timetagger_instance
            if _timetagger_instance is not None:
                import TimeTagger as tt
                tt.freeTimeTagger(_timetagger_instance)
                _timetagger_instance = None
                print("✓ TimeTagger disconnected")
    except Exception as exc:
        print(f"Measurement-device disconnect warning: {exc}")


def signal_handler(sig, frame):
    print("\n\n⚠️ Interrupted by user")
    cleanup_all_devices()
    sys.exit(0)


atexit.register(cleanup_all_devices)
signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# 4. STAGE MOTION AND MEASUREMENT
# =============================================================================

def rotate_stage(stage_identifier: Union[str, int], angle_deg: float) -> float:
    """
    Rotate to a RAW hardware angle, wrapped into [0, 360).

    No calibration offset or physical model is applied. For this lookup method,
    raw stage coordinates are the reference frame.
    """
    name, serial = resolve_stage(stage_identifier)
    meta = STAGE_METADATA[name]
    sign = +1.0 if meta.get("clockwise", True) else -1.0
    raw_angle = (sign * float(angle_deg)) % 360.0
    trs.rotate_stage(raw_angle, serial=serial)
    return raw_angle


def get_stage_position(stage_identifier: Union[str, int]) -> Optional[float]:
    name, serial = resolve_stage(stage_identifier)
    try:
        return float(trs.get_position(serial)) % 360.0
    except Exception as exc:
        print(f"⚠️ Could not read {name} position: {exc}")
        return None


def measure_intensity(n_avg: Optional[int] = None, verbose: bool = False) -> float:
    """Measure intensity and subtract the stored background for the active device."""
    device = CURRENT_DEVICE

    if device == "powermeter":
        cfg = HARDWARE_CONFIG["powermeter"]
        delay = float(cfg["stabilization_delay"])
        avg = int(n_avg or cfg["n_averages"])
    elif device == "timetagger":
        cfg = HARDWARE_CONFIG["timetagger"]
        delay = float(cfg["stabilization_delay"])
        avg = int(n_avg or cfg["n_averages"])
    else:
        print(f"❌ Unknown measurement device: {device}")
        return float("nan")

    if verbose:
        print(f"Waiting {delay:g} s, measuring with {device}...", end="", flush=True)
    time.sleep(delay)

    raw = float("nan")
    try:
        if device == "powermeter":
            raw = float(tpm.read_power(unit=cfg["unit"], n_measurements=avg))
        elif device == "timetagger":
            import TimeTagger as tt
            if _timetagger_instance is None:
                raise RuntimeError("TimeTagger is not connected.")
            ch = int(cfg["channel"])
            duration_ps = int(float(cfg["integration_time_s"]) * 1e12)
            vals = []
            for _ in range(avg):
                cr = tt.Countrate(_timetagger_instance, channels=[ch])
                cr.startFor(duration_ps)
                cr.waitUntilFinished()
                vals.append(float(cr.getData()[0]))
                del cr
            raw = float(np.mean(vals))
    except Exception as exc:
        print(f"\n❌ Measurement failed: {exc}")
        return float("nan")

    bg = float(_background_levels.get(device, 0.0))
    val = raw - bg
    if verbose:
        print(f" {val:.6g} {get_unit()}  (raw={raw:.6g}, bg={bg:.6g})")
    return val


def laser_on() -> None:
    input("\n👉 Please UNBLOCK the laser, then press Enter...")


def laser_off() -> None:
    input("\n👉 Please BLOCK the laser, then press Enter...")


def measure_background() -> float:
    print("\n" + "=" * 70)
    print(f"BACKGROUND MEASUREMENT ({CURRENT_DEVICE})")
    print("=" * 70)
    laser_off()

    saved_bg = float(_background_levels.get(CURRENT_DEVICE, 0.0))
    _background_levels[CURRENT_DEVICE] = 0.0
    bg = measure_intensity(verbose=True)
    _background_levels[CURRENT_DEVICE] = saved_bg

    if np.isfinite(bg):
        _background_levels[CURRENT_DEVICE] = float(bg)
        save_background_levels()
        print(f"✓ Background saved: {bg:.6g} {get_unit()} for {CURRENT_DEVICE}")
    else:
        print("❌ Background measurement failed; old background kept.")

    laser_on()
    return bg


# =============================================================================
# 5. 2D LOOKUP CALIBRATION
# =============================================================================

def save_lookup_calibration(base_path: Path,
                            p1_angles: np.ndarray,
                            hwp_angles: np.ndarray,
                            power_grid: np.ndarray,
                            metadata: Dict) -> None:
    """Save compressed NPZ, JSON metadata, long CSV, and update latest links/copies."""
    npz_path = base_path.with_suffix(".npz")
    json_path = base_path.with_suffix(".json")
    csv_path = base_path.with_suffix(".csv")

    np.savez_compressed(
        npz_path,
        p1_angles=np.asarray(p1_angles, dtype=float),
        hwp_angles=np.asarray(hwp_angles, dtype=float),
        power_grid=np.asarray(power_grid, dtype=float),
        metadata_json=json.dumps(metadata),
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    rows = []
    for i, p1 in enumerate(p1_angles):
        for j, hwp in enumerate(hwp_angles):
            rows.append({
                "P1_angle_deg": float(p1),
                "HWP_angle_deg": float(hwp),
                "intensity": float(power_grid[i, j]),
                "unit": metadata.get("unit", get_unit()),
                "device": metadata.get("device", CURRENT_DEVICE),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    shutil.copy2(npz_path, LATEST_CALIBRATION_FILE)
    shutil.copy2(json_path, LATEST_METADATA_FILE)

    print(f"✓ Saved lookup calibration: {npz_path}")
    print(f"✓ Saved CSV copy:           {csv_path}")
    print(f"✓ Updated latest file:      {LATEST_CALIBRATION_FILE}")


def full_2d_lookup_calibration() -> Optional[Dict]:
    """Scan P1 and HWP over the requested 2D grid and save lookup data."""
    print("\n" + "=" * 70)
    print("FULL 2D LINEAR-POLARIZATION LOOKUP CALIBRATION")
    print("=" * 70)
    print("This will measure intensity for every P1/HWP pair.")
    print("P1 defines the requested output linear-polarization angle: P1=0° is vertical.")
    print("No model is fitted; the measured table is used directly for interpolation.")
    print("=" * 70)

    p1_step = safe_float_input("P1 step in degrees", P1_SCAN_STEP_DEG)
    hwp_step = safe_float_input("HWP step in degrees", HWP_SCAN_STEP_DEG)
    if p1_step is None or hwp_step is None:
        return None

    p1_angles = angle_array(P1_SCAN_START_DEG, P1_SCAN_STOP_DEG, p1_step)
    hwp_angles = angle_array(HWP_SCAN_START_DEG, HWP_SCAN_STOP_DEG, hwp_step)

    n_points = len(p1_angles) * len(hwp_angles)
    print(f"\nGrid: {len(p1_angles)} P1 angles × {len(hwp_angles)} HWP angles = {n_points} points")
    print(f"Device: {CURRENT_DEVICE} ({get_unit()})")
    ans = input("Proceed with full 2D calibration? (y/n) [n]: ").strip().lower()
    if ans != "y":
        print("Calibration cancelled.")
        return None

    comment = input("Optional comment for metadata: ").strip()
    base = DATA_DIR / f"linear_polarization_lookup_{timestamp_now()}"
    temp_npz = base.with_name(base.name + "_PARTIAL.npz")

    power_grid = np.full((len(p1_angles), len(hwp_angles)), np.nan, dtype=float)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "comment": comment,
        "device": CURRENT_DEVICE,
        "unit": get_unit(),
        "background_subtracted": True,
        "background": float(_background_levels.get(CURRENT_DEVICE, 0.0)),
        "p1_reference": "0 deg = vertical in P1 stage reference frame",
        "hwp_reference": "raw HWP stage frame; no model/offset used",
        "stage_metadata": STAGE_METADATA,
        "p1_step_deg": float(p1_step),
        "hwp_step_deg": float(hwp_step),
    }

    try:
        for i, p1 in enumerate(p1_angles):
            print("\n" + "-" * 70)
            print(f"P1 row {i + 1}/{len(p1_angles)}: P1 = {p1:.3f}°")
            rotate_stage("P1", p1)
            time.sleep(0.1)

            # Serpentine HWP scan to reduce mechanical travel.
            if i % 2 == 0:
                hwp_order = range(len(hwp_angles))
            else:
                hwp_order = range(len(hwp_angles) - 1, -1, -1)

            for count, j in enumerate(hwp_order, start=1):
                hwp = hwp_angles[j]
                rotate_stage("HWP", hwp)
                val = measure_intensity(verbose=False)
                power_grid[i, j] = val

                if count == 1 or count % 10 == 0 or count == len(hwp_angles):
                    print(
                        f"  HWP step {count:4d}/{len(hwp_angles)}: "
                        f"HWP={hwp:8.3f}°, I={val:12.6g} {get_unit()}"
                    )

            # Save partial file after each P1 row.
            np.savez_compressed(
                temp_npz,
                p1_angles=p1_angles,
                hwp_angles=hwp_angles,
                power_grid=power_grid,
                metadata_json=json.dumps(metadata),
            )
            print(f"  Partial backup updated: {temp_npz}")

    except KeyboardInterrupt:
        print("\n⚠️ Calibration interrupted. Keeping partial backup.")
        return None

    save_lookup_calibration(base, p1_angles, hwp_angles, power_grid, metadata)
    if temp_npz.exists():
        temp_npz.unlink()

    plot_lookup_calibration(p1_angles, hwp_angles, power_grid, title="2D lookup calibration")

    return {
        "p1_angles": p1_angles,
        "hwp_angles": hwp_angles,
        "power_grid": power_grid,
        "metadata": metadata,
    }


def load_lookup_calibration(path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    cal_path = Path(path) if path is not None else LATEST_CALIBRATION_FILE
    if not cal_path.exists():
        raise FileNotFoundError(
            f"No lookup calibration found at {cal_path}. Run menu option 'c' first."
        )
    data = np.load(cal_path, allow_pickle=False)
    p1_angles = np.asarray(data["p1_angles"], dtype=float)
    hwp_angles = np.asarray(data["hwp_angles"], dtype=float)
    power_grid = np.asarray(data["power_grid"], dtype=float)
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    return p1_angles, hwp_angles, power_grid, metadata


def plot_lookup_calibration(p1_angles: np.ndarray,
                            hwp_angles: np.ndarray,
                            power_grid: np.ndarray,
                            title: str = "Linear polarization lookup") -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        power_grid,
        origin="lower",
        aspect="auto",
        extent=[hwp_angles[0], hwp_angles[-1], p1_angles[0], p1_angles[-1]],
    )
    ax.set_xlabel("HWP raw angle (deg)")
    ax.set_ylabel("P1 angle / output linear polarization angle (deg)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(get_unit())
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. LOOKUP INTERPOLATION AND SETTING POLARIZATION
# =============================================================================

def interpolate_power_curve_at_p1(target_p1_angle: float,
                                  p1_angles: np.ndarray,
                                  power_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate the measured 2D grid along the P1 axis.

    The P1 angle is 180-degree periodic for polarization direction. If a row at
    180 degrees exists, it is treated as equivalent to 0 degrees for interpolation.
    """
    target = wrap_180(target_p1_angle)
    p1 = np.asarray(p1_angles, dtype=float)
    grid = np.asarray(power_grid, dtype=float)

    order = np.argsort(p1)
    p1 = p1[order]
    grid = grid[order, :]

    # Remove 180 deg duplicate for periodic interpolation, if present.
    keep = p1 < 180.0 - 1e-9
    if np.any(keep):
        p1_base = p1[keep]
        grid_base = grid[keep, :]
    else:
        p1_base = p1
        grid_base = grid

    # Periodic closure: append first row at +180 deg.
    p1_ext = np.concatenate([p1_base, [p1_base[0] + 180.0]])
    grid_ext = np.vstack([grid_base, grid_base[0:1, :]])

    curve = np.empty(grid_ext.shape[1], dtype=float)
    for j in range(grid_ext.shape[1]):
        y = grid_ext[:, j]
        finite = np.isfinite(y)
        if finite.sum() < 2:
            curve[j] = np.nan
        else:
            curve[j] = np.interp(target, p1_ext[finite], y[finite])
    return curve


def find_min_hwp_for_power(requested_power: float,
                           hwp_angles: np.ndarray,
                           power_curve: np.ndarray,
                           tolerance: float = 1e-12) -> Tuple[Optional[float], Dict]:
    """
    Find the smallest HWP angle at which the interpolated HWP curve reaches the
    requested power. Crossings are linearly interpolated between measured points.
    """
    hwp = np.asarray(hwp_angles, dtype=float)
    y = np.asarray(power_curve, dtype=float)

    finite = np.isfinite(hwp) & np.isfinite(y)
    hwp = hwp[finite]
    y = y[finite]

    order = np.argsort(hwp)
    hwp = hwp[order]
    y = y[order]

    info = {
        "available_min": float(np.nanmin(y)) if len(y) else float("nan"),
        "available_max": float(np.nanmax(y)) if len(y) else float("nan"),
        "n_crossings": 0,
        "crossings": [],
    }

    if len(y) < 2:
        return None, info

    pmin = info["available_min"]
    pmax = info["available_max"]
    if requested_power < pmin - tolerance or requested_power > pmax + tolerance:
        return None, info

    candidates = []

    # Exact grid-point hits.
    exact = np.where(np.isclose(y, requested_power, rtol=0.0, atol=max(tolerance, 1e-9)))[0]
    for idx in exact:
        candidates.append(float(hwp[idx]))

    # Linear crossings between adjacent points.
    for j in range(len(hwp) - 1):
        y0 = y[j]
        y1 = y[j + 1]
        x0 = hwp[j]
        x1 = hwp[j + 1]
        d0 = y0 - requested_power
        d1 = y1 - requested_power

        if d0 == 0 or d1 == 0:
            continue
        if d0 * d1 < 0:
            frac = (requested_power - y0) / (y1 - y0)
            candidates.append(float(x0 + frac * (x1 - x0)))

    # If requested value is numerically equal to pmin/pmax but no crossing was
    # collected due to tolerance, choose the closest grid point.
    if not candidates:
        idx = int(np.nanargmin(np.abs(y - requested_power)))
        if abs(y[idx] - requested_power) <= max(tolerance, 1e-9):
            candidates.append(float(hwp[idx]))

    candidates = sorted(set(round(c, 10) for c in candidates))
    info["n_crossings"] = len(candidates)
    info["crossings"] = [float(c) for c in candidates]

    if not candidates:
        return None, info
    return float(candidates[0]), info


def set_linear_polarization(angle: float = 0.0,
                            power: float = 50.0,
                            measure_after: bool = False,
                            calibration_path: Optional[Union[str, Path]] = None) -> Optional[Dict]:
    """
    Set arbitrary linear polarization with requested power using the measured lookup table.

    Parameters
    ----------
    angle : float
        Linear-polarization angle in the P1 stage reference frame.
        0 deg = vertical. Internally wrapped into [0, 180).
    power : float
        Requested transmitted power/intensity in the calibration unit.
        For powermeter calibration this is mW; for timetagger calibration this is Hz.
    measure_after : bool
        If True, measure and print the final intensity after motion.
    calibration_path : str or Path or None
        Optional specific .npz lookup file. If None, use latest calibration.
    """
    requested_angle = wrap_180(float(angle))
    requested_power = float(power)

    if requested_power < 0:
        print("❌ Requested power must be non-negative.")
        return None

    try:
        p1_angles, hwp_angles, power_grid, metadata = load_lookup_calibration(calibration_path)
    except Exception as exc:
        print(f"❌ Cannot set polarization: {exc}")
        return None

    curve = interpolate_power_curve_at_p1(requested_angle, p1_angles, power_grid)
    hwp_angle, info = find_min_hwp_for_power(requested_power, hwp_angles, curve)

    unit = metadata.get("unit", get_unit())
    device = metadata.get("device", "unknown")

    print("\n" + "=" * 70)
    print("SET LINEAR POLARIZATION FROM LOOKUP TABLE")
    print("=" * 70)
    print(f"Requested angle: {requested_angle:.6f}°  (P1 frame; 0° = vertical)")
    print(f"Requested power: {requested_power:.6g} {unit}")
    print(f"Calibration device/unit: {device} / {unit}")
    print(f"Reachable range at this angle: {info['available_min']:.6g} to {info['available_max']:.6g} {unit}")

    if hwp_angle is None:
        print("❌ Requested power is not reachable at this polarization angle.")
        print("No stages were moved.")
        return None

    print(f"Selected HWP angle: {hwp_angle:.6f}°")
    print(f"Number of interpolated HWP solutions: {info['n_crossings']}")
    print("Choosing the minimal HWP angle, as requested.")
    print("=" * 70)

    # Move P1 first to define the output linear polarization angle, then HWP for power.
    rotate_stage("P1", requested_angle)
    rotate_stage("HWP", hwp_angle)
    time.sleep(0.2)

    measured = None
    if measure_after:
        measured = measure_intensity(verbose=True)

    result = {
        "requested_angle_deg": requested_angle,
        "requested_power": requested_power,
        "unit": unit,
        "p1_angle_deg": requested_angle,
        "hwp_angle_deg": float(hwp_angle),
        "available_min": info["available_min"],
        "available_max": info["available_max"],
        "n_crossings": info["n_crossings"],
        "all_hwp_crossings_deg": info["crossings"],
        "measured_after": measured,
    }

    print(f"✓ Set P1={requested_angle:.6f}°, HWP={hwp_angle:.6f}°")
    if measured is not None:
        print(f"✓ Measured final intensity: {measured:.6g} {get_unit()}")
    return result


def create_vertical_polarization() -> None:
    print("\n--- Set vertical linear polarization ---")
    p = safe_float_input(f"Power ({get_unit()})", 50.0)
    if p is None:
        return
    set_linear_polarization(angle=0.0, power=p, measure_after=False)


def create_custom_polarization() -> None:
    print("\n" + "=" * 70)
    print("CUSTOM LINEAR POLARIZATION")
    print("=" * 70)
    print("Angle is in the P1 stage reference frame: 0° = vertical.")
    angle = safe_float_input("Linear polarization angle in degrees", 0.0)
    power = safe_float_input(f"Requested power ({get_unit()})", 50.0)
    if angle is None or power is None:
        return
    set_linear_polarization(angle=angle, power=power, measure_after=False)


# =============================================================================
# 7. MANUAL SCANS / ROTATION / MAX POWER / HOMING
# =============================================================================

def choose_scan_device_interactive() -> str:
    ans = input("Measurement device for scan: powermeter [Enter] or timetagger [t]: ").strip().lower()
    device = "timetagger" if ans == "t" else "powermeter"
    if device != CURRENT_DEVICE:
        connect_measurement_device(device)
    return device


def scan_stage_interactive() -> Optional[Dict]:
    print("\n" + "=" * 70)
    print("INTERACTIVE STAGE SCAN")
    print("=" * 70)
    choose_scan_device_interactive()

    print("Available stages:", ", ".join(stage_names()))
    sid = input("Stage to scan [P1]: ").strip() or "P1"
    try:
        name, _ = resolve_stage(sid)
    except ValueError as exc:
        print(f"❌ {exc}")
        return None

    start = safe_float_input("Start angle", 0.0)
    stop = safe_float_input("Stop angle", 180.0)
    step = safe_float_input("Step angle", 5.0)
    if start is None or stop is None or step is None:
        return None

    angles = angle_array(start, stop, step)
    vals = []
    print(f"\nScanning {name}: {len(angles)} points with {CURRENT_DEVICE}")
    for i, ang in enumerate(angles):
        rotate_stage(name, ang)
        val = measure_intensity(verbose=False)
        vals.append(val)
        print(f"  {i + 1:4d}/{len(angles)}: {name}={ang:9.3f}°, I={val:12.6g} {get_unit()}")

    vals = np.asarray(vals, dtype=float)
    base = DATA_DIR / f"manual_scan_{name}_{CURRENT_DEVICE}_{timestamp_now()}"
    csv_path = base.with_suffix(".csv")
    pd.DataFrame({
        f"{name}_angle_deg": angles,
        "intensity": vals,
        "unit": get_unit(),
        "device": CURRENT_DEVICE,
    }).to_csv(csv_path, index=False)
    print(f"✓ Saved scan: {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(angles, vals, "o-")
    ax.set_xlabel(f"{name} raw angle (deg)")
    ax.set_ylabel(f"Intensity ({get_unit()})")
    ax.set_title(f"Manual scan: {name} ({CURRENT_DEVICE})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"stage": name, "angles": angles, "intensities": vals, "csv_path": str(csv_path)}


def interactive_rotate_stage() -> None:
    print("\n" + "=" * 70)
    print("INTERACTIVE ROTATE STAGE")
    print("=" * 70)
    print("Available stages:", ", ".join(stage_names()))
    sid = input("Stage to rotate [P1]: ").strip() or "P1"
    angle = safe_float_input("Raw stage angle in degrees", 0.0)
    if angle is None:
        return
    try:
        name, _ = resolve_stage(sid)
        raw = rotate_stage(name, angle)
        print(f"✓ Rotated {name} to raw angle {raw:.6f}°")
    except Exception as exc:
        print(f"❌ Rotation failed: {exc}")


def measure_max_power() -> Optional[float]:
    """
    Move to the globally maximum point in the latest lookup table if available,
    then measure the current maximum intensity. If no lookup exists, measure the
    current stage configuration.
    """
    print("\n" + "=" * 70)
    print("MEASURE MAX POWER")
    print("=" * 70)

    moved_to_lookup_max = False
    try:
        p1_angles, hwp_angles, grid, metadata = load_lookup_calibration()
        idx = np.nanargmax(grid)
        i, j = np.unravel_index(idx, grid.shape)
        p1 = float(p1_angles[i])
        hwp = float(hwp_angles[j])
        print(f"Moving to maximum point from latest lookup: P1={p1:.6f}°, HWP={hwp:.6f}°")
        rotate_stage("P1", p1)
        rotate_stage("HWP", hwp)
        time.sleep(0.3)
        moved_to_lookup_max = True
    except Exception as exc:
        print(f"⚠️ No usable lookup maximum found ({exc}). Measuring current configuration.")

    val = measure_intensity(verbose=True)
    if not np.isfinite(val):
        print("❌ Max-power measurement failed.")
        return None

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": CURRENT_DEVICE,
        "unit": get_unit(),
        "max_power": float(val),
        "moved_to_lookup_max": bool(moved_to_lookup_max),
        "P1_position_deg": get_stage_position("P1"),
        "HWP_position_deg": get_stage_position("HWP"),
    }
    write_header = not MAX_POWER_FILE.exists()
    with open(MAX_POWER_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"✓ Max power measured: {val:.6g} {get_unit()}")
    print(f"✓ Saved to {MAX_POWER_FILE}")
    return float(val)


def home_all_stages() -> None:
    print("\n⚙️ Homing all stages...")
    try:
        trs.home_all()
        print("✓ All stages homed")
    except Exception as exc:
        print(f"❌ Home-all failed: {exc}")


def home_elliptec_stages_only() -> None:
    print("\n⚙️ Homing Elliptec stages only (integer IDs)...")
    any_stage = False
    for name, meta in STAGE_METADATA.items():
        sid = meta["id"]
        if isinstance(sid, int):
            any_stage = True
            try:
                print(f"  Homing {name} (Elliptec ID {sid})...")
                trs.home_stage(sid)
                print(f"  ✓ {name} homed")
            except Exception as exc:
                print(f"  ❌ {name} home failed: {exc}")
    if not any_stage:
        print("No integer-ID Elliptec stages found in STAGE_METADATA.")


def show_latest_lookup_summary() -> None:
    try:
        p1_angles, hwp_angles, grid, metadata = load_lookup_calibration()
    except Exception as exc:
        print(f"No lookup calibration loaded: {exc}")
        return
    print("\nLatest lookup calibration:")
    print(f"  File: {LATEST_CALIBRATION_FILE}")
    print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
    print(f"  Device/unit: {metadata.get('device', 'unknown')} / {metadata.get('unit', get_unit())}")
    print(f"  P1 grid: {len(p1_angles)} points from {p1_angles[0]:.3f}° to {p1_angles[-1]:.3f}°")
    print(f"  HWP grid: {len(hwp_angles)} points from {hwp_angles[0]:.3f}° to {hwp_angles[-1]:.3f}°")
    print(f"  Intensity range: {np.nanmin(grid):.6g} to {np.nanmax(grid):.6g} {metadata.get('unit', get_unit())}")


# =============================================================================
# 8. MAIN MENU
# =============================================================================

def main() -> None:
    global _background_levels
    _background_levels = load_background_levels()

    print("\n" + "=" * 70)
    print("LINEAR POLARIZATION LOOKUP CONTROL")
    print("=" * 70)
    print("No polarization model is used: control is based on measured P1/HWP lookup data.")

    connect_all()
    show_latest_lookup_summary()

    while True:
        print("\n" + "=" * 70)
        print("MAIN MENU")
        print("=" * 70)
        print("  c   - Full 2D Lookup Calibration (P1 × HWP)")
        print("  s   - Scan Stage (Interactive)")
        print("  r   - Rotate Stage (Interactive)")
        print("  v   - Set Vertical Polarization")
        print("  p   - Set Custom Polarization (Interactive)")
        print("  m   - Measure current power")
        print("  b   - Measure Background")
        print("  max - Measure max power")
        print("  h   - Home All Stages")
        print("  he  - Home Elliptec Stages Only")
        print("  q   - Quit")
        print("=" * 70)

        choice = input("\nSelect option: ").strip().lower()

        try:
            if choice == "c":
                full_2d_lookup_calibration()
            elif choice == "s":
                scan_stage_interactive()
            elif choice == "r":
                interactive_rotate_stage()
            elif choice == "v":
                create_vertical_polarization()
            elif choice == "p":
                create_custom_polarization()
            elif choice == "m":
                measure_intensity(verbose=True)
            elif choice == "b":
                measure_background()
            elif choice == "max":
                measure_max_power()
            elif choice == "h":
                home_all_stages()
            elif choice == "he":
                home_elliptec_stages_only()
            elif choice == "q":
                print("\n👋 Exiting...")
                break
            else:
                print("❌ Unknown option.")
        except KeyboardInterrupt:
            print("\n⚠️ Operation interrupted.")
            raise
        except Exception as exc:
            print(f"\n❌ Error: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_all_devices()
