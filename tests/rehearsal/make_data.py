"""Record both angle scans against a synthetic source, with no hardware at all.

The pump mounts and the analyzer plate run as dry runs and the tagger is replaced by
`fake_acquisition`, so the real acquisition loops of `run_experiment` drive the real
`ExperimentConfig` and write real data files: the labels, the CSV logs and
`experiment_config.txt` all come out as they would in the lab.

    python tests/rehearsal/make_data.py           # both scans, into tests/rehearsal/

The source is a mock, but a described one (see MODEL below), so `replay_check.py` can
check what the analysis reads back against what was put in.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import acquisition  # noqa: E402
from experiment_config import ExperimentConfig  # noqa: E402
from hardware import RotationStageConfig  # noqa: E402
import run_experiment  # noqa: E402

REHEARSAL = REPO / "tests" / "rehearsal"
CALIBRATION = REHEARSAL / "calibration"
TRUTH_NAME = "truth.json"

MODES = {1: "H3T", 2: "H3R", 3: "H4T", 4: "H4R"}
FREQUENCY = 18.66e6

# ---------------------------------------------------------------------------
# MODEL - what the synthetic source does
# ---------------------------------------------------------------------------
#
# Brightness, per harmonic, follows a butterfly in the PUMP angle: two lobes for H3
# along 0-180 deg, four for H4 rotated by 45 deg, both on a floor so no angle is dark.
#
# Polarization, per harmonic, is set by an ellipticity that sweeps the whole range as
# the pump turns: linear at 0 deg, circular at 90 deg. What the analyzer plate then
# transmits in front of the fixed polarizer is
#
#     I(theta) = I_las * (1 + m cos(2 pi theta / 90 deg)),   m = (1-eps^2)/(1+eps^2)
#
# which is the Malus curve `pkl_json_analyze.fit_sinusoid` fits, written about its own
# mean, so a replay must return the eps that went in.

BRIGHTNESS = {"H3": 40000.0, "H4": 12000.0}   # counts/s per harmonic at its brightest
ARM_SPLIT = {"T": 0.48, "R": 0.52}            # the beamsplitter, deliberately imperfect
LOBES = {"H3": (2, 0.0), "H4": (4, 45.0)}     # (lobes, offset) of the butterfly
G2 = {"H3": 1.9, "H4": 1.5}                   # target g2(0) of each autocorrelation


def brightness(harmonic: str, laser_angle_deg: float) -> float:
    """The harmonic's count rate at one pump angle: a butterfly on a floor."""
    lobes, offset = LOBES[harmonic]
    shape = np.cos(np.deg2rad(lobes * (laser_angle_deg - offset) / 2.0)) ** 2
    return BRIGHTNESS[harmonic] * (0.08 + 0.92 * shape)


def ellipticity(laser_angle_deg: float) -> float:
    """Linear at 0 deg, circular at 90 deg, and back: the truth of the scan."""
    return float(abs(np.sin(np.deg2rad(laser_angle_deg))))


def transmitted(harmonic: str, laser_angle_deg: float,
                analyzer_angle_deg: float | None) -> float:
    """The count rate reaching the detectors, plate included when there is one."""
    rate = brightness(harmonic, laser_angle_deg)
    if analyzer_angle_deg is None:
        return rate
    eps = ellipticity(laser_angle_deg)
    modulation = (1.0 - eps ** 2) / (1.0 + eps ** 2)
    # The fixed polarizer takes half the light on average, hence the 0.5.
    return 0.5 * rate * (1.0 + modulation * np.cos(2.0 * np.pi * analyzer_angle_deg / 90.0))


# ---------------------------------------------------------------------------
# The fake tagger
# ---------------------------------------------------------------------------

def fake_acquisition(cfg: ExperimentConfig) -> acquisition.AcquisitionResult:
    """`run_acquisition` without a tagger: same files, same schema, made-up counts."""
    # crc32, not hash(): Python salts string hashes per process, and a rehearsal that
    # cannot be reproduced from one run to the next is not much of a reference.
    rng = np.random.default_rng(zlib.crc32(cfg.POWER_LEVEL.encode()))
    channels = list(cfg.REGISTERED_CHANNELS)
    delays = (np.arange(cfg.CORR_N_BINS) - cfg.CORR_N_BINS // 2) * cfg.CORR_BINWIDTH_PS
    rates = {ch: transmitted(cfg.MODES[ch][:2], cfg.LASER_ANGLE_DEG or 0.0,
                             cfg.ANALYZER_ANGLE_DEG) * ARM_SPLIT[cfg.MODES[ch][2]]
             for ch in channels}

    result = acquisition.AcquisitionResult()
    for index, duration in enumerate(cfg.chunk_durations()):
        countrate = {ch: _counts(rng, rate, duration) for ch, rate in rates.items()}
        correlations = {(a, b): (delays, _histogram(rng, cfg, delays, countrate, a, b, duration))
                        for a, b in cfg.correlation_pairs()}
        payload = acquisition.build_payload(
            correlations, countrate, channels,
            parameters=cfg.file_parameters(chunk=index, duration_s=duration),
        )
        result.chunk_paths.append(
            acquisition.save_payload(payload, cfg.chunk_path(index), cfg.EXPORT_FORMAT))
        result.duration_s += duration
    print(f"  {len(result.chunk_paths)} synthetic chunk(s) -> {cfg.chunk_path(0).name} ...")
    return result


def _counts(rng, rate: float, duration_s: float) -> Tuple[float, int]:
    total = int(rng.poisson(max(rate * duration_s, 0.0)))
    return total / duration_s, total


def _histogram(rng, cfg: ExperimentConfig, delays: np.ndarray,
               countrate: Dict[int, Tuple[float, int]], ch_a: int, ch_b: int,
               duration_s: float) -> np.ndarray:
    """A comb of peaks at every multiple of the repetition period.

    The satellites carry the accidental level two independent channels would give; the
    central peak carries `G2` times that, which is what `compute_g2_integration` reads
    back as g2(0). Cross-harmonic pairs get the mean of the two harmonics' g2.
    """
    period_ps = 1e12 / cfg.FREQUENCY
    accidental = (countrate[ch_a][0] * countrate[ch_b][0] * duration_s
                  * cfg.CORR_BINWIDTH_PS * 1e-12)
    harmonics = (cfg.MODES[ch_a][:2], cfg.MODES[ch_b][:2])
    g2 = float(np.mean([G2[h] for h in harmonics]))

    counts = np.zeros_like(delays, dtype=float)
    orders = int(abs(delays[0]) // period_ps)
    for order in range(-orders, orders + 1):
        centre = int(np.argmin(np.abs(delays - order * period_ps)))
        # Three bins per peak: a peak narrower than the integration window of the
        # analysis, so the window catches all of it as it does on real data.
        weight = g2 if order == 0 else 1.0
        for offset, share in ((-1, 0.2), (0, 0.6), (1, 0.2)):
            if 0 <= centre + offset < len(counts):
                counts[centre + offset] += weight * share * accidental * 30.0
    return rng.poisson(counts)


# ---------------------------------------------------------------------------
# The two scans
# ---------------------------------------------------------------------------

def lookup_table() -> Path:
    """A calibration flat in P1: every pump angle can deliver every power."""
    path = CALIBRATION / "linear_polarization_lookup_latest.npz"
    if path.is_file():
        return path
    p1 = np.arange(0.0, 180.0, 10.0)
    hwp = np.arange(0.0, 91.0, 2.5)
    # Power rises monotonically with the HWP angle, so a request has one answer.
    grid = np.tile(100.0 * np.sin(np.deg2rad(hwp)) ** 2, (len(p1), 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, p1_angles=p1, hwp_angles=hwp, power_grid=grid,
             metadata_json=json.dumps({"unit": "mW", "timestamp": "rehearsal"}))
    return path


def _common(folder: str) -> Dict[str, object]:
    return dict(
        MATERIAL="Rehearsal",
        EXPERIENCE_TYPE="Synthetic",
        DATE="30072026",
        DATA_DIR=REHEARSAL / folder / "data",
        RESULTS_DIR=REHEARSAL / folder / "results",
        MODES=MODES,
        FREQUENCY=FREQUENCY,
        # Two chunks, so the analysis has a spread to draw its error bands from, and long
        # enough that Poisson noise does not dominate the fit near a linear polarization -
        # where I_min goes to zero and the square root in the ellipticity amplifies it.
        ACQUISITION_DURATION_S=60.0,
        CHUNK_DURATION_S=30.0,
        SAVE_MERGED=False,
        INTEGRATION_WINDOW=8,
        INTEGRATION_WINDOWS_SWEEP=np.arange(1, 11, 1),
        PUMP_CALIBRATION=lookup_table(),
        PUMP_CALIBRATION_DIR=CALIBRATION,
        PUMP_STAGE_DRY_RUN=True,
        PUMP_P1=RotationStageConfig(serial_number="27000002"),
        PUMP_HWP=RotationStageConfig(serial_number="27000003"),
        RUN_ANALYSIS_AFTER_ACQUIRE=False,
    )


def laser_angle() -> None:
    cfg = ExperimentConfig(
        **_common("laser_angle"),
        POWER_LEVEL="50mW",
        PUMP_POWER=50.0,
        LASER_ANGLE_SCAN=np.arange(0.0, 180.0, 22.5),
    )
    run_experiment.main(cfg)


def ellipticity_scan() -> None:
    cfg = ExperimentConfig(
        **_common("ellipticity"),
        POWER_LEVEL="50mW",
        PUMP_POWER_SCAN=[{"power": "30mW", "requested_power": 30.0},
                         {"power": "50mW", "requested_power": 50.0}],
        ELLIPTICITY_LASER_ANGLES=[0.0, 45.0, 90.0],
        ELLIPTICITY_ANALYZER_ANGLES=np.arange(0.0, 180.0, 15.0),
        ELLIPTICITY_ANALYZER_DRY_RUN=True,
        ELLIPTICITY_ANALYZER=RotationStageConfig(serial_number="27000001"),
    )
    run_experiment.main(cfg)
    _write_truth(cfg)


def _write_truth(cfg: ExperimentConfig) -> Path:
    """The ellipticity that went in, for `replay_check.py` to compare against."""
    path = Path(cfg.DATA_DIR) / TRUTH_NAME
    truth = {f"{angle:g}": ellipticity(angle) for angle in cfg.ellipticity_laser_angles()}
    path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Injected ellipticity -> {path}")
    return path


if __name__ == "__main__":
    # `acquire_one` imports run_acquisition when it is called, so replacing the module
    # attribute is enough: the acquisition loops themselves are the real ones.
    acquisition.run_acquisition = fake_acquisition

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "ellipticity"):
        ellipticity_scan()
    if which in ("both", "laser_angle"):
        laser_angle()
