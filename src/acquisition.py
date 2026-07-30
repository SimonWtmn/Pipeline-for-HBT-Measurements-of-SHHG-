"""Live acquisition from the Time Tagger, and the file format it writes.

One acquisition = one power level (or one polarization angle), cut into independent
chunks. Each chunk gets its own fresh measurements, so the spread between chunks is a
genuine error bar; their sum is the optional merged file, and copies of that running
sum are the stability snapshots.

The second half of the file is the on-disk schema: `build_payload` turns Time Tagger
measurement objects into the exact dict `pkl_json_analyze.get_data` expects, and
`save_payload` writes it.

The TimeTagger import happens inside `run_acquisition`, not at module import, so
everything else in the pipeline works on a machine without the SDK.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from experiment_config import ExperimentConfig
except ImportError:  # pragma: no cover - import style depends on the entry point
    from src.experiment_config import ExperimentConfig

Pair = Tuple[int, int]
# ============================================================================
# PART 1 - the acquisition
# ============================================================================

POLL_INTERVAL_S = 0.05
STALL_TIMEOUT_S = 30.0

_INSTALL_HINT = (
    "The TimeTagger Python API is not available. It ships with the Swabian "
    "Instruments software (system/SDK installer, not pip): install it from "
    "https://www.swabianinstruments.com/time-tagger/downloads/ and make sure this "
    "interpreter is the one the installer registered. Use ANALYZE_ONLY=True to run "
    "the analysis without hardware."
)


@dataclass
class AcquisitionResult:
    chunk_paths: List[Path] = field(default_factory=list)
    merged_path: Optional[Path] = None
    snapshots_path: Optional[Path] = None
    snapshots: Optional[Dict[str, Any]] = None
    duration_s: float = 0.0


def run_acquisition(cfg: ExperimentConfig) -> AcquisitionResult:
    tt = _import_timetagger()
    pairs = cfg.correlation_pairs()
    if not pairs:
        raise ValueError("No correlation pairs: check MODES / REGISTERED_CHANNELS.")

    durations = cfg.chunk_durations()
    print(f"Acquiring {cfg.ACQUISITION_DURATION_S:g}s as {len(durations)} chunk(s) of "
          f"{cfg.CHUNK_DURATION_S:g}s | {len(pairs)} correlation pairs")

    tagger = _connect(tt, cfg)
    cumulative = _Cumulative(pairs)
    snapshots = _SnapshotRecorder(cfg, pairs) if cfg.STABILITY_ENABLED else None
    progress = _ProgressLogger(cfg.ACQUISITION_DURATION_S)
    result = AcquisitionResult()
    writer = None

    try:
        if cfg.SAVE_RAW_TTBIN:
            writer = _start_raw_writer(tt, tagger, cfg)

        for index, chunk_duration in enumerate(durations, start=1):
            elapsed_before = cumulative.duration_s

            def on_poll(chunk_elapsed: float, correlations: Dict[Pair, Any], countrate: Any) -> None:
                total = elapsed_before + chunk_elapsed
                progress.maybe_log(total, countrate)
                if snapshots is not None:
                    snapshots.maybe_record(total, correlations, cumulative)

            countrate, correlations = _acquire_chunk(tt, tagger, cfg, pairs, chunk_duration, on_poll)

            payload = build_payload(
                correlations, countrate, cfg.REGISTERED_CHANNELS,
                parameters=cfg.file_parameters(chunk=index, duration_s=chunk_duration),
            )
            path = save_payload(payload, cfg.chunk_path(index), cfg.EXPORT_FORMAT)
            result.chunk_paths.append(path)
            print(f"  chunk {index}/{len(durations)} -> {path.name}")

            cumulative.add_chunk(correlations, countrate, cfg.REGISTERED_CHANNELS, chunk_duration)
            correlations.clear()  # drop the measurements so the tagger frees their memory

        result.duration_s = cumulative.duration_s

        if cfg.SAVE_MERGED:
            merged_payload = build_payload(
                cumulative.histograms(), cumulative.countrates(), cfg.REGISTERED_CHANNELS,
                parameters=cfg.file_parameters(chunk="merged", duration_s=cumulative.duration_s),
            )
            result.merged_path = save_payload(merged_payload, cfg.merged_path(), cfg.EXPORT_FORMAT)
            print(f"  merged -> {result.merged_path}")

        if snapshots is not None and snapshots.times:
            result.snapshots = snapshots.as_dict()
            result.snapshots_path = _save_snapshots(result.snapshots, cfg)
            print(f"  {len(snapshots.times)} stability snapshots -> {result.snapshots_path}")
        elif snapshots is not None:
            print("  no stability snapshot recorded: the run is shorter than "
                  "STABILITY_SNAPSHOT_INTERVAL_S")

    finally:
        if writer is not None:
            writer.stop()
            print(f"  raw stream: {writer.getTotalSize() / (1024 ** 2):.1f} MiB")
        tt.freeTimeTagger(tagger)
        print("Time Tagger released.")

    return result


# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------

def _import_timetagger():
    try:
        import TimeTagger
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return TimeTagger


def _connect(tt, cfg: ExperimentConfig):
    """Open the tagger and apply trigger level, input delays and dead times."""
    tagger = tt.createTimeTagger(cfg.TAGGER_SERIAL)
    for ch in cfg.REGISTERED_CHANNELS:
        tagger.setTriggerLevel(ch, cfg.TRIGGER_LEVEL_V)
        if ch in cfg.INPUT_DELAYS_PS:
            tagger.setInputDelay(ch, cfg.INPUT_DELAYS_PS[ch])
        if ch in cfg.DEADTIMES_PS:
            tagger.setDeadtime(ch, cfg.DEADTIMES_PS[ch])
    tagger.sync()  # all settings applied before any measurement is created

    labels = [cfg.MODES.get(ch, f"Ch{ch}") for ch in cfg.REGISTERED_CHANNELS]
    print(f"Connected to Time Tagger {tagger.getSerial()} | channels {cfg.REGISTERED_CHANNELS} -> {labels}")
    return tagger


def _start_raw_writer(tt, tagger, cfg: ExperimentConfig):
    path = cfg.DATA_DIR / "raw" / f"{cfg.POWER_LEVEL}.ttbin"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"SAVE_RAW_TTBIN is on: dumping the raw time-tag stream to {path} (can reach several GB)")
    return tt.FileWriter(tagger, str(path), list(cfg.REGISTERED_CHANNELS))


def _acquire_chunk(
    tt,
    tagger,
    cfg: ExperimentConfig,
    pairs: Sequence[Pair],
    duration_s: float,
    on_poll: Callable[[float, Dict[Pair, Any], Any], None],
):
    """Run one chunk on fresh measurements and return them once it is complete."""
    countrate = tt.Countrate(tagger, list(cfg.REGISTERED_CHANNELS))
    correlations = {
        (ch_a, ch_b): tt.Correlation(
            tagger, channel_1=ch_a, channel_2=ch_b,
            binwidth=cfg.CORR_BINWIDTH_PS, n_bins=cfg.CORR_N_BINS,
        )
        for ch_a, ch_b in pairs
    }
    measurements = [countrate, *correlations.values()]
    for measurement in measurements:
        measurement.clear()

    probe = next(iter(correlations.values()))
    # Watchdog on *progress*, not on total wall-clock time: at high count rates the
    # USB link overflows and getCaptureDuration advances slower than real time, so a
    # chunk can legitimately take longer than duration + STALL_TIMEOUT_S while the
    # device is perfectly fine. Only bail out if capture time truly stops advancing.
    last_elapsed = 0.0
    last_progress = time.monotonic()
    while True:
        elapsed = probe.getCaptureDuration() / 1e12
        on_poll(elapsed, correlations, countrate)
        if elapsed >= duration_s:
            break
        now = time.monotonic()
        if elapsed > last_elapsed:
            last_elapsed = elapsed
            last_progress = now
        elif now - last_progress > STALL_TIMEOUT_S:
            raise TimeoutError(
                f"Tagger stalled: capture stuck at {elapsed:.1f}s out of {duration_s:g}s "
                f"for over {STALL_TIMEOUT_S:g}s. Check that the device is still connected."
            )
        time.sleep(POLL_INTERVAL_S)

    for measurement in measurements:
        measurement.stop()
    return countrate, correlations


# ----------------------------------------------------------------------------
# Cumulative histogram (merged file + stability snapshots)
# ----------------------------------------------------------------------------

class _Cumulative:
    """Sum of the chunks recorded so far.

    Correlation histograms are purely additive, so the full-run histogram is the
    sum of the per-chunk ones. That saves running a second set of measurements on
    the tagger just to get the merged file and the snapshots.
    """

    def __init__(self, pairs: Sequence[Pair]) -> None:
        self.delays: Optional[np.ndarray] = None
        self.hist: Dict[Pair, Optional[np.ndarray]] = {pair: None for pair in pairs}
        self.totals: Dict[int, int] = {}
        self.duration_s: float = 0.0

    def add_chunk(self, correlations, countrate, channels: Sequence[int], duration_s: float) -> None:
        for pair, correlation in correlations.items():
            delays, counts = correlation_arrays(correlation)
            if self.delays is None:
                self.delays = delays
            previous = self.hist.get(pair)
            self.hist[pair] = counts.astype(np.int64) if previous is None else previous + counts
        for ch, (_rate, total) in countrate_values(countrate, channels).items():
            self.totals[ch] = self.totals.get(ch, 0) + total
        self.duration_s += duration_s

    def histograms(self) -> Dict[Pair, Tuple[np.ndarray, np.ndarray]]:
        return {pair: (self.delays, counts) for pair, counts in self.hist.items() if counts is not None}

    def countrates(self) -> Dict[int, Tuple[float, int]]:
        """Run-averaged rate (cps) and total counts per channel."""
        duration = self.duration_s or 1.0
        return {ch: (total / duration, total) for ch, total in self.totals.items()}

    def hist_including(self, pair: Pair, live_counts: np.ndarray) -> np.ndarray:
        """Cumulative histogram of `pair` including the chunk currently running.
        Always a new array, so snapshots keep their own copy of the data."""
        finished = self.hist.get(pair)
        return np.array(live_counts) if finished is None else finished + live_counts


# ----------------------------------------------------------------------------
# Stability snapshots
# ----------------------------------------------------------------------------

class _SnapshotRecorder:
    """Copies the cumulative histograms every STABILITY_SNAPSHOT_INTERVAL_S."""

    def __init__(self, cfg: ExperimentConfig, pairs: Sequence[Pair]) -> None:
        self.interval = cfg.STABILITY_SNAPSHOT_INTERVAL_S
        self.times: List[float] = []
        self.hist: Dict[Pair, List[np.ndarray]] = {pair: [] for pair in pairs}
        self.delays: Optional[np.ndarray] = None
        self._next_target = self.interval

    def maybe_record(self, elapsed_s: float, correlations: Dict[Pair, Any], cumulative: "_Cumulative") -> None:
        if elapsed_s < self._next_target:
            return
        for pair, correlation in correlations.items():
            delays, counts = correlation_arrays(correlation)
            if self.delays is None:
                self.delays = delays
            self.hist[pair].append(cumulative.hist_including(pair, counts))
        self.times.append(elapsed_s)
        self._next_target = elapsed_s + self.interval

    def as_dict(self) -> Dict[str, Any]:
        return {
            "times": np.array(self.times),
            "index": np.array([] if self.delays is None else self.delays),
            "hist": {pair: list(series) for pair, series in self.hist.items()},
        }


def _save_snapshots(snapshots: Dict[str, Any], cfg: ExperimentConfig) -> Path:
    path = cfg.stability_dir / "stability_snapshots.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshots, f)
    return path


def load_snapshots(path: Path | str) -> Dict[str, Any]:
    """Reload snapshots saved by a previous run (handy to replot without measuring)."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Progress
# ----------------------------------------------------------------------------

class _ProgressLogger:
    """Prints a progress line every 1% of the run, at most one every 10 s."""

    def __init__(self, total_duration_s: float) -> None:
        self.total = total_duration_s
        self.step = max(total_duration_s / 100.0, 10.0)
        self._next = self.step

    def maybe_log(self, elapsed_s: float, countrate: Any) -> None:
        if elapsed_s < self._next:
            return
        rate = float(np.sum(countrate.getData())) if hasattr(countrate, "getData") else float("nan")
        print(f"  [{elapsed_s / 60:.1f} / {self.total / 60:.1f} min] total countrate: {rate:.2e} cps")
        self._next = elapsed_s + self.step

# ============================================================================
# PART 2 - the on-disk schema
# ============================================================================

def correlation_arrays(correlation: Any) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(correlation, "getIndex"):
        return np.asarray(correlation.getIndex()), np.asarray(correlation.getData())
    delays, counts = correlation
    return np.asarray(delays), np.asarray(counts)


def countrate_values(countrate: Any, channels: Sequence[int]) -> Dict[int, Tuple[float, int]]:
    if hasattr(countrate, "getData"):
        rates = np.asarray(countrate.getData(), dtype=float)
        totals = np.asarray(countrate.getCountsTotal(), dtype=float)
        return {int(ch): (float(rates[i]), int(totals[i])) for i, ch in enumerate(channels)}
    return {int(ch): (float(rate), int(total)) for ch, (rate, total) in countrate.items()}


def build_payload(
    correlations: Mapping[Pair, Any],
    countrate: Any,
    registered_channels: Iterable[int],
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    channels = list(registered_channels)

    stored: Dict[str, Any] = {}
    for (ch_a, ch_b), correlation in correlations.items():
        ch1, ch2 = (int(ch_a), int(ch_b)) if ch_a < ch_b else (int(ch_b), int(ch_a))
        delays, counts = correlation_arrays(correlation)
        stored[f"({ch1}, {ch2})"] = [
            [int(d) for d in delays],
            [int(c) for c in counts],
        ]

    rates = countrate_values(countrate, channels)
    payload: Dict[str, Any] = {
        "Correlation": stored,
        "Countrate": {str(ch): [rate, total] for ch, (rate, total) in sorted(rates.items())},
    }
    if parameters is not None:
        payload = {"Parameters": dict(parameters), **payload}
    return payload


def save_payload(payload: Mapping[str, Any], path: Path | str, format: str = "pkl") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pkl":
        with open(path, "wb") as f:
            pickle.dump(dict(payload), f)
    elif format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(payload), f)
    else:
        raise ValueError(f"Unsupported export format {format!r} (use 'pkl' or 'json')")
    return path
