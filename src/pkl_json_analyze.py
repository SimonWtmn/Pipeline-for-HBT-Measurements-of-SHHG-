"""The analysis: peaks, g2(0), R, and every figure the pipeline produces.

Five families of output, all reading the same `.pkl`/`.json` files:

* `run_analysis(cfg)` - the per-power tree: spectrograms, window sweeps, chunk
  statistics, and the power sweep summary when several powers are analysed.
* `plot_stability(snapshots, cfg)` - g2(0) and R against time, from the snapshots a
  live run recorded while it was measuring.
* `plot_laser_angle_summary(cfg)` - the four polar summaries of one power: intensity
  per harmonic, g²(0) per pair, R per cross pair, harmonics (intensity + auto-g²).
* `plot_laser_angle_campaign(cfg)` - the same for every power in the folder, then
  the multi-power butterfly overlay.
* `plot_ellipticity_campaign(cfg)` - the analyzer sweep recorded at each pump angle:
  the same four summaries per pump angle, the sinusoid fitted to each, and the
  ellipticity those fits give versus pump angle.

The maths lives in one place: the stability, laser-angle and ellipticity halves reuse
the same peak finding and the same g2 definition as the per-power tree, so a point on a
polar figure is the number the per-angle tree would give for that angle.
"""

from pathlib import Path
import csv
import itertools
import json
import pickle
import re
import math
import matplotlib

# Figures are always saved, never shown. Must be set before pyplot is imported, and is
# what makes this module work headless (over SSH, from a scheduler).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import numpy as np
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from experiment_config import (ExperimentConfig, analyzer_angle_tag, harmonic_of,
                                   laser_angle_tag)
    from hardware import EllipticityLog, LaserAngleLog
except ImportError:  # pragma: no cover - import style depends on the entry point
    from src.experiment_config import (ExperimentConfig, analyzer_angle_tag, harmonic_of,
                                       laser_angle_tag)
    from src.hardware import EllipticityLog, LaserAngleLog

from dataclasses import replace

script_path = Path(__file__).resolve().parent.parent

Pair = Tuple[int, int]           # a channel pair, as the stability snapshots key them
ANGLE_SUMMARY_CSV = "angle_summary.csv"
ANGLE_MINIMA_CSV = "interpolated_minima.csv"
ELLIPTICITY_FIT_CSV = "sinusoidal_fit.csv"
ELLIPTICITY_SUMMARY_CSV = "ellipticity_vs_laser_angle.csv"
BAND_ALPHA = 0.25                # opacity of the chunk-spread band on the polar plots
ARM_COLORS = ("tab:orange", "tab:brown")   # the arms drawn beside a harmonic total
SMOOTH_HARMONICS = ("H4",)       # harmonics drawn as a curve interpolated through the samples
SMOOTH_SAMPLES_PER_STEP = 16     # density of that curve, per measured angle step
MIN_TILE_SPAN_DEG = 45.0         # narrower arcs are zoom scans: draw them as measured, never tiled




# ==========================================
# ENTRY POINT
# ==========================================
def main():
    """Standalone defaults, used when this file is run directly.

    The live pipeline calls run_analysis(cfg) from run_experiment.py instead, so
    the metadata is defined in a single place there.
    """
    run_analysis(
        MATERIAL="ZnO100",
        EXPERIENCE_TYPE="David_Setup",
        DATE="11142023",
        FREQUENCY=18.66e6,
        MODES={1: "H3T", 5: "H3R", 10: "H5T", 14: "H5R"},
        POWER_LEVELS=["10mW", "15mW", "20mW", "25mW", "30mW", "35mW", "40mW", "45mW", "50mW", "55mW", "60mW", "65mW", "70mW", "75mW", "80mW", "85mW", "90mW", "95mW"],
        INTEGRATION_WINDOW=8,
        INTEGRATION_WINDOWS_SWEEP=np.arange(1, 31, 1),
    )


def run_analysis(cfg=None, **overrides):
    """Analyze every power level described by `cfg`.

    Input :
    * cfg (ExperimentConfig or namespace) : carries MATERIAL, EXPERIENCE_TYPE, DATE,
        FREQUENCY, MODES, POWER_LEVEL(S), INTEGRATION_WINDOW, INTEGRATION_WINDOWS_SWEEP
        and the optional DATA_DIR / RESULTS_DIR overrides.
    * overrides : any of the same fields, given directly as keyword arguments.
    """
    p = _resolve_analysis_params(cfg, overrides)
    MATERIAL, FREQUENCY, MODES = p.MATERIAL, p.FREQUENCY, p.MODES
    POWER_LEVELS = p.POWER_LEVELS
    DATA_DIR, RESULTS_DIR = p.DATA_DIR, p.RESULTS_DIR
    INTEGRATION_WINDOW = p.INTEGRATION_WINDOW
    INTEGRATION_WINDOWS_SWEEP = p.INTEGRATION_WINDOWS_SWEEP
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {MATERIAL} | {DATA_DIR}")

    global_power_data = {"powers": [], "pairs": {}}

    for POWER_LEVEL in POWER_LEVELS:
        print(f"\n---------------- Processing Power Level: {POWER_LEVEL} ----------------")
        merged_files, chunk_files = get_files(DATA_DIR, pattern=f"*{POWER_LEVEL}*")

        if not merged_files and not chunk_files:
            print(f"[{POWER_LEVEL}] No files found. Skipping.")
            continue
            
        merged_data = get_data(merged_files, MODES, POWER_LEVEL) if merged_files else {}
        chunk_data = get_data(chunk_files, MODES, POWER_LEVEL) if chunk_files else {}
        
        if merged_data:
            merged_data = get_peaks(merged_data, FREQUENCY)
            merged_data = compute_g2_integration(merged_data, INTEGRATION_WINDOW)
            merged_data = compute_g2_counts(merged_data, INTEGRATION_WINDOW)
            merged_data = compute_R_integration(merged_data) 
            merged_data = compute_R_counts(merged_data)      
        
        if chunk_data:
            chunk_data = get_peaks(chunk_data, FREQUENCY)
            chunk_data = compute_g2_integration(chunk_data, INTEGRATION_WINDOW)
            chunk_data = compute_g2_counts(chunk_data, INTEGRATION_WINDOW)
            chunk_data = compute_R_integration(chunk_data) 
            chunk_data = compute_R_counts(chunk_data)   
            
        # Merged gives the cleanest spectrogram, chunks give the error bars: with both
        # on disk each is used for what it is good at, otherwise whichever exists does
        # everything (and a merged-only power level has no spread, so std dev is 0).
        if merged_data and chunk_data:
            print(f"[{POWER_LEVEL}] Found merged and chunks. Using merged for spectrograms, chunks for statistics.")
            plot_correlations_spectrograms(merged_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOW)
            plot_power_summary(chunk_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOWS_SWEEP)
            global_power_data = aggregate_power_data(chunk_data, global_power_data, POWER_LEVEL)
            
        elif chunk_data and not merged_data:
            print(f"[{POWER_LEVEL}] Found only chunks. Using chunks for all plots.")
            plot_correlations_spectrograms(chunk_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOW)
            plot_window_sweep(chunk_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOWS_SWEEP)
            plot_power_summary(chunk_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOWS_SWEEP)
            global_power_data = aggregate_power_data(chunk_data, global_power_data, POWER_LEVEL)
            
        elif merged_data and not chunk_data:
            print(f"[{POWER_LEVEL}] Found only merged data. Using merged for all plots (std dev will be 0).")
            plot_correlations_spectrograms(merged_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOW)
            plot_window_sweep(merged_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOWS_SWEEP)
            plot_power_summary(merged_data, RESULTS_DIR, POWER_LEVEL, INTEGRATION_WINDOWS_SWEEP)
            global_power_data = aggregate_power_data(merged_data, global_power_data, POWER_LEVEL)

    if len(POWER_LEVELS) > 1 and global_power_data["powers"]:
        plot_metrics_vs_power(global_power_data, RESULTS_DIR)

    return global_power_data


def _resolve_analysis_params(cfg, overrides):
    """Merge cfg attributes, keyword overrides and defaults into one namespace."""
    def get(name, default=None):
        value = overrides.get(name, None)
        if value is None:
            value = getattr(cfg, name, None)
        return default if value is None else value

    material = get("MATERIAL", "ZnO100")
    experience_type = get("EXPERIENCE_TYPE", "David_Setup")
    date = get("DATE", "11142023")

    power_levels = get("POWER_LEVELS")
    if power_levels is None:
        single = get("POWER_LEVEL")
        power_levels = [single] if single else []

    return SimpleNamespace(
        MATERIAL=material,
        EXPERIENCE_TYPE=experience_type,
        DATE=date,
        FREQUENCY=get("FREQUENCY", 18.66e6),
        MODES=get("MODES", {1: "H3T", 5: "H3R", 10: "H5T", 14: "H5R"}),
        POWER_LEVELS=list(power_levels),
        DATA_DIR=Path(get("DATA_DIR", script_path / "data" / material / experience_type / date)),
        RESULTS_DIR=Path(get("RESULTS_DIR", script_path / "results" / material / experience_type / date)),
        INTEGRATION_WINDOW=get("INTEGRATION_WINDOW", 8),
        INTEGRATION_WINDOWS_SWEEP=get("INTEGRATION_WINDOWS_SWEEP", np.arange(1, 31, 1)),
    )




# ==========================================
# HELPER FUNCTIONS
# ==========================================

def is_cross_harmonic(pair):
    harm1 = ''.join([c for c in pair[0] if c.isalnum()][:2])
    harm2 = ''.join([c for c in pair[1] if c.isalnum()][:2])
    return harm1 != harm2

def get_color_map(pairs):
    sorted_pairs = sorted(pairs)
    colors = cm.tab20(np.linspace(0, 1, max(1, len(sorted_pairs))))
    return {pair: colors[i] for i, pair in enumerate(sorted_pairs)}

def setup_grid_figure(pairs, title, cols=5, dpi=700, split_auto=True):
    if split_auto:
        auto_pairs = sorted([p for p in pairs if not is_cross_harmonic(p)])
        cross_pairs = sorted([p for p in pairs if is_cross_harmonic(p)])
        n_auto = len(auto_pairs)
        n_cross = len(cross_pairs)
        
        rows = max(max(n_auto, 1), math.ceil(n_cross / (cols - 1)) if n_cross > 0 else 1)
        fig, axes = plt.subplots(rows, cols, figsize=(min(20, cols*4), rows*3.5), dpi=dpi)
        if rows == 1: axes = axes[np.newaxis, :]
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        return fig, axes, auto_pairs, cross_pairs, rows, cols
    else:
        sorted_pairs = sorted(pairs)
        n_total = len(sorted_pairs)
        rows = max(1, math.ceil(n_total / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(min(20, cols*4), rows*3.5), dpi=dpi)
        if rows == 1: axes = axes[np.newaxis, :]
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        return fig, axes, [], sorted_pairs, rows, cols

def place_in_grid(axes, index, is_auto, cols, split_auto=True):
    if split_auto:
        if is_auto: return axes[index, 0]
        else:
            r = index // (cols - 1)
            c = 1 + (index % (cols - 1))
            return axes[r, c]
    else:
        r = index // cols
        c = index % cols
        return axes[r, c]

def hide_empty_grid(axes, n_auto, n_cross, rows, cols, split_auto=True):
    if split_auto:
        for i in range(n_auto, rows): axes[i, 0].set_visible(False)
        total_cross = rows * (cols - 1)
        for i in range(n_cross, total_cross):
            r = i // (cols - 1)
            c = 1 + (i % (cols - 1))
            axes[r, c].set_visible(False)
    else:
        total_cells = rows * cols
        for i in range(n_cross, total_cells): 
            r = i // cols
            c = i % cols
            axes[r, c].set_visible(False)

def apply_shading(ax, metric_type):
    ylim = ax.get_ylim() 
    if metric_type == "g2":
        ax.axhspan(0, 1, color='lightblue', alpha=0.3, zorder=-1)   
        ax.axhspan(1, 2, color='lightyellow', alpha=0.3, zorder=-1) 
        ax.axhspan(2, 4, color='lightcoral', alpha=0.3, zorder=-1)  
        ax.axhspan(4, max(ylim[1], 5), color='lightgray', alpha=0.4, zorder=-1) 
    elif metric_type == "R":
        ax.axhspan(0, 1, color='lightgray', alpha=0.4, zorder=-1)   
        ax.axhspan(1, max(ylim[1], 2), color='lightgreen', alpha=0.3, zorder=-1) 
    ax.set_ylim(ylim) 




# ==========================================
# DATA LOADING & PRIOR PROCESSING
# ==========================================

def get_files(data_dir, pattern="*"):
    merged_dir = data_dir / "merged"
    merged_files = []
    if merged_dir.is_dir():
        merged_files = list(merged_dir.glob(f"{pattern}.json")) + list(merged_dir.glob(f"{pattern}.pkl"))

    chunk_files = list(data_dir.glob(f"{pattern}.pkl"))
    if not chunk_files:
        chunk_files = list(data_dir.glob(f"{pattern}.json"))

    return merged_files, chunk_files

def _parse_pair_string(pair_str):
    ch1_str, ch2_str = pair_str.strip("()").split(",")
    return int(ch1_str), int(ch2_str)

def _normalize_legacy_payload(raw_data, modes):
    file_correlations, file_countrates, file_total_counts = {}, {}, {}

    raw_corr = raw_data.get("Correlation", {})
    for pair_str, data_lists in raw_corr.items():
        ch1, ch2 = _parse_pair_string(pair_str)
        if ch1 > ch2:
            continue
        phys_pair = (modes.get(ch1, f"Ch{ch1}"), modes.get(ch2, f"Ch{ch2}"))

        # Delays are stored in ps and used in ns everywhere downstream.
        bins = np.array(data_lists[0]) / 1000 if len(data_lists) > 0 else np.array([])
        coherences = data_lists[1] if len(data_lists) > 1 else []
        file_correlations[phys_pair] = {"delay_bins": bins, "coherences": coherences}

    raw_counts = raw_data.get("Countrate", {})
    for ch_str, values in raw_counts.items():
        ch_num = int(ch_str)
        phys_name = modes.get(ch_num, f"Ch{ch_num}")
        if isinstance(values, list) and len(values) >= 2:
            file_countrates[phys_name] = values[0]
            file_total_counts[phys_name] = values[1]

    return file_correlations, file_countrates, file_total_counts

def _normalize_nested_payload(raw_data, modes):
    file_correlations, file_countrates, file_total_counts = {}, {}, {}
    payload = raw_data.get("data", {})

    raw_corr = (
        payload.get("correlations_physical")
        or payload.get("correlations_virtual")
        or {}
    )
    for pair_str, corr_data in raw_corr.items():
        ch1, ch2 = _parse_pair_string(pair_str)
        if ch1 > ch2:
            continue
        phys_pair = (modes.get(ch1, f"Ch{ch1}"), modes.get(ch2, f"Ch{ch2}"))

        bins_ps = corr_data.get("time_bins", [])
        coherences = corr_data.get("counts", [])
        file_correlations[phys_pair] = {
            "delay_bins": np.array(bins_ps) / 1000,
            "coherences": coherences,
        }

    raw_rates = (
        payload.get("countrates_physical")
        or payload.get("countrates_virtual")
        or {}
    )
    raw_counts = (
        payload.get("counts_physical")
        or payload.get("counts_virtual")
        or {}
    )
    for ch_str, rate in raw_rates.items():
        ch_num = int(ch_str)
        phys_name = modes.get(ch_num, f"Ch{ch_num}")
        file_countrates[phys_name] = rate
        if ch_str in raw_counts:
            file_total_counts[phys_name] = raw_counts[ch_str]

    return file_correlations, file_countrates, file_total_counts

def _normalize_payload(raw_data, modes):
    if "Correlation" in raw_data or "Countrate" in raw_data:
        return _normalize_legacy_payload(raw_data, modes)
    if isinstance(raw_data.get("data"), dict):
        return _normalize_nested_payload(raw_data, modes)
    raise ValueError(
        "Unsupported analyzer payload format. Expected legacy Correlation/Countrate "
        "keys or nested data/correlations_* keys."
    )

def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    return np.nan if arr.size == 0 or np.isnan(arr).all() else np.nanmean(arr)

def _safe_nanstd(values):
    arr = np.asarray(values, dtype=float)
    return np.nan if arr.size == 0 or np.isnan(arr).all() else np.nanstd(arr)

def _clean_chunk_name(filename, power_level):
    chunk_match = re.search(rf'{re.escape(power_level)}_num\d+_chunk(\d+)', filename)
    if chunk_match:
        return f"chunk_{chunk_match.group(1)}"

    legacy_match = re.search(rf'{re.escape(power_level)}_num(\d+)', filename)
    if legacy_match:
        return f"chunk_{legacy_match.group(1)}"

    return Path(filename).stem

def get_data(target_files, modes, power_level):
    all_data = {}
    for file_path in target_files:
        clean_filename = _clean_chunk_name(file_path.name, power_level)
            
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f: raw_data = json.load(f)
        elif file_path.suffix == ".pkl":
            with open(file_path, "rb") as f: raw_data = pickle.load(f)
        else:
            continue

        file_correlations, file_countrates, file_total_counts = _normalize_payload(raw_data, modes)

        all_data[clean_filename] = {
            "correlations": file_correlations,
            "countrates": file_countrates,
            "total_counts": file_total_counts,
        }
    return all_data

def get_peaks(data_dict, freq_Hz=18.66e6):
    for chunk_name, data in data_dict.items():
        peaks_dict = {}
        for pair, p_data in data["correlations"].items():
            delay_bins = np.array(p_data["delay_bins"])
            coherences = np.array(p_data["coherences"])
            
            if len(delay_bins) < 2:
                peaks_dict[pair] = np.array([])
                continue
                
            axis_range = delay_bins[-1] - delay_bins[0]
            if axis_range < 10000:
                T_rep = 1e9 / freq_Hz
            else:
                T_rep = 1e12 / freq_Hz
                
            center_mask = (delay_bins >= -T_rep/2) & (delay_bins <= T_rep/2)
            t0 = 0
            
            if np.any(center_mask):
                center_delays = delay_bins[center_mask]
                center_cohs = coherences[center_mask]
                
                peaks, _ = find_peaks(center_cohs, prominence=max(2, 0.05 * np.max(center_cohs)))
                if len(peaks) > 0:
                    idx_closest = peaks[np.argmin(np.abs(center_delays[peaks]))]
                    t0 = center_delays[idx_closest]
                else:
                    t0 = center_delays[np.argmax(center_cohs)]
            
            shifted_delay_bins = delay_bins - t0
            p_data["delay_bins"] = shifted_delay_bins.tolist()
            
            ideal_peak_indices = []
            min_k = int(np.floor(shifted_delay_bins[0] / T_rep))
            max_k = int(np.ceil(shifted_delay_bins[-1] / T_rep))
            
            for k in range(min_k, max_k + 1):
                target_time = k * T_rep
                if shifted_delay_bins[0] <= target_time <= shifted_delay_bins[-1]:
                    closest_idx = np.argmin(np.abs(shifted_delay_bins - target_time))
                    ideal_peak_indices.append(closest_idx)
                    
            peaks_dict[pair] = np.array(ideal_peak_indices)
            
        data["peaks"] = peaks_dict
    return data_dict





# ==========================================
# COMPUTING METRICS
# ==========================================

def compute_g2_integration(data_dict, integration_window):
    for chunk_name, data in data_dict.items():
        g2_dict = {}
        for correlation_pair, correlation_data in data["correlations"].items():
            delay_bins = np.array(correlation_data["delay_bins"])
            coherences = np.array(correlation_data["coherences"])
            peaks_indices = data["peaks"][correlation_pair]

            if len(peaks_indices) == 0:
                g2_dict[correlation_pair] = np.nan
                continue

            center_peak_idx = peaks_indices[np.argmin(np.abs(delay_bins[peaks_indices]))]
            
            c_start = max(0, center_peak_idx - integration_window // 2)
            c_end = min(len(delay_bins), c_start + max(1, integration_window))
            central_integral = np.sum(coherences[c_start:c_end])

            side_integrals = []
            for peak_index in peaks_indices:
                if peak_index == center_peak_idx: continue
                s_start = max(0, peak_index - integration_window // 2)
                s_end = min(len(delay_bins), s_start + max(1, integration_window))
                side_integrals.append(np.sum(coherences[s_start:s_end]))

            if len(side_integrals) > 0:
                avg_side_integral = np.mean(side_integrals)
                g2_value = central_integral / avg_side_integral if avg_side_integral != 0 else np.nan
            else:
                g2_value = np.nan
            g2_dict[correlation_pair] = g2_value
        data_dict[chunk_name]["g2_integrated"] = g2_dict
    return data_dict

def compute_g2_counts(data_dict, integration_window):
    for chunk_name, data in data_dict.items():
        g2_counts_dict = {}
        total_counts_map = data["total_counts"]
        Np = data.get("Np", 60e12 * 1e-12 * (18.66 * 1e6)) 
        
        for correlation_pair, correlation_data in data["correlations"].items():
            ch1, ch2 = correlation_pair
            n1 = float(total_counts_map.get(ch1, 0))
            n2 = float(total_counts_map.get(ch2, 0))
            if n1 == 0 or n2 == 0:
                g2_counts_dict[correlation_pair] = np.nan
                continue

            delay_bins = np.array(correlation_data["delay_bins"])
            coherences = np.array(correlation_data["coherences"])
            peaks_indices = data["peaks"][correlation_pair]
            if len(peaks_indices) == 0:
                g2_counts_dict[correlation_pair] = np.nan
                continue
                
            center_peak_idx = peaks_indices[np.argmin(np.abs(delay_bins[peaks_indices]))]
            c_start = max(0, center_peak_idx - integration_window // 2)
            c_end = min(len(delay_bins), c_start + max(1, integration_window))
            n_12 = float(np.sum(coherences[c_start:c_end]))
            
            g2_val = (Np * n_12) / (n1 * n2) if (n1 * n2) > 0 else np.nan
            g2_counts_dict[correlation_pair] = g2_val
        data_dict[chunk_name]["g2_counts"] = g2_counts_dict
    return data_dict

def get_auto_pair(ch_name, available_pairs):
    harm = ''.join([c for c in ch_name if c.isalnum()][:2]) 
    for pair in available_pairs:
        if pair[0].startswith(harm) and pair[1].startswith(harm):
            return pair
    return None

def compute_R_integration(data_dict):
    for chunk_name, data in data_dict.items():
        R_dict = {}
        g2_integrated = data.get("g2_integrated", {})
        available_pairs = list(g2_integrated.keys())
        
        for g2_pair, g2_val in g2_integrated.items():
            ch1, ch2 = g2_pair
            pair_ii = get_auto_pair(ch1, available_pairs)
            pair_jj = get_auto_pair(ch2, available_pairs)
            if not is_cross_harmonic(g2_pair): continue    

            g2_ii = g2_integrated.get(pair_ii, np.nan) if pair_ii else np.nan
            g2_jj = g2_integrated.get(pair_jj, np.nan) if pair_jj else np.nan
            
            if g2_val is None or np.isnan(g2_val) or np.isnan(g2_ii) or np.isnan(g2_jj) or (g2_ii * g2_jj) == 0:
                R_dict[g2_pair] = np.nan
            else:
                R_dict[g2_pair] = (g2_val ** 2) / (g2_ii * g2_jj)
        data_dict[chunk_name]["R_integrated"] = R_dict
    return data_dict

def compute_R_counts(data_dict):
    for chunk_name, data in data_dict.items():
        R_dict = {}
        g2_counts = data.get("g2_counts", {})
        available_pairs = list(g2_counts.keys())
        
        for g2_pair, g2_val in g2_counts.items():
            ch1, ch2 = g2_pair
            pair_ii = get_auto_pair(ch1, available_pairs)
            pair_jj = get_auto_pair(ch2, available_pairs)
            if not is_cross_harmonic(g2_pair): continue
            
            g2_ii = g2_counts.get(pair_ii, np.nan) if pair_ii else np.nan
            g2_jj = g2_counts.get(pair_jj, np.nan) if pair_jj else np.nan
            
            if g2_val is None or np.isnan(g2_val) or np.isnan(g2_ii) or np.isnan(g2_jj) or (g2_ii * g2_jj) == 0:
                R_dict[g2_pair] = np.nan
            else:
                R_dict[g2_pair] = (g2_val ** 2) / (g2_ii * g2_jj)
        data_dict[chunk_name]["R_counts"] = R_dict
    return data_dict





# ==========================================
# PLOTTING FUNCTIONS (Grids & Summaries)
# ==========================================

def plot_correlations_spectrograms(data_dict, results_dir, power_level, integration_window):
    for chunk_name, data in data_dict.items():
        save_dir = results_dir / power_level / chunk_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_pairs = list(data["correlations"].keys())
        if not all_pairs: continue
        
        fig, axes, auto_pairs, cross_pairs, rows, cols = setup_grid_figure(
            all_pairs, f"Spectrograms - {chunk_name} ({power_level}, Window: {integration_window} ns)", split_auto=True
        )
        
        for i, pair in enumerate(auto_pairs):
            ax = place_in_grid(axes, i, True, cols, split_auto=True)
            _plot_single_spectro(ax, pair, data, integration_window)
            
        for i, pair in enumerate(cross_pairs):
            ax = place_in_grid(axes, i, False, cols, split_auto=True)
            _plot_single_spectro(ax, pair, data, integration_window)
            
        hide_empty_grid(axes, len(auto_pairs), len(cross_pairs), rows, cols, split_auto=True)
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(save_dir / f"{chunk_name}_spectrograms_grid.png")
        plt.close(fig)

def _plot_single_spectro(ax, pair, data, integration_window):
    c_data = data["correlations"][pair]
    d_bins = np.array(c_data["delay_bins"])
    cohs = np.array(c_data["coherences"])
    p_idx = data["peaks"][pair]
    
    g2_int = data.get("g2_integrated", {}).get(pair, np.nan)
    g2_counts = data.get("g2_counts", {}).get(pair, np.nan)

    ax.plot(d_bins, cohs, zorder=2)
    hw = integration_window // 2
    for p in p_idx:
        s_idx, e_idx = max(0, p - hw), min(len(d_bins) - 1, p + hw)
        ax.axvspan(d_bins[s_idx], d_bins[e_idx], color='orange', alpha=0.5, zorder=1)
        
    ax.scatter(d_bins[p_idx], cohs[p_idx], marker="x", color="red", zorder=5)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5, zorder=0)

    ax.text(0.02, 0.95, f"g2_i: {g2_int:.2f}\ng2_c: {g2_counts:.2f}", 
            transform=ax.transAxes, verticalalignment='top', 
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(str(pair), fontsize=10, fontweight='bold' if not is_cross_harmonic(pair) else 'normal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-750, 750)

def plot_window_sweep(data_dict, results_dir, power_level, windows):
    sweep_results = {}
    for chunk_name, data in data_dict.items():
        sweep_results[chunk_name] = {
            "g2_int": defaultdict(list), "g2_counts": defaultdict(list),
            "R_int": defaultdict(list), "R_counts": defaultdict(list)
        }

    for w in windows:
        data_dict = compute_g2_integration(data_dict, w)
        data_dict = compute_g2_counts(data_dict, w)
        data_dict = compute_R_integration(data_dict)
        data_dict = compute_R_counts(data_dict)
        for chunk_name, data in data_dict.items():
            for pair in data["correlations"].keys():
                sweep_results[chunk_name]["g2_int"][pair].append(data.get("g2_integrated", {}).get(pair, np.nan))
                sweep_results[chunk_name]["g2_counts"][pair].append(data.get("g2_counts", {}).get(pair, np.nan))
                sweep_results[chunk_name]["R_int"][pair].append(data.get("R_integrated", {}).get(pair, np.nan))
                sweep_results[chunk_name]["R_counts"][pair].append(data.get("R_counts", {}).get(pair, np.nan))

    w_arr = np.array(windows)
    for chunk_name, data in data_dict.items():
        save_dir = results_dir / power_level / chunk_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_pairs = list(data["correlations"].keys())
        pair_colors = get_color_map(all_pairs)
        
        # --- 1. GRID: g2 vs Window ---
        fig, axes, auto_pairs, cross_pairs, rows, cols = setup_grid_figure(
            all_pairs, f"$g^{{(2)}}(0)$ Sweeps - {chunk_name} ({power_level})", split_auto=True
        )
        for i, pair in enumerate(auto_pairs):
            ax = place_in_grid(axes, i, True, cols, split_auto=True)
            _plot_sweep_ax(ax, w_arr, sweep_results[chunk_name], pair, "g2")
        for i, pair in enumerate(cross_pairs):
            ax = place_in_grid(axes, i, False, cols, split_auto=True)
            _plot_sweep_ax(ax, w_arr, sweep_results[chunk_name], pair, "g2")
        hide_empty_grid(axes, len(auto_pairs), len(cross_pairs), rows, cols, split_auto=True)
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(save_dir / f"{chunk_name}_g2_sweep_grid.png")
        plt.close(fig)
        
        # --- 2. GRID: R vs Window ---
        if cross_pairs:
            fig, axes, _, cp, rows_r, cols = setup_grid_figure(
                cross_pairs, f"Parameter R Sweeps - {chunk_name} ({power_level})", split_auto=False
            )
            for i, pair in enumerate(cp):
                ax = place_in_grid(axes, i, False, cols, split_auto=False)
                _plot_sweep_ax(ax, w_arr, sweep_results[chunk_name], pair, "R")
            hide_empty_grid(axes, 0, len(cp), rows_r, cols, split_auto=False)
            plt.tight_layout(); plt.subplots_adjust(top=0.92)
            plt.savefig(save_dir / f"{chunk_name}_R_sweep_grid.png")
            plt.close(fig)

        # --- 3 & 4. OVERLAP SUMMARIES ---
        _plot_overlap_summary(save_dir / f"{chunk_name}_summary_g2_overlap.png", 
                              w_arr, all_pairs, pair_colors, sweep_results[chunk_name], "g2", f"({chunk_name})")
        if cross_pairs:
            _plot_overlap_summary(save_dir / f"{chunk_name}_summary_R_overlap.png", 
                                  w_arr, cross_pairs, pair_colors, sweep_results[chunk_name], "R", f"({chunk_name})")

def _plot_sweep_ax(ax, w_arr, res_dict, pair, m_type):
    ax.plot(w_arr, res_dict[f"{m_type}_counts"][pair], label='Counts', color='red', marker='.')
    ax.plot(w_arr, res_dict[f"{m_type}_int"][pair], label='Int', color='blue', marker='.')
    ax.set_title(str(pair), fontsize=10, fontweight='bold' if not is_cross_harmonic(pair) else 'normal')
    ax.set_xlabel("Integration Window (ns)")
    apply_shading(ax, m_type)
    ax.grid(True, alpha=0.3)
    if (not is_cross_harmonic(pair) or ax.get_subplotspec().colspan.start == 1) and ax.get_subplotspec().rowspan.start == 0:
        ax.legend(fontsize=8)

def _plot_overlap_summary(path, x_arr, pairs, p_colors, res_dict, m_type, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=700)
    for pair in pairs:
        c = p_colors[pair]
        ax1.plot(x_arr, res_dict[f"{m_type}_counts"][pair], color=c, label=str(pair))
        ax2.plot(x_arr, res_dict[f"{m_type}_int"][pair], color=c, label=str(pair))
    
    sym = "$g^{(2)}(0)$" if m_type == "g2" else "R"
    ax1.set_title(f"All {sym} (Counts) {title_suffix}"); apply_shading(ax1, m_type); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Integration Window (ns)")
    ax2.set_title(f"All {sym} (Integrated) {title_suffix}"); apply_shading(ax2, m_type); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Integration Window (ns)")
    ax1.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def plot_power_summary(data_dict, results_dir, power_level, windows):
    if not data_dict:
        return
        
    first_chunk = list(data_dict.keys())[0]
    all_pairs = list(data_dict[first_chunk]["correlations"].keys())
    pair_colors = get_color_map(all_pairs)

    aggregated = {
        p: {"g2_int": {w: [] for w in windows}, "g2_counts": {w: [] for w in windows},
            "R_int": {w: [] for w in windows}, "R_counts": {w: [] for w in windows}} 
        for p in all_pairs
    }

    for w in windows:
        data_dict = compute_g2_integration(data_dict, w)
        data_dict = compute_g2_counts(data_dict, w)
        data_dict = compute_R_integration(data_dict)
        data_dict = compute_R_counts(data_dict)
        for chunk_name, data in data_dict.items():
            for p in all_pairs:
                aggregated[p]["g2_int"][w].append(data.get("g2_integrated", {}).get(p, np.nan))
                aggregated[p]["g2_counts"][w].append(data.get("g2_counts", {}).get(p, np.nan))
                aggregated[p]["R_int"][w].append(data.get("R_integrated", {}).get(p, np.nan))
                aggregated[p]["R_counts"][w].append(data.get("R_counts", {}).get(p, np.nan))

    save_dir = results_dir / power_level / "power_summary"
    save_dir.mkdir(parents=True, exist_ok=True)
    w_arr = np.array(windows)
    
    plot_data = {}
    for p in all_pairs:
        plot_data[p] = {
            'g2_int_m': np.array([_safe_nanmean(aggregated[p]["g2_int"][w]) for w in windows]),
            'g2_int_s': np.array([_safe_nanstd(aggregated[p]["g2_int"][w]) for w in windows]),
            'g2_counts_m': np.array([_safe_nanmean(aggregated[p]["g2_counts"][w]) for w in windows]),
            'g2_counts_s': np.array([_safe_nanstd(aggregated[p]["g2_counts"][w]) for w in windows]),
            'R_int_m': np.array([_safe_nanmean(aggregated[p]["R_int"][w]) for w in windows]),
            'R_int_s': np.array([_safe_nanstd(aggregated[p]["R_int"][w]) for w in windows]),
            'R_counts_m': np.array([_safe_nanmean(aggregated[p]["R_counts"][w]) for w in windows]),
            'R_counts_s': np.array([_safe_nanstd(aggregated[p]["R_counts"][w]) for w in windows]),
        }

    # --- 1. GRID: Averaged g2 vs Window ---
    fig, axes, auto_pairs, cross_pairs, rows, cols = setup_grid_figure(
        all_pairs, f"Averaged $g^{{(2)}}(0)$ vs Window - All Chunks ({power_level})", split_auto=True
    )
    for i, pair in enumerate(auto_pairs): _plot_averaged_ax(place_in_grid(axes, i, True, cols, True), w_arr, plot_data[pair], pair, "g2")
    for i, pair in enumerate(cross_pairs): _plot_averaged_ax(place_in_grid(axes, i, False, cols, True), w_arr, plot_data[pair], pair, "g2")
    hide_empty_grid(axes, len(auto_pairs), len(cross_pairs), rows, cols, split_auto=True)
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(save_dir / "averaged_g2_sweep_grid.png")
    plt.close(fig)
    
    # --- 2. GRID: Averaged R vs Window ---
    if cross_pairs:
        fig, axes, _, cp, rows_r, cols = setup_grid_figure(
            cross_pairs, f"Averaged R vs Window - All Chunks ({power_level})", split_auto=False
        )
        for i, pair in enumerate(cp): _plot_averaged_ax(place_in_grid(axes, i, False, cols, False), w_arr, plot_data[pair], pair, "R")
        hide_empty_grid(axes, 0, len(cp), rows_r, cols, split_auto=False)
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(save_dir / "averaged_R_sweep_grid.png")
        plt.close(fig)

    # --- 3 & 4. OVERLAP SUMMARIES ---
    _plot_overlap_summary_errors(save_dir / "summary_all_g2_overlap.png", w_arr, all_pairs, pair_colors, plot_data, "g2", f"({power_level})")
    if cross_pairs:
        _plot_overlap_summary_errors(save_dir / "summary_all_R_overlap.png", w_arr, cross_pairs, pair_colors, plot_data, "R", f"({power_level})")

def _plot_averaged_ax(ax, x_arr, d, pair, m_type):
    ax.plot(x_arr, d[f'{m_type}_counts_m'], color='red', label='Counts', marker='.')
    ax.fill_between(x_arr, d[f'{m_type}_counts_m'] - d[f'{m_type}_counts_s'], d[f'{m_type}_counts_m'] + d[f'{m_type}_counts_s'], color='red', alpha=0.2)
    ax.plot(x_arr, d[f'{m_type}_int_m'], color='blue', label='Int', marker='.')
    ax.fill_between(x_arr, d[f'{m_type}_int_m'] - d[f'{m_type}_int_s'], d[f'{m_type}_int_m'] + d[f'{m_type}_int_s'], color='blue', alpha=0.2)
    ax.set_title(str(pair), fontsize=10, fontweight='bold' if not is_cross_harmonic(pair) else 'normal')
    ax.set_xlabel("Integration Window (ns)")
    apply_shading(ax, m_type)
    ax.grid(True, alpha=0.3)
    if (not is_cross_harmonic(pair) or ax.get_subplotspec().colspan.start == 1) and ax.get_subplotspec().rowspan.start == 0:
        ax.legend(fontsize=8)

def _plot_overlap_summary_errors(path, x_arr, pairs, p_colors, plot_data, m_type, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=700)
    for p in pairs:
        c = p_colors[p]
        d = plot_data[p]
        ax1.plot(x_arr, d[f'{m_type}_counts_m'], color=c, label=str(p))
        ax1.fill_between(x_arr, d[f'{m_type}_counts_m'] - d[f'{m_type}_counts_s'], d[f'{m_type}_counts_m'] + d[f'{m_type}_counts_s'], color=c, alpha=0.15)
        ax2.plot(x_arr, d[f'{m_type}_int_m'], color=c, label=str(p))
        ax2.fill_between(x_arr, d[f'{m_type}_int_m'] - d[f'{m_type}_int_s'], d[f'{m_type}_int_m'] + d[f'{m_type}_int_s'], color=c, alpha=0.15)
        
    sym = "$g^{(2)}(0)$" if m_type == "g2" else "R"
    ax1.set_title(f"Averaged {sym} (Counts) {title_suffix}"); apply_shading(ax1, m_type); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Integration Window (ns)")
    ax2.set_title(f"Averaged {sym} (Integrated) {title_suffix}"); apply_shading(ax2, m_type); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Integration Window (ns)")
    ax1.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def aggregate_power_data(data_dict, global_data, power_level):
    match = re.search(r'\d+(?:\.\d+)?', power_level)
    power_val = float(match.group()) if match else 0.0
    global_data["powers"].append(power_val)
    
    first_chunk = list(data_dict.keys())[0]
    all_pairs = list(data_dict[first_chunk]["correlations"].keys())
    
    for pair in all_pairs:
        if pair not in global_data["pairs"]:
            global_data["pairs"][pair] = defaultdict(list)
        
        g2_i = [data.get("g2_integrated", {}).get(pair, np.nan) for data in data_dict.values()]
        g2_c = [data.get("g2_counts", {}).get(pair, np.nan) for data in data_dict.values()]
        r_i = [data.get("R_integrated", {}).get(pair, np.nan) for data in data_dict.values()]
        r_c = [data.get("R_counts", {}).get(pair, np.nan) for data in data_dict.values()]
        
        global_data["pairs"][pair]["g2_int_m"].append(_safe_nanmean(g2_i))
        global_data["pairs"][pair]["g2_int_s"].append(_safe_nanstd(g2_i))
        global_data["pairs"][pair]["g2_counts_m"].append(_safe_nanmean(g2_c))
        global_data["pairs"][pair]["g2_counts_s"].append(_safe_nanstd(g2_c))
        global_data["pairs"][pair]["R_int_m"].append(_safe_nanmean(r_i))
        global_data["pairs"][pair]["R_int_s"].append(_safe_nanstd(r_i))
        global_data["pairs"][pair]["R_counts_m"].append(_safe_nanmean(r_c))
        global_data["pairs"][pair]["R_counts_s"].append(_safe_nanstd(r_c))
    return global_data

def plot_metrics_vs_power(global_data, results_dir):
    save_dir = results_dir / "power_sweep_summary"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    powers = np.array(global_data["powers"])
    sort_idx = np.argsort(powers)
    powers = powers[sort_idx]
    
    all_pairs = list(global_data["pairs"].keys())
    pair_colors = get_color_map(all_pairs)
    
    plot_data = {}
    for p, metrics in global_data["pairs"].items():
        plot_data[p] = {k: np.array(v)[sort_idx] for k, v in metrics.items()}

    # --- 1. GRID: g2 vs Power ---
    fig, axes, auto_pairs, cross_pairs, rows, cols = setup_grid_figure(
        all_pairs, f"$g^{{(2)}}(0)$ vs Power Level", split_auto=True
    )
    for i, pair in enumerate(auto_pairs): _plot_averaged_ax_power(place_in_grid(axes, i, True, cols, True), powers, plot_data[pair], pair, "g2")
    for i, pair in enumerate(cross_pairs): _plot_averaged_ax_power(place_in_grid(axes, i, False, cols, True), powers, plot_data[pair], pair, "g2")
    hide_empty_grid(axes, len(auto_pairs), len(cross_pairs), rows, cols, split_auto=True)
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(save_dir / "g2_vs_power_grid.png")
    plt.close(fig)

    # --- 2. GRID: R vs Power ---
    cross_pairs_only = [p for p in all_pairs if is_cross_harmonic(p)]
    if cross_pairs_only:
        fig, axes, _, cp, rows_r, cols = setup_grid_figure(
            cross_pairs_only, f"Parameter R vs Power Level", split_auto=False
        )
        for i, pair in enumerate(cp): _plot_averaged_ax_power(place_in_grid(axes, i, False, cols, False), powers, plot_data[pair], pair, "R")
        hide_empty_grid(axes, 0, len(cp), rows_r, cols, split_auto=False)
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(save_dir / "R_vs_power_grid.png")
        plt.close(fig)

    # --- 3 & 4. OVERLAP SUMMARIES ---
    _plot_overlap_summary_errors_power(save_dir / "summary_all_g2_vs_power.png", powers, all_pairs, pair_colors, plot_data, "g2", "(vs Power)")
    if cross_pairs_only:
        _plot_overlap_summary_errors_power(save_dir / "summary_all_R_vs_power.png", powers, cross_pairs_only, pair_colors, plot_data, "R", "(vs Power)")

def _plot_averaged_ax_power(ax, x_arr, d, pair, m_type):
    ax.plot(x_arr, d[f'{m_type}_counts_m'], color='red', label='Counts', marker='.')
    ax.fill_between(x_arr, d[f'{m_type}_counts_m'] - d[f'{m_type}_counts_s'], d[f'{m_type}_counts_m'] + d[f'{m_type}_counts_s'], color='red', alpha=0.2)
    ax.plot(x_arr, d[f'{m_type}_int_m'], color='blue', label='Int', marker='.')
    ax.fill_between(x_arr, d[f'{m_type}_int_m'] - d[f'{m_type}_int_s'], d[f'{m_type}_int_m'] + d[f'{m_type}_int_s'], color='blue', alpha=0.2)
    ax.set_title(str(pair), fontsize=10, fontweight='bold' if not is_cross_harmonic(pair) else 'normal')
    ax.set_xlabel("Power (mW)")
    apply_shading(ax, m_type)
    ax.grid(True, alpha=0.3)
    if (not is_cross_harmonic(pair) or ax.get_subplotspec().colspan.start == 1) and ax.get_subplotspec().rowspan.start == 0:
        ax.legend(fontsize=8)

def _plot_overlap_summary_errors_power(path, x_arr, pairs, p_colors, plot_data, m_type, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=700)
    for p in pairs:
        c = p_colors[p]
        d = plot_data[p]
        ax1.plot(x_arr, d[f'{m_type}_counts_m'], color=c, label=str(p))
        ax1.fill_between(x_arr, d[f'{m_type}_counts_m'] - d[f'{m_type}_counts_s'], d[f'{m_type}_counts_m'] + d[f'{m_type}_counts_s'], color=c, alpha=0.15)
        ax2.plot(x_arr, d[f'{m_type}_int_m'], color=c, label=str(p))
        ax2.fill_between(x_arr, d[f'{m_type}_int_m'] - d[f'{m_type}_int_s'], d[f'{m_type}_int_m'] + d[f'{m_type}_int_s'], color=c, alpha=0.15)
        
    sym = "$g^{(2)}(0)$" if m_type == "g2" else "R"
    ax1.set_title(f"Averaged {sym} (Counts) {title_suffix}"); apply_shading(ax1, m_type); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Power (mW)")
    ax2.set_title(f"Averaged {sym} (Integrated) {title_suffix}"); apply_shading(ax2, m_type); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Power (mW)")
    ax1.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


# ============================================================================
# STABILITY: g2(0) AND R VERSUS TIME (live snapshots)
# ============================================================================

def plot_stability(snapshots: Dict[str, Any], cfg: ExperimentConfig, results_dir: Path | str | None = None) -> Optional[Dict[str, Any]]:
    """Compute and plot g2(0)(t) and R(t). Returns the series, or None if there
    are not enough snapshots. Figures are saved, never shown (headless-friendly)."""
    if not snapshots or len(snapshots.get("times", [])) < 2:
        print("Stability: fewer than 2 snapshots, nothing to plot.")
        return None

    out_dir = Path(results_dir) if results_dir is not None else cfg.stability_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    times_min = np.asarray(snapshots["times"], dtype=float) / 60.0
    delays = np.asarray(snapshots["index"], dtype=float)
    hist = snapshots["hist"]
    pairs = list(hist.keys())

    t_rep_ps = 1e12 / cfg.FREQUENCY
    orders = [n for n in range(-cfg.STABILITY_PEAKS, cfg.STABILITY_PEAKS + 1) if n != 0]
    window_ns = cfg.STABILITY_INTEGRATION_WINDOW_NS
    # Center each pair on its own measured delay, read on the richest histogram.
    centered = {pair: delays - measured_delay(hist[pair][-1], delays, t_rep_ps) for pair in pairs}

    g2_cum = _stability_g2_series(hist, centered, t_rep_ps, window_ns, orders, window_snaps=None)
    g2_roll = _stability_g2_series(hist, centered, t_rep_ps, window_ns, orders, window_snaps=cfg.STABILITY_ROLLING_WINDOW_SNAPS)
    r_cum = _stability_r_series(g2_cum, cfg)
    r_roll = _stability_r_series(g2_roll, cfg)

    labels = {pair: _stability_pair_label(pair, cfg) for pair in pairs}
    r_labels = {pair: f"R {_stability_pair_label(pair, cfg)}" for pair in r_cum}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    _plot_stability_series(ax, times_min, g2_cum, labels, r"$g^{(2)}(0)$",
                 f"Cumulative $g^{{(2)}}(0)$ vs time | {window_ns:g} ns window")
    _save_stability_figure(fig, out_dir / "stability_g2.png")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    _plot_stability_series(ax, times_min, r_cum, r_labels, r"$R = g_{nm}^2 / (g_{nn} g_{mm})$",
                 "Cumulative R vs time")
    _save_stability_figure(fig, out_dir / "stability_R.png")

    window = cfg.STABILITY_ROLLING_WINDOW_SNAPS
    fig, (ax_g2, ax_r) = plt.subplots(1, 2, figsize=(20, 6), dpi=300, sharex=True)
    _plot_stability_series(ax_g2, times_min, g2_roll, labels, r"$g^{(2)}(0)$",
                 f"Rolling $g^{{(2)}}(0)$ (last {window} snapshots)")
    _plot_stability_series(ax_r, times_min, r_roll, r_labels, r"$R$", f"Rolling R (last {window} snapshots)")
    _save_stability_figure(fig, out_dir / "stability_rolling.png")

    print(f"Stability plots saved to {out_dir}")
    return {"times_min": times_min, "g2": g2_cum, "R": r_cum, "g2_rolling": g2_roll, "R_rolling": r_roll}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def measured_delay(counts: np.ndarray, delays_ps: np.ndarray, t_rep_ps: float) -> float:
    """Position of the zero-delay coincidence peak, searched within half a laser
    period of zero (same rule as `pkl_json_analyze.get_peaks`).

    The global maximum is not usable here: for an antibunched source the central
    peak is the *smallest* one, so the maximum would lock onto a satellite and
    force g2(0) to ~1. This assumes cable delays are compensated through
    INPUT_DELAYS_PS so the real peak sits near zero.
    """
    counts = np.asarray(counts, dtype=float)
    center = np.abs(delays_ps) <= t_rep_ps / 2
    if not np.any(center):
        return float(delays_ps[int(np.argmax(counts))])
    return float(delays_ps[center][int(np.argmax(counts[center]))])


def g2_from_histogram(
    counts: np.ndarray,
    delays_ps: np.ndarray,
    t_rep_ps: float,
    window_ns: float,
    peak_orders: Sequence[int],
) -> float:
    """Central peak area divided by the mean satellite peak area.

    `delays_ps` must already be centered on the coincidence peak. Peaks are taken
    at the ideal positions n * t_rep_ps, which is robust even when a satellite
    peak is too weak to be detected on its own. Orders that fall outside the
    recorded delay range are dropped, otherwise they would count as empty peaks
    and inflate g2 whenever STABILITY_PEAKS asks for more peaks than the
    histogram holds.
    """
    half = window_ns * 1e3 / 2.0
    central = float(np.sum(counts[np.abs(delays_ps) <= half]))

    sides = []
    for order in peak_orders:
        center = order * t_rep_ps
        if center - half < delays_ps[0] or center + half > delays_ps[-1]:
            continue
        sides.append(float(np.sum(counts[np.abs(delays_ps - center) <= half])))

    mean_side = float(np.mean(sides)) if sides else 0.0
    return central / mean_side if mean_side > 0 else np.nan


def _stability_g2_series(
    hist: Dict[Pair, List[np.ndarray]],
    centered: Dict[Pair, np.ndarray],
    t_rep_ps: float,
    window_ns: float,
    peak_orders: Sequence[int],
    window_snaps: Optional[int],
) -> Dict[Pair, np.ndarray]:
    """g2(0) at every snapshot, cumulative (window_snaps=None) or rolling."""
    n_snaps = len(next(iter(hist.values())))
    series = {pair: np.full(n_snaps, np.nan) for pair in hist}

    for pair, snaps in hist.items():
        for k in range(n_snaps):
            counts = np.asarray(snaps[k], dtype=float)
            if window_snaps is not None:
                if k - window_snaps < 0:
                    continue  # window not full yet
                counts = counts - np.asarray(snaps[k - window_snaps], dtype=float)
            series[pair][k] = g2_from_histogram(counts, centered[pair], t_rep_ps, window_ns, peak_orders)
    return series


def _stability_r_series(g2: Dict[Pair, np.ndarray], cfg: ExperimentConfig) -> Dict[Pair, np.ndarray]:
    """R = g2_cross^2 / (g2_autoA * g2_autoB) for every cross-harmonic pair."""
    harmonics = cfg.active_harmonics()
    autos = {name: _stability_lookup(g2, channels[0], channels[1]) for name, channels in harmonics.items()}

    r: Dict[Pair, np.ndarray] = {}
    for name_a, name_b in itertools.combinations(harmonics, 2):
        auto_a, auto_b = autos[name_a], autos[name_b]
        if auto_a is None or auto_b is None:
            continue
        denominator = np.asarray(auto_a, dtype=float) * np.asarray(auto_b, dtype=float)
        for ch_a, ch_b in itertools.product(harmonics[name_a], harmonics[name_b]):
            cross = _stability_lookup(g2, ch_a, ch_b)
            if cross is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r[(min(ch_a, ch_b), max(ch_a, ch_b))] = np.where(
                    denominator > 0, np.asarray(cross, dtype=float) ** 2 / denominator, np.nan
                )
    return r


def _stability_lookup(g2: Dict[Pair, np.ndarray], ch_a: int, ch_b: int) -> Optional[np.ndarray]:
    return g2.get((ch_a, ch_b), g2.get((ch_b, ch_a)))


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def _stability_pair_label(pair: Pair, cfg: ExperimentConfig) -> str:
    return " - ".join(cfg.MODES.get(ch, f"Ch{ch}") for ch in pair)


def _plot_stability_series(ax, times_min, series, labels, ylabel: str, title: str) -> None:
    for i, (pair, values) in enumerate(sorted(series.items())):
        ax.plot(times_min, values, color=f"C{i % 10}", linewidth=1.2, label=labels.get(pair, str(pair)))
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if series:
        ax.legend(ncol=2, fontsize=8)


def _save_stability_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# ============================================================================
# LASER ANGLE SCAN: CIRCLE / BUTTERFLY SUMMARIES VERSUS ANGLE
# ============================================================================
#
# Four figures per power, under results/.../laser_angle/{power}/summary/:
#   1. intensity_vs_angle.png  — per HARMONIC (total + its two arms): polar on top,
#                                  the same curves on linear axes below. The angles of
#                                  the interpolated minima land in interpolated_minima.csv
#   2. g2_vs_angle.png         — one row per harmonic: its auto pair, then a whole
#                                  cross family (33 | 34…, 44 | 35…, 55 | 45…)
#   3. R_vs_angle.png          — one row per cross family (34…, 35…, 45…)
#   4. harmonics_vs_angle.png  — per HARMONIC: intensity on top, auto-g² below
#                                  (2×N grid → 4 panels for two harmonics, 6 for three)
#
# When several powers share a DATE folder, the same four layouts are redrawn under
# results/.../laser_angle/overlay/ with every power as a coloured curve on each panel.
#
# The four figures are written by `plot_angle_summary`, which knows nothing about WHICH
# angle is on the axis: an ellipticity scan reuses them for its analyzer sweeps.


def plot_laser_angle_summary(cfg: ExperimentConfig,
                             results_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Write the four polar summaries versus pump angle for one power."""
    power = cfg.base_power
    points = _laser_angle_points(cfg, power=power)
    if not points:
        print(f"Laser angle summary [{power}]: no angles configured.")
        return None

    series = collect_angle_series(cfg, points, power_label=power, angle_name="laser angle")
    if len(series["angles"]) < 2:
        print(f"Laser angle summary [{power}]: fewer than 2 angles with data, nothing to plot.")
        return None

    save_dir = Path(results_dir) if results_dir else cfg.laser_angle_summary_dir(power)
    plot_angle_summary(series, save_dir)
    print(f"Laser angle summary [{power}]: {len(series['angles'])} angles -> {save_dir}")
    return series


def plot_angle_summary(series: Dict[str, Any], save_dir: Path) -> Path:
    """The four vs-angle figures of one series, whatever angle it scanned."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_angle_intensity(series, save_dir)
    plot_angle_g2(series, save_dir)
    plot_angle_r(series, save_dir)
    plot_angle_harmonics(series, save_dir)
    return save_dir


def plot_laser_angle_campaign(cfg: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """Per-power summaries for every power in the folder, then the overlay.

    Powers come from `PUMP_POWERS` when set, otherwise from the files on disk (and
    falling back to the single `POWER_LEVEL` of this config).
    """
    powers = list(cfg.PUMP_POWERS) if cfg.PUMP_POWERS else discover_laser_angle_powers(cfg.DATA_DIR)
    if not powers:
        powers = [cfg.POWER_LEVEL]

    series_by_power: Dict[str, Dict[str, Any]] = {}
    for power in powers:
        power_cfg = cfg.for_pump_power(power)
        if cfg.LASER_ANGLE_SCAN is not None:
            # Keep the planned angles so a folder can be replotted without the CSV log.
            power_cfg = replace(power_cfg, LASER_ANGLE_SCAN=list(cfg.LASER_ANGLE_SCAN))
        series = plot_laser_angle_summary(power_cfg)
        if series is not None:
            series_by_power[power] = series

    if len(series_by_power) > 1:
        plot_laser_angle_overlay(series_by_power, cfg.laser_angle_overlay_dir)
    elif len(series_by_power) == 1:
        print("Laser angle overlay: only one power present, skipped.")

    return series_by_power


def discover_laser_angle_powers(data_dir: Path) -> List[str]:
    """Power labels that have at least one `{power}_lasXXX_num*` file in the folder.

    The `_num` anchor is what keeps an ellipticity scan's `{power}_lasXXX_hwpYYY_num*`
    files out of the list when both scans share a DATE folder.
    """
    return _discover_powers(data_dir, "*_las*_num*", r"(.+)_las\d{3}p\d_num")


def discover_ellipticity_powers(data_dir: Path) -> List[str]:
    """Power labels that have at least one `{power}_lasXXX_hwpYYY_num*` file."""
    return _discover_powers(data_dir, "*_las*_hwp*_num*", r"(.+)_las\d{3}p\d_hwp\d{3}p\d_num")


def _discover_powers(data_dir: Path, glob: str, pattern: str) -> List[str]:
    found: List[str] = []
    for path in Path(data_dir).glob(glob):
        match = re.match(pattern, path.name)
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return sorted(found, key=_power_sort_key)


def _power_sort_key(power: str) -> float:
    digits = "".join(c for c in power if c.isdigit() or c == ".")
    return float(digits) if digits else 0.0


def _laser_angle_points(cfg: ExperimentConfig, power: Optional[str] = None) -> List[Tuple[float, str]]:
    """Angles to summarise for one power: CSV log first, else the planned scan."""
    power = power or cfg.base_power
    log_path = cfg.laser_angle_log_path()
    if log_path.is_file():
        recorded = LaserAngleLog(log_path).recorded_labels(power=power)
        if recorded:
            return recorded
    # Rebuild labels for this power even if cfg.POWER_LEVEL was rewritten.
    if cfg.LASER_ANGLE_SCAN is not None:
        return [(float(angle), f"{power}_{laser_angle_tag(angle)}")
                for angle in cfg.LASER_ANGLE_SCAN]
    return []


# ----------------------------------------------------------------------------
# Loading and averaging
# ----------------------------------------------------------------------------

def collect_angle_series(cfg: ExperimentConfig, points: Sequence[Tuple[float, str]],
                         power_label: str, angle_name: str = "angle",
                         title: Optional[str] = None) -> Dict[str, Any]:
    """Mean and std over the chunks of each angle of one scan.

    `points` is [(angle, file label)]; `angle_name` names the axis ("laser angle",
    "analyzer angle") and `title` overrides what the figures are labelled with, so an
    ellipticity sweep can say which pump angle it belongs to.
    """
    angles: List[float] = []
    n_chunks: List[int] = []
    channels: Dict[str, List[Tuple[float, float]]] = {}
    g2: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    r_values: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    harmonics: Dict[str, List[Tuple[float, float]]] = {}

    for angle, label in points:
        chunk_data = _load_angle_point(cfg, label)
        if not chunk_data:
            print(f"  [{label}] no files found, angle skipped.")
            continue

        angles.append(float(angle))
        n_chunks.append(len(chunk_data))
        chunks = list(chunk_data.values())

        rates = {name: [chunk["countrates"].get(name, np.nan) for chunk in chunks]
                 for name in _countrate_names(chunks, "countrates")}
        for name, values in rates.items():
            channels.setdefault(name, []).append(_mean_std(values))

        for pair in _metric_pairs(chunks, "g2_integrated"):
            g2.setdefault(pair, []).append(
                _mean_std([chunk.get("g2_integrated", {}).get(pair, np.nan) for chunk in chunks]))

        # R exists for the cross pairs only: compute_R_integration skips the autos.
        for pair in _metric_pairs(chunks, "R_integrated"):
            r_values.setdefault(pair, []).append(
                _mean_std([chunk.get("R_integrated", {}).get(pair, np.nan) for chunk in chunks]))

        # Harmonic intensity = sum of its arms, summed per chunk so the spread stays
        # a spread of the total.
        for harmonic, names in _harmonic_channels(rates).items():
            totals = [np.nansum([rates[name][i] for name in names]) for i in range(len(chunks))]
            harmonics.setdefault(harmonic, []).append(_mean_std(totals))

    order = np.argsort(angles)
    return {
        "angles": np.asarray(angles, dtype=float)[order],
        "n_chunks": np.asarray(n_chunks)[order],
        "channels": _series_arrays(channels, order),
        "pairs": _series_arrays(g2, order),
        "R": _series_arrays(r_values, order),
        "harmonics": _series_arrays(harmonics, order),
        "power_label": power_label,
        "angle_name": angle_name,
        "title": title or power_label,
        "clockwise": cfg.PUMP_P1.clockwise if cfg.PUMP_P1 else True,
    }


def _load_angle_point(cfg: ExperimentConfig, label: str) -> Dict[str, Any]:
    """The chunks of one point of a scan, keyed as the per-power tree keys them.

    The `_num` / `_merged` anchors matter: a bare `*{label}*` glob would also pick up
    every `{label}_hwpYYY` file of an ellipticity scan sharing the folder, and average a
    whole analyzer sweep into the point.
    """
    _merged, chunk_files = get_files(cfg.DATA_DIR, pattern=f"*{label}_num*")
    merged_files, _chunks = get_files(cfg.DATA_DIR, pattern=f"*{label}_merged*")
    files = chunk_files or merged_files
    if not files:
        return {}
    data = get_data(files, cfg.MODES, label)
    if not data:
        return {}
    data = get_peaks(data, cfg.FREQUENCY)
    data = compute_g2_integration(data, cfg.INTEGRATION_WINDOW)
    data = compute_R_integration(data)
    return data


def _countrate_names(chunks: Sequence[Dict[str, Any]], key: str) -> List[str]:
    names: List[str] = []
    for chunk in chunks:
        for name in chunk.get(key, {}):
            if name not in names:
                names.append(name)
    return sorted(names)


def _metric_pairs(chunks: Sequence[Dict[str, Any]], key: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for chunk in chunks:
        for pair in chunk.get(key, {}):
            if pair not in pairs:
                pairs.append(pair)
    return sorted(pairs)


def _harmonic_channels(rates: Dict[str, Any]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name in sorted(rates):
        groups.setdefault(harmonic_of(name), []).append(name)
    return groups


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(array)):
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array))


def _series_arrays(collected: Dict[Any, List[Tuple[float, float]]], order) -> Dict[Any, Dict[str, np.ndarray]]:
    out = {}
    for name, values in collected.items():
        mean = np.asarray([v[0] for v in values], dtype=float)[order]
        std = np.asarray([v[1] for v in values], dtype=float)[order]
        out[name] = {"mean": mean, "std": std}
    return out


# ----------------------------------------------------------------------------
# Polar drawing helpers
# ----------------------------------------------------------------------------

def _pair_family(pair: Tuple[str, str]) -> Tuple[str, str]:
    """('H4R', 'H3T') -> ('H3', 'H4'): the two harmonics of a pair, channel order dropped.

    Pairs are keyed in tagger-channel order, so the higher harmonic can come first.
    """
    first, second = sorted((harmonic_of(pair[0]), harmonic_of(pair[1])))
    return first, second


def _pair_label(pair: Tuple[str, str]) -> str:
    """('H4R', 'H3T') -> 'H3T–H4R': a title reading in harmonic order, not channel order."""
    first, second = sorted(pair, key=lambda name: (harmonic_of(name), name))
    return f"{first}–{second}"


def _pairs_by_family(pairs: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for pair in pairs:
        grouped[_pair_family(pair)].append(pair)
    # Sorted the way the panels are titled, so a row reads H3R-H4R, H3R-H4T, H3T-H4R...
    return {family: sorted(members, key=_pair_label) for family, members in grouped.items()}


def _auto_then_cross_rows(pairs: Sequence[Tuple[str, str]]) -> List[List[Optional[Tuple[str, str]]]]:
    """One row per harmonic: its auto pair in column 0, then one whole cross family.

    With H3/H4/H5 that reads "33 then all 34", "44 then all 35", "55 then all 45", so
    the left column walks the auto-correlations while each row carries a single cross
    family. Column 0 stays reserved even when a row has no auto pair, so the families
    keep their own rows.
    """
    grouped = _pairs_by_family(pairs)
    autos = sorted(family for family in grouped if family[0] == family[1])
    crosses = sorted(family for family in grouped if family[0] != family[1])

    rows: List[List[Optional[Tuple[str, str]]]] = []
    for index in range(max(len(autos), len(crosses))):
        auto = grouped[autos[index]] if index < len(autos) else [None]
        cross = grouped[crosses[index]] if index < len(crosses) else []
        rows.append([*auto, *cross])
    return rows


def _cross_family_rows(pairs: Sequence[Tuple[str, str]]) -> List[List[Optional[Tuple[str, str]]]]:
    """One row per cross family: all 34, then all 35, then all 45."""
    grouped = _pairs_by_family(pairs)
    return [grouped[family] for family in sorted(grouped) if family[0] != family[1]]


def _row_grid(rows: Sequence[Sequence[Optional[Tuple[str, str]]]], title: str, dpi: int = 300):
    """Polar axes laid out exactly as `rows` says, `None` leaving a cell empty.

    Returns the figure and a {pair: ax} map.
    """
    n_rows = max(1, len(rows))
    n_cols = max(1, max((len(row) for row in rows), default=1))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.8 * n_rows), dpi=dpi,
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(np.asarray(axes)).reshape(n_rows, n_cols)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax_by_pair: Dict[Tuple[str, str], Any] = {}
    for row_index in range(n_rows):
        row = list(rows[row_index]) if row_index < len(rows) else []
        for col_index in range(n_cols):
            pair = row[col_index] if col_index < len(row) else None
            if pair is None:
                axes[row_index, col_index].set_visible(False)
            else:
                ax_by_pair[pair] = axes[row_index, col_index]
    return fig, ax_by_pair


def _two_row_grid(n_cols: int, title: str, dpi: int = 300):
    """A polar row on top and a Cartesian row below it, one column per harmonic."""
    n_cols = max(1, n_cols)
    fig = plt.figure(figsize=(4.4 * n_cols, 9.4), dpi=dpi)
    grid = fig.add_gridspec(2, n_cols)
    polar_axes = [fig.add_subplot(grid[0, column], projection="polar") for column in range(n_cols)]
    linear_axes = [fig.add_subplot(grid[1, column]) for column in range(n_cols)]
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig, polar_axes, linear_axes


def _fill_polar_circle(angles: np.ndarray, *series: np.ndarray):
    """Tile a partial scan by its own span so a polar plot covers the whole turn.

    Linear polarization repeats every 180 deg, so a 0-180 scan already samples every
    distinct state; copying that arc onto the empty half draws the familiar full
    butterfly instead of a bare semicircle. The same tiling completes a 0-90 scan
    (four copies) or any arc whose span divides 360 evenly, while a genuine 0-360 scan
    is already full and is left untouched.

    A narrow zoom scan (a few close angles taken to resolve one minimum) is not a
    periodic unit of the pattern, so it is drawn as the measured arc rather than
    replicated into a full circle: any span below MIN_TILE_SPAN_DEG is left as is.
    """
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 2:
        return (angles, *series)
    step = float(np.median(np.diff(angles)))
    span = angles[-1] - angles[0] + step
    if not (step > 0) or span >= 359.9:
        return (angles, *series)  # already (near) a full turn: nothing to copy
    if span < MIN_TILE_SPAN_DEG:
        return (angles, *series)  # a zoom scan, not a period to tile: draw the arc as measured
    copies = 360.0 / span
    n = int(round(copies))
    if n < 2 or abs(copies - n) > 1e-3:
        return (angles, *series)  # span does not tile 360 cleanly: leave the arc as is
    filled_angles = np.concatenate([angles + k * span for k in range(n)])
    filled_series = [np.concatenate([np.asarray(s, dtype=float)] * n) for s in series]
    return (filled_angles, *filled_series)


def _close_polar_circle(angles: np.ndarray, *series: np.ndarray):
    """Repeat the first point at +360 deg when the scan covers the whole turn."""
    if len(angles) < 3:
        return (angles, *series)
    step = float(np.median(np.diff(angles))) if len(angles) > 1 else 0.0
    if angles[-1] - angles[0] + step < 359.9:
        return (angles, *series)
    closed_angles = np.append(angles, angles[0] + 360.0)
    return (closed_angles, *[np.append(s, s[0]) for s in series])


def _circle_arrays(angles: np.ndarray, *values: np.ndarray):
    """The samples as the panels draw them: tiled over the whole turn, then closed."""
    return _close_polar_circle(*_fill_polar_circle(angles, *values))


def _fit_circle_spline(angles: np.ndarray, values: np.ndarray):
    """A cubic spline through the samples, and the angles it was fitted on.

    H4 nearly extinguishes between its lobes, and with one sample every 15 deg the
    straight segments between points cut the minimum off at whichever sample happened
    to sit closest to it. Interpolating (periodically when the scan covers the turn, so
    the curve joins itself smoothly at 0 deg) puts the missing curvature back without
    moving any measured point. Returns (None, None) when there is too little to fit.
    """
    angles = np.asarray(angles, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return None, None
    angles, values = angles[finite], values[finite]

    periodic = angles[-1] - angles[0] >= 359.9 and np.isclose(values[0], values[-1])
    return CubicSpline(angles, values, bc_type="periodic" if periodic else "natural"), angles


def _smooth_circle(angles: np.ndarray, values: np.ndarray, floor: Optional[float] = None):
    """The interpolated curve, sampled densely enough to draw. None when it cannot fit."""
    spline, fitted = _fit_circle_spline(angles, values)
    if spline is None:
        return None
    dense = np.linspace(fitted[0], fitted[-1], (len(fitted) - 1) * SMOOTH_SAMPLES_PER_STEP + 1)
    smoothed = spline(dense)
    if floor is not None:
        smoothed = np.clip(smoothed, floor, None)
    return dense, smoothed


def _curve_minima(angles: np.ndarray, values: np.ndarray,
                  floor: Optional[float] = None) -> List[Tuple[float, float]]:
    """Angle and depth of every local minimum of the interpolated curve.

    The roots of the spline derivative give the extrema in closed form, so a minimum is
    located between the measured angles instead of being pinned to whichever one came
    closest. Only minima in the lower half of the curve's range are kept: a spline
    through noisy samples also wobbles near the lobe tops, and those wobbles are not
    extinction minima. Returns (angle_deg, value) sorted by angle.
    """
    spline, fitted = _fit_circle_spline(angles, values)
    if spline is None:
        return []

    curve = spline(fitted)
    midpoint = float(curve.min() + 0.5 * (curve.max() - curve.min()))

    minima: Dict[float, float] = {}
    for root in spline.derivative().roots(extrapolate=False):
        if spline(root, 2) <= 0:            # second derivative <= 0: a maximum, not a dip
            continue
        value = float(spline(root))
        if value > midpoint:
            continue
        if floor is not None:
            value = max(value, floor)
        minima[round(float(root) % 360.0, 1)] = value
    return sorted(minima.items())


def _polar_series(ax, angles: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  color: str, label: Optional[str] = None, alpha: float = BAND_ALPHA,
                  smooth: bool = False, floor: Optional[float] = None,
                  linewidth: float = 1.6, markersize: float = 3.5) -> None:
    angles, mean, std = _circle_arrays(angles, mean, std)
    ax.fill_between(np.deg2rad(angles), np.clip(mean - std, 0.0, None), mean + std,
                    color=color, alpha=alpha, linewidth=0)
    _draw_curve(ax, angles, mean, color, label, smooth, floor, linewidth, markersize,
                to_x=np.deg2rad)


def _linear_series(ax, angles: np.ndarray, mean: np.ndarray, std: np.ndarray,
                   color: str, label: Optional[str] = None, alpha: float = BAND_ALPHA,
                   smooth: bool = False, floor: Optional[float] = None,
                   linewidth: float = 1.6, markersize: float = 3.5) -> None:
    """The same curve on Cartesian axes: angle across, value up."""
    angles, mean, std = _circle_arrays(angles, mean, std)
    ax.fill_between(angles, np.clip(mean - std, 0.0, None), mean + std,
                    color=color, alpha=alpha, linewidth=0)
    _draw_curve(ax, angles, mean, color, label, smooth, floor, linewidth, markersize)


def _draw_curve(ax, angles: np.ndarray, mean: np.ndarray, color: str, label: Optional[str],
                smooth: bool, floor: Optional[float], linewidth: float, markersize: float,
                to_x=None) -> None:
    """Points joined by segments, or by an interpolated curve with the points on top.

    `to_x` maps degrees to the axis coordinate (radians on a polar panel, degrees on a
    linear one), so both kinds of panel share the interpolation.
    """
    to_x = to_x if to_x is not None else (lambda degrees: degrees)
    curve = _smooth_circle(angles, mean, floor=floor) if smooth else None
    if curve is None:
        ax.plot(to_x(angles), mean, color=color, marker="o", markersize=markersize,
                linewidth=linewidth, label=label)
        return
    dense_angles, dense_mean = curve
    ax.plot(to_x(dense_angles), dense_mean, color=color, linewidth=linewidth, label=label)
    ax.plot(to_x(angles), mean, color=color, marker="o", markersize=markersize, linestyle="none")


def _draw_sinusoid(ax, fit: Dict[str, Any], angles: np.ndarray, to_x=None,
                   color: str = "0.15") -> None:
    """The fitted sinusoid over the drawn angle range, labelled with what it measures.

    Drawn instead of the cubic spline of SMOOTH_HARMONICS: an analyzer sweep is a
    sinusoid by construction, so the fit IS the curve, and its depth is the number the
    ellipticity comes from.
    """
    to_x = to_x if to_x is not None else (lambda degrees: degrees)
    dense = np.linspace(float(angles[0]), float(angles[-1]), 361)
    values = np.clip(sinusoid(dense, fit), 0.0, None)
    label = (f"fit: m={fit['modulation']:.3f}, "
             f"$\\epsilon$={fit['ellipticity']:.3f}")
    ax.plot(to_x(dense), values, color=color, linestyle="--", linewidth=1.4,
            label=label, zorder=4)


def _mark_minima(ax, minima: Sequence[Tuple[float, float]], color: str = "0.2",
                 to_x=None, annotate: bool = False, label: Optional[str] = None) -> None:
    """Ring the interpolated minima, and on a linear panel write the angle of each."""
    if not minima:
        return
    to_x = to_x if to_x is not None else (lambda degrees: degrees)
    angles = np.array([angle for angle, _ in minima], dtype=float)
    values = np.array([value for _, value in minima], dtype=float)
    ax.plot(to_x(angles), values, linestyle="none", marker="o", markersize=7,
            markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.3,
            label=label, zorder=5)
    if not annotate:
        return
    for angle, value in minima:
        ax.annotate(f"{angle:g}°", (angle, value), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=6.5, color=color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))


def _polar_style(ax, title: str, clockwise: bool = True) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1 if clockwise else 1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=14)
    ax.set_rlabel_position(112.5)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


def _linear_style(ax, title: str, ylabel: str, angles: np.ndarray,
                  angle_name: str = "angle") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(f"{angle_name} (deg)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    first, last = float(angles[0]), float(angles[-1])
    ax.set_xlim(first, last)
    ax.set_xticks(np.arange(45.0 * math.ceil(first / 45.0), last + 1e-6, 45.0))
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


def _unity_circle(ax, level: float = 1.0) -> None:
    """Dashed g²=1 / R=1 reference ring. The radial limits come from `_focus_radial`."""
    theta = np.linspace(0, 2 * np.pi, 181)
    ax.plot(theta, np.full_like(theta, level), color="0.45", linestyle="--", linewidth=0.9, zorder=1)


def _data_limits(curves: Sequence[np.ndarray], floor: Optional[float] = None,
                 pad: float = 0.08) -> Optional[Tuple[float, float]]:
    """Limits set by the means alone, ignoring how far the spread band reaches.

    A few angles can carry a chunk spread of several hundred percent, and letting the
    autoscale fit those bands squashes every curve on the panel into a dot near the
    centre. Fitting the means keeps the measured shape readable and simply clips the
    tall bands at the rim. `floor` (0 for an intensity, 1 for g²/R) anchors the inner
    edge when the data stays above it, and is dropped when the data goes below.
    """
    finite = [np.asarray(curve, dtype=float)[np.isfinite(np.asarray(curve, dtype=float))]
              for curve in curves]
    values = np.concatenate(finite) if any(part.size for part in finite) else np.array([])
    if values.size == 0:
        return None

    low, high = float(values.min()), float(values.max())
    anchored = floor is not None and low >= floor
    if anchored:
        low = float(floor)
    span = (high - low) or abs(high) or 1.0
    return (low if anchored else low - pad * span), high + pad * span


def _focus_radial(ax, curves: Sequence[np.ndarray], floor: Optional[float] = None) -> None:
    limits = _data_limits(curves, floor=floor)
    if limits:
        ax.set_rlim(*limits)


def _focus_vertical(ax, curves: Sequence[np.ndarray], floor: Optional[float] = None) -> None:
    limits = _data_limits(curves, floor=floor)
    if limits:
        ax.set_ylim(*limits)


def _save_polar_figure(fig, path: Path) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def _power_colors(powers: Sequence[str]) -> Dict[str, Any]:
    colors = cm.viridis(np.linspace(0.15, 0.85, max(len(powers), 1)))
    return {power: colors[i] for i, power in enumerate(powers)}


def _power_legend(fig, colors: Dict[str, Any]) -> None:
    """One legend for the whole figure: a per-panel one would land on a neighbour."""
    handles = [plt.Line2D([], [], color=color, marker="o", markersize=4, label=power)
               for power, color in colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=True,
               title="power", title_fontsize=9)


# ----------------------------------------------------------------------------
# The four figures (single power)
# ----------------------------------------------------------------------------

def plot_angle_intensity(series: Dict[str, Any], save_dir: Path) -> None:
    """Per harmonic: the polar butterfly on top, the same curves on linear axes below.

    Each column carries the harmonic total (the sum of its arms, as in
    harmonics_vs_angle.png) plus the arms themselves as thin curves, so a lopsided pair
    of arms shows up without a figure of its own. The linear row is the same data read
    the other way: it separates lobes of similar radius that the polar panel overlaps.

    The interpolation of SMOOTH_HARMONICS is applied to the sum only, and its minima are
    ringed on both rows, labelled with their angle on the linear one, and written to
    interpolated_minima.csv beside the figure.

    An ellipticity sweep also carries `series["fits"]`: the sinusoid fitted to each
    harmonic total is then drawn on the linear panel, with the modulation and the
    ellipticity it gives, since that curve is the measurement.
    """
    harmonics = sorted(series["harmonics"])
    if not harmonics:
        return

    angle_name = series.get("angle_name", "angle")
    fits = series.get("fits") or {}
    arms_of = {harmonic: sorted(name for name in series["channels"] if harmonic_of(name) == harmonic)
               for harmonic in harmonics}
    fig, polar_axes, linear_axes = _two_row_grid(
        len(harmonics), f"Intensity vs {angle_name} — {series.get('title', series['power_label'])}")
    circle_angles = _circle_arrays(series["angles"])[0]
    minima_rows: List[Dict[str, Any]] = []

    for column, harmonic in enumerate(harmonics):
        total = series["harmonics"][harmonic]
        smooth = harmonic in SMOOTH_HARMONICS and harmonic not in fits
        for ax, draw in ((polar_axes[column], _polar_series), (linear_axes[column], _linear_series)):
            draw(ax, series["angles"], total["mean"], total["std"], "tab:blue",
                 label=f"{harmonic} total", smooth=smooth, floor=0.0)
            for name, color in zip(arms_of[harmonic], ARM_COLORS):
                arm = series["channels"][name]
                draw(ax, series["angles"], arm["mean"], arm["std"], color, label=name,
                     floor=0.0, alpha=0.12, linewidth=1.1, markersize=2.5)

        if smooth:
            minima = _harmonic_minima(series, harmonic)
            _mark_minima(polar_axes[column], minima, to_x=np.deg2rad)
            _mark_minima(linear_axes[column], minima, annotate=True,
                         label="interpolated minimum")
            minima_rows.extend(_minima_rows(series["power_label"], harmonic, total["mean"], minima))

        fit = fits.get(harmonic)
        if fit is not None:
            _draw_sinusoid(linear_axes[column], fit, circle_angles)
            _draw_sinusoid(polar_axes[column], fit, circle_angles, to_x=np.deg2rad)

        curves = [total["mean"]] + [series["channels"][name]["mean"] for name in arms_of[harmonic]]
        _focus_radial(polar_axes[column], curves, floor=0.0)
        _polar_style(polar_axes[column], f"{harmonic} intensity (counts/s)", series["clockwise"])
        _focus_vertical(linear_axes[column], curves, floor=0.0)
        _linear_style(linear_axes[column], f"{harmonic} intensity — linear axes",
                      "counts/s", circle_angles, angle_name)
        linear_axes[column].legend(fontsize=7, loc="best", framealpha=0.8)

    _save_polar_figure(fig, save_dir / "intensity_vs_angle.png")
    write_minima_csv(minima_rows, save_dir / ANGLE_MINIMA_CSV)


def _harmonic_minima(series: Dict[str, Any], harmonic: str) -> List[Tuple[float, float]]:
    """The interpolated minima of a harmonic's summed intensity, over the drawn circle."""
    angles, mean = _circle_arrays(series["angles"], series["harmonics"][harmonic]["mean"])
    return _curve_minima(angles, mean, floor=0.0)


def _minima_rows(power_label: str, harmonic: str, mean: np.ndarray,
                 minima: Sequence[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """One CSV row per minimum, with its depth relative to the brightest measured angle."""
    peak = float(np.nanmax(mean)) if np.any(np.isfinite(mean)) else float("nan")
    if minima:
        print(f"  {harmonic} interpolated minima: "
              + ", ".join(f"{angle:g}° ({value:.4g} counts/s)" for angle, value in minima))
    return [{
        "power_label": power_label,
        "harmonic": harmonic,
        "angle_deg": f"{angle:g}",
        "value": f"{value:.6g}",
        "fraction_of_max": f"{value / peak:.4g}" if peak else "",
    } for angle, value in minima]


def plot_angle_g2(series: Dict[str, Any], save_dir: Path) -> None:
    """g²(0) vs angle: one row per harmonic, its auto pair first, then a cross family.

    Row by row that reads 33 | all 34, 44 | all 35, 55 | all 45.
    """
    pairs = series["pairs"]
    if not pairs:
        return
    fig, ax_by_pair = _row_grid(
        _auto_then_cross_rows(pairs),
        f"$g^{{(2)}}(0)$ vs {series.get('angle_name', 'angle')} — "
        f"{series.get('title', series['power_label'])}")
    for pair, ax in ax_by_pair.items():
        color = "tab:red" if is_cross_harmonic(pair) else "tab:green"
        _polar_series(ax, series["angles"], pairs[pair]["mean"], pairs[pair]["std"], color)
        _unity_circle(ax)
        _focus_radial(ax, [pairs[pair]["mean"]], floor=1.0)
        _polar_style(ax, _pair_label(pair), series["clockwise"])
    _save_polar_figure(fig, save_dir / "g2_vs_angle.png")


def plot_angle_r(series: Dict[str, Any], save_dir: Path) -> None:
    """R = g²_cross² / (g²_auto g²_auto) vs angle: one row per cross family (34, 35, 45).

    R is only defined for cross pairs, so there is no auto column here.
    """
    r_values = series.get("R") or {}
    if not r_values:
        return
    fig, ax_by_pair = _row_grid(
        _cross_family_rows(r_values),
        f"$R = g^{{(2)}}_{{nm}}{{}}^2 / (g^{{(2)}}_{{nn}} g^{{(2)}}_{{mm}})$ vs "
        f"{series.get('angle_name', 'angle')} — {series.get('title', series['power_label'])}")
    for pair, ax in ax_by_pair.items():
        _polar_series(ax, series["angles"], r_values[pair]["mean"], r_values[pair]["std"],
                      "tab:purple")
        _unity_circle(ax)
        _focus_radial(ax, [r_values[pair]["mean"]], floor=1.0)
        _polar_style(ax, f"R  {_pair_label(pair)}", series["clockwise"])
    _save_polar_figure(fig, save_dir / "R_vs_angle.png")


def plot_angle_harmonics(series: Dict[str, Any], save_dir: Path) -> None:
    """2×N polar grid: per harmonic, intensity on top and its auto-g²(0) below.

    Two harmonics → 4 panels; three harmonics → 6 panels.
    """
    harmonics = sorted(series["harmonics"])
    auto_pairs = {harmonic_of(pair[0]): pair for pair in series["pairs"]
                  if not is_cross_harmonic(pair)}
    if not harmonics:
        return

    cols = len(harmonics)
    fig, axes = plt.subplots(2, cols, figsize=(4.4 * cols, 9.0), dpi=300,
                             subplot_kw={"projection": "polar"})
    axes = np.asarray(axes).reshape(2, cols)
    fig.suptitle(f"Harmonics vs {series.get('angle_name', 'angle')} — "
                 f"{series.get('title', series['power_label'])}",
                 fontsize=15, fontweight="bold")

    for column, harmonic in enumerate(harmonics):
        signal = series["harmonics"][harmonic]
        smooth = harmonic in SMOOTH_HARMONICS and harmonic not in (series.get("fits") or {})
        _polar_series(axes[0, column], series["angles"], signal["mean"], signal["std"], "tab:blue",
                      smooth=smooth, floor=0.0)
        if smooth:
            _mark_minima(axes[0, column], _harmonic_minima(series, harmonic), to_x=np.deg2rad)
        _focus_radial(axes[0, column], [signal["mean"]], floor=0.0)
        _polar_style(axes[0, column], f"{harmonic} intensity (counts/s)", series["clockwise"])

        pair = auto_pairs.get(harmonic)
        ax = axes[1, column]
        if pair is None:
            ax.set_visible(False)
            continue
        g2 = series["pairs"][pair]
        _polar_series(ax, series["angles"], g2["mean"], g2["std"], "tab:green")
        _unity_circle(ax)
        _focus_radial(ax, [g2["mean"]], floor=1.0)
        _polar_style(ax, f"{harmonic} $g^{{(2)}}(0)$  ({pair[0]}–{pair[1]})", series["clockwise"])

    _save_polar_figure(fig, save_dir / "harmonics_vs_angle.png")


# ----------------------------------------------------------------------------
# Multi-power butterfly overlay
# ----------------------------------------------------------------------------

def plot_laser_angle_overlay(series_by_power: Dict[str, Dict[str, Any]], save_dir: Path) -> None:
    """Same four layouts, every power drawn on each panel as a coloured butterfly.

    The arm curves of the intensity figure are dropped here: with one colour per power
    already on every panel, three more curves per harmonic would only be a thicket. The
    interpolation of SMOOTH_HARMONICS is applied to each power's sum, so the overlay
    carries one interpolated curve (and one set of minima) per power.
    """
    if len(series_by_power) < 2:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    powers = list(series_by_power)
    colors = _power_colors(powers)
    first = next(iter(series_by_power.values()))
    clockwise = first.get("clockwise", True)
    angle_name = first.get("angle_name", "angle")
    title_suffix = ", ".join(powers)

    # --- intensity: per harmonic, polar on top and the same curves linear below ---
    harmonics = sorted({h for series in series_by_power.values() for h in series["harmonics"]})
    if harmonics:
        fig, polar_axes, linear_axes = _two_row_grid(
            len(harmonics), f"Intensity vs {angle_name} — {title_suffix}")
        minima_rows: List[Dict[str, Any]] = []
        for column, harmonic in enumerate(harmonics):
            smooth = harmonic in SMOOTH_HARMONICS
            curves = []
            widest = None
            for power, series in series_by_power.items():
                if harmonic not in series["harmonics"]:
                    continue
                d = series["harmonics"][harmonic]
                curves.append(d["mean"])
                for ax, draw in ((polar_axes[column], _polar_series),
                                 (linear_axes[column], _linear_series)):
                    draw(ax, series["angles"], d["mean"], d["std"], colors[power],
                         label=power, alpha=0.18, smooth=smooth, floor=0.0)
                if smooth:
                    # Rings only, no angle labels: with one set of minima per power the
                    # texts would land on each other. The angles are in the CSV.
                    minima = _harmonic_minima(series, harmonic)
                    _mark_minima(polar_axes[column], minima, color=colors[power], to_x=np.deg2rad)
                    _mark_minima(linear_axes[column], minima, color=colors[power])
                    minima_rows.extend(_minima_rows(power, harmonic, d["mean"], minima))
                angles = _circle_arrays(series["angles"])[0]
                if widest is None or angles[-1] - angles[0] > widest[-1] - widest[0]:
                    widest = angles

            _focus_radial(polar_axes[column], curves, floor=0.0)
            _polar_style(polar_axes[column], f"{harmonic} intensity (counts/s)", clockwise)
            _focus_vertical(linear_axes[column], curves, floor=0.0)
            if widest is not None:
                _linear_style(linear_axes[column], f"{harmonic} intensity — linear axes",
                              "counts/s", widest, angle_name)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "intensity_vs_angle.png")
        write_minima_csv(minima_rows, save_dir / ANGLE_MINIMA_CSV)

    # --- g2: one row per harmonic, its auto pair then a cross family ---
    pair_names = sorted({pair for series in series_by_power.values() for pair in series["pairs"]})
    if pair_names:
        fig, ax_by_pair = _row_grid(_auto_then_cross_rows(pair_names),
                                    f"$g^{{(2)}}(0)$ vs {angle_name} — {title_suffix}")
        for pair, ax in ax_by_pair.items():
            curves = _overlay_panel(ax, series_by_power, "pairs", pair, colors)
            _unity_circle(ax)
            _focus_radial(ax, curves, floor=1.0)
            _polar_style(ax, _pair_label(pair), clockwise)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "g2_vs_angle.png")

    # --- R: one row per cross family ---
    r_names = sorted({pair for series in series_by_power.values() for pair in series.get("R", {})})
    if r_names:
        fig, ax_by_pair = _row_grid(
            _cross_family_rows(r_names),
            f"$R = g^{{(2)}}_{{nm}}{{}}^2 / (g^{{(2)}}_{{nn}} g^{{(2)}}_{{mm}})$ vs "
            f"{angle_name} — {title_suffix}")
        for pair, ax in ax_by_pair.items():
            curves = _overlay_panel(ax, series_by_power, "R", pair, colors)
            _unity_circle(ax)
            _focus_radial(ax, curves, floor=1.0)
            _polar_style(ax, f"R  {pair[0]}–{pair[1]}", clockwise)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "R_vs_angle.png")

    # --- harmonics: intensity on top, auto-g2 below ---
    if harmonics:
        cols = len(harmonics)
        fig, axes = plt.subplots(2, cols, figsize=(4.4 * cols, 9.0), dpi=300,
                                 subplot_kw={"projection": "polar"})
        axes = np.asarray(axes).reshape(2, cols)
        fig.suptitle(f"Harmonics vs {angle_name} — {title_suffix}",
                     fontsize=15, fontweight="bold")
        for column, harmonic in enumerate(harmonics):
            curves = []
            for power, series in series_by_power.items():
                if harmonic in series["harmonics"]:
                    d = series["harmonics"][harmonic]
                    curves.append(d["mean"])
                    smooth = harmonic in SMOOTH_HARMONICS
                    _polar_series(axes[0, column], series["angles"], d["mean"], d["std"],
                                  colors[power], label=power, alpha=0.18,
                                  smooth=smooth, floor=0.0)
                    if smooth:
                        _mark_minima(axes[0, column], _harmonic_minima(series, harmonic),
                                     color=colors[power], to_x=np.deg2rad)
            _focus_radial(axes[0, column], curves, floor=0.0)
            _polar_style(axes[0, column], f"{harmonic} intensity (counts/s)", clockwise)

            auto_pair = None
            for series in series_by_power.values():
                for pair in series["pairs"]:
                    if not is_cross_harmonic(pair) and harmonic_of(pair[0]) == harmonic:
                        auto_pair = pair
                        break
                if auto_pair:
                    break
            ax = axes[1, column]
            if auto_pair is None:
                ax.set_visible(False)
                continue
            curves = _overlay_panel(ax, series_by_power, "pairs", auto_pair, colors)
            _unity_circle(ax)
            _focus_radial(ax, curves, floor=1.0)
            _polar_style(ax, f"{harmonic} $g^{{(2)}}(0)$  ({auto_pair[0]}–{auto_pair[1]})", clockwise)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "harmonics_vs_angle.png")

    print(f"Laser angle overlay ({len(powers)} powers) -> {save_dir}")


def _overlay_panel(ax, series_by_power: Dict[str, Dict[str, Any]], key: str,
                   target: Any, colors: Dict[str, Any]) -> List[np.ndarray]:
    """Draw one quantity of every power on one panel; return the means for the limits."""
    curves = []
    for power, series in series_by_power.items():
        entry = series.get(key, {}).get(target)
        if entry is None:
            continue
        curves.append(entry["mean"])
        _polar_series(ax, series["angles"], entry["mean"], entry["std"], colors[power],
                      label=power, alpha=0.18)
    return curves


# ----------------------------------------------------------------------------
# The numbers behind the figures
# ----------------------------------------------------------------------------

def write_minima_csv(rows: Sequence[Dict[str, Any]], path: Path) -> Optional[Path]:
    """The angle and depth of every interpolated minimum: the numbers behind the dips."""
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["power_label", "harmonic", "angle_deg", "value", "fraction_of_max"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}")
    return path


def write_angle_summary_csv(series: Dict[str, Any], path: Path) -> Path:
    """Every quantity of every angle of one series, one row per (angle, quantity)."""
    rows = []
    for angle_index, angle in enumerate(series["angles"]):
        common = {
            "power_label": series["power_label"],
            "angle_deg": f"{angle:g}",
            "n_chunks": int(series["n_chunks"][angle_index]),
        }
        for quantity, entries in (("countrate", series["channels"]),
                                  ("harmonic_countrate", series["harmonics"]),
                                  ("g2", series["pairs"]),
                                  ("R", series["R"])):
            for name, values in sorted(entries.items(), key=lambda item: str(item[0])):
                rows.append({
                    **common,
                    "quantity": quantity,
                    "target": name if isinstance(name, str) else f"{name[0]}-{name[1]}",
                    "mean": f"{values['mean'][angle_index]:.6g}",
                    "std": f"{values['std'][angle_index]:.6g}",
                })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["power_label", "angle_deg", "quantity", "target",
                        "mean", "std", "n_chunks"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}")
    return path


# ============================================================================
# ELLIPTICITY SCAN: THE ANALYZER SWEEP RECORDED AT EACH LASER ANGLE
# ============================================================================
#
# The harmonics leave the crystal, pass a half-wave plate, then a polarizer that never
# moves. Turning the plate turns the polarization in front of that polarizer, so the
# intensity it transmits traces a sinusoid of period 90 deg in plate angle:
#
#     I(theta) = offset + amplitude * cos(2*pi*(theta - phase) / 90 deg)
#
# What the shape says about the emission is entirely in the DEPTH of that sinusoid:
#
#   * a curve reaching zero  -> one axis carries nothing         -> linear
#   * a flat curve           -> both axes carry the same         -> circular
#
# Written as the modulation m = amplitude / offset, and the two intensities the fit
# reaches, I_max = offset + amplitude and I_min = offset - amplitude:
#
#     attenuation = I_min / I_max = (1 - m) / (1 + m)
#     ellipticity = sqrt(attenuation)      i.e. the ratio of the ellipse's axes
#
# so ellipticity 0 is linear and 1 is circular. Repeating the sweep at several pump
# (laser) angles is what gives ellipticity versus laser angle, the point of the scan.
#
# Per power, under results/.../ellipticity/{power}/:
#   lasXXXpY/  — the four vs-analyzer-angle figures of that sweep, the fitted sinusoid
#                  drawn on the intensity panels, and sinusoidal_fit.csv
#   summary/   — ellipticity_vs_laser_angle.png / .csv, modulation_vs_laser_angle.png,
#                  sinusoid_fits_{harmonic}.png (every sweep and its fit side by side),
#                  and the sinusoidal_fit.csv of the whole power
#   ../overlay/ — ellipticity and modulation versus laser angle, every power together


def sinusoid(angles_deg: np.ndarray, fit: Dict[str, Any]) -> np.ndarray:
    """The fitted curve evaluated at `angles_deg`."""
    phase = 2.0 * np.pi * (np.asarray(angles_deg, dtype=float) - fit["phase_deg"]) / fit["period_deg"]
    return fit["offset"] + fit["amplitude"] * np.cos(phase)


def fit_sinusoid(angles_deg: Sequence[float], values: Sequence[float],
                 period_deg: float = 90.0) -> Optional[Dict[str, Any]]:
    """Fit `offset + amplitude*cos(2 pi (angle - phase)/period)` at a FIXED period.

    Holding the period at the 90 deg a half-wave plate imposes leaves three linear
    parameters, so the fit is a least-squares solve rather than an iteration that can
    fail to converge on a nearly flat (circular) curve. The uncertainties are the usual
    residual-based ones, propagated into the modulation and the ellipticity.

    Returns None when fewer than four angles carry a finite value.
    """
    angles = np.asarray(angles_deg, dtype=float)
    signal = np.asarray(values, dtype=float)
    finite = np.isfinite(angles) & np.isfinite(signal)
    if finite.sum() < 4:
        return None
    angles, signal = angles[finite], signal[finite]

    turns = 2.0 * np.pi * angles / float(period_deg)
    design = np.column_stack([np.cos(turns), np.sin(turns), np.ones_like(turns)])
    (cosine, sine, offset), *_ = np.linalg.lstsq(design, signal, rcond=None)
    amplitude = float(np.hypot(cosine, sine))
    phase_deg = float(np.rad2deg(np.arctan2(sine, cosine)) * period_deg / 360.0 % period_deg)

    residuals = signal - design @ np.array([cosine, sine, offset])
    rss = float(residuals @ residuals)
    total = float(np.sum((signal - signal.mean()) ** 2))
    dof = len(signal) - 3
    covariance = np.linalg.pinv(design.T @ design) * (rss / dof if dof > 0 else np.nan)

    # The amplitude is not a fitted parameter but the length of (cosine, sine), so its
    # variance - and its covariance with the offset - come from that of the two.
    if amplitude > 0:
        amplitude_var = float(
            cosine ** 2 * covariance[0, 0] + sine ** 2 * covariance[1, 1]
            + 2.0 * cosine * sine * covariance[0, 1]) / amplitude ** 2
        amplitude_offset_cov = float(cosine * covariance[0, 2] + sine * covariance[1, 2]) / amplitude
    else:
        amplitude_var = amplitude_offset_cov = float("nan")

    modulation = amplitude / offset if offset > 0 else float("nan")
    modulation_var = (modulation ** 2 * (amplitude_var / amplitude ** 2
                                         + covariance[2, 2] / offset ** 2
                                         - 2.0 * amplitude_offset_cov / (amplitude * offset))
                      if offset > 0 and amplitude > 0 else float("nan"))
    modulation_std = float(np.sqrt(modulation_var)) if modulation_var >= 0 else float("nan")
    attenuation, ellipticity, ellipticity_std = ellipticity_of(modulation, modulation_std)

    return {
        "n_points": int(len(signal)),
        "period_deg": float(period_deg),
        "offset": float(offset),
        "amplitude": amplitude,
        "phase_deg": phase_deg,
        "modulation": float(modulation),
        "modulation_std": modulation_std,
        "ellipticity": ellipticity,
        "ellipticity_std": ellipticity_std,
        "attenuation": attenuation,
        "mean_intensity": float(offset),
        "r_squared": float(1.0 - rss / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(rss / len(signal))),
    }


def ellipticity_of(modulation: float, modulation_std: float = float("nan")) -> Tuple[float, float, float]:
    """(attenuation, ellipticity, ellipticity std) of a modulation depth.

    attenuation = I_min/I_max = (1-m)/(1+m), and the ellipticity is its square root: the
    ratio of the ellipse's axes, 0 for a linear polarization and 1 for a circular one.
    Noise can push a fitted m just past 1 (a curve reaching slightly below zero), which
    is a linear polarization measured with a small negative excursion: it is clipped to
    0 rather than turned into a NaN.
    """
    if not np.isfinite(modulation):
        return float("nan"), float("nan"), float("nan")
    attenuation = float(np.clip((1.0 - modulation) / (1.0 + modulation), 0.0, 1.0))
    ellipticity = float(np.sqrt(attenuation))
    if ellipticity > 0 and np.isfinite(modulation_std):
        # d(eps)/dm = -1 / ((1+m)^2 eps)
        ellipticity_std = float(modulation_std / ((1.0 + modulation) ** 2 * ellipticity))
    else:
        ellipticity_std = float("nan")
    return attenuation, ellipticity, ellipticity_std


def fit_angle_series(series: Dict[str, Any], period_deg: float = 90.0) -> Dict[str, Dict[str, Any]]:
    """Fit the sinusoid of every harmonic total and every single channel of a sweep.

    The harmonic totals are what the ellipticity is quoted from; the per-channel fits
    are kept because a transmitted and a reflected arm disagreeing is how a
    mis-set analyzer or a clipped beam shows up.
    """
    fits: Dict[str, Dict[str, Any]] = {"harmonics": {}, "channels": {}}
    for kind, key in (("harmonics", "harmonics"), ("channels", "channels")):
        for name, entry in sorted(series[key].items()):
            fit = fit_sinusoid(series["angles"], entry["mean"], period_deg)
            if fit is not None:
                fits[kind][name] = fit
    return fits


# ----------------------------------------------------------------------------
# One power: every sweep, then ellipticity versus laser angle
# ----------------------------------------------------------------------------

def plot_ellipticity_power(cfg: ExperimentConfig,
                           power: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Every analyzer sweep of one power, its fit, and the summary versus laser angle."""
    power = power or cfg.base_power
    period = float(cfg.ELLIPTICITY_FIT_PERIOD_DEG)
    laser_angles = _ellipticity_laser_angles(cfg, power)
    if not laser_angles:
        print(f"Ellipticity [{power}]: no laser angles configured.")
        return None

    sweeps: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    for laser_angle in laser_angles:
        points = _ellipticity_points(cfg, power, laser_angle)
        series = collect_angle_series(cfg, points, power_label=power,
                                      angle_name="analyzer angle",
                                      title=f"{power}, laser angle {laser_angle:g}°")
        if len(series["angles"]) < 4:
            print(f"  [{power} @ {laser_angle:g}°] fewer than 4 analyzer angles with data, "
                  "sweep skipped.")
            continue

        fits = fit_angle_series(series, period)
        series["fits"] = fits["harmonics"]
        save_dir = cfg.ellipticity_laser_dir(power, laser_angle)
        print(f"\n  ---- {power}, laser angle {laser_angle:g}° "
              f"({len(series['angles'])} analyzer angles) ----")
        plot_angle_summary(series, save_dir)

        rows = _fit_rows(power, laser_angle, fits)
        write_sinusoid_csv(rows, save_dir / ELLIPTICITY_FIT_CSV)
        fit_rows.extend(rows)
        sweeps.append({"laser_angle_deg": float(laser_angle), "series": series,
                       "fits": fits["harmonics"]})
        for harmonic, fit in sorted(fits["harmonics"].items()):
            print(f"    {harmonic}: modulation {fit['modulation']:.4f}, "
                  f"ellipticity {fit['ellipticity']:.4f} "
                  f"(R² {fit['r_squared']:.4f})")

    if not sweeps:
        print(f"Ellipticity [{power}]: no sweep held enough data.")
        return None

    summary_dir = cfg.ellipticity_summary_dir(power)
    summary_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  ---- {power} summary ----")
    plot_ellipticity_vs_laser_angle(sweeps, power, summary_dir)
    plot_modulation_vs_laser_angle(sweeps, power, summary_dir)
    for harmonic in sorted({h for sweep in sweeps for h in sweep["fits"]}):
        plot_sinusoid_fits(sweeps, harmonic, power, summary_dir)
    write_ellipticity_csv(sweeps, power, summary_dir / ELLIPTICITY_SUMMARY_CSV)
    write_sinusoid_csv(fit_rows, summary_dir / ELLIPTICITY_FIT_CSV)

    print(f"Ellipticity [{power}]: {len(sweeps)} laser angle(s) -> {summary_dir.parent}")
    return {"power_label": power, "sweeps": sweeps}


def plot_ellipticity_campaign(cfg: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """Every power of the folder, then the overlay comparing them.

    Powers come from `PUMP_POWERS` when set, otherwise from the files on disk (and
    falling back to the single `POWER_LEVEL` of this config).
    """
    powers = list(cfg.PUMP_POWERS) if cfg.PUMP_POWERS else discover_ellipticity_powers(cfg.DATA_DIR)
    if not powers:
        powers = [cfg.POWER_LEVEL]

    by_power: Dict[str, Dict[str, Any]] = {}
    for power in powers:
        print(f"\n================ ellipticity: {power} ================")
        campaign = plot_ellipticity_power(cfg.for_pump_power(power), power)
        if campaign is not None:
            by_power[power] = campaign

    if len(by_power) > 1:
        plot_ellipticity_overlay(by_power, cfg.ellipticity_overlay_dir)
    elif len(by_power) == 1:
        print("Ellipticity overlay: only one power present, skipped.")
    return by_power


def _ellipticity_laser_angles(cfg: ExperimentConfig, power: str) -> List[float]:
    """Laser angles to summarise for one power: CSV log first, else the planned scan."""
    log_path = cfg.ellipticity_log_path()
    if log_path.is_file():
        recorded = EllipticityLog(log_path).recorded_points(power=power)
        angles = sorted({laser for laser, _analyzer, _label in recorded})
        if angles:
            return angles
    return cfg.ellipticity_laser_angles()


def _ellipticity_points(cfg: ExperimentConfig, power: str,
                        laser_angle_deg: float) -> List[Tuple[float, str]]:
    """[(analyzer angle, file label)] of one sweep: CSV log first, else the plan."""
    tag = laser_angle_tag(laser_angle_deg)
    log_path = cfg.ellipticity_log_path()
    if log_path.is_file():
        recorded = [(analyzer, label)
                    for laser, analyzer, label in EllipticityLog(log_path).recorded_points(power=power)
                    if laser_angle_tag(laser) == tag]
        if recorded:
            return recorded
    return [(float(analyzer), f"{power}_{tag}_{analyzer_angle_tag(analyzer)}")
            for analyzer in cfg.ELLIPTICITY_ANALYZER_ANGLES]


# ----------------------------------------------------------------------------
# The summary figures
# ----------------------------------------------------------------------------

def plot_ellipticity_vs_laser_angle(sweeps: Sequence[Dict[str, Any]], power: str,
                                    save_dir: Path) -> None:
    """The point of the scan: how elliptical the emission is at each pump angle."""
    _plot_fit_quantity(sweeps, power, save_dir / "ellipticity_vs_laser_angle.png",
                       key="ellipticity",
                       title=f"Ellipticity of the harmonics vs laser angle — {power}",
                       ylabel="ellipticity  $\\epsilon = \\sqrt{I_{min}/I_{max}}$",
                       limits=(0.0, 1.05),
                       guides=((0.0, "linear"), (1.0, "circular")))


def plot_modulation_vs_laser_angle(sweeps: Sequence[Dict[str, Any]], power: str,
                                   save_dir: Path) -> None:
    """The depth of each fitted sinusoid: the number the ellipticity is derived from."""
    _plot_fit_quantity(sweeps, power, save_dir / "modulation_vs_laser_angle.png",
                       key="modulation",
                       title=f"Modulation of the analyzer sweep vs laser angle — {power}",
                       ylabel="modulation  $m = A / I_0$",
                       limits=(0.0, 1.05),
                       guides=((1.0, "linear"), (0.0, "circular")))


def _plot_fit_quantity(sweeps: Sequence[Dict[str, Any]], power: str, path: Path, key: str,
                       title: str, ylabel: str, limits: Tuple[float, float],
                       guides: Sequence[Tuple[float, str]] = ()) -> None:
    """One fitted quantity per harmonic against the laser angle: linear, then polar."""
    harmonics = sorted({harmonic for sweep in sweeps for harmonic in sweep["fits"]})
    if not harmonics:
        return

    fig = plt.figure(figsize=(11.0, 4.8), dpi=300)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    linear_ax = fig.add_subplot(grid[0, 0])
    polar_ax = fig.add_subplot(grid[0, 1], projection="polar")
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors = cm.tab10(np.linspace(0, 1, 10))
    for index, harmonic in enumerate(harmonics):
        angles, values, errors = _fit_series(sweeps, harmonic, key)
        if not len(angles):
            continue
        color = colors[index % len(colors)]
        linear_ax.errorbar(angles, values, yerr=errors, color=color, marker="o",
                           markersize=4, linewidth=1.5, capsize=2.5, label=harmonic)
        polar_angles, polar_values = _circle_arrays(angles, values)
        polar_ax.plot(np.deg2rad(polar_angles), polar_values, color=color, marker="o",
                      markersize=3.5, linewidth=1.5, label=harmonic)

    for level, note in guides:
        linear_ax.axhline(level, color="0.6", linestyle=":", linewidth=1.0)
        linear_ax.annotate(note, (1.0, level), xycoords=("axes fraction", "data"),
                           textcoords="offset points", xytext=(-3, 3), ha="right",
                           fontsize=7, color="0.35")

    linear_ax.set_xlabel("laser angle (deg)", fontsize=9)
    linear_ax.set_ylabel(ylabel, fontsize=9)
    linear_ax.set_ylim(*limits)
    linear_ax.grid(True, alpha=0.35)
    linear_ax.tick_params(labelsize=8)
    linear_ax.legend(fontsize=8, loc="best", framealpha=0.85)

    polar_ax.set_rlim(*limits)
    _polar_style(polar_ax, "same, on the pump's circle",
                 sweeps[0]["series"].get("clockwise", True))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_sinusoid_fits(sweeps: Sequence[Dict[str, Any]], harmonic: str, power: str,
                       save_dir: Path) -> None:
    """Every analyzer sweep of one harmonic and its fit, side by side.

    The figure the ellipticity has to be read against: a curve grazing zero is linear,
    a flat one is circular, and a fit missing its points is a number not to trust.
    """
    usable = [sweep for sweep in sweeps if harmonic in sweep["fits"]]
    if not usable:
        return

    cols = min(4, len(usable))
    rows = int(np.ceil(len(usable) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows), dpi=300,
                             squeeze=False)
    fig.suptitle(f"{harmonic}: analyzer sweep and its sinusoidal fit — {power}",
                 fontsize=13, fontweight="bold")

    for index, sweep in enumerate(usable):
        ax = axes[index // cols][index % cols]
        series, fit = sweep["series"], sweep["fits"][harmonic]
        angles = series["angles"]
        entry = series["harmonics"][harmonic]
        ax.errorbar(angles, entry["mean"], yerr=entry["std"], color="tab:blue",
                    marker="o", markersize=4, linestyle="none", capsize=2.5,
                    label="measured")
        _draw_sinusoid(ax, fit, angles)
        ax.set_title(f"laser angle {sweep['laser_angle_deg']:g}°", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("analyzer angle (deg)", fontsize=8)
        ax.set_ylabel("counts/s", fontsize=8)
        ax.set_ylim(0.0, None)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best", framealpha=0.85)

    for empty in range(len(usable), rows * cols):
        axes[empty // cols][empty % cols].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(save_dir) / f"sinusoid_fits_{harmonic}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_ellipticity_overlay(by_power: Dict[str, Dict[str, Any]], save_dir: Path) -> None:
    """Ellipticity and modulation versus laser angle, one colour per power."""
    if len(by_power) < 2:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    colors = _power_colors(list(by_power))
    markers = ("o", "s", "^", "D", "v")

    for key, ylabel, name in (
            ("ellipticity", "ellipticity  $\\epsilon = \\sqrt{I_{min}/I_{max}}$",
             "ellipticity_vs_laser_angle.png"),
            ("modulation", "modulation  $m = A / I_0$", "modulation_vs_laser_angle.png")):
        harmonics = sorted({harmonic for campaign in by_power.values()
                            for sweep in campaign["sweeps"] for harmonic in sweep["fits"]})
        fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=300)
        fig.suptitle(f"{key.capitalize()} vs laser angle — {', '.join(by_power)}",
                     fontsize=13, fontweight="bold")
        for power, campaign in by_power.items():
            for index, harmonic in enumerate(harmonics):
                angles, values, errors = _fit_series(campaign["sweeps"], harmonic, key)
                if not len(angles):
                    continue
                ax.errorbar(angles, values, yerr=errors, color=colors[power],
                            marker=markers[index % len(markers)], markersize=4,
                            linewidth=1.4, capsize=2.5, label=f"{power} {harmonic}")
        ax.set_xlabel("laser angle (deg)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.85)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(save_dir / name, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}")

    print(f"Ellipticity overlay ({len(by_power)} powers) -> {save_dir}")


def _fit_series(sweeps: Sequence[Dict[str, Any]], harmonic: str,
                key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(laser angles, fitted quantity, its std) of one harmonic, in angular order."""
    rows = [(sweep["laser_angle_deg"], sweep["fits"][harmonic][key],
             sweep["fits"][harmonic].get(f"{key}_std", float("nan")))
            for sweep in sweeps if harmonic in sweep["fits"]]
    rows.sort()
    angles = np.array([row[0] for row in rows], dtype=float)
    values = np.array([row[1] for row in rows], dtype=float)
    errors = np.array([row[2] for row in rows], dtype=float)
    # errorbar refuses NaN error bars, and a fit of four points can legitimately have no
    # usable covariance; drawing those points without a bar is better than dropping them.
    return angles, values, np.nan_to_num(errors, nan=0.0)


# ----------------------------------------------------------------------------
# The numbers behind the ellipticity figures
# ----------------------------------------------------------------------------

SINUSOID_CSV_COLUMNS = [
    "power_label", "laser_angle_deg", "kind", "target", "n_points", "period_deg",
    "offset", "amplitude", "phase_deg", "modulation", "modulation_std", "ellipticity",
    "ellipticity_std", "attenuation", "r_squared", "rmse",
]

ELLIPTICITY_CSV_COLUMNS = [
    "power_label", "laser_angle_deg", "harmonic", "ellipticity", "ellipticity_std",
    "modulation", "modulation_std", "attenuation", "mean_intensity", "r_squared",
    "n_analyzer_angles",
]


def _fit_rows(power: str, laser_angle_deg: float,
              fits: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per fitted target of one sweep: the harmonic totals, then the channels."""
    rows: List[Dict[str, Any]] = []
    for kind, entries in (("harmonic", fits["harmonics"]), ("channel", fits["channels"])):
        for target, fit in sorted(entries.items()):
            rows.append({
                "power_label": power,
                "laser_angle_deg": f"{laser_angle_deg:g}",
                "kind": kind,
                "target": target,
                **{key: _number(fit[key]) for key in
                   ("n_points", "period_deg", "offset", "amplitude", "phase_deg",
                    "modulation", "modulation_std", "ellipticity", "ellipticity_std",
                    "attenuation", "r_squared", "rmse")},
            })
    return rows


def write_sinusoid_csv(rows: Sequence[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Every fitted sinusoid: what the modulation and the ellipticity were read from."""
    return _write_csv(rows, path, SINUSOID_CSV_COLUMNS)


def write_ellipticity_csv(sweeps: Sequence[Dict[str, Any]], power: str,
                          path: Path) -> Optional[Path]:
    """The one table the scan is for: ellipticity per harmonic, per laser angle."""
    rows = []
    for sweep in sweeps:
        for harmonic, fit in sorted(sweep["fits"].items()):
            rows.append({
                "power_label": power,
                "laser_angle_deg": f"{sweep['laser_angle_deg']:g}",
                "harmonic": harmonic,
                **{key: _number(fit[key]) for key in
                   ("ellipticity", "ellipticity_std", "modulation", "modulation_std",
                    "attenuation", "mean_intensity", "r_squared")},
                "n_analyzer_angles": fit["n_points"],
            })
    return _write_csv(rows, path, ELLIPTICITY_CSV_COLUMNS)


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path,
               columns: Sequence[str]) -> Optional[Path]:
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}")
    return path


def _number(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return "" if value is None or not np.isfinite(value) else f"{float(value):.6g}"


if __name__ == "__main__":
    main()
