"""The analysis: peaks, g2(0), R, and every figure the pipeline produces.

Four families of output, all reading the same `.pkl`/`.json` files:

* `run_analysis(cfg)` - the per-power tree: spectrograms, window sweeps, chunk
  statistics, and the power sweep summary when several powers are analysed.
* `plot_stability(snapshots, cfg)` - g2(0) and R against time, from the snapshots a
  live run recorded while it was measuring.
* `plot_polarization_summary(cfg)` - the three polar summaries of one power:
  intensity per channel, g²(0) per pair, harmonics (intensity + auto-g²).
* `plot_polarization_campaign(cfg)` - the same for every power in the folder, then
  the multi-power butterfly overlay.

The maths lives in one place: the stability and polarization halves reuse the same
peak finding and the same g2 definition as the per-power tree, so a point on a polar
figure is the number the per-angle tree would give for that angle.
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
from scipy.signal import find_peaks
import numpy as np
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from experiment_config import ExperimentConfig, angle_tag, harmonic_of
    from hardware import ScanLog
except ImportError:  # pragma: no cover - import style depends on the entry point
    from src.experiment_config import ExperimentConfig, angle_tag, harmonic_of
    from src.hardware import ScanLog

from dataclasses import replace

script_path = Path(__file__).resolve().parent.parent

Pair = Tuple[int, int]           # a channel pair, as the stability snapshots key them
POLARIZATION_SUMMARY_CSV = "polarization_summary.csv"
BAND_ALPHA = 0.25                # opacity of the chunk-spread band on the polar plots




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

def get_data(target_files, modes, power_level):
    all_data = {}
    for file_path in target_files:
        original_filename = file_path.name
        match = re.search(rf'({power_level}_num\d+)', original_filename)
        if match:
            chunk_number = re.search(r'num(\d+)', match.group(1)).group(1)
            clean_filename = "chunk_" + chunk_number
        else:
            clean_filename = file_path.stem 
            
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f: raw_data = json.load(f)
        elif file_path.suffix == ".pkl":
            with open(file_path, "rb") as f: raw_data = pickle.load(f)
        else:
            continue

        file_correlations, file_countrates, file_total_counts = {}, {}, {}

        raw_corr = raw_data.get("Correlation", {})
        for pair_str, data_lists in raw_corr.items():
            ch1_str, ch2_str = pair_str.strip("()").split(",")
            ch1, ch2 = int(ch1_str), int(ch2_str)
            if ch1 > ch2: continue
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
            'g2_int_m': np.array([np.nanmean(aggregated[p]["g2_int"][w]) for w in windows]),
            'g2_int_s': np.array([np.nanstd(aggregated[p]["g2_int"][w]) for w in windows]),
            'g2_counts_m': np.array([np.nanmean(aggregated[p]["g2_counts"][w]) for w in windows]),
            'g2_counts_s': np.array([np.nanstd(aggregated[p]["g2_counts"][w]) for w in windows]),
            'R_int_m': np.array([np.nanmean(aggregated[p]["R_int"][w]) for w in windows]),
            'R_int_s': np.array([np.nanstd(aggregated[p]["R_int"][w]) for w in windows]),
            'R_counts_m': np.array([np.nanmean(aggregated[p]["R_counts"][w]) for w in windows]),
            'R_counts_s': np.array([np.nanstd(aggregated[p]["R_counts"][w]) for w in windows]),
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
    match = re.search(r'\d+', power_level)
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
        
        global_data["pairs"][pair]["g2_int_m"].append(np.nanmean(g2_i))
        global_data["pairs"][pair]["g2_int_s"].append(np.nanstd(g2_i))
        global_data["pairs"][pair]["g2_counts_m"].append(np.nanmean(g2_c))
        global_data["pairs"][pair]["g2_counts_s"].append(np.nanstd(g2_c))
        global_data["pairs"][pair]["R_int_m"].append(np.nanmean(r_i))
        global_data["pairs"][pair]["R_int_s"].append(np.nanstd(r_i))
        global_data["pairs"][pair]["R_counts_m"].append(np.nanmean(r_c))
        global_data["pairs"][pair]["R_counts_s"].append(np.nanstd(r_c))
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
# POLARIZATION SCAN: CIRCLE / BUTTERFLY SUMMARIES VERSUS ANGLE
# ============================================================================
#
# Three figures per power, under results/.../polarization/{power}/summary/:
#   1. intensity_vs_angle.png  — one polar panel per CHANNEL
#   2. g2_vs_angle.png         — one polar panel per PAIR
#   3. harmonics_vs_angle.png  — per HARMONIC: intensity on top, auto-g² below
#                                  (2×N grid → 4 panels for two harmonics, 6 for three)
#
# When several powers share a DATE folder, the same three layouts are redrawn under
# results/.../polarization/overlay/ with every power as a coloured curve on each panel.


def plot_polarization_summary(cfg: ExperimentConfig, results_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Write the three polar summaries for one power. Returns the series used."""
    power = cfg.POLARIZATION_BASE_POWER or cfg.POWER_LEVEL
    points = _polarization_points(cfg, power=power)
    if not points:
        print(f"Polarization summary [{power}]: no angles configured.")
        return None

    series = collect_polarization_series(cfg, points, power_label=power)
    if len(series["angles"]) < 2:
        print(f"Polarization summary [{power}]: fewer than 2 angles with data, nothing to plot.")
        return None

    save_dir = Path(results_dir) if results_dir else cfg.polarization_summary_dir(power)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_polarization_intensity(series, save_dir)
    plot_polarization_g2(series, save_dir)
    plot_polarization_harmonics(series, save_dir)
    write_polarization_csv(series, save_dir / POLARIZATION_SUMMARY_CSV)

    print(f"Polarization summary [{power}]: {len(series['angles'])} angles -> {save_dir}")
    return series


def plot_polarization_campaign(cfg: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """Per-power summaries for every polarization power in the folder, then the overlay.

    Powers come from `POLARIZATION_POWERS` when set, otherwise from the files on disk
    (and falling back to the single `POWER_LEVEL` of this config).
    """
    powers = list(cfg.POLARIZATION_POWERS) if cfg.POLARIZATION_POWERS else discover_polarization_powers(cfg.DATA_DIR)
    if not powers:
        powers = [cfg.POWER_LEVEL]

    series_by_power: Dict[str, Dict[str, Any]] = {}
    for power in powers:
        power_cfg = cfg.for_polarization_power(power)
        if cfg.POLARIZATION_SCAN is not None:
            # Keep the planned angles so a folder can be replotted without the CSV log.
            power_cfg = replace(power_cfg, POLARIZATION_SCAN=list(cfg.POLARIZATION_SCAN))
        series = plot_polarization_summary(power_cfg)
        if series is not None:
            series_by_power[power] = series

    if len(series_by_power) > 1:
        plot_polarization_overlay(series_by_power, cfg.polarization_overlay_dir)
    elif len(series_by_power) == 1:
        print("Polarization overlay: only one power present, skipped.")

    return series_by_power


def discover_polarization_powers(data_dir: Path) -> List[str]:
    """Power labels that have at least one `{power}_polXXX_num*.pkl` in the folder."""
    found: List[str] = []
    for path in Path(data_dir).glob("*_pol*_num*"):
        match = re.match(r"(.+)_pol\d{3}p\d", path.name)
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return sorted(found, key=_power_sort_key)


def _power_sort_key(power: str) -> float:
    digits = "".join(c for c in power if c.isdigit() or c == ".")
    return float(digits) if digits else 0.0


def _polarization_points(cfg: ExperimentConfig, power: Optional[str] = None) -> List[Tuple[float, str]]:
    """Angles to summarise for one power: CSV log first, else the planned scan."""
    power = power or cfg.POLARIZATION_BASE_POWER or cfg.POWER_LEVEL
    log_path = cfg.polarization_log_path()
    if log_path.is_file():
        recorded = ScanLog(log_path).recorded_labels(power=power)
        if recorded:
            return recorded
    # Rebuild labels for this power even if cfg.POWER_LEVEL was rewritten.
    if cfg.POLARIZATION_SCAN is not None:
        return [(float(angle), f"{power}_{angle_tag(angle)}") for angle in cfg.POLARIZATION_SCAN]
    return []


# ----------------------------------------------------------------------------
# Loading and averaging
# ----------------------------------------------------------------------------

def collect_polarization_series(cfg: ExperimentConfig, points: Sequence[Tuple[float, str]],
                                power_label: str) -> Dict[str, Any]:
    """Mean and std over the chunks of each angle, for one power."""
    angles: List[float] = []
    n_chunks: List[int] = []
    channels: Dict[str, List[Tuple[float, float]]] = {}
    g2: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    harmonics: Dict[str, List[Tuple[float, float]]] = {}

    for angle, label in points:
        chunk_data = _load_polarization_angle(cfg, label)
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
        "harmonics": _series_arrays(harmonics, order),
        "power_label": power_label,
        "clockwise": cfg.POLARIZATION_P1.clockwise if cfg.POLARIZATION_P1 else True,
    }


def _load_polarization_angle(cfg: ExperimentConfig, label: str) -> Dict[str, Any]:
    merged_files, chunk_files = get_files(cfg.DATA_DIR, pattern=f"*{label}*")
    files = chunk_files or merged_files
    if not files:
        return {}
    data = get_data(files, cfg.MODES, label)
    if not data:
        return {}
    data = get_peaks(data, cfg.FREQUENCY)
    data = compute_g2_integration(data, cfg.INTEGRATION_WINDOW)
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

def _polar_grid(n_panels: int, title: str, max_cols: int = 4, dpi: int = 300):
    cols = max(1, min(max_cols, n_panels))
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.8 * rows), dpi=dpi,
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(np.asarray(axes)).reshape(rows, cols)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for index in range(n_panels, rows * cols):
        axes.flat[index].set_visible(False)
    return fig, axes


def _channel_arm(name: str) -> str:
    """"H3T" -> "T": the part of a channel label after its harmonic (e.g. the arm)."""
    return name[len(harmonic_of(name)):] or name


def _intensity_grid(channel_names: Sequence[str], title: str, dpi: int = 300):
    """A polar grid with one column per harmonic and one row per arm.

    The two arms of a harmonic (e.g. H3T and H3R) sit one above the other, so each
    column reads as "this harmonic" and each row as "this arm". Returns the figure and
    a {channel: ax} map; panels with no channel are hidden.
    """
    names = sorted(channel_names)
    harmonics = sorted({harmonic_of(name) for name in names})
    arms = sorted({_channel_arm(name) for name in names})
    cols, rows = max(1, len(harmonics)), max(1, len(arms))

    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.8 * rows), dpi=dpi,
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(np.asarray(axes)).reshape(rows, cols)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax_by_name: Dict[str, Any] = {}
    used = set()
    for name in names:
        row, col = arms.index(_channel_arm(name)), harmonics.index(harmonic_of(name))
        ax_by_name[name] = axes[row, col]
        used.add((row, col))
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in used:
                axes[row, col].set_visible(False)
    return fig, ax_by_name


def _fill_polar_circle(angles: np.ndarray, *series: np.ndarray):
    """Tile a partial scan by its own span so a polar plot covers the whole turn.

    Linear polarization repeats every 180 deg, so a 0-180 scan already samples every
    distinct state; copying that arc onto the empty half draws the familiar full
    butterfly instead of a bare semicircle. The same tiling completes a 0-90 scan
    (four copies) or any arc whose span divides 360 evenly, while a genuine 0-360 scan
    is already full and is left untouched.
    """
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 2:
        return (angles, *series)
    step = float(np.median(np.diff(angles)))
    span = angles[-1] - angles[0] + step
    if not (step > 0) or span >= 359.9:
        return (angles, *series)  # already (near) a full turn: nothing to copy
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


def _polar_series(ax, angles: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  color: str, label: Optional[str] = None, alpha: float = BAND_ALPHA) -> None:
    angles, mean, std = _fill_polar_circle(angles, mean, std)
    angles, mean, std = _close_polar_circle(angles, mean, std)
    theta = np.deg2rad(angles)
    low = np.clip(mean - std, 0.0, None)
    ax.fill_between(theta, low, mean + std, color=color, alpha=alpha, linewidth=0)
    ax.plot(theta, mean, color=color, marker="o", markersize=3.5, linewidth=1.6, label=label)


def _polar_style(ax, title: str, clockwise: bool = True) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1 if clockwise else 1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=14)
    ax.set_rlabel_position(112.5)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


def _unity_circle(ax, level: float = 1.0) -> None:
    """Dashed g²=1 reference, and start the radial axis at that level.

    g²(0) for these signals sits at or above 1, so anchoring the inner edge at 1
    (rather than 0) spends the whole panel on the bunching region and reads the
    unity circle as the axis origin. Called after the data so it fixes the lower
    limit without the autoscale pulling it back to 0.
    """
    theta = np.linspace(0, 2 * np.pi, 181)
    ax.plot(theta, np.full_like(theta, level), color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_rmin(level)


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
# The three figures (single power)
# ----------------------------------------------------------------------------

def plot_polarization_intensity(series: Dict[str, Any], save_dir: Path) -> None:
    """Grid of polar plots: count rate vs angle, one column per harmonic.

    The two arms of a harmonic (e.g. H3T and H3R) are stacked one above the other.
    """
    channels = series["channels"]
    if not channels:
        return
    fig, ax_by_name = _intensity_grid(
        channels, f"Intensity vs polarization angle — {series['power_label']}")
    for name, ax in ax_by_name.items():
        _polar_series(ax, series["angles"], channels[name]["mean"], channels[name]["std"], "tab:blue")
        _polar_style(ax, f"{name} (counts/s)", series["clockwise"])
    _save_polar_figure(fig, save_dir / "intensity_vs_angle.png")


def plot_polarization_g2(series: Dict[str, Any], save_dir: Path) -> None:
    """Grid of polar plots: g²(0) vs angle, one panel per pair."""
    pairs = series["pairs"]
    if not pairs:
        return
    names = sorted(pairs, key=lambda pair: (is_cross_harmonic(pair), pair))
    fig, axes = _polar_grid(
        len(names),
        f"$g^{{(2)}}(0)$ vs polarization angle — {series['power_label']}",
        max_cols=min(4, len(names)),
    )
    for index, pair in enumerate(names):
        ax = axes.flat[index]
        color = "tab:red" if is_cross_harmonic(pair) else "tab:green"
        _polar_series(ax, series["angles"], pairs[pair]["mean"], pairs[pair]["std"], color)
        _unity_circle(ax)
        _polar_style(ax, f"{pair[0]}–{pair[1]}", series["clockwise"])
    _save_polar_figure(fig, save_dir / "g2_vs_angle.png")


def plot_polarization_harmonics(series: Dict[str, Any], save_dir: Path) -> None:
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
    fig.suptitle(f"Harmonics vs polarization angle — {series['power_label']}",
                 fontsize=15, fontweight="bold")

    for column, harmonic in enumerate(harmonics):
        signal = series["harmonics"][harmonic]
        _polar_series(axes[0, column], series["angles"], signal["mean"], signal["std"], "tab:blue")
        _polar_style(axes[0, column], f"{harmonic} intensity (counts/s)", series["clockwise"])

        pair = auto_pairs.get(harmonic)
        ax = axes[1, column]
        if pair is None:
            ax.set_visible(False)
            continue
        g2 = series["pairs"][pair]
        _polar_series(ax, series["angles"], g2["mean"], g2["std"], "tab:green")
        _unity_circle(ax)
        _polar_style(ax, f"{harmonic} $g^{{(2)}}(0)$  ({pair[0]}–{pair[1]})", series["clockwise"])

    _save_polar_figure(fig, save_dir / "harmonics_vs_angle.png")


# ----------------------------------------------------------------------------
# Multi-power butterfly overlay
# ----------------------------------------------------------------------------

def plot_polarization_overlay(series_by_power: Dict[str, Dict[str, Any]], save_dir: Path) -> None:
    """Same three layouts, every power drawn on each panel as a coloured butterfly."""
    if len(series_by_power) < 2:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    powers = list(series_by_power)
    colors = _power_colors(powers)
    clockwise = next(iter(series_by_power.values())).get("clockwise", True)
    title_suffix = ", ".join(powers)

    # --- intensity: one column per harmonic, arms stacked ---
    channel_names = sorted({name for series in series_by_power.values() for name in series["channels"]})
    if channel_names:
        fig, ax_by_name = _intensity_grid(channel_names,
                                          f"Intensity vs polarization angle — {title_suffix}")
        for name, ax in ax_by_name.items():
            for power, series in series_by_power.items():
                if name not in series["channels"]:
                    continue
                d = series["channels"][name]
                _polar_series(ax, series["angles"], d["mean"], d["std"], colors[power],
                              label=power, alpha=0.18)
            _polar_style(ax, f"{name} (counts/s)", clockwise)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "intensity_vs_angle.png")

    # --- g2: one panel per pair ---
    pair_names = sorted({pair for series in series_by_power.values() for pair in series["pairs"]},
                        key=lambda pair: (is_cross_harmonic(pair), pair))
    if pair_names:
        fig, axes = _polar_grid(len(pair_names),
                                f"$g^{{(2)}}(0)$ vs polarization angle — {title_suffix}",
                                max_cols=min(4, len(pair_names)))
        for index, pair in enumerate(pair_names):
            ax = axes.flat[index]
            for power, series in series_by_power.items():
                if pair not in series["pairs"]:
                    continue
                d = series["pairs"][pair]
                _polar_series(ax, series["angles"], d["mean"], d["std"], colors[power],
                              label=power, alpha=0.18)
            _unity_circle(ax)
            _polar_style(ax, f"{pair[0]}–{pair[1]}", clockwise)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "g2_vs_angle.png")

    # --- harmonics: intensity on top, auto-g2 below ---
    harmonics = sorted({h for series in series_by_power.values() for h in series["harmonics"]})
    if harmonics:
        cols = len(harmonics)
        fig, axes = plt.subplots(2, cols, figsize=(4.4 * cols, 9.0), dpi=300,
                                 subplot_kw={"projection": "polar"})
        axes = np.asarray(axes).reshape(2, cols)
        fig.suptitle(f"Harmonics vs polarization angle — {title_suffix}",
                     fontsize=15, fontweight="bold")
        for column, harmonic in enumerate(harmonics):
            for power, series in series_by_power.items():
                if harmonic in series["harmonics"]:
                    d = series["harmonics"][harmonic]
                    _polar_series(axes[0, column], series["angles"], d["mean"], d["std"],
                                  colors[power], label=power, alpha=0.18)
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
            for power, series in series_by_power.items():
                if auto_pair not in series["pairs"]:
                    continue
                d = series["pairs"][auto_pair]
                _polar_series(ax, series["angles"], d["mean"], d["std"], colors[power],
                              label=power, alpha=0.18)
            _unity_circle(ax)
            _polar_style(ax, f"{harmonic} $g^{{(2)}}(0)$  ({auto_pair[0]}–{auto_pair[1]})", clockwise)
        _power_legend(fig, colors)
        _save_polar_figure(fig, save_dir / "harmonics_vs_angle.png")

    print(f"Polarization overlay ({len(powers)} powers) -> {save_dir}")


# ----------------------------------------------------------------------------
# The numbers behind the figures
# ----------------------------------------------------------------------------

def write_polarization_csv(series: Dict[str, Any], path: Path) -> Path:
    rows = []
    for angle_index, angle in enumerate(series["angles"]):
        common = {
            "power_label": series["power_label"],
            "polarization_angle_deg": f"{angle:g}",
            "n_chunks": int(series["n_chunks"][angle_index]),
        }
        for quantity, entries in (("countrate", series["channels"]),
                                  ("harmonic_countrate", series["harmonics"]),
                                  ("g2", series["pairs"])):
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
            fieldnames=["power_label", "polarization_angle_deg", "quantity", "target",
                        "mean", "std", "n_chunks"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}")
    return path


if __name__ == "__main__":
    main()
