# Pipeline for HBT Measurement of SHHG using SPADs

One command connects to the tagger, records coincidence histograms in time chunks, writes them in the format the analyzer reads, checks their stability over the run, and produces the full plot tree.

---

## Table of contents

1. [What is being measured](#1-what-is-being-measured)
2. [Repository map](#2-repository-map)
3. [Installation and running](#3-installation-and-running)
4. [The configuration object](#4-the-configuration-object)
5. [The pipeline, step by step](#5-the-pipeline-step-by-step)
6. [File formats on disk](#6-file-formats-on-disk)
7. [Units and naming conventions](#7-units-and-naming-conventions)
8. [Module reference](#8-module-reference)
9. [What the analyzer does](#9-what-the-analyzer-does)
10. [Troubleshooting](#10-troubleshooting)
11. [Known limitations](#11-known-limitations)

---

## 1. What is being measured

Each harmonic of the drive laser is split into a **transmitted** and a **reflected** arm, each arm landing on one detector, i.e. one Time Tagger channel. Channels are labelled `H3T`, `H3R`, `H4T`, `H4R`, `H5T`, `H5R`: harmonic order plus arm.

For every pair of channels the tagger builds a **coincidence histogram**: how many times a click on channel A was followed (or preceded) by a click on channel B at a given delay. Because the laser is pulsed at `FREQUENCY`, that histogram is a comb of peaks spaced by the repetition period

$$
T_{\text{rep}} = \frac{1}{\text{FREQUENCY}} = \frac{1}{18.66 \text{ MHz}} = 53.59 \text{ ns}
$$

* the **central peak** at zero delay holds coincidences from the *same* laser shot,
* the **satellite peaks** at ±k·T_rep hold coincidences from *different* shots, which are uncorrelated by construction and therefore measure the accidental level.

Hence the second-order correlation, normalised so that a coherent source gives 1:

$$
g^{(2)}(0) = \frac{\text{area of the central peak}}{\text{mean area of the satellite peaks}}
$$

| Type of light | $g^{(2)}(0)$ value |
| :--- | :--- |
| **Coherent** (Laser) | $g^{(2)}(0) = 1$ |
| **Thermal / Chaotic** (Thermal source, pseudo-thermal light) | $g^{(2)}(0) = 2$ |
| **Single-photon** (Fock state / Quantum emitter) | $g^{(2)}(0) = 0$ |

For two different harmonics n and m, the Cauchy–Schwarz ratio tests whether the cross-correlation exceeds what any classical field could produce:

$$
R = \frac{(g^{(2)}_{nm})^2}{g^{(2)}_{nn} \, g^{(2)}_{mm}} \quad \text{($R > 1$ violates the classical bound)}
$$

This is why the pipeline only correlates harmonics for which both the T and the R channel are declared.

---

## 2. Repository map

```
HBT_Code/
├── src/
│   ├── run_experiment.py     ← the only file you normally edit; entry point
│   ├── experiment_config.py    all knobs, derived paths, and experiment_config.txt
│   ├── hardware.py             the Thorlabs mounts, and polarization on top of them
│   ├── acquisition.py          live tagger acquisition, chunk loop, on-disk schema
│   ├── pkl_json_analyze.py     the analysis: peaks, g², R, and every figure
│   ├── config_examples.py      ready-made CONFIG blocks, one per kind of run (§3.7)
│   └── Polarization_calibration_v20_linear_only.py
│                               the lab's calibration script — read, never imported
├── data/      {MATERIAL}/{EXPERIENCE_TYPE}/{DATE}/…   inputs and recordings
├── results/   {MATERIAL}/{EXPERIENCE_TYPE}/{DATE}/…   figures
├── polarization_calibration/                          the P1/HWP lookup tables (§3.6)
└── requirements.txt
```

`data/` and `results/` are git-ignored: the repository holds code, the disk holds
measurements. `polarization_calibration/` is the exception — the lookup tables are
committed, because every polarization scan reuses one instead of recording its own
(§3.6).

Dependency direction — nothing ever points back up, so any module can be imported and poked at on its own:

```
run_experiment
   ├── acquisition ─────────┐
   ├── pkl_json_analyze ────┴──→ experiment_config ──→ hardware
   └── hardware ──────────────────────────────────────→ (numpy and the stdlib only)
```

Each module is written in two or three clearly banded parts, listed at the top of the
file :

| Module | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| `experiment_config.py` | the `ExperimentConfig` dataclass | writing `experiment_config.txt` | naming helpers |
| `hardware.py` | one Thorlabs rotation mount | linear polarization from the lookup table | – |
| `acquisition.py` | the live acquisition | the on-disk schema (`build_payload`) | – |
| `pkl_json_analyze.py` | the per-power analysis and figures | g²(0) and R versus time | the polar summaries versus angle |

---

## 3. Installation and running

### 3.1 Python packages

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib (+ notebook extras)
```

The repository already carries a virtual environment at `.HBTVenv/`; if you use it, call it explicitly so you are certain which interpreter runs:

```bash
.HBTVenv/bin/python src/run_experiment.py
```

### 3.2 The Time Tagger API is not a pip package

`TimeTagger` ships with the Swabian Instruments software package (<https://www.swabianinstruments.com/time-tagger/downloads/>). Install that system-wide and use the interpreter it registers.

The import is **lazy**: it happens inside `acquisition.run_acquisition`, not at module import time. Therefore everything except the acquisition — analysis, stability replotting, config-file writing — works on a laptop with no SDK and no hardware. If the import does fail you get an actionable message rather than a traceback:

```
ImportError: The TimeTagger Python API is not available. It ships with the Swabian
Instruments software (system/SDK installer, not pip): install it from
https://www.swabianinstruments.com/time-tagger/downloads/ and make sure this
interpreter is the one the installer registered. Use ANALYZE_ONLY=True to run the
analysis without hardware.
```

### 3.3 Running

Edit the `CONFIG` block at the top of `src/run_experiment.py`, then:

```bash
python src/run_experiment.py        # or
python -m src.run_experiment        # both work, see §8.5
```

Seven useful modes, all selected from `CONFIG`:

| Goal | `ANALYZE_ONLY` | `RUN_ANALYSIS_AFTER_ACQUIRE` | Needs hardware |
| --- | --- | --- | --- |
| Full run : acquire then analyze | `False` | `True` | yes |
| Acquire now, analyze later | `False` | `False` | yes |
| Automated power scan (§3.4) | `False` + `POWER_SCAN=[…]` | `True` | yes + one mount |
| Polarization scan (§3.6) | `False` + `POLARIZATION_SCAN=[…]` | `True` | yes + two mounts |
| Reprocess existing data | `True` | *(ignored)* | no |
| Replay a whole power sweep | `True` + `POWER_LEVELS=[…]` | *(ignored)* | no |
| Multi-power butterflies (§3.6) | `True` + `POLARIZATION_POWERS=[…]` | *(ignored)* | no |

Copy-paste blocks for each of those live in `src/config_examples.py` (§3.7).

`matplotlib` is switched to the `Agg` backend at the top of `run_experiment.py`: figures are always written to disk, never shown. The pipeline therefore runs over SSH, in a screen session, or from a scheduler without a display.

### 3.4 Power scan

#### By hand :
Set `POWER_LEVEL`, run, change `POWER_LEVEL`, run again. Everything else stays untouched. Each run adds its own files (`70mW_num1.pkl`, …) plus one `[run 70mW]` section in the folder's configuration file; the description of the setup is written once and shared. 

If a field describing the setup changed since a previous run in that folder, the pipeline warns instead of silently mixing two configurations (§6.2). When the scan is finished, one analysis pass covers all of it:

```python
ANALYZE_ONLY=True,
POWER_LEVELS=["65mW", "70mW", "75mW"],
```


#### Automatically : 
Listing the points in `POWER_SCAN` does the same thing in one command: for each point the rotation stage moves to the given angle, the normal acquisition records that power level, and after the last point a single analysis pass covers the whole scan, summary figures included.

```python
CONFIG = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="24052024",
    POWER_LEVEL="65mW",          # unused while POWER_SCAN is set
    ACQUISITION_DURATION_S=600.0,
    CHUNK_DURATION_S=60.0,

    ROTATION_STAGE_ENABLED=True,
    ROTATION_STAGE=RotationStageConfig(serial_number="27261601"),
    POWER_SCAN=[
        {"power": "25mW", "angle_deg": 8.0},
        {"power": "35mW", "angle_deg": 14.2},
        {"power": "45mW", "angle_deg": 21.0},
    ],

    RUN_ANALYSIS_AFTER_ACQUIRE=True,
    ANALYZE_ONLY=False,
)
```

Three points, ten minutes each: `python src/run_experiment.py` then takes half an hour and leaves `25mW_num1.pkl … 45mW_num10.pkl` in one folder, one `[run <power>]` section per point with the angle it was taken at, and the sweep figures under `results/.../power_sweep_summary/`.

> **Power and angle are unrelated as far as the software is concerned.** `"25mW"` is a **label** — it names files, legends and result
> folders, nothing more. No Malus law, no calibration curve, no power meter: whatever you measured at 8.0° is what `"25mW"` means.

Because each point carries its own angle, the two lists cannot fall out of step — which is exactly why the shape is a list of pairs rather than two parallel lists. The configuration is checked before anything moves: a missing key, an angle that is not a number, or the same power label twice (which would overwrite the first point's files) all raise immediately (§4.6).

Other fields of `CONFIG` behave as usual, per point: chunking, `SAVE_MERGED`, and `STABILITY_ENABLED`, which produces one set of stability figures under `results/.../{power}/stability/`.

If a point fails — a stalled tagger, a stage that never arrives — the run is marked `status = failed` in `experiment_config.txt`. `POWER_SCAN_STOP_ON_ERROR=True` (the default) then re-raises so that the sample is not left drifting through a scan nobody watches; set it to `False` to record what is left of the scan and analyse the points that did work. The stage is disconnected and the tagger released either way.

To reorder or repeat a scan, edit the list: points are recorded in the order given, and re-running an existing power label overwrites its files and updates its section.

### 3.5 The rotation stage needs Kinesis and pythonnet

Part 1 of `src/hardware.py` drives the mount through the **Kinesis .NET API**, reached from Python with **pythonnet**. Two installs, and only one of them is pip:

```bash
pip install pythonnet          # the Python ↔ .NET bridge
```

Kinesis itself is a Windows installer from Thorlabs (<https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control>). It is deliberately **not** in `requirements.txt`: nothing else in the repository needs
it, and the pipeline must stay installable on the analysis laptop. The assemblies are loaded from `C:\Program Files\Thorlabs\Kinesis`; point `RotationStageConfig.kinesis_path` elsewhere for a non-standard install.

Like the Time Tagger, the import is lazy — inside `RotationStage.connect()` — so a machine without Kinesis still analyses data, and a missing dependency gives an actionable message instead of a traceback.

The controller family is guessed from the first two digits of the serial number: `55…` is a K10CR1 cage rotator, `27…` a KDC101 K-cube (a PRM1Z8 mount, typically), `83…` a TDC001, `26…` a KST101, `40…` a benchtop stepper. Set `device_type="K10CR1"` to override the guess, or add an entry to `DEVICE_CLASSES` for a controller that is not listed — the .NET class name and its assembly are all that is needed.

Angles are **degrees, absolute, in the controller's own coordinate system**: they mean whatever the last homing made them mean. Home once with `home_on_connect=True` (or from the Kinesis GUI) before calibrating the (power, angle) pairs, and do not home again in the middle of a campaign.

**No hardware?** `ROTATION_STAGE_DRY_RUN=True` prints each move and skips every import, which is how a scan is rehearsed on a laptop:

```
[stage] dry run: not connecting (serial 27261601)
[stage] dry run: would move to 8 deg
```

It overrides `ROTATION_STAGE_ENABLED`, so a rehearsal costs one line and no edits to the rest of the block: the scan loop, the configuration file and the angles are all exercised, only the mount stays still (§10.16). `ANALYZE_ONLY=True` skips motion and acquisition entirely.

### 3.6 Polarization scan

The question is how the signal depends on the direction the crystal is driven along, so the polarization must turn while **the power stays constant** — otherwise every curve also carries the intensity dependence. Two mounts do that together:

* **P1**, the polarizer, turns to the requested polarization angle;
* the **half-wave plate** turns to whatever angle delivers the requested power *at that polarization*, read off the lookup table measured beforehand.

There is no optical model anywhere in this. The table is a grid of measured powers over (P1 angle, HWP angle), recorded by the lab's calibration script (`Polarization_calibration_v20_linear_only.py`, menu option `c`), and the pipeline only interpolates it.

**A calibration is recorded once and then reused.** The scan takes hours, so do not repeat it before every campaign: the tables in `polarization_calibration/` are committed with the repository and the defaults already point at `linear_polarization_lookup_latest.npz`, which is the current good one. Recalibrate only when the beam path changes — a new alignment, a swapped wave plate, a moved power meter. The calibration script writes into the same folder, so a freshly recorded table becomes the new `_latest` and is picked up with no config change at all.

The pipeline reads the `.npz` that script writes, with the same four keys (`p1_angles`, `hwp_angles`, `power_grid`, `metadata_json`) and the same rule for choosing an HWP angle — the *smallest* one whose interpolated power matches the request.

```python
CONFIG = ExperimentConfig(
    MATERIAL="ZnO100",
    EXPERIENCE_TYPE="David_Setup",
    DATE="27072026",
    POWER_LEVEL="50mW",                  # the label every angle of the scan shares
    ACQUISITION_DURATION_S=300.0,        # per angle: × 24 angles is the whole scan
    CHUNK_DURATION_S=60.0,

    POLARIZATION_SCAN=np.arange(0.0, 360.0, 15.0),
    POLARIZATION_POWER=50.0,             # in the table's unit (mW), not a label
    POLARIZATION_CALIBRATION_DIR=None,   # None → polarization_calibration/, reused as is
    POLARIZATION_STAGE_ENABLED=True,
    POLARIZATION_P1=RotationStageConfig(serial_number="27260002", clockwise=True),
    POLARIZATION_HWP=RotationStageConfig(serial_number="27260003", clockwise=True,
                                         home_on_connect=True),

    RUN_ANALYSIS_AFTER_ACQUIRE=True,
)
```

Homing is a property of a mount, not of the run: it belongs in the `RotationStageConfig`
of the mount you want homed, as above.

**What one command then does.** Before anything moves, a *preflight* asks the table whether the requested power is reachable at every angle, and says so:

```
[pol] lookup: linear_polarization_lookup_latest.npz: 37 P1 angles (0 to 180 deg)
      x 31 HWP angles (0 to 90 deg), 0 to 100 mW, recorded 2026-07-27T10:15:00
[pol] preflight: 22/24 angles can deliver 50 mW
  skipped - 90 deg: 50 mW out of reach (0 to 48 mW)
```

Then, per angle: P1 and the HWP move, the settling time passes, the normal acquisition runs, and the angle is written to the per-angle log. Angles the table cannot serve are skipped rather than recorded at the wrong power — a gap in the polar figure is honest, a point at the wrong power is not.

**How the files are named.** Each angle gets a label of its own, the shared power label plus the angle: `50mW_pol000p0`, `50mW_pol015p0`, … `50mW_pol345p0`. Fixed width, so no label is a prefix of another (which would make the analyser's `*{label}*` glob pick up two angles at once) and they sort by angle. Every angle is therefore an ordinary single-power run as far as the rest of the pipeline is concerned: its own chunk files, its own merged file, its own full analysis tree — the latter under `results/…/polarization/50mW/angles/50mW_pol015p0/`.

**Angle conventions**, which have to match the calibration script or the table means
nothing:

* an angle is a **polarization direction**, 0° = vertical in P1's frame;
* mounts receive `(sign × angle) mod 360`, the sign from `clockwise` — the calibration script uses clockwise on both mounts, and so     must the config;
* the table covers P1 over `[0, 180)` because a linear polarization repeats every 180°, so 190° reuses the HWP setting measured at 10°. The scan itself still runs the full 0–360°: the two halves are an independent measurement of the same physics, and their disagreement is a useful check on drift.

**The summary figures**, written to `results/…/polarization/{POWER}/summary/`, are the point of the exercise. Every point is the mean over that angle's chunks, and the shaded band is their standard deviation — the same chunk-to-chunk spread the per-power figures use as an error bar:

| Figure | One polar panel per | Shows |
| --- | --- | --- |
| `intensity_vs_angle.png` | channel | count rate (counts/s) |
| `g2_vs_angle.png` | pair | g²(0), green for auto pairs, red for cross pairs, dashed unit circle |
| `harmonics_vs_angle.png` | harmonic | a 2×N grid: the harmonic's total signal on top, its auto-g²(0) directly below |

g²(0) on these figures is the **integration** estimator only — the counts-based one is not comparable across angles while `Np` is hard-coded (§11.1).

`polarization_summary.csv` next to them holds the same numbers as text, and `data/…/polarization_scan.csv` records what each angle was actually set to (§6.4). The polar plots turn in the same direction as the mounts, so a lobe on the figure points where the polarizer pointed.

**Several powers: the butterfly overlay.** Record one scan per power *in the same `DATE` folder*, changing `POWER_LEVEL` and `POLARIZATION_POWER` between runs, then replay the folder with the powers listed:

```python
ANALYZE_ONLY=True,
POLARIZATION_SCAN=np.arange(0.0, 360.0, 15.0),
POLARIZATION_POWERS=["25mW", "45mW", "65mW"],
```

That rebuilds each power's three summaries and then redraws the same three layouts with every power as a coloured curve on each panel, under `results/…/polarization/overlay/`. Leaving `POLARIZATION_POWERS=None` makes the pipeline discover the powers from the file names instead, which is what happens automatically at the end of an acquisition. With a single power the overlay is skipped and says so.

The full result tree is in §6.5.

### 3.7 Ready-made configs

`src/config_examples.py` holds one complete `CONFIG` per kind of run — single power,
power scan, polarization scan, multi-power overlay, analysis only — aimed at the real
folders and the real serial numbers. Running it builds every one of them, so a typo or
an impossible combination shows up in a second rather than at the start of a six-hour
scan:

```bash
python src/config_examples.py
```

It prints, per example, the folders it resolves to, the correlation pairs it will
record and how many hours of acquisition it adds up to. Copy the body of the one you
want between `CONFIG = ExperimentConfig(` and `)`.

Nothing there touches hardware: building an `ExperimentConfig` only runs the validation
of §4.8. Before a session, that plus a dry run of the mounts (§3.5, §10.16) is the
cheapest way to find a mistake while it still costs a second.

---

## 4. The configuration object

`ExperimentConfig` (in `src/experiment_config.py`) is a dataclass (field names are upper-case so they read like lab-notebook entries). `run_experiment.py` builds one instance and passes that same object to every stage, so the metadata written on disk cannot drift from the parameters that were actually used.

### 4.1 Identity and paths

| Field | Default | Meaning |
| --- | --- | --- |
| `MATERIAL` | `"ZnO100"` | sample; first level of `data/` and `results/` |
| `EXPERIENCE_TYPE` | `"David_Setup"` | setup name; second level |
| `DATE` | `"11142023"` | third level, free-form string |
| `POWER_LEVEL` | `"65mW"` | the power of **this** acquisition |
| `POWER_LEVELS` | `None` → `[POWER_LEVEL]` | analysis only: list of powers to replay |
| `FILE_PREFIX` | `""` | prepended to exported names, e.g. `"David_"` |
| `CHUNK_INDEX_DIGITS` | `1` | `1` → `…_num7.pkl`, `3` → `…_num007.pkl` |
| `DATA_DIR` | `None` → derived | override the input/output data folder |
| `RESULTS_DIR` | `None` → derived | override the figure folder |

When left at `None`, the two directories are derived from the repository root, which is `Path(__file__).resolve().parent.parent` — exactly the same rule the analyzer has always used:

```
DATA_DIR    = <repo>/data/{MATERIAL}/{EXPERIENCE_TYPE}/{DATE}
RESULTS_DIR = <repo>/results/{MATERIAL}/{EXPERIENCE_TYPE}/{DATE}
```

### 4.2 Optics and metadata

These never reach the analyzer except for `FREQUENCY` and `MODES`. All of it is written to `experiment_config.txt`.

| Field | Default | Meaning |
| --- | --- | --- |
| `FREQUENCY` | `18.66e6` | laser repetition rate in **Hz**; sets `T_rep` |
| `MODES` | `{1: "H3T", 5: "H3R", 10: "H5T", 14: "H5R"}` | tagger channel → physical label |
| `P1`, `P2`, `P3` | `{"present": False, "angle_deg": 0.0}` | polarizers |
| `HWP_ANGLE_DEG` | `0.0` | half-wave plate angle |
| `COVER` | `{"present": False, "description": "between harmonics"}` | beam block |
| `FILTERS` | `{"H3": {"separation": …, "filter": "700-40"}, …}` | per harmonic |

`MODES` is the most load-bearing entry of the whole config: it decides which channels are read, which pairs are correlated, how the harmonics are grouped, and which labels appear on every figure. See §7.3.

### 4.3 Hardware

| Field | Default | Meaning |
| --- | --- | --- |
| `REGISTERED_CHANNELS` | `None` → `sorted(MODES)` | channels to configure and read |
| `TRIGGER_LEVEL_V` | `0.5` | trigger threshold applied to every channel |
| `INPUT_DELAYS_PS` | `{}` | `{channel: ps}` cable/detector delay compensation |
| `DEADTIMES_PS` | `{}` | `{channel: ps}` per-channel dead time |
| `TAGGER_SERIAL` | `""` | `""` selects the first device found |
| `CORR_BINWIDTH_PS` | `300` | histogram bin width in picoseconds |
| `CORR_N_BINS` | `2009` | number of bins, centred on zero delay |
| `ACQUISITION_DURATION_S` | `600.0` | total measurement time |
| `CHUNK_DURATION_S` | `60.0` | one exported file per chunk |
| `SAVE_MERGED` | `True` | also write the cumulative full-run histogram |
| `EXPORT_FORMAT` | `"pkl"` | `"pkl"` or `"json"` |
| `SAVE_RAW_TTBIN` | `False` | debug only; writes the raw time-tag stream |

The histogram geometry follows from the two `CORR_*` values:

```
delay span    = CORR_N_BINS × CORR_BINWIDTH_PS = 2009 × 300 ps = 602.7 ns
delay axis    = −301.35 ns … +301.35 ns
peaks in view = ±301.35 / 53.59 = ±5 orders  →  1 central + 10 satellite peaks
```

These defaults reproduce the geometry of the historical David-setup files (2009 bins of 300 ps), which is why old and new data can be analysed side by side.Widen `CORR_N_BINS` if you want more satellite peaks — see the trap in §10.7.

### 4.4 Stability (live runs only)

| Field | Default | Meaning |
| --- | --- | --- |
| `STABILITY_ENABLED` | `False` | take snapshots during the acquisition |
| `STABILITY_SNAPSHOT_INTERVAL_S` | `60.0` | capture-time between two snapshots |
| `STABILITY_INTEGRATION_WINDOW_NS` | `4.0` | peak window, in **nanoseconds** |
| `STABILITY_PEAKS` | `15` | satellite orders requested on each side |
| `STABILITY_ROLLING_WINDOW_SNAPS` | `10` | width of the rolling view, in snapshots |

Choose the interval so that `ACQUISITION_DURATION_S / interval` is comfortably larger than `STABILITY_ROLLING_WINDOW_SNAPS`, otherwise the rolling figure is empty (§10.9).

### 4.5 Analysis

| Field | Default | Meaning |
| --- | --- | --- |
| `INTEGRATION_WINDOW` | `8` | peak window in **bins** (not ns — see §7.1) |
| `INTEGRATION_WINDOWS_SWEEP` | `np.arange(1, 31, 1)` | windows swept in the sweep figures |
| `RUN_ANALYSIS_AFTER_ACQUIRE` | `True` | chain the analyzer after the acquisition |
| `ANALYZE_ONLY` | `False` | skip the hardware entirely |

### 4.6 Rotation stage and power scan

| Field | Default | Meaning |
| --- | --- | --- |
| `POWER_SCAN` | `None` | list of `{"power": …, "angle_deg": …}`; `None` → single power |
| `POWER_SCAN_STOP_ON_ERROR` | `True` | abort the scan on the first failed point |
| `ROTATION_STAGE_ENABLED` | `False` | actually move the stage |
| `ROTATION_STAGE_DRY_RUN` | `False` | log the moves, import nothing, move nothing |
| `ROTATION_STAGE` | `None` | a `RotationStageConfig` (below) |
| `ROTATION_ANGLE_DEG` | `None` | the angle of **this** acquisition |

`ROTATION_ANGLE_DEG` is normally set by the scan loop, one step at a time, and ends up in the run's section and in each data file's `Parameters`. Setting it by hand together with `ROTATION_STAGE_ENABLED=True` and no `POWER_SCAN` is the single-point case: move once, then record one power level.

`RotationStageConfig` (in `src/hardware.py`) describes the mount itself, and the same dataclass is used for the two mounts of a polarization scan:

| Field | Default | Meaning |
| --- | --- | --- |
| `serial_number` | `""` | printed on the controller; required to move |
| `device_type` | `None` → from the serial | `"K10CR1"`, `"KDC101"`, `"TDC001"`, … |
| `channel` | `1` | benchtop controllers only: which axis |
| `scale_deg_per_count` | `None` | leave it; Kinesis already reports degrees |
| `home_on_connect` | `False` | home before the first move |
| `move_timeout_s` | `120.0` | per move, and for homing |
| `settle_time_s` | `0.5` | extra wait after the target is reached |
| `angle_tolerance_deg` | `0.05` | how close counts as arrived |
| `max_velocity_deg_s` | `None` | `None` keeps the controller's setting |
| `acceleration_deg_s2` | `None` | idem |
| `kinesis_path` | `C:\Program Files\Thorlabs\Kinesis` | where the assemblies live |
| `poll_interval_s` | `0.2` | how often the position is read while moving |
| `clockwise` | `True` | `False` negates every requested angle |

A move is commanded and then **polled**: `move_to_angle_deg` returns once the position is within `angle_tolerance_deg` of the target, waits `settle_time_s` for the mount to stop ringing, and raises `TimeoutError` after `move_timeout_s`. Tolerance and settling time therefore define "arrived", not the controller's own idea of a finished move — a mount that hunts around its target still satisfies it.

`clockwise` must match the mount the calibration was taken on: with `False`, 15° is sent to the controller as 345°. Angles are wrapped into `[0, 360)` and compared the short way round, so a move from 359° to 1° arrives instead of timing out at the seam.

### 4.7 Polarization scan

| Field | Default | Meaning |
| --- | --- | --- |
| `POLARIZATION_SCAN` | `None` | the angles to record, e.g. `np.arange(0, 360, 15)` |
| `POLARIZATION_POWER` | `0.0` | power held at every angle, in the **table's unit** |
| `POLARIZATION_POWERS` | `None` | analysis only: the powers to summarise and overlay |
| `POLARIZATION_CALIBRATION` | `None` | a specific lookup file; `None` → `linear_polarization_lookup_latest.npz` in the folder below, else the newest table there |
| `POLARIZATION_CALIBRATION_DIR` | `None` → `polarization_calibration/` | where the reused tables live |
| `POLARIZATION_STAGE_ENABLED` | `False` | actually move P1 and the HWP |
| `POLARIZATION_STAGE_DRY_RUN` | `False` | log the moves, import nothing, move nothing |
| `POLARIZATION_P1` | `None` | `RotationStageConfig` of the polarizer |
| `POLARIZATION_HWP` | `None` | `RotationStageConfig` of the half-wave plate |
| `POLARIZATION_SETTLE_TIME_S` | `0.2` | extra pause after both mounts have arrived |
| `POLARIZATION_ANGLE_DEG` | `None` | the angle of **this** acquisition, set by the loop |
| `POLARIZATION_BASE_POWER` | `None` | the power label of **this** acquisition, set by the loop |

`POLARIZATION_POWER` is a **number in the calibration's unit** (mW for a power-meter
table), not a label like `"50mW"`: it is looked up in the table. `POWER_LEVEL` remains
the label that names the files, shared by every angle of the scan.

The last two fields are bookkeeping the scan loop fills in, not knobs. While one angle
is being recorded, `POWER_LEVEL` *is* that angle's label (`50mW_pol015p0`), so
`POLARIZATION_BASE_POWER` is what remembers that the campaign is called `50mW` — which
is how the results of every angle end up under the same power folder.

`POLARIZATION_POWERS` is the multi-power replay of §3.6: it names the power labels
already recorded in the folder, and it is the only field the butterfly overlay needs.

### 4.8 Validation and derived values

`__post_init__` fills the defaults that depend on other fields and rejects mistakes
early: an `EXPORT_FORMAT` that is neither `pkl` nor `json`, a non-positive duration,
and four ways of getting a power scan wrong.

| Situation | Why it is refused |
| --- | --- |
| an entry without `"power"` or `"angle_deg"` | the pair would be incomplete |
| `"angle_deg"` that is not a number | the move would fail after the tagger connected |
| the same power label twice | the second point would overwrite the first one's files |
| one power label inside another (`"5mW"` and `"45mW"`) | the `*{label}*` glob cannot tell their files apart, so the two powers would be averaged into one point — pad them, `"05mW"` |
| `POWER_SCAN` while the stage is disabled | every power would be recorded at one angle |
| `ROTATION_STAGE_ENABLED` with no serial number | nothing to connect to |

and five ways of getting a polarization scan wrong:

| Situation | Why it is refused |
| --- | --- |
| `POWER_SCAN` and `POLARIZATION_SCAN` together | that is one acquisition per (power, angle): run one scan at a time |
| two angles closer than 0.1° once wrapped | they would share a file label, hence their files |
| `POLARIZATION_SCAN` without `POLARIZATION_POWER` | there would be nothing to look up in the table |
| `POLARIZATION_SCAN` while the mounts are disabled | every angle would be recorded at whatever polarization the mounts happen to hold |
| a mount enabled with no serial number | nothing to connect to |
| `POLARIZATION_POWERS` with one label inside another | same glob ambiguity as above |

The hardware checks are skipped when `ANALYZE_ONLY=True` (reprocessing never moves
anything) or when the matching `*_DRY_RUN` is set (an explicit "I know there is no
hardware"). When either scan is configured, `POWER_LEVELS` is **derived from it**, in
order, so the analysis covers exactly the points that were recorded.

Everything else is computed on demand:

| Member | Returns |
| --- | --- |
| `merged_dir` | `DATA_DIR/merged` |
| `power_results_dir` | `RESULTS_DIR/{POWER_LEVEL}` |
| `stability_dir` | `RESULTS_DIR/{POWER_LEVEL}/stability` |
| `chunk_path(i)` | `DATA_DIR/{PREFIX}{POWER}_num{i}.{ext}` |
| `merged_path()` | `DATA_DIR/merged/{PREFIX}{POWER}_merged.{ext}` |
| `config_file_path()` | `DATA_DIR/experiment_config.txt` |
| `n_chunks` | `ceil(ACQUISITION_DURATION_S / CHUNK_DURATION_S)` |
| `chunk_durations()` | list of chunk lengths, last one shortened to fit |
| `is_power_scan()` | `True` when `POWER_SCAN` holds at least one point |
| `power_scan_points()` | `[("25mW", 8.0), …]` in the configured order |
| `for_power(power, angle)` | a copy of the config describing one point of the scan |
| `is_polarization_scan()` | `True` when `POLARIZATION_SCAN` is set |
| `polarization_scan_points()` | `[(0.0, "50mW_pol000p0"), …]` in the configured order |
| `for_polarization(angle)` | a copy describing one angle, with its label as `POWER_LEVEL` |
| `for_polarization_power(power)` | a copy aimed at one power of a multi-power campaign |
| `polarization_log_path()` | `DATA_DIR/polarization_scan.csv` |
| `polarization_root` | `RESULTS_DIR/polarization` |
| `polarization_summary_dir(power)` | `RESULTS_DIR/polarization/{power}/summary` |
| `polarization_overlay_dir` | `RESULTS_DIR/polarization/overlay` |
| `active_harmonics()` | `{"H3": (1, 5), "H5": (10, 14)}` — complete harmonics only |
| `active_channels()` | flattened channels of the complete harmonics |
| `correlation_pairs()` | every unordered pair of those, always `ch1 < ch2` |
| `file_parameters(...)` | the tiny per-file metadata block |

`active_harmonics()` groups the channels by the first two characters of their label
(`H3T` → `H3`) and **keeps only groups with at least two channels**, because R needs
an autocorrelation per harmonic. A lone channel is therefore dropped from the
correlations; if no harmonic is complete at all, the code falls back to correlating
every registered channel so that nothing is silently lost.

With the default `MODES`, `correlation_pairs()` returns the six pairs

```
(1, 5)  H3T-H3R   autocorrelation H3
(10,14) H5T-H5R   autocorrelation H5
(1,10) (1,14) (5,10) (5,14)        the four cross terms H3×H5
```

---

## 5. The pipeline, step by step

```
                          run_experiment.main(cfg)
                                     │
              ┌──────────────────────┴───────────────────────┐
      ANALYZE_ONLY=True                              ANALYZE_ONLY=False
              │                                              │
              │                              open_stage(cfg)  (scan or single angle)
              │                                              │
              │                            ┌─── per power scan point ──────────────┐
              │                            │  stage.move_to_angle_deg              │
   write_configuration(cfg)                │  record_run(step)  (running)          │
              │                                              │
              │                              acquisition.run_acquisition(step)
              │                                 ├─ connect + configure channels
              │                                 ├─ for each chunk:
              │                                 │    fresh Countrate + Correlations
              │                                 │    poll until the chunk is full
              │                                 │    export  {POWER}_num{i}.pkl
              │                                 │    add to the software sum
              │                                 ├─ merged/{POWER}_merged.pkl
              │                                 ├─ stability_snapshots.pkl
              │                                 └─ freeTimeTagger (always)
              │                                              │
              │                            │  record_run(step, result)  (complete) │
              │                            │  plot_stability(snapshots, step)      │
              │                            └───────────────────────────────────────┘
              │                                              │
              │                              stage.disconnect()  (always)
              └──────────────────────┬───────────────────────┘
                                     │
                    pkl_json_analyze.run_analysis(cfg)
                                     │
                             print_summary(cfg, runs)
```

Without a power scan the loop runs once, on `POWER_LEVEL`, and nothing moves unless
the stage is enabled with an explicit `ROTATION_ANGLE_DEG` — which is why adding the
scan changed no behaviour for a single-power run. `run_analysis(cfg)` is called **once**
either way: with a scan, `POWER_LEVELS` already holds every point, so the sweep figures
come out of the same pass.

A **polarization scan** replaces the middle of that diagram with its own loop
(`acquire_polarization_scan`): the preflight first, then per angle P1 and the HWP move,
the same `run_acquisition` records the angle's label as if it were a power level, and
the angle is written to the CSV log instead of a `[run …]` section. Its analysis differs
too: `run_analysis` is called once **per angle** — each angle is one acquisition, and a
single pass over 24 of them would build a 24-point "power sweep" out of angles — and
the polar summaries and the overlay come after (§3.6).

On the `ANALYZE_ONLY` branch the configuration file is written only if the folder does
not have one yet. A replay config usually carries the analysis fields and not the optics
or the tagger settings, and rewriting the file would replace the record of what was
actually measured by this config's defaults.

### 5.1 Connect

`createTimeTagger(TAGGER_SERIAL)`, then for every registered channel
`setTriggerLevel`, plus `setInputDelay` and `setDeadtime` when that channel appears in
the corresponding dict. `sync()` guarantees the settings are live before any
measurement object exists. The serial and the channel → label mapping are printed so
the log identifies the device.

### 5.2 The chunk loop

`ACQUISITION_DURATION_S` is cut into pieces of `CHUNK_DURATION_S` (600 s / 60 s → ten
chunks; a remainder becomes a shorter last chunk). For **each** chunk:

1. a fresh `Countrate` over all registered channels and a fresh `Correlation` per pair
   are created and `clear()`ed;
2. the loop polls `getCaptureDuration()` — hardware capture time, not wall clock —
   every `POLL_INTERVAL_S = 0.05 s`, calling back for progress and snapshots;
3. when the chunk is full every measurement is stopped, `build_payload` turns them
   into the schema of §6.1 and the file is written;
4. the histograms and the channel totals are added to the running software sum, then
   the measurement objects are dropped so the tagger frees their memory.

Recreating the measurements is what makes the chunks **statistically independent**:
the spread between chunks is a genuine error bar, which is what the analyzer turns
into the shaded bands of the summary figures. The price is a few milliseconds of dead
time at each boundary.

If capture time stops advancing — the usual symptom of a device that dropped off the
bus — the loop raises `TimeoutError` after `STALL_TIMEOUT_S = 30 s` of grace instead
of hanging forever.

### 5.3 Merged file and snapshots come for free

Coincidence histograms are purely additive, so the full-run histogram is just the sum
of the chunk histograms and the cumulative histogram at time *t* is
`(sum of finished chunks) + (current chunk so far)`. Both are computed in software by
the `_Cumulative` helper. Nothing extra runs on the tagger: one set of measurements
serves the chunk files, the merged file and the stability snapshots at once.

The merged countrates are recomputed as `total counts / total duration`, i.e. the
average over the whole run.

### 5.4 Snapshots

When `STABILITY_ENABLED`, every `STABILITY_SNAPSHOT_INTERVAL_S` of capture time the
recorder appends, for every pair, a copy of the cumulative histogram together with
the elapsed capture time. Copies are always fresh arrays, so nothing aliases the
tagger's buffers. The result is pickled to

```
results/{MATERIAL}/{EXPERIENCE_TYPE}/{DATE}/{POWER}/stability/stability_snapshots.pkl
```

It lives under `results/` on purpose: any `.pkl` sitting in `DATA_DIR` whose name
contains the power level would be picked up by the analyzer's chunk glob and would
show up as a bogus chunk.

### 5.5 Stability plots, then analysis

Stability is plotted **before** the analysis: it takes a second and tells you whether
the run is usable, while the full analysis can take minutes. Three figures land in the
`stability/` folder next to the snapshots — `stability_g2.png`, `stability_R.png` and
`stability_rolling.png` (§8.5).

Finally `run_analysis(cfg)` runs the analyzer over `POWER_LEVELS` and
`print_summary` lists every path produced.

---

## 6. File formats on disk

```
data/{MATERIAL}/{EXPERIENCE_TYPE}/{DATE}/
    experiment_config.txt          the setup, written once for the whole folder
    65mW_num1.pkl                  chunk 1  → analyzer key "chunk_1"
    65mW_num2.pkl                  chunk 2  → analyzer key "chunk_2"
    …
    merged/65mW_merged.pkl         cumulative full-run histogram
    raw/65mW.ttbin                 only if SAVE_RAW_TTBIN=True

results/{MATERIAL}/{EXPERIENCE_TYPE}/{DATE}/
    65mW/65mW_merged/…             spectrograms, drawn from the merged file
    65mW/chunk_1/…                 per-chunk figures (only without a merged file)
    65mW/power_summary/…           chunk-averaged figures
    65mW/stability/…               stability figures + snapshots
    power_sweep_summary/…          only when several powers are analysed
    polarization/…                 only for a polarization scan (§6.5)
```

The name of a figure folder is the analyzer's key for the file it was drawn from
(§7.2), so with `SAVE_MERGED=True` the spectrograms sit in `65mW_merged/` and there are
no `chunk_N/` folders at all — the chunks are used for the error bars in
`power_summary/`, not for figures of their own (§9, step 6).

### 6.1 A data file

Both `.pkl` and `.json` hold the same dict. This schema is dictated by
`pkl_json_analyze.get_data()` and must be respected exactly:

```python
{
  "Parameters": {"power_level": "65mW", "chunk": 1, "duration_s": 60.0,
                 "rotation_angle_deg": 8.0},   # only when an angle was used

  "Correlation": {
      "(1, 5)":  [[-301200, -300900, …], [3, 3, 2, 4, …]],   # delays in ps, counts
      "(1, 10)": [[…], […]],
      …
  },

  "Countrate": {
      "1": [21446.21, 1286773],     # [rate in counts/s, total counts]
      "5": [23359.71, 1401583],
      …
  },
}
```

Details that matter:

* **Pair keys** are strings `"(ch1, ch2)"` with a comma **and a space**, always with
  `ch1 < ch2`. The analyzer skips any key where `ch1 > ch2`, so a mirrored duplicate
  would simply be ignored.
* **Delays are picoseconds** in the file. `get_data()` divides by 1000, so everything
  downstream of it is in nanoseconds.
* **Counts are the raw histogram**, no normalisation, no background subtraction.
* Everything is plain Python `int`/`float` lists, which keeps the `.pkl` and the
  `.json` byte-for-byte equivalent in content.
* **`Parameters` deliberately holds almost nothing** — only what makes this file
  differ from its neighbours: its power level, its chunk number, how long it lasted,
  and the stage angle when one was used. The analyzer ignores the key entirely; it is
  there so a file found on its own can still be identified.

### 6.2 `experiment_config.txt`

Everything shared by the files of a folder lives here instead of being duplicated in
each of them. The format is INI, so it opens in any editor and is still parseable
with `configparser` — which is how a new run appends itself without disturbing the
runs already listed.

```ini
# HBT experiment configuration
# Written by src/run_experiment.py - shared by every power level of this folder.
# The data files themselves carry no metadata: everything about the setup is here.

[identity]
material        = ZnO100
experience_type = David_Setup
date            = 11142023

[optics]
laser_frequency = 18.66 MHz
hwp_angle_deg   = 0
cover           = absent (between harmonics)
polarizer_p1    = absent
polarizer_p2    = absent
polarizer_p3    = present at 45 deg
filter_h3       = per-channel, 700-40
filter_h5       = per-channel, 400-20

[channels]
1  = H3T
5  = H3R
10 = H5T
14 = H5R

[hardware]
tagger_serial     = (first device)
trigger_level_v   = 0.5
corr_binwidth_ps  = 300
corr_n_bins       = 2009
input_delays_ps   = none
deadtimes_ps      = 1: 25900, 5: 26200
correlation_pairs = (1, 5), (1, 10), (1, 14), (5, 10), (5, 14), (10, 14)

[analysis]
integration_window_bins   = 8
integration_windows_sweep = 1 to 30 step 1

[power_scan]
enabled       = yes
stage_serial  = 27261601
stage_motion  = yes
stop_on_error = yes
points        = 25mW @ 8 deg, 35mW @ 14.2 deg, 45mW @ 21 deg

[run 25mW]
status             = complete
started            = 2026-07-27 15:16:27
duration_s         = 600
chunk_duration_s   = 60
chunks             = 10
export_format      = pkl
merged             = yes
raw_ttbin          = no
stability          = every 60 s, 8 ns window, 15 peaks, rolling 10 snapshots
rotation_angle_deg = 8
finished           = 2026-07-27 15:26:31
```

* The shared sections are rewritten from the config at every run; the `[run <power>]`
  sections accumulate and are sorted by numeric power.
* Each run is recorded **twice**: once before acquiring (`status = running`, planned
  duration and chunk count) and once after (`status = complete`, what was really
  recorded, plus `finished`). An interrupted run therefore still leaves a trace, and
  `started` is preserved across the update. A power point that raised is rewritten as
  `status = failed`, keeping the numbers it was *asked* for.
* `rotation_angle_deg` appears whenever an angle was used, so a folder of runs is also
  the record of the calibration the scan was based on.
* `[power_scan]` describes the scan as configured, including whether the stage really
  moved (`stage_motion = dry run (angles logged, nothing moved)` after a rehearsal).
  It only appears when `POWER_SCAN` is set.
* `ANALYZE_ONLY` writes the shared sections only, and only when the folder has no
  configuration file yet — reprocessing never invents a run that did not happen, and
  never overwrites the record of one that did.
* If a shared value differs from what the file already says, the pipeline prints a
  warning naming each field rather than overwriting in silence:

  ```
  WARNING: experiment_config.txt already describes a different configuration:
    optics.filter_h3: 'per-channel, 700-40' -> 'per-channel, 650-40'
    The runs listed in this folder are no longer comparable. Consider using another
    DATE / EXPERIENCE_TYPE folder.
  ```

  `[power_scan]` and `[polarization_scan]` are exempt from that check: extending a
  scan, reordering it or rehearsing it as a dry run are all legitimate, and each
  point already records the angle it was taken at — in its own section for a power
  scan, in the CSV log for a polarization scan (§6.4).
* Reprocessing (`ANALYZE_ONLY=True`) rewrites nothing at all when the file exists, so
  a replay config that only carries the analysis fields cannot erase the record of the
  run (§5).

### 6.3 `stability_snapshots.pkl`

```python
{
  "times": np.ndarray,                    # capture seconds at each snapshot
  "index": np.ndarray,                    # shared delay axis, picoseconds
  "hist":  {(1, 5): [np.ndarray, …], …},  # cumulative histogram per snapshot
}
```

Keys of `hist` are `(chA, chB)` integer tuples — not the strings used in the data
files. Reload it with `acquisition.load_snapshots(path)` and feed it straight back to
`pkl_json_analyze.plot_stability(snapshots, cfg)` to redraw without measuring again.

### 6.4 `polarization_scan.csv`

A polarization scan is dozens of acquisitions, so its per-angle record is one table
rather than dozens of `[run …]` sections. It lives next to the data, in
`data/…/polarization_scan.csv`, one row per angle:

```csv
polarization_angle_deg,power_label,requested_power,unit,p1_angle_deg,hwp_angle_deg,reachable_min,reachable_max,status,chunks,duration_s,recorded_at,note
0,50mW_pol000p0,50,mW,0,25.4006,0,100,complete,5,300,2026-07-27 15:16:27,
15,50mW_pol015p0,50,mW,15,24.8112,0,98,complete,5,300,2026-07-27 15:22:04,
90,50mW_pol090p0,50,mW,,,0,48,unreachable,,,2026-07-27 15:27:41,
```

* `p1_angle_deg` and `hwp_angle_deg` are what the mounts were **actually set to**, read
  off the lookup table — the record of the calibration this scan relied on.
* `reachable_min` / `reachable_max` are what the table could deliver at that angle, so
  an `unreachable` row explains itself: the requested power was outside that range and
  nothing was moved.
* `status` is `complete`, `failed` or `unreachable`. Only `complete` rows are used when
  the summaries are rebuilt, which is why a failed angle leaves an honest gap in the
  polar figures instead of a point at the wrong power.
* Re-running an angle replaces its row; rows stay sorted by angle.

### 6.5 A polarization scan's results

```
results/{MATERIAL}/{EXPERIENCE_TYPE}/{DATE}/polarization/
    25mW/summary/intensity_vs_angle.png     one polar panel per channel
    25mW/summary/g2_vs_angle.png            one polar panel per pair
    25mW/summary/harmonics_vs_angle.png     2×N: intensity on top, auto-g² below
    25mW/summary/polarization_summary.csv   the same numbers as text
    25mW/angles/25mW_pol000p0/…             the ordinary analysis tree of one angle
    25mW/angles/25mW_pol015p0/…
    45mW/…                                  one folder per power in this DATE folder
    65mW/…
    overlay/intensity_vs_angle.png          the same three layouts, every power
    overlay/g2_vs_angle.png                   overlaid as coloured butterflies
    overlay/harmonics_vs_angle.png
```

Each power is self-contained, so a scan can be added to the folder later and only the
overlay has to be redrawn. `overlay/` appears only when at least two powers are present.

---

## 7. Units and naming conventions

### 7.1 Units

Most confusion in this codebase comes from three different delay units and two
different meanings of "integration window". The table is the reference:

| Quantity | Config | On disk | In the analyzer |
| --- | --- | --- | --- |
| Delay axis | `CORR_BINWIDTH_PS` (ps) | picoseconds | **nanoseconds** (`get_data` divides by 1000) |
| Capture time | seconds | seconds | – |
| Tagger capture duration | – | – | picoseconds from the API, converted on read |
| Laser rate | `FREQUENCY` in Hz | `18.66 MHz` in the txt | `T_rep` in ns |
| Analysis window | `INTEGRATION_WINDOW` | – | **number of bins** |
| Stability window | `STABILITY_INTEGRATION_WINDOW_NS` | – | **nanoseconds** |

`INTEGRATION_WINDOW` counting bins is a historical property of `pkl_json_analyze.py`,
where the value indexes samples around a peak while the figure titles call it "ns".
It has been left untouched to keep old results reproducible. To make the two windows
describe the same physical width:

```
STABILITY_INTEGRATION_WINDOW_NS  ≈  INTEGRATION_WINDOW × CORR_BINWIDTH_PS / 1000
                              2.4  =  8 × 300 / 1000
```

### 7.2 Chunk file names

`get_data()` searches the file name for `{POWER_LEVEL}_num<digits>` and turns those
digits into its internal key:

```
65mW_num1.pkl                                        → chunk_1
David_65mW_num7.pkl                                  → chunk_7
11142023_ZnO_…_res_65mW_num3.json                    → chunk_3
65mW_num007.pkl                                      → chunk_007   (CHUNK_INDEX_DIGITS=3)
anything_without_the_pattern.pkl                     → the file stem, as-is
```

The key becomes the sub-folder name under `results/{POWER}/`, so it is worth keeping
tidy. Any prefix is allowed as long as `{POWER}_num<N>` appears somewhere, which is
why the historical David-setup files load unchanged.

### 7.3 Channels, labels, harmonics

* `MODES` maps the **physical tagger channel number** to a label. Channel numbers do
  not have to be contiguous: the default `{1, 5, 10, 14}` reflects actual cabling.
* A label is `H<order><arm>`, arm being `T` or `R`. The **harmonic** is the first two
  alphanumeric characters, `H3T` → `H3`; `experiment_config.harmonic_of()` and the
  analyzer's `is_cross_harmonic()` apply the same rule.
* A pair is **auto** when both labels share the harmonic (`H3T`,`H3R`) and **cross**
  otherwise. Only cross pairs get an R value.
* Downstream of `get_data()`, pairs are tuples of labels — `("H3T", "H5R")` — not
  channel numbers. Figures are titled with those tuples.
* A channel listed in `REGISTERED_CHANNELS` but absent from `MODES` is still read for
  its countrate, but takes no part in the correlations, and the analyzer names it
  `Ch<number>`.

### 7.4 Folder-name case

`EXPERIENCE_TYPE` is used verbatim for both trees. The legacy folders disagree —
`data/…/David_Setup/` versus `results/…/David_setup/`. macOS is case-insensitive and
resolves both to the same directory; Linux will not, so rename them if you migrate.

---

## 8. Module reference

Five files. Each is long on purpose: the code that draws a figure, writes a config
file or talks to a mount lives next to the rest of that concern, not in a sixth file.

### 8.1 `experiment_config.py`

**Part 1 — `ExperimentConfig`.** The dataclass of §4, plus `REPO_ROOT`,
`angle_tag` and `harmonic_of`. Pure data and derivations.

**Part 2 — `experiment_config.txt`.** The INI writer that used to live in its own
module (`write_configuration`, `record_run`, `path_for`). Same semantics as §6.2:
shared sections rewritten every run, `[run <power>]` sections accumulated,
`[power_scan]` / `[polarization_scan]` exempt from the drift warning.

### 8.2 `hardware.py`

**Part 1 — one Thorlabs mount.** `RotationStageConfig` (§4.6) and `RotationStage`
(`connect` / `disconnect` / `get_angle_deg` / `move_to_angle_deg` / `home`, context
manager, dry-run). Details that are not obvious from the Kinesis documentation:

* `LoadMotorConfiguration` is what makes positions **degrees** instead of raw counts.
* Positions cross the .NET boundary as `System.Decimal` — hence `float(str(...))`.
* `MoveTo(position, 0)` returns immediately; the polling loop decides arrival (§4.6).
* Benchtop controllers expose one channel per axis: the object that moves is
  `device.GetChannel(channel)`, the one that disconnects is the controller.

**Part 2 — linear polarization.** `LookupTable`, `PolarizationController`,
`preflight`, `ScanLog`. Reads the P1/HWP lookup table, picks the smallest HWP angle
that delivers the requested power, moves P1 then the HWP. Same conventions as the
lab's calibration script (§3.6).

Nothing in this file imports the tagger or the analysis.

### 8.3 `acquisition.py`

**Part 1 — the live run.** `run_acquisition(cfg) → AcquisitionResult`, the chunk
loop, `_Cumulative`, `_SnapshotRecorder`, `_ProgressLogger`, the stall timeout
(`POLL_INTERVAL_S = 0.05`, `STALL_TIMEOUT_S = 30`). Frees the tagger in a `finally`.

**Part 2 — the on-disk schema.** `build_payload`, `save_payload`, `correlation_arrays`,
`countrate_values`. Duck-typed so a software-summed merged file and a live
`Correlation` go through the same path. No analysis, no physics.

Typical log of a live run:

```
Acquiring 600s as 10 chunk(s) of 60s | 6 correlation pairs
Connected to Time Tagger 1740000JG3 | channels [1, 5, 10, 14] -> ['H3T', 'H3R', 'H5T', 'H5R']
  [0.2 / 10.0 min] total countrate: 4.82e+04 cps
  chunk 1/10 -> 65mW_num1.pkl
  …
  merged -> …/data/…/merged/65mW_merged.pkl
  10 stability snapshots -> …/results/…/65mW/stability/stability_snapshots.pkl
Time Tagger released.
```

### 8.4 `pkl_json_analyze.py`

**Part 1 — the per-power tree.** `run_analysis(cfg)`, peak finding, g²/R maths,
spectrograms, window sweeps, power summaries (§9).

**Part 2 — stability versus time.** `plot_stability(snapshots, cfg)` writes
`stability_g2.png`, `stability_R.png`, `stability_rolling.png`. Peak centring uses
`measured_delay` (strongest bin within half a laser period of zero — same rule as
`get_peaks`), and out-of-range satellites are skipped so they cannot inflate g²(0)
(§10.7).

**Part 3 — polar summaries versus angle.** `plot_polarization_summary(cfg)` writes the
three figures of §3.6 plus `polarization_summary.csv` for one power;
`plot_polarization_campaign(cfg)` does that for every power of the folder and then
`plot_polarization_overlay` redraws the same three layouts with all powers on each panel
(§6.5). `discover_polarization_powers` is what finds the powers when they are not listed.

All of it reuses `get_data` / `get_peaks` / `compute_g2_integration`, so a point on a
polar figure is the number the per-angle tree would give for that angle, and the band
around it is the standard deviation over that angle's chunks.

### 8.5 `run_experiment.py`

The `CONFIG` block and the orchestration around it:

| Function | Role |
| --- | --- |
| `main(cfg)` | acquire (unless `ANALYZE_ONLY`), analyse, print the summary |
| `acquire(cfg)` | power scan, polarization scan, or single power |
| `acquire_one(cfg)` | one acquisition + optional stability plots |
| `acquire_polarization_scan(cfg)` | preflight, move, record, CSV log |
| `open_stage` / `open_polarizer` | dry-run aware hardware handles |
| `describe_configuration(cfg)` | the folder's config file, written only if absent |
| `analyze_polarization_scan(cfg)` | one `run_analysis` per angle of each power, then the summaries and the overlay |
| `print_summary(cfg, runs)` | one block per recorded point |

`matplotlib.use("Agg")` runs before anything imports `pyplot`. Every module imports
its siblings through a `try/except ImportError` pair so both
`python src/run_experiment.py` and `python -m src.run_experiment` work.

---

## 9. What the analyzer does

`run_analysis(cfg)` loops over `POWER_LEVELS`, and for each one:

**1. Find files** — `get_files(DATA_DIR, pattern="*{POWER}*")`:

* merged: `DATA_DIR/merged/*{POWER}*.json` **and** `*.pkl`;
* chunks: `DATA_DIR/*{POWER}*.pkl`, and only if that comes back empty,
  `DATA_DIR/*{POWER}*.json`.

**2. Load** — `get_data(files, MODES, POWER)` builds, per chunk key,
`correlations` (`{(labelA, labelB): {"delay_bins": ns, "coherences": counts}}`),
`countrates` and `total_counts`, both keyed by label.

**3. Find peaks** — `get_peaks(data, FREQUENCY)`:
looks for the peak nearest zero within ±T_rep/2 using
`scipy.signal.find_peaks` (prominence `max(2, 5 % of the maximum)`, falling back to
the plain maximum of that window), shifts the whole delay axis so this peak sits at
zero, then marks the *ideal* peak positions at every multiple of T_rep across the
axis — 11 peaks with the default geometry.

**4. Two g² estimators**

* `compute_g2_integration(data, w)` — the peak-ratio estimator of §1: sum of `w` bins
  around the central peak divided by the mean of the same sums on the satellites.
  Self-normalising, no external input.
* `compute_g2_counts(data, w)` — the pulsed-source estimator
  `g² = Np · n₁₂ / (n₁ · n₂)`, where `n₁`, `n₂` are the channel totals and `Np` the
  number of laser pulses during the measurement. **`Np` is hard-coded** to
  `60 s × 18.66 MHz = 1.12e9` (§11.1).

**5. R** — `compute_R_integration` / `compute_R_counts` compute
`g²_cross² / (g²_autoA · g²_autoB)` for cross pairs only, finding each
autocorrelation with `get_auto_pair`, which looks for the pair whose two labels share
the harmonic prefix.

**6. Plot, by what is available**

| Files present | Spectrograms | Per-chunk sweeps | Chunk-averaged | Contributes to power sweep |
| --- | --- | --- | --- | --- |
| merged **and** chunks | from merged | **not produced** | from chunks | chunks |
| chunks only | from chunks | from chunks | from chunks | chunks |
| merged only | from merged | from merged | from merged (σ = 0) | merged |

That first row is the normal outcome of a live run with `SAVE_MERGED=True`: the
merged file gives clean spectrograms, the chunks give the error bars, and the
per-chunk sweep grids are intentionally skipped.

Figures written:

```
{POWER}/{chunk}/{chunk}_spectrograms_grid.png       histogram + peaks + windows
{POWER}/{chunk}/{chunk}_g2_sweep_grid.png           g² vs integration window
{POWER}/{chunk}/{chunk}_R_sweep_grid.png            R vs integration window
{POWER}/{chunk}/{chunk}_summary_g2_overlap.png      all pairs overlaid
{POWER}/{chunk}/{chunk}_summary_R_overlap.png
{POWER}/power_summary/averaged_g2_sweep_grid.png    mean ± σ over chunks
{POWER}/power_summary/averaged_R_sweep_grid.png
{POWER}/power_summary/summary_all_g2_overlap.png
{POWER}/power_summary/summary_all_R_overlap.png
power_sweep_summary/g2_vs_power_grid.png            only with several powers
power_sweep_summary/R_vs_power_grid.png
power_sweep_summary/summary_all_g2_vs_power.png
power_sweep_summary/summary_all_R_vs_power.png
```

In the grids, autocorrelation pairs go in the first column and cross pairs fill the
rest; the coloured bands are reading aids (g² below 1, 1–2, above 2; R below and
above 1). Every figure is rendered at dpi 700, which is the main reason the analysis
is slow.

---

## 10. Troubleshooting

### 10.1 `ImportError: The TimeTagger Python API is not available`

Expected on any machine without the Swabian SDK (§3.2). Set `ANALYZE_ONLY=True` to
work on existing data. If the SDK *is* installed, you are probably running a
different interpreter than the one it registered — check `sys.executable`.

### 10.2 `[65mW] No files found. Skipping.`

`get_files` matched nothing. In order of likelihood:

1. `POWER_LEVEL` / `POWER_LEVELS` does not appear in the file names — the glob is
   `*{POWER}*`, and `"65mw"` will not match `65mW`;
2. `MATERIAL`, `EXPERIENCE_TYPE` or `DATE` point at the wrong folder — the summary
   prints the resolved `DATA_DIR`, compare it with reality;
3. the files are `.json` but a `.pkl` matching the same power also exists (§10.4).

### 10.3 Results land in `chunk_001` or in a folder named after the file

The analyzer keys chunks on the **literal digits** following `num`. With
`CHUNK_INDEX_DIGITS = 3` you get `chunk_001`; that works, it is just a different
folder name. If the pattern `{POWER}_num<N>` is missing entirely, the key becomes the
file stem and the file is treated as a single unnumbered chunk.

### 10.4 A `.json` export is ignored

`get_files` tries `*.pkl` first and only falls back to `*.json` **when no `.pkl`
matched**. One stray `.pkl` in the folder therefore hides every `.json` of the same
power. Keep one `EXPORT_FORMAT` per folder.

### 10.5 A stray file appears as an extra chunk

Everything matching `DATA_DIR/*{POWER}*.pkl` is loaded as a chunk. A merged file left
in the main folder instead of `merged/`, a copy, a backup, or a snapshot pickle whose
name contains the power will all show up. This is why snapshots are written under
`results/`. Symptoms: an unexpected sub-folder in `results/{POWER}/`, or a chunk whose
figures are empty.

### 10.6 Every g²(0) is ≈ 1, or the satellite peaks look asymmetric

The delay axis is not centred on the true zero-delay peak. Either the coincidence peak
sits further than half a laser period from zero — compensate the cable and detector
delays with `INPUT_DELAYS_PS` — or the central window is too noisy for `find_peaks`,
in which case the analyzer falls back to the maximum of that window. Look at
`{chunk}_spectrograms_grid.png`: the red crosses must sit on the peaks and the orange
bands must cover them.

### 10.7 `STABILITY_PEAKS` seems to be ignored, or g²(0) is inflated

Only satellite orders that fit inside the recorded delay axis are used. With the
default geometry the axis spans ±301.35 ns and `T_rep` is 53.59 ns, so **±5 orders**
are available no matter that `STABILITY_PEAKS` says 15:

```
usable orders ≈ (span/2 − window/2) / T_rep = (301.35 − 2) / 53.59 = 5.6 → ±5
```

To really average 15 satellites you need `CORR_N_BINS ≥ 2 × 15 × T_rep /
CORR_BINWIDTH_PS ≈ 5360` bins at 300 ps. Note that `get_peaks` uses a heuristic that
assumes the axis spans less than 10 µs; keep `CORR_N_BINS × CORR_BINWIDTH_PS` below
`1e7` ps (33 000 bins at 300 ps) or its repetition period will be computed in the
wrong unit.

### 10.8 `R` is `NaN`

R needs three g² values. It is `NaN` when the pair is not cross-harmonic (auto pairs
never get an R by design), when a harmonic has no autocorrelation pair — both arms
must be declared in `MODES` with matching labels such as `H5T`/`H5R` — or when either
autocorrelation is itself `NaN` or zero. Check the auto-pair panels in the g² grid
first: if `H5T-H5R` is `NaN`, every R involving H5 follows.

### 10.9 `stability_rolling.png` is empty

The rolling value at snapshot *k* needs snapshot *k − W*, so it stays `NaN` until
snapshot `W + 1`. With the shipped defaults, a 600 s run sampled every 60 s yields
exactly 10 snapshots while `STABILITY_ROLLING_WINDOW_SNAPS = 10` — nothing is
plottable. Either shorten the interval or reduce the window:

```python
STABILITY_SNAPSHOT_INTERVAL_S=20.0,   # 600 s / 20 s = 30 snapshots
STABILITY_ROLLING_WINDOW_SNAPS=10,    # 20 valid rolling points
```

Rule of thumb: `ACQUISITION_DURATION_S / interval ≥ 3 × window`.

### 10.10 `Stability: fewer than 2 snapshots, nothing to plot.`

The run was shorter than twice `STABILITY_SNAPSHOT_INTERVAL_S`. The acquisition prints
a matching note when it recorded none at all.

### 10.11 `TimeoutError: Tagger stalled`

Capture time stopped advancing for more than 30 s. The device dropped off the bus, was
claimed by another process (only one client at a time), or the machine suspended.
Chunks already exported are valid and the tagger is still freed properly.

### 10.12 The analysis takes forever

It is dominated by the window sweep: the g²/R maths is recomputed for every value in
`INTEGRATION_WINDOWS_SWEEP`, for every chunk and every pair, and all figures render at
dpi 700. While troubleshooting, use `np.arange(1, 6)` and a single power level; a full
sweep over many chunks is a coffee break.

### 10.13 Warnings about a different configuration

`_warn_on_drift` compares the shared values with what the folder already says (§6.2).
Either you changed the setup and should use a new `DATE`/`EXPERIENCE_TYPE` folder, or
you fixed a typo in the metadata, in which case the warning is harmless and appears
only once.

### 10.14 `RuntimeWarning: Mean of empty slice`

Emitted by the analyzer's `np.nanmean` when a pair has no R value at all — normal for
autocorrelation pairs, which are excluded from R by design. Harmless.

### 10.15 No figure window opens

By design: the backend is `Agg` and nothing calls `plt.show()`, so the pipeline runs
over SSH or from a scheduler. Open the PNGs under `results/`.

### 10.16 Rehearsing a run without hardware

Three of the four things a campaign does can be checked on a laptop, and each costs
one line:

| To check | Set | What still happens |
| --- | --- | --- |
| the configuration is valid | *(nothing)* | `python src/config_examples.py`, or just build your `ExperimentConfig`: §4.8 runs at construction |
| the scan plan and the angles | `ROTATION_STAGE_DRY_RUN=True`, `POLARIZATION_STAGE_DRY_RUN=True` | the loop, the angle for every point, the preflight against the lookup table, `experiment_config.txt` — every move is printed, no mount is touched, and no Kinesis import is attempted |
| the analysis and the figures | `ANALYZE_ONLY=True` | the whole plot tree, on data already in `DATA_DIR` |

Only the acquisition itself needs the tagger, so a polarization scan can be rehearsed
end to end but for the counts:

```python
POLARIZATION_SCAN=np.arange(0.0, 360.0, 15.0),
POLARIZATION_POWER=50.0,
POLARIZATION_STAGE_DRY_RUN=True,     # prints every P1 / HWP angle, moves nothing
```

That prints the preflight — which angles the table can serve at that power, and which
would be skipped — before you commit hours of beam time to a scan with holes in it.

### 10.17 Two powers came out as one curve

Power labels are matched with a `*{label}*` glob, so `"5mW"` also matches every
`45mW_num*.pkl` and the two powers end up averaged into one point. A scan or a
`POLARIZATION_POWERS` list containing labels nested like that is refused at config time
(§4.8); a `POWER_LEVELS` list is not, so pad the labels to the same width — `"05mW"`,
`"45mW"` — or rename them. Symptom before it is spotted: suspiciously smooth data and a
chunk count that is a multiple of what was recorded.

### 10.18 The polarization summary is missing angles

Only rows with `status = complete` in `polarization_scan.csv` (§6.4) are summarised. An
angle can be missing because the requested power was out of reach there
(`status = unreachable`, and the preflight said so before the scan started), because its
acquisition raised (`failed`), or because its files were moved. The CSV names the reason
per angle; the polar figures leave the gap rather than interpolating over it.

If *every* angle is missing, the powers were probably not discovered: check that the
file names still contain the power label, or list the powers explicitly with
`POLARIZATION_POWERS`.

### 10.19 `ImportError: Reaching the Thorlabs stage needs pythonnet`

`pip install pythonnet`, and check that Kinesis itself is installed (§3.5). To work
without any of it, `ROTATION_STAGE_DRY_RUN=True`.

### 10.20 `ImportError: Could not load the Kinesis assembly …`

pythonnet is there but the .NET assembly is not. Either Kinesis is installed somewhere
else — set `RotationStageConfig.kinesis_path` — or the controller family was guessed
wrong from the serial number, in which case the message lists the known families and
`device_type` picks one explicitly. A controller that is not listed at all needs an
entry in `DEVICE_CLASSES` (§8.2).

### 10.21 `TimeoutError: Rotation stage did not reach … deg`

The stage stopped short of `angle_tolerance_deg`. In order of likelihood: it was never
homed, so its coordinates are not what you think; the target is outside its travel
range; something is mechanically blocking it; or the tolerance is tighter than the
mount's own resolution — `0.05°` suits a K10CR1 or a PRM1Z8, a coarser mount needs
more. The printed "moving to … (from …)" line tells you where it started.

### 10.22 A scan recorded every power at the same angle

The stage never moved. Either `ROTATION_STAGE_DRY_RUN=True` was left on from a
rehearsal — `stage_motion` in `[power_scan]` says so — or the angles are all within
`angle_tolerance_deg` of each other. Since v1 the combination "`POWER_SCAN` set,
`ROTATION_STAGE_ENABLED=False`" is refused outright, which used to be the third way of
getting this.

### 10.23 A power label is missing from the sweep figures

Check its section in `experiment_config.txt`. `status = failed` means the point raised
and `POWER_SCAN_STOP_ON_ERROR=False` let the scan continue; the analyzer then finds no
files for it and prints `No files found. Skipping.` (§10.2). Re-run just that point
with a single-power `CONFIG`, then replay the whole folder with `ANALYZE_ONLY=True` and
the full `POWER_LEVELS`.

---

## 11. Known limitations

### 11.1 `Np` in the counts-based g² is hard-coded

`compute_g2_counts` uses `Np = 60 s × 18.66 MHz` for the number of laser pulses. It
ignores both `FREQUENCY` and the real measurement duration, so what it reports is

```
g2_counts_reported = g2_true × (60 s / duration) × (18.66 MHz / FREQUENCY)
```

With the defaults (60 s chunks at 18.66 MHz) both factors are 1 and the estimator is
exact — change either and only the integration-based g² stays trustworthy. The merged
file covers 600 s, so its `g2_counts` comes out ten times too small for that reason
alone; its integration-based g² is unaffected. Fixing it means setting
`data["Np"]` per chunk from the recorded duration, which the analyzer already reads
through `data.get("Np", …)`; the duration is stored in each file's `Parameters` for
exactly that purpose.

### 11.2 The analysis window is in bins

See §7.1. `INTEGRATION_WINDOW` and `INTEGRATION_WINDOWS_SWEEP` are bin counts, so
their physical width depends on `CORR_BINWIDTH_PS`. Comparing runs recorded with
different bin widths at the same "window" compares different time windows.

### 11.3 Chunk boundaries lose a little time

Measurements are recreated between chunks, which costs a few milliseconds each time.
Irrelevant for 60 s chunks, noticeable if you ever go below a second.

### 11.4 Merged countrates are run averages

The merged file reports `total counts / total duration`. A source that drifted during
the run is summarised by its mean rate; the per-chunk files and the stability figures
are where drift is visible.

### 11.5 One client per tagger

The Time Tagger accepts a single connection. A forgotten Python session, a notebook,
or the Swabian GUI will make `createTimeTagger` fail or stall.

### 11.6 Nothing verifies that a power label is true

The (power, angle) pairs of a scan are a calibration you did once, by hand. No power
meter is read, no Malus law is applied, and the stage reports an angle rather than a
power. If the beam path changes, the mount is re-homed, or the laser drifts, the labels
keep looking right while meaning something else. Recalibrate after touching the optics,
and treat the labels as names, not measurements.

### 11.7 A scan reconnects the tagger at every point

`run_acquisition` connects and frees the tagger per power level, which costs a second
or so per point and is why an already-open session breaks a scan halfway through
(§11.5) rather than at the start. Passing an open tagger through the loop would fix
both, at the price of a wider interface.

### 11.8 The polarization is set open-loop

The HWP angle comes from a table measured earlier; no power meter is read during the
scan, so nothing checks that the power actually held at every angle. That is deliberate
— the meter usually sits in the beam only during the calibration — but it means the
lookup table is the whole guarantee. Recalibrate after touching the optics, and use the
`intensity_vs_angle.png` panels as the sanity check: a channel whose total signal tracks
the HWP setting rather than the physics is the symptom of a stale table.

### 11.9 A polarization scan is analysed angle by angle

`run_analysis` runs once per angle, so a 24-angle scan renders 24 full analysis trees at
dpi 700 — long. Set `RUN_ANALYSIS_AFTER_ACQUIRE=False` to record first and replay later,
or call `pkl_json_analyze.plot_polarization_campaign(cfg)` directly to get only the
circle plots and skip the per-angle trees.
