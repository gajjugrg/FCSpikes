# FCSpikes

## Directory layout

- `data/beam-YYYY-MM-DD.csv`
- `data/tco-YYYY-MM-DD.csv`
- `BPcurrentStudy/data/current_discharge_study/` (state for Beam Plug current discharge study)
- `BPcurrentStudy/figures/analysis/current_discharge_study/` (Beam Plug current discharge study plots)
- `figures/daily/<Month>/` and `figures/aggregates/...` (outputs from other FC spike scripts)

## Python environment

These scripts require `numpy`, `matplotlib`, and `scipy`.

Tip: most scripts use repo-relative paths (no hard-coded absolute paths). If you want to point
The code at a different input tree, set:

```bash
export FCSPIKES_DATA_DIR=/path/to/data
```

(`src/utilities.py` resolves inputs through that variable.)

Example runs:

```bash
# HV ramp plotting + plateau summary
python src/hv_ramp_plateau_summary.py

# HV ramp plateau summary (custom overrides)
# - Remove termination column from output CSV (now default)
# - Force a specific plateau end time by start timestamp
# - Optionally cap the last plateau end time
python src/hv_ramp_plateau_summary.py \
  --force-plateau-end "2025-12-02 12:04:14=2025-12-03 12:27" \
  --force-last-plateau-end "2025-12-16 15:00"

# HV ramping FC spike summary (Top/Bottom; plateau-binned only)
# Reads: figures/analysis/ramping_hv/HV_ramp_stable_voltage_summary.csv
# Writes: figures/analysis/hv_ramping_spike_study/
python src/hv_ramping_top_bottom_spike_summary.py --save-plots --show-plots

# HV ramping FC spike summary with exclusive spikes and exp-fit charge
# - Remove coincident Top/Bottom spikes
# - Use exponential-fit charge and save diagnostics
python src/hv_ramping_top_bottom_spike_summary.py \
  --exclusive-spikes \
  --coincidence-window-s 2 \
  --charge-method exp_fit \
  --exp-fit-diagnostics save \
  --exp-fit-diagnostics-max 10 \
  --save-plots

# HV ramping (Top detector only; plateau-binned, with tau fits)
# Writes: figures/analysis/hv_ramping_spike_study/
python src/hv_ramping_top_detector_by_plateau.py --save-plots

# Calibration fit/summary
python src/calibration_fit_show_control.py
```

## Beam Plug current/discharge study

Implementation:
- `BPcurrentStudy/src/currentDischargeStudy.py`

Code layout (refactored):
- `BPcurrentStudy/src/currentDischargeStudy.py`: thin CLI entrypoint/orchestrator
- `BPcurrentStudy/src/cds_data.py`: CSV parsing + timestamp extraction; cleans `±inf` and records saturation flags
- `BPcurrentStudy/src/cds_signal.py`: smoothing helpers + onset/timing helpers + auto-classification
- `BPcurrentStudy/src/cds_cache.py`: safe NPZ cache for charge + timing metrics
- `BPcurrentStudy/src/cds_plotting.py`: plotting (overlays, scatter, coincidences)

Entry point (recommended; keeps older commands working):
- `src/currentDischargeStudy.py`

Default outputs/state:
 Plots: `BPcurrentStudy/figures/analysis/current_discharge_study/`
 State: `BPcurrentStudy/data/current_discharge_study/`

Common usage:

```bash
# Scan and generate plots for the default folder/pattern
python BPcurrentStudy/src/currentDischargeStudy.py --n-files 25

# Use a local folder (example: gapped_files in this repo)
python BPcurrentStudy/src/currentDischargeStudy.py --folder BPcurrentStudy/data/current_discharge_study/gapped_files --pattern 'SaveOnEvent*.csv'
```

Notes:
- macOS may create `._*.csv` AppleDouble files on external drives; the scripts generally ignore these.
- CH3-vs-CH1 scatter uses a cache in `BPcurrentStudy/data/current_discharge_study/` to speed reruns.
- CH3/CH1 charge integration for the CH3-vs-CH1 scatter matches the per-event overlays: integrate the **signed baseline-subtracted** smoothed waveform in a fixed time window `TIME ∈ [0, tmax]` (negative-going CH3 pulses yield negative charge).
- CSV loading removes `±inf` samples (which can appear on saturated scope channels) and keeps `df.attrs["ch*_saturated"]` flags for QA.

### Auto-classification (waveform shape)

Classification labels are stored in:
- `BPcurrentStudy/data/current_discharge_study/file_status.json`

Run it like:

```bash
# classify only currently-unclassified files (recommended)
python BPcurrentStudy/src/currentDischargeStudy.py --auto-classify

# force re-classification of all files
python BPcurrentStudy/src/currentDischargeStudy.py --auto-classify --auto-all
```

How classification works (high level):
- Load one file, clean NaN/±inf, keep raw channels (`CH1_raw`, `CH3_raw`).
- Compute a **baseline** from the first `--auto-baseline-window` samples (default 500).
  - The window is interpreted as “N samples”; internally we convert to a time span using the file’s median `Δt`.
- Estimate baseline noise using a robust MAD-based sigma.
- Subtract baseline and look at the max positive and negative excursions.
- If neither excursion exceeds `--auto-noise-sigma × sigma` → `noise`
- If both exceed the threshold and are comparable (ratio ≥ `--auto-bipolar-ratio`) → `bipolar`
- Otherwise → `unipolar_positive` or `unipolar_negative`

If the classifications don’t match what you expect, the most important tuning knobs are:
- `--auto-noise-sigma` (lower = more things become “real”, higher = more “noise”)
- `--auto-baseline-window` (baseline duration; too short can mis-estimate noise, too long can include the event)
- `--auto-channel` (CH3 is usually best)

### Interactive classification (manual)

You can also label events yourself in an interactive matplotlib window.

This mode **autosaves** labels to:
- `BPcurrentStudy/data/current_discharge_study/file_status.json`

Run it like:

```bash
# classify only currently-unclassified files (default behavior)
python BPcurrentStudy/src/currentDischargeStudy.py --interactive-classify --n-files 200 --no-ch3-vs-ch1

# review/re-label *all* files (including already-classified)
python BPcurrentStudy/src/currentDischargeStudy.py --interactive-classify --interactive-all --n-files -1 --no-ch3-vs-ch1

# don’t save per-event annotated overlay plots while you label
python BPcurrentStudy/src/currentDischargeStudy.py --interactive-classify --interactive-all --interactive-no-save-plots --no-ch3-vs-ch1
```

Keybinds in the plot window:
- `1` → `bipolar`
- `2` → `unipolar_positive`
- `3` → `unipolar_negative`
- `0` → `noise`
- `s` / `space` → skip (no change)
- `backspace` / left arrow → previous file
- `q` / `esc` → quit

Notes:
- `--interactive-classify` forces `--show` internally (it needs a GUI backend).
- If you see “No files matched the interactive classification filter”, you likely already labeled everything; add `--interactive-all`.

## One-off helper scripts

These are small diagnostics/legacy studies.

- `src/spike_diagnostics_oct26_halfday.py`: prints spike stats for Oct 26 AM/PM.
- `src/october_spike_study_highlight_oct29_evening.py`: October study with Oct 29 evening highlighted.

### Notes on spike detection

Several scripts share the same convention:

- Baseline is estimated with Savitzky–Golay smoothing.
- Spikes are detected as peaks in `baseline - current` (so a downward dip becomes a positive peak).

Path resolution helpers live in `src/utilities.py`:
- `find_beam_csv(tag)` / `find_tco_csv(tag)`
- `fcspikes_root()` (respects `$FCSPIKES_DATA_DIR`)

### Loading and combining the currents

Function: `load_combined_data(beam_file, tco_file)`

- Loads each CSV with `np.loadtxt(..., delimiter=',', skiprows=2, dtype=str)`
- De-quotes string fields (`"..."` or `'...'`) to make float parsing robust.
- Parses timestamps with:
  - `datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f')`

For each timestamp `t`, the script computes:
- BEAM current at `t` = `col1 + col2 + col3`
- TCO current at `t`  = `col1 + col2 + col3`

Then it intersects timestamps and returns:
- `dates`: sorted timestamps common to BEAM and TCO
- `total_currents`: array of `(beam_current + tco_current)` at those times

### Spike detection logic

Function: `detect_spikes_with_savgol(dates, currents_uA, ...)`

1. Compute baseline using Savitzky–Golay filtering:
   - `baseline = savgol_filter(currents_uA, window_length, polyorder, mode='nearest')`
   - The helper `_adaptive_savgol` ensures the window length is valid for short arrays.

2. Define residual:

   `residual = baseline - currents_uA`

   Interpretation:
   - A downward excursion of the current (dip) makes the `baseline - current` positive.
   - Spikes are detected as positive peaks in this residual.

3. Peak finding:

   `peaks, props = find_peaks(residual, prominence=threshold_uA, width=1)`

   Key parameters:
   - `prominence=THRESHOLD_UA` sets how large the dip must be (in µA).
   - `width=1` is a minimal width constraint.

The function returns a list of spikes where each spike record includes:
- spike timestamp
- spike magnitude (`prominence`)
- current value at that point
- baseline value at that point

### Optional exclusion windows

Dictionary: `EXCLUDE_SPIKE_WINDOWS_BY_FILE`

- Keys are BEAM basenames like `"beam-2025-12-05.csv"`.
- Values are lists of `(start_datetime, end_datetime)`.

In `summarize_dataset(...)`, after spike detection:
- It looks up windows by `os.path.basename(beam_file)`.
- It removes spikes whose timestamps fall inside any configured window.

This is intentionally per-file and opt-in.

### Metrics reported per dataset

Inside `summarize_dataset(...)`:

- Convert current to µA:

  `currents_uA = currents * 1e6`

- Duration:

  `duration_hours = (dates[-1] - dates[0]).total_seconds() / 3600.0`

- Spike rate:

  `spikes_per_hour = N / duration_hours`

- Average spike magnitude (µA): mean of prominences

- Fixed shaping time constant:

  `TAU_FIXED = 6.6  # seconds`

- “Average spike charge” (in µC) is computed as:

  `avg_spike_charge_uC = avg_magnitude_uA * TAU_FIXED`

  (This is a model-based conversion using a fixed time constant.)

#### Uncertainties

- Charge uncertainty uses the **standard error of the mean** (if `N>1`):

  `std(mag, ddof=1) * TAU_FIXED / sqrt(N)`

- Rate uncertainty uses a **Poisson** approximation:

  `sqrt(N) / duration_hours`

## Charge + uncertainty in the Top-by-plateau tau-fit script

Script: `src/hv_ramping_top_detector_by_plateau.py`

This script computes charge per spike using the detected spike prominence (in µA) and an exponential relaxation time $\tau$ (in s).

### Per-spike charge

1) Detect spikes as peaks in the residual:

`residual = baseline - current`

Spike “magnitude” is the `find_peaks(..., prominence=...)` prominence in µA.

2) Fit an exponential decay to the post-peak residual:

$$r(t) = A e^{-t/\tau} + C$$

Fit is performed for $t \ge 0$ (after the peak) up to `--fit-max-s` seconds.

3) Compute charge using the fitted $\tau$:

$$Q_{\text{spike}}\;[\mu\text{C}] = \mathrm{prominence}\;[\mu\text{A}] \times \tau\;[\text{s}]$$

If the exponential fit fails, the script falls back to `--tau-fixed-s` for charge estimation.

### Tau per spike + mean tau per plateau

- Each spike may have a fitted $\tau$ from the exponential decay fit (stored per spike in the output CSV `top_spike_tau_fits_<tag>.csv` as `tau_s`).
- For the per-plateau plots, the script also computes a **mean fitted tau** per plateau using only valid fits (finite, $\tau>0$, and $\tau\le 100$ s) and annotates it on the plot.

Notes:
- Fit overlays, and the tau histogram ignore fitted values with $\tau > 100$ s (to avoid plotting obviously-bad fits).
- Spikes with prominence above `--max-prominence-uA` (default 1.5 µA) are rejected.

### Uncertainty/error bars in the summary plot

The summary plot reports metrics vs Mean E (V/cm):

- Spike rate (spikes/hour) error bar uses Poisson counting statistics:

  $$\sigma_{\text{rate}} \approx \frac{\sqrt{N}}{T}\;\;\text{where }T\text{ is plateau duration in hours}$$

- Average charge per plateau is the mean of per-spike charges; its error bar is the **SEM** across spikes in that plateau:

  $$\sigma_{\bar{Q}} = \frac{s_Q}{\sqrt{N}}$$

  where $s_Q$ is the sample standard deviation of per-spike charges within the plateau.

### Plot outputs

- Per-dataset plots (if enabled):
  - `<label>_baseline.png`
  - `<label>_threshold.png`

- Summary plots:
  - `voltage_scan.png`
  - `voltage_scan_overlay.png`

By default, outputs go to:
- `figures/analysis/voltage_scan/`

You can override per-dataset plot output with `PLOTS_DIR`:
- If `PLOTS_DIR` is a non-empty string, per-dataset plots go there.
- If `PLOTS_DIR` is empty, per-dataset plots go to `figures/analysis/voltage_scan/`.

### Overlay logic

`plot_vs_voltage_overlay(...)` plots:
- the current scan results (treated as “20 m cable”) and
- the provided `REFERENCE_CABLE_10M`

on the same axes for a direct comparison.

## Heinzinger ↔ Slow Control calibration

Script: `src/calibration_fit_show_control.py`

Goal: compute a calibration mapping from **Slow Control** readings (x) to the **Heinzinger readout** (y), treating Heinzinger as the reference.

The default model fit is:

$$\text{Heinzinger} = a \cdot \text{SlowControl} + b$$

where:
- `a` is the gain (calibration factor)
- `b` is an offset

The script can also optionally compute a “through-origin” fit (forces $b=0$) if you pass `--through-origin`.

### Input

Default input file:
- `data/calibration_data.txt`

This is a whitespace/tab-separated table with a header. Required columns:
- `slow_control_readout_kV`
- `Heinzinger_readout_kV`

Optional (kept for bookkeeping but not used in plots/fit):
- `set_voltage_kV`

### Outputs

Default output directory:
- `figures/analysis/calibration_show_control/`

Files written:
- `calibration_summary.json`
- `scatter_fit.png`
- `residual_vs_slowcontrol.png`

### Usage

Default (reads `data/calibration_data.txt` and writes outputs to `figures/analysis/calibration_show_control/`):

```bash
python src/calibration_fit_show_control.py
```

Also compute a forced-through-origin fit:

```bash
python src/calibration_fit_show_control.py --through-origin
```

## Shared utilities

All shared helper functions live in `src/utilities.py`:
- `project_root()` - returns the FCSpikes project root
- `fcspikes_root()` - returns the data directory (respects `FCSPIKES_DATA_DIR` env var)
- `beam_csv_glob()` / `tco_csv_glob()` - glob patterns for finding CSV files
- `find_beam_csv(tag)` / `find_tco_csv(tag)` - locate specific CSV files by day tag (YYYY-MM-DD or YYYYMMDD)
