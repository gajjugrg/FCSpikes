# Copilot instructions (FCSpikes)

## Project purpose
This repo contains Python analysis scripts for FC/Beam Plug current waveforms (spike finding, plateau summaries, Beam Plug waveform classification).

## Source of truth: paths & layout
- Use `src/utilities.py` for all path resolution.
- Data root is `utilities.fcspikes_root()`:
  - Defaults to `<repo>/data`
  - Can be overridden with `$FCSPIKES_DATA_DIR`
- Figures go under `<repo>/figures` (helpers: `utilities.figures_root()`, `utilities.daily_plot_path(...)`).
- Beam/TCO inputs are expected under:
  - `<data-root>/csv/beam/<Month>/BEAM_*.csv`
  - `<data-root>/csv/tco/<Month>/TCO_*.csv`

## Common gotchas
- macOS external drives may contain AppleDouble files like `._*.csv`. `find` will count them, but glob patterns like `*.csv` typically ignore dotfiles. Don’t treat the mismatch as missing data.
- When writing outputs (plots, timestamp text files, caches), always `mkdir(parents=True, exist_ok=True)` on the parent directory first.

## Coding conventions
- Keep scripts runnable as standalone modules: `if __name__ == "__main__": main()`.
- Prefer small, composable functions. Avoid wrappers/compat shims; update call sites instead.
- Use matplotlib’s non-interactive backend when `--show`/interactive display is not requested.
- Spike detection convention used in this repo:
  - Compute a smooth baseline (Savitzky–Golay)
  - Detect negative spikes on the residual `baseline - current` (e.g., via `scipy.signal.find_peaks`)

## BPcurrentStudy conventions
- Beam Plug waveform classification state lives under `BPcurrentStudy/data/current_discharge_study/` (e.g., `file_status.json`).
- Outputs for that study live under `BPcurrentStudy/figures/analysis/current_discharge_study/`.

## Caching
- Prefer safe, inspectable cache formats (e.g., `npz` + JSON metadata) over pickle-only caches.
- Cache invalidation should consider input file mtimes/sizes and relevant analysis parameters; provide a CLI `--refresh-cache`/equivalent escape hatch.
