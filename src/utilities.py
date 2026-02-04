from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

def project_root() -> Path:
    # FCSpikes/src/<thisfile>.py -> FCSpikes/
    return Path(__file__).resolve().parents[1]


def fcspikes_root() -> Path:
    """Return root directory for FC/BEAM plug datasets.

    Resolution order:
    1) $FCSPIKES_DATA_DIR (absolute or relative)
    2) <FCSpikes>/data
    """

    override = os.environ.get("FCSPIKES_DATA_DIR")
    if override:
        p = Path(override)
        return (p if p.is_absolute() else (project_root() / p)).resolve()
    return (project_root() / "data").resolve()


def beam_csv_glob() -> str:
    # New layout: <root>/beam-YYYY-MM-DD.csv
    return str(fcspikes_root() / "beam-*.csv")


def tco_csv_glob() -> str:
    return str(fcspikes_root() / "tco-*.csv")


def find_beam_csv(day_tag: str) -> Path:
    """Find a BEAM csv by tag, e.g. '2025-07-27' -> beam-2025-07-27.csv."""
    tag = day_tag.strip()
    iso_match = re.fullmatch(r"\d{4}-\d{2}-\d{2}", tag)
    compact_match = re.fullmatch(r"\d{8}", tag)
    if iso_match or compact_match:
        iso_tag = tag if iso_match else f"{tag[0:4]}-{tag[4:6]}-{tag[6:8]}"
        pattern = f"beam-{iso_tag}.csv"
        matches = list(fcspikes_root().glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Could not find {pattern} under {fcspikes_root()}")
        return sorted(matches)[0]

    pattern = f"BEAM_{day_tag}.csv"
    matches = list((fcspikes_root() / "csv" / "beam").glob(f"*/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {fcspikes_root()}/csv/beam")
    return sorted(matches)[0]


def find_tco_csv(day_tag: str) -> Path:
    tag = day_tag.strip()
    iso_match = re.fullmatch(r"\d{4}-\d{2}-\d{2}", tag)
    compact_match = re.fullmatch(r"\d{8}", tag)
    if iso_match or compact_match:
        iso_tag = tag if iso_match else f"{tag[0:4]}-{tag[4:6]}-{tag[6:8]}"
        pattern = f"tco-{iso_tag}.csv"
        matches = list(fcspikes_root().glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Could not find {pattern} under {fcspikes_root()}")
        return sorted(matches)[0]

    pattern = f"TCO_{day_tag}.csv"
    matches = list((fcspikes_root() / "csv" / "tco").glob(f"*/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {fcspikes_root()}/csv/tco")
    return sorted(matches)[0]


def spike_timestamps_path(filename: str) -> Path:
    return results_root() / "txt" / "spike_timestamps" / filename


def results_root() -> Path:
    return (project_root() / "results").resolve()


def figures_root() -> Path:
    """Backward-compatible alias for results_root()."""
    return results_root()


def _month_folder_from_label(label: str) -> str:
    # label examples: "Oct11_AM", "Oct29_Evening", "July26", "Jul26", "Dec01_PM"
    prefix = label.split("_", 1)[0]
    m = re.match(r"([A-Za-z]+)", prefix)
    if not m:
        return "special"
    month = m.group(1)
    if month.startswith("July"):
        return "July"
    if month == "Jul":
        return "July"
    return month[:3].title()


def daily_figures_dir(label: str) -> Path:
    return figures_root() / "daily" / _month_folder_from_label(label)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def daily_plot_path(label: str, filename: str) -> str:
    return str(_ensure_dir(daily_figures_dir(label)) / filename)

def ensure_exists(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(str(p) for p in missing))


def adaptive_savgol(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Adaptive Savitzky-Golay smoothing that tolerates short arrays."""

    y = np.asarray(y)
    n = y.size
    if n == 0:
        return y

    max_odd = n if (n % 2 == 1) else (n - 1)
    wl = min(int(window_length), max_odd)

    if wl <= polyorder:
        wl = polyorder + 1 if ((polyorder + 1) % 2 == 1) else (polyorder + 2)
        wl = min(wl, max_odd)
        if wl <= polyorder:
            return y.copy()

    try:
        return savgol_filter(y, wl, polyorder, mode="nearest")
    except Exception:
        return y.copy()


def fill_nan_1d(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs so downstream filters stay stable."""

    y = np.asarray(y, dtype=float)
    if y.size == 0 or np.all(np.isfinite(y)):
        return y
    x = np.arange(y.size, dtype=float)
    finite_mask = np.isfinite(y)
    if not np.any(finite_mask):
        return np.zeros_like(y)
    result = y.copy()
    result[~finite_mask] = np.interp(x[~finite_mask], x[finite_mask], y[finite_mask])
    return result


def baseline_median_savgol(
    y: np.ndarray,
    *,
    median_kernel: int,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """Median-filtered Savitzky-Golay baseline (robust to spikes)."""

    y = fill_nan_1d(y)
    n = y.size
    if n == 0:
        return y

    k = int(median_kernel)
    if k < 1:
        return adaptive_savgol(y, window_length=window_length, polyorder=polyorder)
    if k % 2 == 0:
        k += 1
    if k > n:
        k = n if (n % 2 == 1) else (n - 1)
    if k <= 1:
        y_med = y
    else:
        y_med = medfilt(y, kernel_size=k)

    return adaptive_savgol(y_med, window_length=window_length, polyorder=polyorder)


def baseline_asls(
    y: np.ndarray,
    *,
    lam: float,
    p: float,
    niter: int,
) -> np.ndarray:
    """Asymmetric least squares baseline tailored for downward spikes."""

    y = fill_nan_1d(y)
    n = int(y.size)
    if n < 3:
        return y.copy()

    lam = float(lam)
    p = float(p)
    niter = int(niter)
    if not np.isfinite(lam) or lam <= 0:
        return y.copy()
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return y.copy()
    if niter <= 0:
        return y.copy()

    y_flip = -y

    data = np.vstack(
        [
            np.ones(n, dtype=float),
            -2.0 * np.ones(n, dtype=float),
            np.ones(n, dtype=float),
        ]
    )
    D = sparse.spdiags(data, np.array([0, 1, 2], dtype=int), n - 2, n).tocsc()
    DTD = D.T @ D

    w = np.ones(n, dtype=float)
    z = np.zeros(n, dtype=float)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = spsolve(W + lam * DTD, w * y_flip)
        w = np.where(y_flip > z, p, 1.0 - p)

    return -np.asarray(z, dtype=float)


def compute_baseline(
    y: np.ndarray,
    *,
    method: str,
    window_length: int,
    polyorder: int,
    median_kernel: int,
    asls_lam: float,
    asls_p: float,
    asls_niter: int,
) -> np.ndarray:
    """Unified baseline selector for different smoothing strategies."""

    method = str(method).lower()
    if method == "median_savgol":
        return baseline_median_savgol(
            y,
            median_kernel=median_kernel,
            window_length=window_length,
            polyorder=polyorder,
        )
    if method == "asls":
        return baseline_asls(y, lam=asls_lam, p=asls_p, niter=asls_niter)
    return adaptive_savgol(fill_nan_1d(y), window_length=window_length, polyorder=polyorder)


def channel_cols(prefix: str, channels: Iterable[int]) -> list[str]:
    """Build the channel column names for a given prefix and channel set."""

    return [f"{prefix}_Channel{ch:02d}" for ch in channels]


def plot_spike_distribution(
    spikes: list[tuple[datetime, float, float, float]],
    input_label: str,
    threshold_uA: float,
    bins: int = 30,
    value: str = "magnitude",
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Histogram of spike magnitudes or raw current values."""

    if not spikes:
        return
    if value == "magnitude":
        spike_values = [mag for _, mag, _, _ in spikes]
        x_label = "Magnitude (baseline - spike) [µA]"
        title = "Distribution of Spike Magnitudes"
    elif value == "spike":
        spike_values = [val for _, _, val, _ in spikes]
        x_label = "Spike Current (µA)"
        title = "Distribution of Raw Spike Values"
    else:
        raise ValueError("value must be 'magnitude' or 'spike'")

    plt.figure(figsize=(8, 5))
    plt.hist(spike_values, bins=bins, histtype="step", facecolor="lightblue", edgecolor="black")
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    stats_text = (
        f"{input_label}\nThreshold: {threshold_uA:.3f} µA\nSpikes: {len(spikes)}"
    )
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9, va="top", ha="right")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_spikes_with_baseline(
    dates: list[datetime] | np.ndarray,
    currents_uA: np.ndarray,
    spikes_info: list[tuple[datetime, float, float, float]],
    baseline_uA: np.ndarray,
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Show currents, baseline, and spike magnitudes on the same timeline."""

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, currents_uA, label="Current", alpha=0.7)
    ax.plot(dates, baseline_uA, label="Baseline (rolling)", linestyle="--", color="black")

    spike_plotted = False
    magnitude_plotted = False
    for ts, mag, spike_val, baseline_val in spikes_info:
        x_vals = np.array([ts], dtype="datetime64[ns]")
        y_spike = np.array([spike_val], dtype=float)
        y_baseline = np.array([baseline_val], dtype=float)
        if not spike_plotted:
            ax.scatter(x_vals, y_spike, color="red", zorder=5, label="Spike")
            spike_plotted = True
        else:
            ax.scatter(x_vals, y_spike, color="red", zorder=5)
        if not magnitude_plotted:
            ax.vlines(x_vals, y_spike, y_baseline, color="orange", linestyle=":", label="Magnitude")
            magnitude_plotted = True
        else:
            ax.vlines(x_vals, y_spike, y_baseline, color="orange", linestyle=":")

    ax.set_xlabel("Time")
    ax.set_ylabel("Current (µA)")
    ax.set_title("Detected Spikes with Rolling Baseline")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved spike/baseline plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def exponential_pdf(t: np.ndarray, tau: float) -> np.ndarray:
    return (1.0 / tau) * np.exp(-t / tau)


def plot_spike_intervals_and_fit(
    spikes: list[tuple[datetime, float, float, float]],
    bins: int = 30,
    time_unit: str = "s",
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Histogram inter-spike intervals and overlay an exponential fit."""

    if len(spikes) < 2:
        return
    times = [entry[0] for entry in spikes]
    delta_ts = np.diff([t.timestamp() for t in times])
    factor = {'s': 1, 'min': 60, 'h': 3600}[time_unit]
    delta_ts /= factor
    hist_vals, bin_edges = np.histogram(delta_ts, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_dt = np.mean(delta_ts)
    popt, pcov = curve_fit(exponential_pdf, bin_centers, hist_vals, p0=[mean_dt])
    tau_fit = popt[0]
    tau_err = np.sqrt(np.diag(pcov))[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_ts, bins=bins, density=True, alpha=0.5, histtype='step', label="Data", color="gray", edgecolor='black')
    t_fit = np.linspace(min(delta_ts), max(delta_ts), 300)
    ax.plot(t_fit, exponential_pdf(t_fit, *popt), 'r--', label=f"Fit: τ = {tau_fit:.3f} ± {tau_err:.3f} ({time_unit})")
    ax.set_xlabel(f"Interval between spikes ({time_unit})")
    ax.set_ylabel("Probability Density")
    ax.set_title("Spike Interval Distribution with Exponential Fit")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_overall_intervals(
    times: list[datetime],
    bins: int = 30,
    time_unit: str = 'min',
    title: str | None = None,
    save_path: Path | None = None,
    show_plot: bool = True,
    x_min: float = 0,
    x_max: float = 500,
) -> tuple[float, float]:
    """Plot histogram of intervals and fit a model to the aggregated data."""

    if len(times) < 2:
        print("Not enough timestamps to compute overall intervals.")
        return float('nan'), float('nan')
    times_sorted = sorted(times)
    deltas = np.diff([t.timestamp() for t in times_sorted])
    factor = {'s': 1, 'min': 60, 'h': 3600}[time_unit]
    deltas /= factor
    in_range = (deltas >= x_min) & (deltas <= x_max)
    deltas_plot = deltas[in_range] if np.any(in_range) else deltas
    hist_vals, bin_edges = np.histogram(deltas_plot, bins=bins, range=(x_min, x_max), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_dt = np.mean(deltas_plot) if deltas_plot.size else np.mean(deltas)
    try:
        popt, pcov = curve_fit(exponential_pdf, bin_centers, hist_vals, p0=[mean_dt])
        tau_fit = popt[0]
        tau_err = np.sqrt(np.diag(pcov))[0]
    except Exception as exc:
        print(f"Exponential fit failed: {exc}")
        tau_fit, tau_err = float('nan'), float('nan')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas_plot, bins=bins, range=(x_min, x_max), density=True, alpha=0.5, histtype='step', color='gray', edgecolor='black', label='Data')
    if np.isfinite(tau_fit):
        t_fit = np.linspace(x_min, x_max, 300)
        ax.plot(t_fit, exponential_pdf(t_fit, tau_fit), 'r--', label=f"Fit: τ = {tau_fit:.3f} ± {tau_err:.3f} {time_unit}")
    ax.set_xlabel(f"Interval between spikes ({time_unit})")
    ax.set_ylabel("Probability Density")
    ax.set_title(title or "Overall Inter-spike Interval Distribution")
    ax.set_xlim(x_min, x_max)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved overall interval plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"Overall mean interval: {mean_dt:.3f} {time_unit}")
    if np.isfinite(tau_fit):
        print(f"Overall fitted tau: {tau_fit:.4f} ± {tau_err:.4f} {time_unit}")
    return tau_fit, tau_err


def compute_total_discharge_charge(spikes: list[tuple[datetime, float, float, float]], tau_fixed: float = 6.6) -> float:
    """Return the aggregate charge under the assumption of a fixed tau."""

    total_charge = sum(mag * tau_fixed for _, mag, _, _ in spikes)
    if spikes:
        magnitudes = np.array([mag for _, mag, _, _ in spikes])
        rms_magnitude = np.sqrt(np.mean(magnitudes**2))
        error_estimate = rms_magnitude * tau_fixed / np.sqrt(len(spikes))
        times = [ts for ts, *_ in spikes]
        if len(times) >= 2:
            duration_sec = (max(times) - min(times)).total_seconds()
            duration_hr = duration_sec / 3600.0 if duration_sec > 0 else float('nan')
            n_spikes = len(spikes)
            freq_per_hr = n_spikes / duration_hr if duration_hr > 0 else float('nan')
            freq_per_hr_err = np.sqrt(n_spikes) / duration_hr if duration_hr > 0 else float('nan')
    return total_charge
