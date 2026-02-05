"""HV ramping spike analysis for Top detector only, grouped by HV plateau.

Inputs:
- BEAM_HVRamping.csv, BEAM_HVRamping2.csv
- TCO_HVRamping.csv,  TCO_HVRamping2.csv
- HV_ramp_stable_voltage_summary.csv (plateau definitions)

This script:
- Focuses exclusively on the Top detector (TCO ch 01,03,05 + BEAM ch 03,04,06).
- Detects negative spikes via Savitzky-Golay baseline + `find_peaks` on residual.
- Groups spikes by HV plateau regions defined in the stable voltage summary CSV.
- Produces one plot per plateau showing spike waveforms overlaid.

Outputs:
- Plots per plateau: figures/analysis/hv_ramping_spike_study/top_by_plateau/<plateau_label>.png
- Summary plot: figures/analysis/hv_ramping_spike_study/top_plateau_summary.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from utilities import channel_cols, compute_baseline, fcspikes_root, figures_root


# Data quality exclusions (same as main script).
EXCLUDED_TIME_INTERVALS: list[tuple[pd.Timestamp, pd.Timestamp]] = [
    # Short transient windows to exclude from plots + all derived time-based metrics.
    (pd.Timestamp("2025-12-10 09:00"), pd.Timestamp("2025-12-10 17:30")),
    (pd.Timestamp("2025-12-12 10:39"), pd.Timestamp("2025-12-12 10:50")),
]


def _overlap_seconds(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    if right <= left:
        return 0.0
    return float((right - left).total_seconds())


def _excluded_seconds_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    exclusions: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    if end <= start:
        return 0.0
    total = 0.0
    for x0, x1 in exclusions:
        total += _overlap_seconds(start, end, pd.Timestamp(x0), pd.Timestamp(x1))
    return float(total)


def _live_seconds_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    exclusions: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    raw = float((end - start).total_seconds())
    if raw <= 0:
        return 0.0
    return max(0.0, raw - _excluded_seconds_between(start, end, exclusions=exclusions))


def _fit_interarrival_tau_s(dt_s: np.ndarray) -> tuple[float, float]:
    """Fit exponential inter-arrival tau from a histogram; fallback to mean(dt)."""

    dt = np.asarray(dt_s, dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 2:
        return float("nan"), float("nan")

    tau_mle = float(np.mean(dt))
    tau_mle_err = float(tau_mle / np.sqrt(dt.size))

    if dt.size >= 20:
        hi = float(np.quantile(dt, 0.95))
        dt_fit = dt[dt <= hi]
        if dt_fit.size >= 5:
            dt = dt_fit

    bins = int(max(8, min(40, round(np.sqrt(dt.size) * 2))))
    counts, edges = np.histogram(dt, bins=bins)
    used = counts > 0
    if int(used.sum()) < 3:
        return tau_mle, tau_mle_err

    idx = np.where(used)[0]
    c = counts.astype(float)[used]
    e0 = np.asarray(edges, dtype=float)[idx]
    e1 = np.asarray(edges, dtype=float)[idx + 1]

    def _bin_model(_x: np.ndarray, n0: float, tau: float) -> np.ndarray:
        return n0 * (np.exp(-e0 / tau) - np.exp(-e1 / tau))

    try:
        p0 = [float(np.sum(c)), max(1e-6, tau_mle)]
        bounds = ([0.0, 1e-6], [np.inf, np.inf])
        popt, pcov = curve_fit(
            _bin_model,
            xdata=np.zeros_like(c),
            ydata=c,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        tau = float(popt[1])
        if not np.isfinite(tau) or tau <= 0:
            return tau_mle, tau_mle_err
        tau_err = float(np.sqrt(pcov[1, 1])) if (pcov is not None and pcov.shape == (2, 2)) else tau_mle_err
        if not np.isfinite(tau_err) or tau_err <= 0:
            tau_err = tau_mle_err
        return tau, tau_err
    except Exception:
        return tau_mle, tau_mle_err

# Top detector channel definitions.
BEAM_TOP_CHANNELS = {3, 4, 6}
TCO_TOP_CHANNELS = {1, 3, 5}


# Baseline configuration (this script uses median+SavGol only).
MEDIAN_KERNEL_DEFAULT = 301


@dataclass(frozen=True)
class Spike:
    ts: datetime
    magnitude_uA: float
    current_uA: float
    baseline_uA: float
    charge_uC: float
    tau_s: float | None = None
    fit_r2: float | None = None  # exp fit R^2
    fit_A_uA: float | None = None
    fit_C_uA: float | None = None
    lin_r2: float | None = None
    lin_m: float | None = None
    lin_b: float | None = None
    fit_model: str | None = None  # 'exp', 'linear', or None


def _linear_model(t: np.ndarray, m: float, b: float) -> np.ndarray:
    return m * t + b


def _strip_outer_quotes_series(s: pd.Series) -> pd.Series:
    """Strip whitespace and surrounding quote characters from a string series."""
    s2 = s.astype(str).str.strip()
    s2 = s2.str.strip('"').str.strip("'")
    return s2


def _read_hvramping_csv(path: Path) -> pd.DataFrame:
    """Read the ramping CSV format with two header rows."""
    # Read only the timestamp + channel columns (much faster/less memory than pulling everything).
    header = pd.read_csv(path, header=1, nrows=0)
    cols = list(header.columns)
    if not cols:
        raise ValueError(f"No columns found in {path}")
    time_col = cols[0]
    channel_cols = [c for c in cols[1:] if (str(c).strip() != "" and not str(c).startswith("Unnamed") and "Channel" in str(c))]
    usecols = [time_col] + channel_cols

    # Fast path: let pandas parse types; only do quote-stripping if it appears necessary.
    df = pd.read_csv(path, header=1, usecols=usecols, low_memory=False)
    df = df.rename(columns={time_col: "timestamp"})

    ts = df["timestamp"]
    try:
        sample = ts.dropna().astype(str).head(1000)
        needs_strip = bool(
            (
                sample.str.startswith(("\"", "'"), na=False)
                | sample.str.endswith(("\"", "'"), na=False)
            ).any()
        )
    except Exception:
        needs_strip = True

    if needs_strip:
        ts = _strip_outer_quotes_series(ts.astype(str))
    df["timestamp"] = pd.to_datetime(ts, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _top_detector_current_a(beam_df: pd.DataFrame, tco_df: pd.DataFrame) -> pd.Series:
    """Combine BEAM + TCO top channels."""
    beam_cols = [c for c in channel_cols("FC_Beam", BEAM_TOP_CHANNELS) if c in beam_df.columns]
    tco_cols = [c for c in channel_cols("TCO", TCO_TOP_CHANNELS) if c in tco_df.columns]

    if not beam_cols:
        raise ValueError("No BEAM top channel columns found.")
    if not tco_cols:
        raise ValueError("No TCO top channel columns found.")

    beam_sum = beam_df[beam_cols].sum(axis=1, min_count=1)
    tco_sum = tco_df[tco_cols].sum(axis=1, min_count=1)

    common = beam_sum.index.intersection(tco_sum.index)
    if common.empty:
        raise ValueError("No overlapping timestamps between BEAM and TCO.")

    return beam_sum.loc[common] + tco_sum.loc[common]


def _detect_spikes(
    times: np.ndarray,
    currents_uA: np.ndarray,
    threshold_uA: float,
    window_length: int,
    polyorder: int,
    *,
    tau_fixed_s: float,
):
    """Detect negative spikes using median+SavGol baseline + find_peaks on residual."""
    baseline = compute_baseline(
        currents_uA,
        method="median_savgol",
        window_length=window_length,
        polyorder=polyorder,
        median_kernel=MEDIAN_KERNEL_DEFAULT,
        asls_lam=1e7,
        asls_p=0.01,
        asls_niter=10,
    )
    residual = baseline - currents_uA
    peaks, props = find_peaks(residual, prominence=float(threshold_uA), width=1)

    prominences = props.get("prominences", np.array([]))
    times_pd = pd.to_datetime(times)
    spikes: list[Spike] = []

    for j, idx in enumerate(peaks.tolist()):
        mag = float(prominences[j]) if j < len(prominences) else float(residual[idx])

        spikes.append(
            Spike(
                ts=times_pd[idx].to_pydatetime(),
                magnitude_uA=mag,
                current_uA=float(currents_uA[idx]),
                baseline_uA=float(baseline[idx]),
                charge_uC=float(mag * float(tau_fixed_s)),
            )
        )

    return spikes, baseline


def _exp_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    return A * np.exp(-t / tau) + C


def _fit_tau_for_spike(
    *,
    times_pd: pd.DatetimeIndex,
    currents_uA: np.ndarray,
    baseline_uA: np.ndarray,
    spike_ts: datetime,
    window_pre_s: float,
    window_post_s: float,
    fit_max_s: float,
    fit_min_uA: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Fit exponential + linear models on residual after the spike peak.

    Exponential: residual(t) = A * exp(-t/tau) + C
    Linear:      residual(t) = m * t + b

    Both fits use the same post-peak samples in t in [0, fit_max_s] with residual>0.
    Returns (tau_s, exp_r2, A, C, lin_r2, m, b). Any failed fit yields Nones.
    """

    if times_pd.size == 0:
        return None, None, None, None, None, None, None

    times_ns = times_pd.values.astype("datetime64[ns]").astype("int64")
    times_s = times_ns / 1e9

    spike_ts64 = np.datetime64(spike_ts)
    spike_idx = int(np.argmin(np.abs(times_pd.values - spike_ts64)))
    spike_t = float(times_s[spike_idx])

    window_pre_s = max(0.0, float(window_pre_s))
    window_post_s = max(0.0, float(window_post_s))
    fit_max_s = max(0.5, float(fit_max_s))
    fit_min_uA = max(0.0, float(fit_min_uA))

    window_mask = (times_s >= (spike_t - window_pre_s)) & (times_s <= (spike_t + window_post_s))
    if not np.any(window_mask):
        return None, None, None, None, None, None, None

    t_rel = times_s[window_mask] - spike_t
    residual = np.asarray(baseline_uA[window_mask] - currents_uA[window_mask], dtype=float)

    # Use only post-peak samples for the decay fit.
    fit_mask = (t_rel >= 0.0) & (t_rel <= fit_max_s) & np.isfinite(residual) & np.isfinite(t_rel)
    if not np.any(fit_mask):
        return None, None, None, None, None, None, None

    t_fit = np.asarray(t_rel[fit_mask], dtype=float)
    y_fit = np.asarray(residual[fit_mask], dtype=float)

    # Keep only positive residual values (we're fitting a decay back to baseline).
    pos = y_fit > 0.0
    t_fit = t_fit[pos]
    y_fit = y_fit[pos]
    if t_fit.size < 6:
        return None, None, None, None, None, None, None

    peak_amp = float(np.nanmax(y_fit))
    if not np.isfinite(peak_amp) or peak_amp < fit_min_uA:
        return None, None, None, None, None, None, None

    # Linear fit.
    lin_r2: float | None = None
    lin_m: float | None = None
    lin_b: float | None = None
    try:
        A_mat = np.vstack([t_fit, np.ones_like(t_fit)]).T
        sol, *_ = np.linalg.lstsq(A_mat, y_fit, rcond=None)
        lin_m = float(sol[0])
        lin_b = float(sol[1])
        y_lin = _linear_model(t_fit, lin_m, lin_b)
        ss_res_lin = float(np.sum((y_fit - y_lin) ** 2))
        ss_tot_lin = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
        if ss_tot_lin > 0 and np.isfinite(ss_res_lin) and np.isfinite(ss_tot_lin):
            lin_r2 = 1.0 - ss_res_lin / ss_tot_lin
    except Exception:
        lin_r2 = None
        lin_m = None
        lin_b = None

    # Initial guess: A ~ peak, tau ~ 5s, C ~ 0.
    p0 = (peak_amp, 5.0, 0.0)
    bounds = ([0.0, 1e-3, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt, _pcov = curve_fit(_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None, None, None, None, lin_r2, lin_m, lin_b

    A, tau, C = (float(popt[0]), float(popt[1]), float(popt[2]))
    if not np.isfinite(tau) or tau <= 0:
        return None, None, None, None, lin_r2, lin_m, lin_b

    y_pred = _exp_decay(t_fit, A, tau, C)
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    r2 = None
    if ss_tot > 0 and np.isfinite(ss_res) and np.isfinite(ss_tot):
        r2 = 1.0 - ss_res / ss_tot

    return tau, r2, A, C, lin_r2, lin_m, lin_b


def _attach_tau_fits(
    *,
    times: np.ndarray,
    currents_uA: np.ndarray,
    baseline_uA: np.ndarray,
    spikes: list[Spike],
    spike_window_pre_s: float,
    spike_window_post_s: float,
    fit_max_s: float,
    fit_min_uA: float,
    tau_fallback_s: float,
    require_exponential: bool,
    exp_min_r2: float,
    exp_r2_margin_vs_linear: float,
) -> tuple[list[Spike], list[float], list[Spike]]:
    times_pd = pd.to_datetime(times)
    currents_uA = np.asarray(currents_uA, dtype=float)
    baseline_uA = np.asarray(baseline_uA, dtype=float)

    selected: list[Spike] = []
    rejected: list[Spike] = []
    taus: list[float] = []
    for s in spikes:
        tau_s, exp_r2, A, C, lin_r2, lin_m, lin_b = _fit_tau_for_spike(
            times_pd=times_pd,
            currents_uA=currents_uA,
            baseline_uA=baseline_uA,
            spike_ts=s.ts,
            window_pre_s=spike_window_pre_s,
            window_post_s=spike_window_post_s,
            fit_max_s=fit_max_s,
            fit_min_uA=fit_min_uA,
        )

        exp_ok = (
            tau_s is not None
            and exp_r2 is not None
            and np.isfinite(exp_r2)
            and float(exp_r2) >= float(exp_min_r2)
            and (lin_r2 is None or (np.isfinite(lin_r2) and float(exp_r2) >= float(lin_r2) + float(exp_r2_margin_vs_linear)))
        )

        fit_model = "exp" if exp_ok else ("linear" if lin_r2 is not None else None)

        # NOTE: charge is computed at the plateau level using the plateau-mean tau.
        # Here we only attach fit diagnostics; charge is filled later.
        charge_uC = float("nan")

        fitted_spike = (
            Spike(
                ts=s.ts,
                magnitude_uA=s.magnitude_uA,
                current_uA=s.current_uA,
                baseline_uA=s.baseline_uA,
                charge_uC=charge_uC,
                tau_s=tau_s,
                fit_r2=exp_r2,
                fit_A_uA=A,
                fit_C_uA=C,
                lin_r2=lin_r2,
                lin_m=lin_m,
                lin_b=lin_b,
                fit_model=fit_model,
            )
        )

        if require_exponential and not exp_ok:
            rejected.append(fitted_spike)
            continue

        selected.append(fitted_spike)
        if tau_s is not None and np.isfinite(tau_s):
            taus.append(float(tau_s))

    return selected, taus, rejected


def _plateau_mean_tau_s(
    spikes: list[Spike],
    *,
    tau_fallback_s: float,
) -> float:
    """Compute plateau-mean tau from exponential-passing spikes.

    Uses only spikes with fit_model == 'exp' and finite tau_s.
    Falls back to tau_fallback_s if none are available.
    """

    tau_vals = [
        float(s.tau_s)
        for s in spikes
        if (s.fit_model == "exp" and s.tau_s is not None and np.isfinite(s.tau_s) and float(s.tau_s) > 0)
    ]
    if tau_vals:
        return float(np.mean(np.asarray(tau_vals, dtype=float)))
    return float(tau_fallback_s)


def _global_mean_tau_s_for_charge(
    tau_values_s: list[float],
    *,
    tau_fallback_s: float,
) -> float:
    tau_arr = np.asarray([t for t in tau_values_s if np.isfinite(t) and float(t) > 0], dtype=float)
    if tau_arr.size > 0:
        return float(np.mean(tau_arr))
    return float(tau_fallback_s)


def _apply_plateau_mean_tau_charge(
    spikes: list[Spike],
    *,
    plateau_mean_tau_s: float,
) -> list[Spike]:
    """Set per-spike charge as: charge_uC = prominence_uA * plateau_mean_tau_s.

    Only spikes with fit_model == 'exp' get a finite charge; others are set to NaN.
    """

    out: list[Spike] = []
    for s in spikes:
        if s.fit_model == "exp" and np.isfinite(plateau_mean_tau_s) and float(plateau_mean_tau_s) > 0:
            charge_uC = float(s.magnitude_uA * float(plateau_mean_tau_s))
        else:
            charge_uC = float("nan")

        out.append(
            Spike(
                ts=s.ts,
                magnitude_uA=s.magnitude_uA,
                current_uA=s.current_uA,
                baseline_uA=s.baseline_uA,
                charge_uC=charge_uC,
                tau_s=s.tau_s,
                fit_r2=s.fit_r2,
                fit_A_uA=s.fit_A_uA,
                fit_C_uA=s.fit_C_uA,
                lin_r2=s.lin_r2,
                lin_m=s.lin_m,
                lin_b=s.lin_b,
                fit_model=s.fit_model,
            )
        )
    return out


def _gap_threshold_seconds(times: np.ndarray) -> float:
    """Compute a robust gap threshold for segmenting time series."""
    if len(times) < 3:
        return 120.0
    t = pd.to_datetime(times)
    diffs = np.diff(t.values.astype("datetime64[ns]").astype("int64")) / 1e9
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 120.0
    med = float(np.median(diffs))
    return max(5.0 * med, 120.0)


def _detect_spikes_segmented(
    times: np.ndarray,
    currents_uA: np.ndarray,
    *,
    threshold_uA: float,
    window_length: int,
    polyorder: int,
    tau_fixed_s: float,
    gap_threshold_s: float | None = None,
) -> tuple[list[Spike], np.ndarray]:
    """Detect spikes with a baseline that resets across large time gaps."""
    if len(times) == 0:
        return [], np.asarray(currents_uA, dtype=float)

    times_pd = pd.to_datetime(times)
    times_ns = times_pd.values.astype("datetime64[ns]").astype("int64")
    if gap_threshold_s is None:
        gap_threshold_s = _gap_threshold_seconds(times)
    diffs_s = np.diff(times_ns) / 1e9
    gap_starts = np.where(diffs_s > float(gap_threshold_s))[0] + 1

    baseline_all = np.full_like(np.asarray(currents_uA, dtype=float), np.nan)
    spikes_all: list[Spike] = []

    start = 0
    for end in list(gap_starts) + [len(times)]:
        seg_times = times[start:end]
        seg_currents = np.asarray(currents_uA[start:end], dtype=float)
        if seg_currents.size >= 3:
            seg_spikes, seg_baseline = _detect_spikes(
                seg_times,
                seg_currents,
                threshold_uA=threshold_uA,
                window_length=window_length,
                polyorder=polyorder,
                tau_fixed_s=tau_fixed_s,
            )
            spikes_all.extend(seg_spikes)
            baseline_all[start:end] = seg_baseline
        else:
            baseline_all[start:end] = seg_currents
        start = end

    return spikes_all, baseline_all


def _apply_exclusions(df: pd.DataFrame, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    """Drop rows falling within exclusion intervals."""
    if df.empty:
        return df
    mask = np.ones(len(df), dtype=bool)
    for start, end in intervals:
        mask &= ~((df.index >= start) & (df.index < end))
    return df.loc[mask].copy()


def _read_stable_voltage_summary(path: Path) -> pd.DataFrame:
    """Read HV plateau definition CSV."""
    df = pd.read_csv(path, dtype=str)
    if "start" in df.columns:
        df["start"] = _strip_outer_quotes_series(df["start"])
    if "end" in df.columns:
        df["end"] = _strip_outer_quotes_series(df["end"])
    if "mean_E_V_per_cm" in df.columns:
        df["mean_E_V_per_cm"] = _strip_outer_quotes_series(df["mean_E_V_per_cm"])
    if "avg_voltage_kV" in df.columns:
        df["avg_voltage_kV"] = _strip_outer_quotes_series(df["avg_voltage_kV"])

    df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.tz_localize(None)
    df["end"] = pd.to_datetime(df["end"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["start", "end", "mean_E_V_per_cm"]).copy()
    df["start"] = df["start"].astype("datetime64[ns]")
    df["end"] = df["end"].astype("datetime64[ns]")
    df["mean_E_V_per_cm"] = pd.to_numeric(df["mean_E_V_per_cm"], errors="coerce")
    if "avg_voltage_kV" in df.columns:
        df["avg_voltage_kV"] = pd.to_numeric(df["avg_voltage_kV"], errors="coerce")
    df = df.dropna(subset=["mean_E_V_per_cm"]).copy()
    df = df.sort_values("start").reset_index(drop=True)
    return df


def _plot_plateau_spikes(
    plateau_idx: int,
    plateau_start: pd.Timestamp,
    plateau_end: pd.Timestamp,
    avg_voltage_kV: float,
    mean_E: float,
    times: np.ndarray,
    currents_uA: np.ndarray,
    baseline_uA: np.ndarray,
    spikes: list[Spike],
    rejected_spikes: list[Spike] | None,
    spike_window_pre_s: float,
    spike_window_post_s: float,
    fit_max_s: float,
    show_fit_overlay: bool,
    show_tau_annotation: bool,
    make_model_plots: bool,
    save_path: Path | None,
    model_save_path: Path | None,
    show_plot: bool,
):
    """Plot all spikes within a single plateau overlaid."""
    times_pd = pd.to_datetime(times)
    times_s = times_pd.values.astype("datetime64[ns]").astype("int64") / 1e9
    spike_window_pre_s = max(0.0, float(spike_window_pre_s))
    spike_window_post_s = max(0.0, float(spike_window_post_s))

    if times_pd.size == 0:
        print(f"  Plateau {plateau_idx}: No data in window.")
        return
    if not spikes:
        print(f"  Plateau {plateau_idx}: No spikes detected.")
        return

    print(f"  Plateau {plateau_idx}: {len(spikes)} spikes @ {avg_voltage_kV:.1f} kV ({mean_E:.0f} V/cm)")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    
    # Panel 1: Full plateau window with all spikes marked
    # Convert to relative time in hours from plateau start
    rel_times_h = (times_pd - plateau_start).total_seconds() / 3600.0
    
    ax1.plot(rel_times_h, currents_uA, 'k-', linewidth=0.8, label='Current')
    ax1.plot(rel_times_h, baseline_uA, 'g-', linewidth=1.2, label='Baseline')
    
    # Mark spike positions
    spike_rel_times = [(s.ts - plateau_start.to_pydatetime()).total_seconds() / 3600.0 for s in spikes]
    spike_currents = [s.current_uA for s in spikes]
    ax1.plot(spike_rel_times, spike_currents, 'ro', markersize=4, label='Spikes')
    
    ax1.set_xlabel('Time from plateau start (hours)')
    ax1.set_ylabel('Current (µA)')
    ax1.set_title(f'Plateau {plateau_idx}: {avg_voltage_kV:.1f} kV, {mean_E:.0f} V/cm\n'
                  f'{plateau_start.strftime("%Y-%m-%d %H:%M")} to {plateau_end.strftime("%Y-%m-%d %H:%M")} ({len(spikes)} spikes)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Panel 2: Overlaid spike waveforms (aligned at peak)
    # Extract [t0-pre, t0+post] windows around each spike
    colors = cm.viridis(np.linspace(0, 1, len(spikes)))

    for i, spike in enumerate(spikes):
        spike_ts64 = np.datetime64(spike.ts)
        spike_idx = int(np.argmin(np.abs(times_pd.values - spike_ts64)))
        spike_t = float(times_s[spike_idx])
        window_mask = (times_s >= (spike_t - spike_window_pre_s)) & (times_s <= (spike_t + spike_window_post_s))
        if not window_mask.any():
            continue

        t_rel = times_s[window_mask] - spike_t
        residual = baseline_uA[window_mask] - currents_uA[window_mask]
        ax2.plot(t_rel, residual, color=colors[i], alpha=0.5, linewidth=1.0)

        if (
            show_fit_overlay
            and spike.tau_s is not None
            and spike.fit_A_uA is not None
            and spike.fit_C_uA is not None
            and np.isfinite(spike.tau_s)
            and np.isfinite(spike.fit_A_uA)
            and np.isfinite(spike.fit_C_uA)
        ):
            t_fit_mask = (t_rel >= 0.0) & (t_rel <= float(fit_max_s))
            if np.any(t_fit_mask):
                t_fit = np.asarray(t_rel[t_fit_mask], dtype=float)
                y_fit = _exp_decay(t_fit, float(spike.fit_A_uA), float(spike.tau_s), float(spike.fit_C_uA))
                ax2.plot(t_fit, y_fit, color=colors[i], alpha=0.9, linewidth=1.2, linestyle="--")

    if show_tau_annotation:
        taus = [
            s.tau_s
            for s in spikes
            if (s.tau_s is not None and np.isfinite(s.tau_s) and float(s.tau_s) > 0)
        ]
        if taus:
            tau_mean = float(np.mean(np.asarray(taus, dtype=float)))
            ax2.text(
                0.02,
                0.98,
                f"Mean tau: {tau_mean:.2f} s (n={len(taus)})",
                transform=ax2.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
            )
    
    ax2.axvline(0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Peak time')
    ax2.set_xlabel('Time from spike peak (s)')
    ax2.set_ylabel('Residual: baseline - current (µA)')
    ax2.set_title(f'All {len(spikes)} spike waveforms overlaid (aligned at peak)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"    Saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)

    # Optional diagnostic plots: exp-only (log-y) and rejected/linear-only.
    if not make_model_plots:
        return

    # If the user asked for model plots but didn't request show or save,
    # nothing visible would happen; make that explicit.
    if not show_plot and model_save_path is None:
        print("    Note: --plot-model-selection is enabled, but neither --show nor --save was requested; skipping model plots.")
        return

    y_floor = 1e-4  # uA, for log-scale plotting

    def _plot_subset(
        subset: list[Spike],
        *,
        title: str,
        semilogy: bool,
        overlay: str,
        out_suffix: str,
    ) -> None:
        if not subset:
            return

        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors2 = cm.plasma(np.linspace(0, 1, len(subset)))

        for i2, spike in enumerate(subset):
            spike_ts64 = np.datetime64(spike.ts)
            spike_idx = int(np.argmin(np.abs(times_pd.values - spike_ts64)))
            spike_t = float(times_s[spike_idx])
            window_mask = (times_s >= (spike_t - spike_window_pre_s)) & (times_s <= (spike_t + spike_window_post_s))
            if not window_mask.any():
                continue

            t_rel = times_s[window_mask] - spike_t
            residual = np.asarray(baseline_uA[window_mask] - currents_uA[window_mask], dtype=float)
            residual_plot = np.maximum(residual, y_floor)

            if semilogy:
                ax.semilogy(t_rel, residual_plot, color=colors2[i2], alpha=0.5, linewidth=1.0)
            else:
                ax.plot(t_rel, residual, color=colors2[i2], alpha=0.5, linewidth=1.0)

            t_fit_mask = (t_rel >= 0.0) & (t_rel <= float(fit_max_s))
            if not np.any(t_fit_mask):
                continue
            t_fit = np.asarray(t_rel[t_fit_mask], dtype=float)

            if overlay == "exp":
                if (
                    spike.tau_s is None
                    or spike.fit_A_uA is None
                    or spike.fit_C_uA is None
                    or (not np.isfinite(spike.tau_s))
                    or (not np.isfinite(spike.fit_A_uA))
                    or (not np.isfinite(spike.fit_C_uA))
                    or float(spike.tau_s) <= 0
                ):
                    continue
                y_fit = _exp_decay(t_fit, float(spike.fit_A_uA), float(spike.tau_s), float(spike.fit_C_uA))
                if semilogy:
                    y_fit = np.maximum(np.asarray(y_fit, dtype=float), y_floor)
                    ax.semilogy(t_fit, y_fit, color=colors2[i2], alpha=0.9, linewidth=1.2, linestyle="--")
                else:
                    ax.plot(t_fit, y_fit, color=colors2[i2], alpha=0.9, linewidth=1.2, linestyle="--")

            if overlay == "linear":
                if spike.lin_m is None or spike.lin_b is None:
                    continue
                y_lin = _linear_model(t_fit, float(spike.lin_m), float(spike.lin_b))
                if semilogy:
                    y_lin = np.maximum(np.asarray(y_lin, dtype=float), y_floor)
                    ax.semilogy(t_fit, y_lin, color=colors2[i2], alpha=0.9, linewidth=1.2, linestyle="--")
                else:
                    ax.plot(t_fit, y_lin, color=colors2[i2], alpha=0.9, linewidth=1.2, linestyle="--")

        ax.axvline(0, color="r", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Time from spike peak (s)")
        ax.set_ylabel("Residual: baseline - current (µA)" + (" (log scale)" if semilogy else ""))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        base_path = model_save_path if model_save_path is not None else save_path
        if base_path is not None:
            p2 = base_path.with_name(base_path.stem + out_suffix + base_path.suffix)
            p2.parent.mkdir(parents=True, exist_ok=True)
            fig2.savefig(p2, dpi=150)
            print(f"    Saved to {p2}")

        if show_plot:
            plt.show()
        plt.close(fig2)

    exp_subset = [s for s in spikes if s.fit_model == "exp"]
    _plot_subset(
        exp_subset,
        title=f"Plateau {plateau_idx}: exponential-fit spikes only (log Y)",
        semilogy=True,
        overlay="exp",
        out_suffix="_exp_logy",
    )

    rej = rejected_spikes or []
    lin_subset = [
        s
        for s in (list(spikes) + list(rej))
        if (s.fit_model == "linear" or s.lin_r2 is not None)
    ]
    _plot_subset(
        lin_subset,
        title=f"Plateau {plateau_idx}: linear/rejected spikes only (linear overlay)",
        semilogy=False,
        overlay="linear",
        out_suffix="_linear_only",
    )


def _plot_summary(
    mean_Es: list[float],
    spike_rates_per_h: list[float],
    spike_rate_err_per_h: list[float],
    avg_charges_uC: list[float],
    avg_charge_err_uC: list[float],
    tau_values_s: list[float],
    save_path: Path | None,
    show_plot: bool,
):
    """Summary plot: spike rate/charge vs Mean E + relaxation-tau histogram."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    E = np.asarray(mean_Es, dtype=float)
    rates = np.asarray(spike_rates_per_h, dtype=float)
    rate_err = np.asarray(spike_rate_err_per_h, dtype=float)
    charges = np.asarray(avg_charges_uC, dtype=float)
    charge_err = np.asarray(avg_charge_err_uC, dtype=float)

    # Sort by E for cleaner lines.
    order = np.argsort(E)
    E = E[order]
    rates = rates[order]
    rate_err = rate_err[order]
    charges = charges[order]
    charge_err = charge_err[order]

    rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
    rate_err = np.nan_to_num(rate_err, nan=0.0, posinf=0.0, neginf=0.0)
    charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)
    charge_err = np.nan_to_num(charge_err, nan=0.0, posinf=0.0, neginf=0.0)

    # Panel 1: Spike rate vs Mean E
    ax1.errorbar(E, rates, yerr=rate_err, fmt='o', color='steelblue', ecolor='0.3', elinewidth=1.2, capsize=3)
    ax1.set_xlabel('Mean E (V/cm)')
    ax1.set_ylabel('Spike rate (spikes/hour)')
    ax1.set_title('Top detector: Spike rate vs Mean E')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Average charge vs Mean E
    ax2.errorbar(E, charges, yerr=charge_err, fmt='o', color='tab:purple', ecolor='0.3', elinewidth=1.2, capsize=3)
    ax2.set_xlabel('Mean E (V/cm)')
    ax2.set_ylabel('Average charge (µC)')
    ax2.set_title('Top detector: Average spike charge vs Mean E')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Histogram of fitted tau
    tau_arr = np.asarray(
        [t for t in tau_values_s if np.isfinite(t) and t > 0],
        dtype=float,
    )
    if tau_arr.size > 0:
        ax3.hist(tau_arr, bins=30, color='tab:green', alpha=0.7)
        ax3.set_xlabel('Fitted tau (s)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Exponential decay tau distribution (n={tau_arr.size})')
        ax3.grid(True, alpha=0.3, axis='y')
        tau_med = float(np.median(tau_arr))
        ax3.axvline(tau_med, color='k', linestyle='--', linewidth=1.5, label=f'Median {tau_med:.2f}s')
        ax3.legend(loc='best')
    else:
        ax3.text(0.5, 0.5, 'No tau fits available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_axis_off()
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved summary to {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Top detector spikes grouped by HV plateau regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--tag", default="hvramping", help="Tag for file naming (default: hvramping)")
    parser.add_argument("--beam-files", nargs="+", default=["BEAM_HVRamping.csv", "BEAM_HVRamping2.csv"],
                        help="BEAM ramping CSV filenames")
    parser.add_argument("--tco-files", nargs="+", default=["TCO_HVRamping.csv", "TCO_HVRamping2.csv"],
                        help="TCO ramping CSV filenames")
    parser.add_argument("--plateau-summary", default="HV_ramp_stable_voltage_summary.csv",
                        help="Stable voltage summary CSV filename")
    
    # Spike detection parameters (match hv_ramping_top_bottom_spike_summary.py defaults)
    parser.add_argument("--threshold-uA", type=float, default=0.03, help="Spike prominence threshold (µA)")
    parser.add_argument("--window-length", type=int, default=2500, help="Savitzky-Golay window length")
    parser.add_argument("--polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--tau-fixed-s", type=float, default=6.6, help="Fixed tau for charge estimation (s)")

    # Spike waveform overlay window (aligned at peak)
    parser.add_argument("--spike-window-pre-s", type=float, default=2.0, help="Seconds before peak for waveform plot")
    parser.add_argument("--spike-window-post-s", type=float, default=30.0, help="Seconds after peak for waveform plot")

    # Exponential fit for tau
    parser.add_argument(
        "--fit-tau",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit exp decay on each spike residual to estimate tau (default: enabled)",
    )
    parser.add_argument(
        "--fit-max-s",
        type=float,
        default=25.0,
        help="Max time after peak (s) used for exponential fit",
    )
    parser.add_argument(
        "--fit-min-uA",
        type=float,
        default=0.02,
        help="Minimum residual amplitude (uA) required to attempt a tau fit",
    )

    parser.add_argument(
        "--require-exponential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only spikes whose tail is better fit by an exponential than a linear model (default: enabled).",
    )
    parser.add_argument(
        "--exp-min-r2",
        type=float,
        default=0.85,
        help="Minimum exponential-fit R^2 required when --require-exponential is enabled.",
    )
    parser.add_argument(
        "--exp-r2-margin-vs-linear",
        type=float,
        default=0.05,
        help="Require exp R^2 >= linear R^2 + margin when --require-exponential is enabled.",
    )

    parser.add_argument(
        "--show-fit-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay each spike's fitted exp curve on the residual plot (default: enabled)",
    )
    parser.add_argument(
        "--show-tau-annotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate per-plateau plot with median fitted tau (default: enabled)",
    )

    parser.add_argument(
        "--plot-model-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also make per-plateau diagnostic plots: exp-only log-y, and rejected/linear-only overlay (default: disabled).",
    )

    parser.add_argument(
        "--save-summary-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-plateau summary data table to a .txt file (default: enabled).",
    )
    
    # Output control
    parser.add_argument("--save-plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument("--save", dest="save_plots", action="store_true", help="Alias for --save-plots")
    parser.add_argument("--show", dest="show_plots", action="store_true", help="Alias for --show-plots")
    parser.add_argument("--max-plateaus", type=int, default=None, help="Limit number of plateau plots (for testing)")
    
    args = parser.parse_args()

    # Use a non-interactive backend unless explicitly requested.
    if not args.show_plots:
        plt.switch_backend("Agg")
    
    # Locate input files
    data_root = fcspikes_root()
    ramping_dir = data_root / "csv" / "ramping"
    
    beam_files = [ramping_dir / f for f in args.beam_files]
    tco_files = [ramping_dir / f for f in args.tco_files]
    plateau_file = figures_root() / "analysis" / "ramping_hv" / args.plateau_summary
    
    # Verify files exist
    missing = [f for f in beam_files + tco_files + [plateau_file] if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files:\n" + "\n".join(str(f) for f in missing))
    
    print(f"Loading BEAM files: {[f.name for f in beam_files]}")
    print(f"Loading TCO files: {[f.name for f in tco_files]}")
    print(f"Loading plateau definitions: {plateau_file}")
    
    # Load and concatenate BEAM + TCO data
    beam_dfs = [_read_hvramping_csv(f) for f in beam_files]
    tco_dfs = [_read_hvramping_csv(f) for f in tco_files]
    
    beam_df = pd.concat(beam_dfs, axis=0).sort_index()
    tco_df = pd.concat(tco_dfs, axis=0).sort_index()
    
    # Apply exclusions
    beam_df = _apply_exclusions(beam_df, EXCLUDED_TIME_INTERVALS)
    tco_df = _apply_exclusions(tco_df, EXCLUDED_TIME_INTERVALS)
    
    # Combine top detector channels
    print("Combining top detector channels...")
    current_a = _top_detector_current_a(beam_df, tco_df)
    current_uA = np.asarray((current_a * 1e6).to_numpy(), dtype=float)
    times = current_a.index.to_numpy(dtype="datetime64[ns]")
    
    print(f"Top detector: {len(times)} data points from {times[0]} to {times[-1]}")
    
    # Load plateau definitions
    print(f"\nLoading plateau definitions from {plateau_file.name}")
    plateaus = _read_stable_voltage_summary(plateau_file)
    print(f"Found {len(plateaus)} HV plateaus")
    
    # Process each plateau
    output_dir = figures_root() / "analysis" / "hv_ramping_spike_study" / "top_by_plateau"
    
    plateau_indices = []
    avg_voltages_kV = []
    mean_Es = []
    spike_rates_per_h: list[float] = []
    spike_rate_err_per_h: list[float] = []
    avg_charges_uC: list[float] = []
    avg_charge_err_uC: list[float] = []
    all_tau_values_s: list[float] = []
    fit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    tau_for_charge_global_s: list[float] = []
    plateau_exp_magnitudes_uA: list[list[float]] = []
    
    max_to_process = args.max_plateaus if args.max_plateaus is not None else len(plateaus)
    
    print(f"\nProcessing {min(max_to_process, len(plateaus))} plateaus...")
    for i, row in enumerate(plateaus.itertuples(index=False)):
        if i >= max_to_process:
            break

        plateau_start = pd.Timestamp(getattr(row, "start"))
        plateau_end = pd.Timestamp(getattr(row, "end"))
        avg_voltage_kV = float(getattr(row, "avg_voltage_kV"))
        mean_E = float(getattr(row, "mean_E_V_per_cm"))

        seg = current_a[(current_a.index >= plateau_start) & (current_a.index < plateau_end)]
        seg_times = pd.DatetimeIndex(seg.index).to_pydatetime()
        seg_currents_uA = seg.to_numpy(dtype=float) * 1e6

        plateau_spikes, plateau_baseline_uA = _detect_spikes_segmented(
            np.asarray(seg_times),
            np.asarray(seg_currents_uA, dtype=float),
            threshold_uA=float(args.threshold_uA),
            window_length=int(args.window_length),
            polyorder=int(args.polyorder),
            tau_fixed_s=float(args.tau_fixed_s),
        )

        n_detected = int(len(plateau_spikes))

        rejected_spikes: list[Spike] = []
        if args.fit_tau and len(plateau_spikes) > 0:
            plateau_spikes, plateau_taus, rejected_spikes = _attach_tau_fits(
                times=np.asarray(seg_times),
                currents_uA=np.asarray(seg_currents_uA, dtype=float),
                baseline_uA=np.asarray(plateau_baseline_uA, dtype=float),
                spikes=plateau_spikes,
                spike_window_pre_s=float(args.spike_window_pre_s),
                spike_window_post_s=float(args.spike_window_post_s),
                fit_max_s=float(args.fit_max_s),
                fit_min_uA=float(args.fit_min_uA),
                tau_fallback_s=float(args.tau_fixed_s),
                require_exponential=bool(args.require_exponential),
                exp_min_r2=float(args.exp_min_r2),
                exp_r2_margin_vs_linear=float(args.exp_r2_margin_vs_linear),
            )
            all_tau_values_s.extend([t for t in plateau_taus if np.isfinite(t) and float(t) > 0])

            # Global charge tau uses only exp-passing spikes across ALL plateaus.
            tau_for_charge_global_s.extend(
                [
                    float(s.tau_s)
                    for s in plateau_spikes
                    if (s.fit_model == "exp" and s.tau_s is not None and np.isfinite(s.tau_s) and float(s.tau_s) > 0)
                ]
            )

        if bool(args.require_exponential):
            print(
                f"  Plateau {i}: kept {len(plateau_spikes)}/{n_detected} spikes after exp-vs-linear selection"
            )

        if len(plateau_spikes) > 0:
            for s in plateau_spikes:
                fit_rows.append(
                    {
                        "plateau_idx": i,
                        "ts": s.ts.isoformat(timespec="seconds"),
                        "avg_voltage_kV": avg_voltage_kV,
                        "mean_E_V_per_cm": mean_E,
                        "magnitude_uA": s.magnitude_uA,
                        "fit_model": s.fit_model,
                        "exp_r2": s.fit_r2,
                        "lin_r2": s.lin_r2,
                        "tau_s": s.tau_s,
                        "fit_A_uA": s.fit_A_uA,
                        "fit_C_uA": s.fit_C_uA,
                        "charge_uC": float("nan"),
                    }
                )
        
        plateau_indices.append(i)
        avg_voltages_kV.append(avg_voltage_kV)
        mean_Es.append(mean_E)

        plateau_exp_magnitudes_uA.append([s.magnitude_uA for s in plateau_spikes if s.fit_model == "exp" and np.isfinite(s.magnitude_uA)])

        n_spikes = int(len(plateau_spikes))
        duration_s_live = _live_seconds_between(plateau_start, plateau_end, exclusions=EXCLUDED_TIME_INTERVALS)
        duration_h_live = duration_s_live / 3600.0 if duration_s_live > 0 else float("nan")
        rate = float(n_spikes) / duration_h_live if (np.isfinite(duration_h_live) and duration_h_live > 0) else float("nan")
        spike_rates_per_h.append(rate)

        # Poisson counting error propagated to rate.
        if np.isfinite(duration_h_live) and duration_h_live > 0:
            spike_rate_err_per_h.append(float(np.sqrt(n_spikes)) / float(duration_h_live))
        else:
            spike_rate_err_per_h.append(float("nan"))

        # Fill charge later using the GLOBAL mean tau across all plateaus.
        avg_charges_uC.append(float("nan"))
        avg_charge_err_uC.append(float("nan"))

        # Record row for downstream analysis.
        summary_rows.append(
            {
                "plateau_idx": i,
                "plateau_start": plateau_start.isoformat(sep=" "),
                "plateau_end": plateau_end.isoformat(sep=" "),
                "avg_voltage_kV": float(avg_voltage_kV),
                "mean_E_V_per_cm": float(mean_E),
                "n_spikes": int(n_spikes),
                "live_time_s": float(duration_s_live),
                "live_time_h": float(duration_h_live) if np.isfinite(duration_h_live) else float("nan"),
                "spike_rate_per_h": float(rate) if np.isfinite(rate) else float("nan"),
                "spike_rate_err_per_h": float(spike_rate_err_per_h[-1]) if spike_rate_err_per_h else float("nan"),
                "global_mean_tau_s": float("nan"),
                "avg_charge_uC": float("nan"),
                "avg_charge_err_uC": float("nan"),
            }
        )
        
        if len(plateau_spikes) > 0:
            save_path = None
            model_save_path = None
            label = f"plateau_{i:02d}_{avg_voltage_kV:.1f}kV"
            if args.save_plots:
                save_path = output_dir / f"{label}.png"
            if bool(args.plot_model_selection):
                # Ensure model-selection plots can be saved even without --save (user expectation).
                model_save_path = output_dir / f"{label}.png"
            
            _plot_plateau_spikes(
                plateau_idx=i,
                plateau_start=plateau_start,
                plateau_end=plateau_end,
                avg_voltage_kV=avg_voltage_kV,
                mean_E=mean_E,
                times=np.asarray(seg_times),
                currents_uA=np.asarray(seg_currents_uA, dtype=float),
                baseline_uA=np.asarray(plateau_baseline_uA, dtype=float),
                spikes=plateau_spikes,
                rejected_spikes=rejected_spikes,
                spike_window_pre_s=args.spike_window_pre_s,
                spike_window_post_s=args.spike_window_post_s,
                fit_max_s=float(args.fit_max_s),
                show_fit_overlay=bool(args.show_fit_overlay),
                show_tau_annotation=bool(args.show_tau_annotation),
                make_model_plots=bool(args.plot_model_selection),
                save_path=save_path,
                model_save_path=model_save_path,
                show_plot=args.show_plots,
            )
    
    # Finalize charge using one global mean tau across all plateaus.
    global_mean_tau_s = _global_mean_tau_s_for_charge(
        tau_for_charge_global_s,
        tau_fallback_s=float(args.tau_fixed_s),
    )
    print(f"\nGlobal mean tau for charge: {global_mean_tau_s:.6g} s")

    for j in range(len(plateau_exp_magnitudes_uA)):
        mags = np.asarray(plateau_exp_magnitudes_uA[j], dtype=float)
        mags = mags[np.isfinite(mags)]
        charges = mags * float(global_mean_tau_s)
        if charges.size > 0:
            avg_charges_uC[j] = float(np.mean(charges))
            if charges.size >= 2:
                avg_charge_err_uC[j] = float(np.std(charges, ddof=1) / np.sqrt(charges.size))
            else:
                avg_charge_err_uC[j] = 0.0
        else:
            avg_charges_uC[j] = 0.0
            avg_charge_err_uC[j] = 0.0

        summary_rows[j]["global_mean_tau_s"] = float(global_mean_tau_s)
        summary_rows[j]["avg_charge_uC"] = float(avg_charges_uC[j])
        summary_rows[j]["avg_charge_err_uC"] = float(avg_charge_err_uC[j])

    for row in fit_rows:
        try:
            mag_obj = row.get("magnitude_uA")
            mag = float("nan")
            if isinstance(mag_obj, (int, float, str, np.integer, np.floating)):
                mag = float(mag_obj)
            if row.get("fit_model") == "exp" and np.isfinite(mag):
                row["charge_uC"] = float(mag) * float(global_mean_tau_s)
            else:
                row["charge_uC"] = float("nan")
        except Exception:
            row["charge_uC"] = float("nan")

    # Create summary plot
    print("\nCreating summary plot...")
    summary_path = None
    if args.save_plots:
        summary_path = figures_root() / "analysis" / "hv_ramping_spike_study" / f"top_plateau_summary_{args.tag}.png"
    
    _plot_summary(
        mean_Es=mean_Es,
        spike_rates_per_h=spike_rates_per_h,
        spike_rate_err_per_h=spike_rate_err_per_h,
        avg_charges_uC=avg_charges_uC,
        avg_charge_err_uC=avg_charge_err_uC,
        tau_values_s=all_tau_values_s,
        save_path=summary_path,
        show_plot=args.show_plots,
    )

    if bool(args.save_summary_data):
        summary_txt = figures_root() / "analysis" / "hv_ramping_spike_study" / f"top_plateau_summary_data_{args.tag}.txt"
        summary_txt.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_txt, index=False, sep="\t")
        print(f"Saved per-plateau summary table to {summary_txt}")

    if args.save_plots and fit_rows:
        fit_csv = figures_root() / "analysis" / "hv_ramping_spike_study" / f"top_spike_tau_fits_{args.tag}.csv"
        fit_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fit_rows).to_csv(fit_csv, index=False)
        print(f"Saved tau fit table to {fit_csv}")
    
    print("\nDone!")
    print(f"Plateaus analyzed: {len(plateau_indices)}")


if __name__ == "__main__":
    main()
