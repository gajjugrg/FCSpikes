"""HV ramping spike summary for Top/Bottom detectors.

Inputs:
- BEAM_HVRamping.csv, BEAM_HVRamping2.csv
- TCO_HVRamping.csv,  TCO_HVRamping2.csv

This script:
- Loads BEAM + TCO ramping CSVs (these files contain two header rows; the 2nd is used).
- Groups channels into Top/Bottom detectors using the mapping:
  - Top:    TCO channels 01, 03, 05  +  BEAM channels 03, 04, 06
  - Bottom: remaining channels (01–06) in each file.
- Detects negative spikes via a Savitzky–Golay baseline and `find_peaks` on residual = baseline - current.
- Produces summary plots (avg charge/spike and spike rate) for Top and Bottom.

Outputs:
- Summary plots: figures/analysis/hv_ramping_spike_study/summary_<tag>_top.png and ..._bottom.png
- Optional spike timestamps+charge: data/txt/spike_timestamps/spike_timestamps_<tag>_{top|bottom}.txt
- Optional overlay plot: figures/analysis/hv_ramping_spike_study/cathode_voltage_top_bottom_current_<tag>.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any, cast
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from scipy.signal import find_peaks

from utilities import channel_cols, compute_baseline, fcspikes_root, figures_root, spike_timestamps_path


# Hard-coded data quality exclusions / baseline reset points (per logbook notes).
# - 2025-12-10 09:00–17:30: BEAM/TCO values not recorded; a constant last value was repeated.
# - 2025-12-12 10:39: FC termination current setpoint change to 1.2 kV; drop data until 10:50 and
#   reset baseline by segmenting spike detection across this gap.
EXCLUDED_TIME_INTERVALS: list[tuple[pd.Timestamp, pd.Timestamp]] = [
    # Short transient windows to exclude from plots + all derived time-based metrics.
    (pd.Timestamp("2025-12-05 11:28"), pd.Timestamp("2025-12-05 11:30")),
    (pd.Timestamp("2025-12-08 10:39"), pd.Timestamp("2025-12-08 10:41")),
    (pd.Timestamp("2025-12-10 00:05"), pd.Timestamp("2025-12-10 00:06")),
    (pd.Timestamp("2025-12-11 22:05"), pd.Timestamp("2025-12-11 22:10")),
    (pd.Timestamp("2025-12-12 07:43"), pd.Timestamp("2025-12-12 07:45")),
    (pd.Timestamp("2025-12-15 06:43"), pd.Timestamp("2025-12-15 06:45")),
    (pd.Timestamp("2025-12-15 11:00"), pd.Timestamp("2025-12-15 11:02")),
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
    """Fit exponential inter-arrival time constant tau from a histogram.

    Uses a bin-integrated exponential model on histogram counts.
    Falls back to tau = mean(dt) (exponential MLE) if the fit is ill-conditioned.
    Returns (tau_s, tau_err_s).
    """

    dt = np.asarray(dt_s, dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 2:
        return float("nan"), float("nan")

    tau_mle = float(np.mean(dt))
    tau_mle_err = float(tau_mle / np.sqrt(dt.size))

    # Robustify: ignore extreme tail for histogram fit.
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
        from scipy.optimize import curve_fit

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


BEAM_TOP_CHANNELS = {3, 4, 6}
TCO_TOP_CHANNELS = {1, 3, 5}
ALL_CHANNELS = {1, 2, 3, 4, 5, 6}


@dataclass(frozen=True)
class DetectorGrouping:
    beam_channels: set[int]
    tco_channels: set[int]


TOP = DetectorGrouping(beam_channels=BEAM_TOP_CHANNELS, tco_channels=TCO_TOP_CHANNELS)
BOTTOM = DetectorGrouping(beam_channels=ALL_CHANNELS - BEAM_TOP_CHANNELS, tco_channels=ALL_CHANNELS - TCO_TOP_CHANNELS)


@dataclass(frozen=True)
class Spike:
    ts: datetime
    magnitude_uA: float
    current_uA: float
    baseline_uA: float
    charge_uC: float
    tau_s: float | None
    fit_A_uA: float | None
    fit_used_points: int | None


def _to_ns(dt: datetime) -> int:
    return int(np.datetime64(dt, "ns").astype("int64"))


def _exclusive_spikes_by_time(
    spikes_a: list[Spike],
    spikes_b: list[Spike],
    *,
    coincidence_window_s: float,
) -> tuple[list[Spike], list[Spike], list[Spike]]:
    """Split spikes into (a_only, b_only, coincident).

    A spike is considered coincident if there exists any spike in the other list
    within +/- coincidence_window_s of its timestamp.

    Returns:
      a_only: spikes from A with no coincidence in B
      b_only: spikes from B with no coincidence in A
      coincident: spikes from A that were coincident (for debugging/diagnostics)
    """

    tol_ns = int(float(coincidence_window_s) * 1e9)
    if tol_ns <= 0:
        # With zero tolerance, only exact timestamp matches count as coincident.
        tol_ns = 0

    a_sorted = sorted(spikes_a, key=lambda s: s.ts)
    b_sorted = sorted(spikes_b, key=lambda s: s.ts)
    a_ns = np.asarray([_to_ns(s.ts) for s in a_sorted], dtype=np.int64)
    b_ns = np.asarray([_to_ns(s.ts) for s in b_sorted], dtype=np.int64)

    a_is_coincident = np.zeros(len(a_sorted), dtype=bool)
    b_is_coincident = np.zeros(len(b_sorted), dtype=bool)

    if len(b_sorted) > 0:
        for i, t in enumerate(a_ns.tolist()):
            j = int(np.searchsorted(b_ns, t))
            candidates = []
            if 0 <= j < len(b_ns):
                candidates.append(j)
            if 0 <= (j - 1) < len(b_ns):
                candidates.append(j - 1)
            for k in candidates:
                if abs(int(b_ns[k]) - int(t)) <= tol_ns:
                    a_is_coincident[i] = True
                    b_is_coincident[k] = True
                    break

    if len(a_sorted) > 0:
        for j, t in enumerate(b_ns.tolist()):
            i = int(np.searchsorted(a_ns, t))
            candidates = []
            if 0 <= i < len(a_ns):
                candidates.append(i)
            if 0 <= (i - 1) < len(a_ns):
                candidates.append(i - 1)
            for k in candidates:
                if abs(int(a_ns[k]) - int(t)) <= tol_ns:
                    b_is_coincident[j] = True
                    a_is_coincident[k] = True
                    break

    a_only = [s for s, m in zip(a_sorted, a_is_coincident.tolist(), strict=False) if not m]
    b_only = [s for s, m in zip(b_sorted, b_is_coincident.tolist(), strict=False) if not m]
    coincident = [s for s, m in zip(a_sorted, a_is_coincident.tolist(), strict=False) if m]
    return a_only, b_only, coincident


def _plot_exp_fit_diagnostic(
    *,
    detector_name: str,
    segment_label: str,
    spike_time: datetime,
    spike_window_x_s: np.ndarray,
    spike_window_current_uA: np.ndarray,
    spike_window_baseline_uA: np.ndarray,
    x_s: np.ndarray,
    y_uA: np.ndarray,
    used_mask: np.ndarray,
    A_uA: float,
    tau_s: float,
    charge_uC: float,
    save_path: Path | None,
    show_plot: bool,
) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7.0), sharex=False)

    # Panel 1: actual spike shape in current (with baseline overlay)
    ax0.plot(spike_window_x_s, spike_window_current_uA, color="black", linewidth=1.2, label="Current")
    ax0.plot(spike_window_x_s, spike_window_baseline_uA, color="tab:green", linewidth=1.6, label="Baseline")
    ax0.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2, label="Peak t0")
    ax0.set_ylabel("µA")
    ax0.set_title(
        f"Exp fit diagnostic ({detector_name}; {segment_label})\n{spike_time.strftime('%Y-%m-%d %H:%M:%S')}  Q={charge_uC:.4g} µC"
    )
    ax0.grid(True)
    ax0.legend(loc="best")

    # Panel 2: residual tail + fit, shown on semilog so tau is visually the slope.
    ax1.semilogy(x_s, np.clip(y_uA, 1e-12, None), color="black", linewidth=1.2, label="Residual (baseline - current)")
    if used_mask.size == y_uA.size:
        ax1.scatter(
            x_s[used_mask],
            np.clip(y_uA[used_mask], 1e-12, None),
            s=14,
            color="tab:blue",
            label="Fit points",
        )
    x_fit = np.linspace(float(np.min(x_s)), float(np.max(x_s)), 200) if x_s.size > 1 else x_s
    y_fit = float(A_uA) * np.exp(-x_fit / float(tau_s))
    ax1.semilogy(
        x_fit,
        np.clip(y_fit, 1e-12, None),
        color="tab:orange",
        linewidth=2.0,
        label=f"Fit: A*exp(-t/tau)\nA={A_uA:.3g} µA, tau={tau_s:.3g} s",
    )
    ax1.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("t - t0 (s)")
    ax1.set_ylabel("Residual (µA) [log]")
    ax1.grid(True, which="both")
    ax1.legend(loc="best")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved exp-fit diagnostic plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _read_hvramping_csv(path: Path) -> pd.DataFrame:
    """Read the ramping CSV format with two header rows.

    These files contain a long DCS header (row 0) and a simplified header (row 1).
    We use row 1 as the header.
    """

    df = pd.read_csv(path, header=1)

    # Drop trailing empty column (these files commonly end lines with a comma).
    df = df.drop(columns=[c for c in df.columns if str(c).strip() == "" or str(c).startswith("Unnamed")], errors="ignore")

    time_col = df.columns[0]
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _read_hv_ramp_voltage_kv(paths: list[Path]) -> pd.Series:
    """Read HV ramp CSV(s) and return cathode voltage in kV as a single time series."""

    frames = []
    for path in paths:
        df = pd.read_csv(path, header=1)
        df = df.drop(columns=[c for c in df.columns if str(c).strip() == "" or str(c).startswith("Unnamed")], errors="ignore")

        time_col = df.columns[0]
        df = df.rename(columns={time_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

        if df.empty:
            continue

        # Normalize column name whitespace to find the voltage column.
        col_map = {str(c).strip(): c for c in df.columns}
        if "Voltage [V]" in col_map:
            voltage_col = col_map["Voltage [V]"]
        else:
            candidates = [orig for name, orig in col_map.items() if ("Voltage" in name and "Current" not in name)]
            if not candidates:
                raise ValueError(f"No voltage column found in {path}. Columns: {list(df.columns)}")
            voltage_col = candidates[0]

        v = pd.to_numeric(df[voltage_col], errors="coerce").dropna()
        if not v.empty:
            frames.append(v)

    if not frames:
        return pd.Series(dtype=float)

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    merged = merged.dropna()
    merged = merged / 1000.0
    merged.name = "cathode_voltage_kV"
    return merged


def _plot_cathode_voltage_and_termination_currents(
    voltage_kv: pd.Series,
    top_current_uA: pd.Series,
    bottom_current_uA: pd.Series,
    *,
    title: str,
    save_path: Path | None,
    show_plot: bool,
) -> None:
    if voltage_kv.empty or top_current_uA.empty or bottom_current_uA.empty:
        print("Skipping cathode overlay plot: missing voltage or current data.")
        return

    # Restrict to the common time span for a cleaner overlay.
    t0 = max(voltage_kv.index.min(), top_current_uA.index.min(), bottom_current_uA.index.min())
    t1 = min(voltage_kv.index.max(), top_current_uA.index.max(), bottom_current_uA.index.max())
    if t1 <= t0:
        print("Skipping cathode overlay plot: no overlapping time range.")
        return

    v = voltage_kv[(voltage_kv.index >= t0) & (voltage_kv.index <= t1)]
    top = top_current_uA[(top_current_uA.index >= t0) & (top_current_uA.index <= t1)]
    bottom = bottom_current_uA[(bottom_current_uA.index >= t0) & (bottom_current_uA.index <= t1)]

    fig, ax_v = plt.subplots(figsize=(14, 6))
    ax_i = ax_v.twinx()

    line_v = ax_v.plot(v.index, v.to_numpy(dtype=float), color="tab:blue", lw=1.2, label="Cathode voltage")
    line_t = ax_i.plot(top.index, top.to_numpy(dtype=float), color="tab:red", lw=1.0, label="Top termination current")
    line_b = ax_i.plot(bottom.index, bottom.to_numpy(dtype=float), color="tab:green", lw=1.0, label="Bottom termination current")

    ax_v.set_xlabel("Time")
    ax_v.set_ylabel("Cathode voltage [kV]", color="tab:blue")
    ax_i.set_ylabel("Termination current [µA]", color="black")
    ax_v.grid(True, alpha=0.25)

    lines = line_v + line_t + line_b
    labels = [l.get_label() for l in lines]
    ax_v.legend(lines, labels, loc="best")
    ax_v.set_title(title)

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved cathode overlay plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _combined_detector_current_a(
    beam_df: pd.DataFrame,
    tco_df: pd.DataFrame,
    grouping: DetectorGrouping,
) -> pd.Series:
    beam_cols = [c for c in channel_cols("FC_Beam", grouping.beam_channels) if c in beam_df.columns]
    tco_cols = [c for c in channel_cols("TCO", grouping.tco_channels) if c in tco_df.columns]

    if not beam_cols:
        raise ValueError("No BEAM channel columns found for requested grouping.")
    if not tco_cols:
        raise ValueError("No TCO channel columns found for requested grouping.")

    beam_sum = beam_df[beam_cols].sum(axis=1, min_count=1)
    tco_sum = tco_df[tco_cols].sum(axis=1, min_count=1)

    common = beam_sum.index.intersection(tco_sum.index)
    if common.empty:
        raise ValueError("No overlapping timestamps between BEAM and TCO after parsing.")

    return beam_sum.loc[common] + tco_sum.loc[common]


def _detect_spikes_with_savgol(
    times: np.ndarray,
    currents_uA: np.ndarray,
    threshold_uA: float,
    window_length: int,
    polyorder: int,
    *,
    baseline_method: str,
    median_kernel: int,
    asls_lam: float,
    asls_p: float,
    asls_niter: int,
    charge_method: str,
    tau_fixed_s: float,
    exp_fit_min_points: int,
    exp_fit_max_window_s: float,
    exp_fit_frac_min: float,
    exp_fit_tau_min_s: float,
    exp_fit_tau_max_s: float,
    exp_fit_diagnostics: str,
    exp_fit_diag_dir: Path | None,
    exp_fit_diag_budget: dict[str, int] | None,
    exp_fit_diag_min_mag_uA: float,
    exp_fit_diag_pre_s: float,
    exp_fit_diag_post_s: float,
    exp_fit_diag_include: str,
    detector_name: str,
    segment_label: str,
):
    baseline = compute_baseline(
        currents_uA,
        method=baseline_method,
        window_length=window_length,
        polyorder=polyorder,
        median_kernel=median_kernel,
        asls_lam=asls_lam,
        asls_p=asls_p,
        asls_niter=asls_niter,
    )
    residual = baseline - currents_uA
    peaks, props = find_peaks(residual, prominence=threshold_uA, width=1)

    prominences = props.get("prominences", np.array([]))
    left_ips = props.get("left_ips", np.array([]))
    right_ips = props.get("right_ips", np.array([]))

    times_pd = pd.to_datetime(times)
    times_s = times_pd.values.astype("datetime64[ns]").astype("int64") / 1e9

    def fit_exponential_charge(
        idx: int, peak_uA: float, right_bound: int | None
    ) -> tuple[float, float | None, float | None, int | None]:
        # Fit y(t) = A * exp(-(t-t0)/tau) for t>=t0.
        t0 = float(times_s[idx])
        end_by_time = int(np.searchsorted(times_s, t0 + float(exp_fit_max_window_s), side="right"))
        end = min(end_by_time, len(residual))
        if right_bound is not None:
            end = min(end, right_bound + 1)
        if end <= idx + 2:
            return peak_uA * tau_fixed_s, None, None, None

        x = np.asarray(times_s[idx:end] - t0, dtype=float)
        y = np.asarray(residual[idx:end], dtype=float)
        if not np.isfinite(peak_uA) or peak_uA <= 0:
            return peak_uA * tau_fixed_s, None, None, None
        y_min = max(float(exp_fit_frac_min) * float(peak_uA), 0.0)
        m = np.isfinite(x) & np.isfinite(y) & (y > y_min)
        used_n = int(np.sum(m))
        if used_n < int(exp_fit_min_points):
            return peak_uA * tau_fixed_s, None, None, None

        x_fit = x[m]
        logy = np.log(y[m])
        # logy = logA - x/tau
        slope, intercept = np.polyfit(x_fit, logy, 1)
        if not np.isfinite(slope) or slope >= 0:
            return peak_uA * tau_fixed_s, None, None, None
        tau = -1.0 / float(slope)
        if not np.isfinite(tau) or tau <= 0:
            return peak_uA * tau_fixed_s, None, None, None
        A = float(np.exp(intercept))
        if not np.isfinite(A) or A <= 0:
            return peak_uA * tau_fixed_s, None, None, None
        charge = A * tau  # µA*s == µC

        accepted = (float(exp_fit_tau_min_s) <= float(tau) <= float(exp_fit_tau_max_s))

        if (
            exp_fit_diagnostics != "none"
            and exp_fit_diag_budget is not None
            and exp_fit_diag_budget.get("remaining", 0) > 0
            and float(peak_uA) >= float(exp_fit_diag_min_mag_uA)
            and (exp_fit_diag_include == "all" or accepted)
        ):
            exp_fit_diag_budget["remaining"] = int(exp_fit_diag_budget.get("remaining", 0)) - 1
            exp_fit_diag_budget["created"] = int(exp_fit_diag_budget.get("created", 0)) + 1
            save_path = None
            show_plot = exp_fit_diagnostics == "show"
            if exp_fit_diagnostics == "save" and exp_fit_diag_dir is not None:
                status = "ok" if accepted else "rej"
                fn = (
                    f"expfit_{segment_label}_{detector_name.lower()}_{times_pd[idx].strftime('%Y%m%d_%H%M%S')}"
                    f"_i{idx}_{status}.png"
                )
                save_path = exp_fit_diag_dir / fn

            # Window around spike peak showing raw current + baseline.
            pre_s = max(0.0, float(exp_fit_diag_pre_s))
            post_s = max(0.0, float(exp_fit_diag_post_s))
            w0 = int(np.searchsorted(times_s, t0 - pre_s, side="left"))
            w1 = int(np.searchsorted(times_s, t0 + post_s, side="right"))
            w0 = max(0, min(w0, len(times_s)))
            w1 = max(0, min(w1, len(times_s)))
            spike_window_x = np.asarray(times_s[w0:w1] - t0, dtype=float)
            spike_window_current = np.asarray(currents_uA[w0:w1], dtype=float)
            spike_window_baseline = np.asarray(baseline[w0:w1], dtype=float)

            _plot_exp_fit_diagnostic(
                detector_name=detector_name,
                segment_label=segment_label,
                spike_time=times_pd[idx].to_pydatetime(),
                spike_window_x_s=spike_window_x,
                spike_window_current_uA=spike_window_current,
                spike_window_baseline_uA=spike_window_baseline,
                x_s=x,
                y_uA=y,
                used_mask=m,
                A_uA=A,
                tau_s=tau,
                charge_uC=charge,
                save_path=save_path,
                show_plot=show_plot,
            )

        if not accepted:
            return peak_uA * tau_fixed_s, None, None, None

        return charge, tau, A, used_n

    spikes: list[Spike] = []
    for j, idx in enumerate(peaks.tolist()):
        mag = float(prominences[j]) if j < len(prominences) else float(residual[idx])

        rb: int | None = None
        if j < len(right_ips):
            rb = int(np.ceil(float(right_ips[j])))
        if rb is not None:
            rb = max(0, min(rb, len(residual) - 1))

        if charge_method == "exp_fit":
            charge_uC, tau_s, fit_A_uA, fit_used_points = fit_exponential_charge(idx, mag, rb)
        else:
            charge_uC, tau_s = (mag * float(tau_fixed_s), None)
            fit_A_uA = None
            fit_used_points = None

        spikes.append(
            Spike(
                ts=times_pd[idx].to_pydatetime(),
                magnitude_uA=mag,
                current_uA=float(currents_uA[idx]),
                baseline_uA=float(baseline[idx]),
                charge_uC=float(charge_uC),
                tau_s=tau_s,
                fit_A_uA=fit_A_uA,
                fit_used_points=fit_used_points,
            )
        )

    return spikes, baseline


def _apply_exclusions(df: pd.DataFrame, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for left, right in intervals:
        mask &= ~((df.index >= left) & (df.index < right))
    return df.loc[mask]


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
    # Treat anything > max(5*median cadence, 2 minutes) as a gap.
    return max(5.0 * med, 120.0)


def _detect_spikes_with_savgol_segmented(
    times: np.ndarray,
    currents_uA: np.ndarray,
    *,
    threshold_uA: float,
    window_length: int,
    polyorder: int,
    gap_threshold_s: float | None = None,
    baseline_method: str,
    median_kernel: int,
    asls_lam: float,
    asls_p: float,
    asls_niter: int,
    charge_method: str,
    tau_fixed_s: float,
    exp_fit_min_points: int,
    exp_fit_max_window_s: float,
    exp_fit_frac_min: float,
    exp_fit_tau_min_s: float,
    exp_fit_tau_max_s: float,
    exp_fit_diagnostics: str,
    exp_fit_diag_dir: Path | None,
    exp_fit_diag_budget: dict[str, int] | None,
    exp_fit_diag_min_mag_uA: float,
    exp_fit_diag_pre_s: float,
    exp_fit_diag_post_s: float,
    exp_fit_diag_include: str,
    detector_name: str,
    segment_label: str,
):
    """Detect spikes with a baseline that resets across large time gaps.

    This prevents the Savitzky-Golay baseline from "bridging" gaps created by
    data quality exclusions or setpoint changes.
    """

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
            seg_spikes, seg_baseline = _detect_spikes_with_savgol(
                seg_times,
                seg_currents,
                threshold_uA=threshold_uA,
                window_length=window_length,
                polyorder=polyorder,
                baseline_method=baseline_method,
                median_kernel=median_kernel,
                asls_lam=asls_lam,
                asls_p=asls_p,
                asls_niter=asls_niter,
                charge_method=charge_method,
                tau_fixed_s=tau_fixed_s,
                exp_fit_min_points=exp_fit_min_points,
                exp_fit_max_window_s=exp_fit_max_window_s,
                exp_fit_frac_min=exp_fit_frac_min,
                exp_fit_tau_min_s=exp_fit_tau_min_s,
                exp_fit_tau_max_s=exp_fit_tau_max_s,
                exp_fit_diagnostics=exp_fit_diagnostics,
                exp_fit_diag_dir=exp_fit_diag_dir,
                exp_fit_diag_budget=exp_fit_diag_budget,
                exp_fit_diag_min_mag_uA=exp_fit_diag_min_mag_uA,
                exp_fit_diag_pre_s=exp_fit_diag_pre_s,
                exp_fit_diag_post_s=exp_fit_diag_post_s,
                exp_fit_diag_include=exp_fit_diag_include,
                detector_name=detector_name,
                segment_label=segment_label,
            )
            spikes_all.extend(seg_spikes)
            baseline_all[start:end] = seg_baseline
        else:
            baseline_all[start:end] = seg_currents
        start = end

    return spikes_all, baseline_all


def _bin_spikes(
    spikes,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bin_by: str,
    bin_minutes: int,
    tau_fixed_s: float,
):
    if start >= end:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if bin_by == "day":
        start_day = start.normalize()
        # Ensure the last day is included as a right edge.
        end_day = end.normalize() + pd.Timedelta(days=1)
        edges = pd.date_range(start=start_day, end=end_day, freq="1D")
    elif bin_by == "minutes":
        edges = pd.date_range(start=start, end=end, freq=f"{bin_minutes}min")
    else:
        raise ValueError(f"Unknown bin_by={bin_by!r} (expected 'day' or 'minutes')")

    if len(edges) < 2:
        edges = pd.DatetimeIndex([start, end])

    mids = []
    avg_charge_uC = []
    err_charge_uC = []
    rate_per_hour = []
    err_rate_per_hour = []

    # Pre-extract arrays for fast filtering.
    spike_times = pd.to_datetime([s.ts for s in spikes])
    spike_charges = np.array([s.charge_uC for s in spikes], dtype=float)

    for left, right in zip(edges[:-1], edges[1:]):
        mask = (spike_times >= left) & (spike_times < right)
        n = int(mask.sum())
        duration_h = (right - left).total_seconds() / 3600.0
        mid = left + (right - left) / 2

        mids.append(mid)
        if duration_h > 0:
            rate_per_hour.append(n / duration_h)
            err_rate_per_hour.append(np.sqrt(n) / duration_h)
        else:
            rate_per_hour.append(0.0)
            err_rate_per_hour.append(0.0)

        if n == 0:
            avg_charge_uC.append(np.nan)
            err_charge_uC.append(np.nan)
            continue

        charges = spike_charges[np.asarray(mask)]
        avg_charge_uC.append(float(np.mean(charges)))
        if n > 1:
            err_charge_uC.append(float(np.std(charges, ddof=1) / np.sqrt(n)))
        else:
            err_charge_uC.append(0.0)

    return (
        np.asarray(mids, dtype="datetime64[ns]"),
        np.asarray(avg_charge_uC, dtype=float),
        np.asarray(err_charge_uC, dtype=float),
        np.asarray(rate_per_hour, dtype=float),
        np.asarray(err_rate_per_hour, dtype=float),
    )


def _read_stable_voltage_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize to timezone-naive datetime64[ns] up front so downstream iteration
    # sees real datetimes (avoids pandas Scalar/bytes typing ambiguity).
    df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.tz_localize(None)
    df["end"] = pd.to_datetime(df["end"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["start", "end", "mean_E_V_per_cm"]).copy()
    df["start"] = df["start"].astype("datetime64[ns]")
    df["end"] = df["end"].astype("datetime64[ns]")
    df["mean_E_V_per_cm"] = pd.to_numeric(df["mean_E_V_per_cm"], errors="coerce")
    df = df.dropna(subset=["mean_E_V_per_cm"]).copy()
    df = df.sort_values("start").reset_index(drop=True)
    return df


def _bin_spikes_by_intervals(
    spikes,
    intervals_df: pd.DataFrame,
    tau_fixed_s: float,
):
    """Bin spikes inside explicit [start,end) intervals (plateaus).

    Returns:
    times_mid (datetime64), avg_charge_uC, err_charge_uC, rate_per_hour, err_rate_per_hour, mean_E_V_per_cm, avg_voltage_kV
    """

    if intervals_df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    mids = []
    avg_charge_uC = []
    err_charge_uC = []
    rate_per_hour = []
    err_rate_per_hour = []
    mean_E = []
    avg_voltage_kV = []

    spike_times = pd.to_datetime([s.ts for s in spikes])
    spike_charges = np.array([s.charge_uC for s in spikes], dtype=float)

    for left_raw, right_raw, mean_E_raw, avg_v_raw in intervals_df[["start", "end", "mean_E_V_per_cm", "avg_voltage_kV"]].itertuples(
        index=False, name=None
    ):
        left = pd.Timestamp(left_raw)
        right = pd.Timestamp(right_raw)
        if right <= left:
            continue

        mask = (spike_times >= left) & (spike_times < right)
        n = int(mask.sum())
        duration_h = (right - left).total_seconds() / 3600.0
        mid = left + (right - left) / 2

        mids.append(mid)
        mean_E.append(float(mean_E_raw))
        avg_voltage_kV.append(float(avg_v_raw))
        duration_s_live = _live_seconds_between(left, right, exclusions=EXCLUDED_TIME_INTERVALS)
        duration_h_live = duration_s_live / 3600.0 if duration_s_live > 0 else 0.0
        if duration_h_live > 0:
            rate_per_hour.append(n / duration_h_live)
            err_rate_per_hour.append(np.sqrt(n) / duration_h_live)
        else:
            rate_per_hour.append(0.0)
            err_rate_per_hour.append(0.0)

        if n == 0:
            avg_charge_uC.append(np.nan)
            err_charge_uC.append(np.nan)
            continue

        charges = spike_charges[np.asarray(mask)]
        avg_charge_uC.append(float(np.mean(charges)))
        if n > 1:
            err_charge_uC.append(float(np.std(charges, ddof=1) / np.sqrt(n)))
        else:
            err_charge_uC.append(0.0)

    return (
        np.asarray(mids, dtype="datetime64[ns]"),
        np.asarray(avg_charge_uC, dtype=float),
        np.asarray(err_charge_uC, dtype=float),
        np.asarray(rate_per_hour, dtype=float),
        np.asarray(err_rate_per_hour, dtype=float),
        np.asarray(mean_E, dtype=float),
        np.asarray(avg_voltage_kV, dtype=float),
    )


def _summarize_plateaus_from_window_detection(
    current_a: pd.Series,
    other_current_a: pd.Series | None,
    intervals_df: pd.DataFrame,
    *,
    exclusive: bool,
    coincidence_window_s: float,
    threshold_uA: float,
    window_length: int,
    polyorder: int,
    baseline_method: str,
    median_kernel: int,
    asls_lam: float,
    asls_p: float,
    asls_niter: int,
    charge_method: str,
    tau_fixed_s: float,
    exp_fit_min_points: int,
    exp_fit_max_window_s: float,
    exp_fit_frac_min: float,
    exp_fit_tau_min_s: float,
    exp_fit_tau_max_s: float,
    exp_fit_diagnostics: str,
    exp_fit_diag_dir: Path | None,
    exp_fit_diag_budget: dict | None,
    exp_fit_diag_min_mag_uA: float,
    exp_fit_diag_pre_s: float,
    exp_fit_diag_post_s: float,
    exp_fit_diag_include: str,
    detector_name: str,
    tag: str,
):
    """Compute plateau-binned summary by re-detecting spikes within each plateau window.

    This matches the baseline plateau plots when they run spike detection on cropped windows.

    Returns:
    times_mid (datetime64), n_spikes, avg_charge_uC, err_charge_uC, rate_per_hour, err_rate_per_hour, mean_E_V_per_cm, avg_voltage_kV
    """

    if intervals_df is None or intervals_df.empty:
        return (
            np.array([]),
            np.array([], dtype=int),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    mids: list[pd.Timestamp] = []
    n_spikes: list[int] = []
    avg_charge_uC: list[float] = []
    err_charge_uC: list[float] = []
    rate_per_hour: list[float] = []
    err_rate_per_hour: list[float] = []
    mean_E: list[float] = []
    avg_voltage_kV: list[float] = []

    for left_raw, right_raw, mean_E_raw, avg_v_raw in intervals_df[
        ["start", "end", "mean_E_V_per_cm", "avg_voltage_kV"]
    ].itertuples(index=False, name=None):
        left = pd.Timestamp(left_raw)
        right = pd.Timestamp(right_raw)
        if right <= left:
            continue

        # Any time-based metric uses "live" time with excluded windows removed.
        duration_s_live = _live_seconds_between(left, right, exclusions=EXCLUDED_TIME_INTERVALS)
        duration_h_live = duration_s_live / 3600.0 if duration_s_live > 0 else 0.0
        mid = left + (right - left) / 2

        seg = current_a[(current_a.index >= left) & (current_a.index < right)]
        seg_times = pd.DatetimeIndex(seg.index).to_pydatetime()
        seg_currents_uA = seg.to_numpy(dtype=float) * 1e6

        # Detect on the cropped plateau window.
        seg_spikes, _seg_baseline = _detect_spikes_with_savgol_segmented(
            seg_times,
            seg_currents_uA,
            threshold_uA=threshold_uA,
            window_length=window_length,
            polyorder=polyorder,
            charge_method=charge_method,
            tau_fixed_s=tau_fixed_s,
            exp_fit_min_points=exp_fit_min_points,
            exp_fit_max_window_s=exp_fit_max_window_s,
            exp_fit_frac_min=exp_fit_frac_min,
            exp_fit_tau_min_s=exp_fit_tau_min_s,
            exp_fit_tau_max_s=exp_fit_tau_max_s,
            exp_fit_diagnostics=exp_fit_diagnostics,
            exp_fit_diag_dir=exp_fit_diag_dir,
            exp_fit_diag_budget=exp_fit_diag_budget,
            exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag_uA),
            exp_fit_diag_pre_s=exp_fit_diag_pre_s,
            exp_fit_diag_post_s=exp_fit_diag_post_s,
            exp_fit_diag_include=exp_fit_diag_include,
            detector_name=detector_name,
            segment_label=f"{tag}_plateau_{left.strftime('%Y%m%d_%H%M')}",
            baseline_method=baseline_method,
            median_kernel=median_kernel,
            asls_lam=asls_lam,
            asls_p=asls_p,
            asls_niter=asls_niter,
        )

        if exclusive and other_current_a is not None:
            other_seg = other_current_a[(other_current_a.index >= left) & (other_current_a.index < right)]
            other_times = pd.DatetimeIndex(other_seg.index).to_pydatetime()
            other_currents_uA = other_seg.to_numpy(dtype=float) * 1e6
            other_spikes, _ = _detect_spikes_with_savgol_segmented(
                other_times,
                other_currents_uA,
                threshold_uA=threshold_uA,
                window_length=window_length,
                polyorder=polyorder,
                charge_method=charge_method,
                tau_fixed_s=tau_fixed_s,
                exp_fit_min_points=exp_fit_min_points,
                exp_fit_max_window_s=exp_fit_max_window_s,
                exp_fit_frac_min=exp_fit_frac_min,
                exp_fit_tau_min_s=exp_fit_tau_min_s,
                exp_fit_tau_max_s=exp_fit_tau_max_s,
                exp_fit_diagnostics="none",
                exp_fit_diag_dir=None,
                exp_fit_diag_budget=None,
                exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag_uA),
                exp_fit_diag_pre_s=exp_fit_diag_pre_s,
                exp_fit_diag_post_s=exp_fit_diag_post_s,
                exp_fit_diag_include=exp_fit_diag_include,
                detector_name=("Other"),
                segment_label=f"{tag}_coincidence_{left.strftime('%Y%m%d_%H%M')}",
                baseline_method=baseline_method,
                median_kernel=median_kernel,
                asls_lam=asls_lam,
                asls_p=asls_p,
                asls_niter=asls_niter,
            )
            seg_spikes, _other_only, _coincident = _exclusive_spikes_by_time(
                seg_spikes,
                other_spikes,
                coincidence_window_s=float(coincidence_window_s),
            )

        n = int(len(seg_spikes))

        mids.append(mid)
        mean_E.append(float(mean_E_raw))
        avg_voltage_kV.append(float(avg_v_raw))
        n_spikes.append(n)

        if duration_h_live > 0:
            rate_per_hour.append(n / duration_h_live)
            err_rate_per_hour.append(np.sqrt(n) / duration_h_live)
        else:
            rate_per_hour.append(0.0)
            err_rate_per_hour.append(0.0)

        if n == 0:
            avg_charge_uC.append(np.nan)
            err_charge_uC.append(np.nan)
            continue

        charges = np.asarray([s.charge_uC for s in seg_spikes], dtype=float)
        avg_charge_uC.append(float(np.mean(charges)))
        if n > 1:
            err_charge_uC.append(float(np.std(charges, ddof=1) / np.sqrt(n)))
        else:
            err_charge_uC.append(0.0)

    return (
        np.asarray(mids, dtype="datetime64[ns]"),
        np.asarray(n_spikes, dtype=int),
        np.asarray(avg_charge_uC, dtype=float),
        np.asarray(err_charge_uC, dtype=float),
        np.asarray(rate_per_hour, dtype=float),
        np.asarray(err_rate_per_hour, dtype=float),
        np.asarray(mean_E, dtype=float),
        np.asarray(avg_voltage_kV, dtype=float),
    )


def _plot_summary(
    times_mid,
    avg_charge_uC,
    err_charge_uC,
    rate_per_hour,
    err_rate_per_hour,
    rate_ylabel: str,
    title: str,
    save_path: Path | None,
    show_plot: bool,
    vertical_lines: list[pd.Timestamp] | list[tuple[pd.Timestamp, str]] | None = None,
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] | None = None,
    span_labels: list[tuple[pd.Timestamp, str, str]] | None = None,
):
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if spans:
        for start, end, color in spans:
            if end <= start:
                continue
            ax[0].axvspan(start, end, color=color, alpha=0.08, zorder=0)
            ax[1].axvspan(start, end, color=color, alpha=0.08, zorder=0)

    if span_labels:
        for t, text, color in span_labels:
            ax[0].text(
                t,
                0.92,
                text,
                color=color,
                fontsize=10,
                ha="center",
                va="center",
                transform=ax[0].get_xaxis_transform(),
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=3),
            )
            ax[1].text(
                t,
                0.08,
                text,
                color=color,
                fontsize=9,
                ha="center",
                va="center",
                transform=ax[1].get_xaxis_transform(),
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=3),
            )

    ax[0].errorbar(times_mid, avg_charge_uC, yerr=err_charge_uC, fmt="o", color="black", capsize=4)
    ax[0].set_ylabel("Avg charge / spike (µC)")
    ax[0].set_title(title)

    ax[1].errorbar(times_mid, rate_per_hour, yerr=err_rate_per_hour, fmt="s", linestyle="None", color="black", capsize=3)
    ax[1].set_ylabel(rate_ylabel)
    ax[1].set_xlabel("Time")

    if vertical_lines:
        for item in vertical_lines:
            if isinstance(item, tuple):
                t, c = item
            else:
                t, c = item, "tab:blue"
            # Mirror the October spike study style: dashed vlines on both subplots.
            ax[0].axvline(t, color=c, linestyle="--", linewidth=1.5, label="_nolegend_")
            ax[1].axvline(t, color=c, linestyle="--", linewidth=1.5)

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved summary plot to {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def _plot_summary_vs_mean_e(
    mean_E_V_per_cm: np.ndarray,
    avg_charge_uC: np.ndarray,
    err_charge_uC: np.ndarray,
    rate_per_hour: np.ndarray,
    err_rate_per_hour: np.ndarray,
    *,
    title: str,
    save_path: Path | None,
    show_plot: bool,
):
    order = np.argsort(mean_E_V_per_cm)
    x = np.asarray(mean_E_V_per_cm)[order]
    y_charge = np.asarray(avg_charge_uC)[order]
    y_charge_err = np.asarray(err_charge_uC)[order]
    y_rate = np.asarray(rate_per_hour)[order]
    y_rate_err = np.asarray(err_rate_per_hour)[order]

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax[0].errorbar(x, y_charge, yerr=y_charge_err, fmt="o", color="black", capsize=4)
    ax[0].set_ylabel("Avg charge / spike (µC)")
    ax[0].set_title(title)

    ax[1].errorbar(x, y_rate, yerr=y_rate_err, fmt="s", linestyle="None", color="black", capsize=3)
    ax[1].set_ylabel("Spikes / hour")
    ax[1].set_xlabel("Mean E (V/cm)")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved mean-E summary plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_summary_vs_voltage_kV(
    avg_voltage_kV: np.ndarray,
    avg_charge_uC: np.ndarray,
    err_charge_uC: np.ndarray,
    rate_per_hour: np.ndarray,
    err_rate_per_hour: np.ndarray,
    *,
    title: str,
    save_path: Path | None,
    show_plot: bool,
):
    order = np.argsort(avg_voltage_kV)
    x = np.asarray(avg_voltage_kV)[order]
    y_charge = np.asarray(avg_charge_uC)[order]
    y_charge_err = np.asarray(err_charge_uC)[order]
    y_rate = np.asarray(rate_per_hour)[order]
    y_rate_err = np.asarray(err_rate_per_hour)[order]

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax[0].errorbar(x, y_charge, yerr=y_charge_err, fmt="o", color="black", capsize=4)
    ax[0].set_ylabel("Avg charge / spike (µC)")
    ax[0].set_title(title)

    ax[1].errorbar(x, y_rate, yerr=y_rate_err, fmt="s", linestyle="None", color="black", capsize=3)
    ax[1].set_ylabel("Spikes / hour")
    ax[1].set_xlabel("Avg HV (kV)")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved voltage summary plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_spikes_with_baseline(
    dates,
    currents_uA: np.ndarray,
    spikes_info,
    baseline_uA: np.ndarray,
    *,
    title: str,
    save_path: Path | None,
    show_plot: bool,
    vertical_lines: list[pd.Timestamp] | None = None,
    top_annotations: list[tuple[pd.Timestamp, str]] | None = None,
):
    """Mirror the baseline/spike plot format used in the October spike study."""

    plt.figure(figsize=(14, 5))
    n_spikes = len(spikes_info) if spikes_info is not None else 0
    plt.plot(dates, currents_uA, label="Current", alpha=0.7)
    plt.plot(dates, baseline_uA, label="Baseline (rolling)", linestyle="--", color="black")

    for sp in spikes_info:
        plt.scatter(
            sp.ts,
            sp.current_uA,
            color="red",
            s=36,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="Spike" if "Spike" not in plt.gca().get_legend_handles_labels()[1] else "",
        )
        plt.vlines(
            sp.ts,
            sp.current_uA,
            sp.baseline_uA,
            color="orange",
            linestyle=":",
            label="Magnitude" if "Magnitude" not in plt.gca().get_legend_handles_labels()[1] else "",
        )

    if vertical_lines:
        for t in vertical_lines:
            # Use Matplotlib date numbers to satisfy type checkers.
            plt.axvline(date2num(pd.Timestamp(t).to_pydatetime()), color="tab:blue", linestyle="--", linewidth=1.5, label="_nolegend_")

    if top_annotations:
        for t, text in top_annotations:
            plt.text(
                date2num(pd.Timestamp(t).to_pydatetime()),
                0.92,
                text,
                fontsize=10,
                ha="center",
                va="center",
                transform=plt.gca().get_xaxis_transform(),
                bbox=cast(Any, dict(facecolor="white", alpha=0.6, edgecolor="none", pad=3)),
            )

    plt.xlabel("Time")
    plt.ylabel("Current (uA)")
    plt.title(f"{title}  (n_spikes={n_spikes})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved baseline+spike plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close()


def _spikes_projected_onto_window(
    spikes: list[Spike],
    window_times: np.ndarray,
    window_currents_uA: np.ndarray,
    window_baseline_uA: np.ndarray,
) -> list[Spike]:
    """Project spike timestamps onto a specific plotted window.

    This keeps spike timestamps (and charges) from the global detection while
    using the window's current/baseline arrays for plotting markers/vlines.
    """

    if not spikes:
        return []
    if len(window_times) == 0:
        return []

    wt = pd.to_datetime(window_times).values.astype("datetime64[ns]")
    wc = np.asarray(window_currents_uA, dtype=float)
    wb = np.asarray(window_baseline_uA, dtype=float)
    if wt.size != wc.size or wt.size != wb.size:
        return spikes

    out: list[Spike] = []
    for sp in spikes:
        t = np.datetime64(pd.Timestamp(sp.ts).to_datetime64())
        i = int(np.searchsorted(wt, t, side="left"))
        if i <= 0:
            j = 0
        elif i >= wt.size:
            j = wt.size - 1
        else:
            # Choose nearest of i-1 and i
            j = i - 1 if abs((t - wt[i - 1]).astype("timedelta64[ns]").astype(int)) <= abs((wt[i] - t).astype("timedelta64[ns]").astype(int)) else i
        out.append(
            Spike(
                ts=sp.ts,
                magnitude_uA=sp.magnitude_uA,
                current_uA=float(wc[j]),
                baseline_uA=float(wb[j]),
                charge_uC=sp.charge_uC,
                tau_s=sp.tau_s,
                fit_A_uA=sp.fit_A_uA,
                fit_used_points=sp.fit_used_points,
            )
        )
    return out


def _write_spike_timestamps(
    path: Path,
    spikes: list[Spike],
    *,
    charge_method: str,
    tau_fixed_s: float,
    exp_fit_min_points: int,
    exp_fit_max_window_s: float,
    exp_fit_frac_min: float,
    exp_fit_tau_min_s: float,
    exp_fit_tau_max_s: float,
    threshold_uA: float,
    window_length: int,
    polyorder: int,
    baseline_method: str,
    median_kernel: int,
    asls_lam: float,
    asls_p: float,
    asls_niter: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# Generated by: {Path(__file__).name}\n")
        f.write("# Columns: timestamp(YYYY-mm-dd HH:MM:SS)  charge_uC\n")
        f.write(f"# charge_method = {charge_method}\n")
        if charge_method == "tau_fixed":
            f.write(f"# charge_uC = magnitude_uA * tau_fixed_s; tau_fixed_s = {tau_fixed_s:.6g}\n")
        else:
            f.write(
                f"# charge_uC = A_uA * tau_s from fit: residual = A*exp(-(t-t0)/tau); min_points={exp_fit_min_points}, max_window_s={exp_fit_max_window_s:.6g}, frac_min={exp_fit_frac_min:.6g}, tau_range_s=[{exp_fit_tau_min_s:.6g}, {exp_fit_tau_max_s:.6g}]\n"
            )
        f.write(
            "# spike_magnitude_uA is the prominence returned by scipy.signal.find_peaks on residual = baseline_uA - current_uA\n"
        )
        f.write(
            f"# Baseline: method={baseline_method}, window_length={window_length}, polyorder={polyorder}, median_kernel={median_kernel}, asls_lam={asls_lam:.6g}, asls_p={asls_p:.6g}, asls_niter={asls_niter}\n"
        )
        f.write(f"# Spike finding params: threshold_uA={threshold_uA:.6g}, window_length={window_length}, polyorder={polyorder}\n")
        for sp in spikes:
            f.write(pd.Timestamp(sp.ts).strftime("%Y-%m-%d %H:%M:%S") + f"  {sp.charge_uC:.6f}\n")
    print(f"Wrote spike timestamps to {path}")


def main() -> None:
    ramp_dir = fcspikes_root() / "csv" / "ramping"
    default_hv_candidates = [
        ramp_dir / "HV_ramp_Nov27.csv",
        ramp_dir / "HV_ramp_Dec6.csv",
        ramp_dir / "HV_ramp_Dec15.csv",
    ]
    default_hv_paths = [p for p in default_hv_candidates if p.exists()]
    if not default_hv_paths:
        default_hv_paths = sorted(ramp_dir.glob("HV_ramp_*.csv"))

    parser = argparse.ArgumentParser(
        description="HV ramping spike summary for Top/Bottom (BEAM+TCO combined).",
    )
    parser.add_argument(
        "--beam",
        nargs="+",
        default=[str(ramp_dir / "BEAM_HVRamping.csv"), str(ramp_dir / "BEAM_HVRamping2.csv")],
        help="One or more BEAM ramping CSV files (default: BEAM_HVRamping(.csv|2.csv)).",
    )
    parser.add_argument(
        "--tco",
        nargs="+",
        default=[str(ramp_dir / "TCO_HVRamping.csv"), str(ramp_dir / "TCO_HVRamping2.csv")],
        help="One or more TCO ramping CSV files (default: TCO_HVRamping(.csv|2.csv)).",
    )
    parser.add_argument(
        "--hv-ramp",
        nargs="*",
        default=None,
        help="HV ramp CSV(s) used for cathode voltage overlay (default: HV_ramp_*.csv under data/csv/ramping).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Tag used in output filenames.",
    )
    parser.add_argument(
        "--threshold-uA",
        type=float,
        default=0.03,
        help="Spike prominence threshold in µA (applied on baseline-current residual).",
    )
    parser.add_argument(
        "--exclusive-spikes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, remove coincident spikes (present in both Top and Bottom) and output Top-only/Bottom-only plots.",
    )
    parser.add_argument(
        "--coincidence-window-s",
        type=float,
        default=2.0,
        help="Time window (s) for considering Top/Bottom spikes coincident when --exclusive-spikes is enabled.",
    )
    parser.add_argument(
        "--charge-method",
        choices=["tau_fixed", "exp_fit"],
        default="tau_fixed",
        help="Charge calculation method: legacy magnitude*tau_fixed or exponential fit on residual tail.",
    )
    parser.add_argument(
        "--tau-fixed-s",
        type=float,
        default=6.6,
        help="Fixed time constant used to convert spike magnitude to charge: charge(µC)=magnitude(µA)*tau(s).",
    )
    parser.add_argument(
        "--exp-fit-min-points",
        type=int,
        default=10,
        help="Minimum points above threshold to fit exponential tail (used when --charge-method=exp_fit).",
    )
    parser.add_argument(
        "--exp-fit-max-window-s",
        type=float,
        default=60.0,
        help="Max seconds after peak to include in exponential fit window (used when --charge-method=exp_fit).",
    )
    parser.add_argument(
        "--exp-fit-frac-min",
        type=float,
        default=0.05,
        help="Fit uses points where residual > frac_min * peak_height (used when --charge-method=exp_fit).",
    )
    parser.add_argument(
        "--exp-fit-tau-min-s",
        type=float,
        default=5.0,
        help="Minimum allowed tau (s) for exponential fit; outside range falls back to tau_fixed.",
    )
    parser.add_argument(
        "--exp-fit-tau-max-s",
        type=float,
        default=20.0,
        help="Maximum allowed tau (s) for exponential fit; outside range falls back to tau_fixed.",
    )
    parser.add_argument(
        "--exp-fit-diagnostics",
        choices=["none", "save", "show"],
        default="none",
        help="Optional diagnostic plots showing the exponential-fit tail (limited by --exp-fit-diagnostics-max).",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-max",
        type=int,
        default=20,
        help="Maximum number of exponential-fit diagnostic plots to create total (across Top+Bottom).",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-max-per-detector",
        type=int,
        default=None,
        help="If set, overrides --exp-fit-diagnostics-max with a per-detector limit.",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-min-mag-uA",
        type=float,
        default=None,
        help="Only create diagnostic plots for spikes with magnitude >= this value (µA). Default: uses --threshold-uA.",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-dir",
        default=None,
        help="Directory to save diagnostic plots when --exp-fit-diagnostics=save (default: figures/.../plots_exp_fit).",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-include",
        choices=["accepted", "all"],
        default="accepted",
        help="Which fits to plot: only accepted fits (default) or all attempted fits (including tau out-of-range).",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-scope",
        choices=["full", "plateau", "day", "both"],
        default="plateau",
        help="Where to generate exp-fit diagnostic plots. Use 'plateau' to match the plateau baseline plots.",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-pre-s",
        type=float,
        default=20.0,
        help="Seconds before spike peak to show in diagnostic plot (current + baseline panel).",
    )
    parser.add_argument(
        "--exp-fit-diagnostics-post-s",
        type=float,
        default=60.0,
        help="Seconds after spike peak to show in diagnostic plot (current + baseline panel).",
    )
    parser.add_argument("--window-length", type=int, default=2500)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument(
        "--baseline-method",
        choices=["savgol", "median_savgol", "asls"],
        default="median_savgol",
        help="Baseline method. 'median_savgol' is more robust to spikes than plain SavGol.",
    )
    parser.add_argument(
        "--median-kernel",
        type=int,
        default=301,
        help="Median filter kernel size (odd) used when --baseline-method=median_savgol.",
    )
    parser.add_argument(
        "--asls-lam",
        type=float,
        default=1e7,
        help="AsLS smoothness (lambda) used when --baseline-method=asls.",
    )
    parser.add_argument(
        "--asls-p",
        type=float,
        default=0.01,
        help="AsLS asymmetry p used when --baseline-method=asls (applied on negated signal for negative spikes).",
    )
    parser.add_argument(
        "--asls-niter",
        type=int,
        default=10,
        help="Number of AsLS reweighting iterations used when --baseline-method=asls.",
    )
    parser.add_argument(
        "--stable-summary-csv",
        default=str(figures_root() / "analysis" / "ramping_hv" / "HV_ramp_stable_voltage_summary.csv"),
        help="CSV with stable-voltage plateau intervals (must include start,end,mean_E_V_per_cm).",
    )
    parser.add_argument(
        "--baseline-plots",
        choices=["none", "plateau", "day"],
        default="plateau",
        help="Save baseline+spike detection plots per plateau (default), per day, or not at all.",
    )
    parser.add_argument(
        "--show-baseline-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show baseline+spike plots interactively (off by default).",
    )
    parser.add_argument(
        "--write-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write spike timestamps+charge text files under data/txt/spike_timestamps/.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save summary plots under figures/analysis/hv_ramping_spike_study/.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show summary plots interactively.",
    )
    parser.add_argument(
        "--plot-cathode-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot cathode voltage with top/bottom termination currents in one overlay plot.",
    )

    args = parser.parse_args()

    global_exp_fit_diag_budget: dict[str, int] | None = None
    if args.exp_fit_diagnostics != "none":
        global_exp_fit_diag_budget = {"remaining": int(args.exp_fit_diagnostics_max), "created": 0}

    beam_paths = [Path(p) for p in args.beam]
    tco_paths = [Path(p) for p in args.tco]

    def load_merged_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        if not beam_paths:
            raise SystemExit("Provide at least one BEAM file via --beam")
        if not tco_paths:
            raise SystemExit("Provide at least one TCO file via --tco")

        beam_frames = []
        for b in beam_paths:
            print(f"Loading BEAM: {b.name}")
            beam_frames.append(_read_hvramping_csv(b))
        beam_df = pd.concat(beam_frames).sort_index()
        beam_df = beam_df[~beam_df.index.duplicated(keep="first")]

        beam_df = _apply_exclusions(beam_df, EXCLUDED_TIME_INTERVALS)

        tco_frames = []
        for t in tco_paths:
            print(f"Loading TCO:  {t.name}")
            tco_frames.append(_read_hvramping_csv(t))
        tco_df = pd.concat(tco_frames).sort_index()
        tco_df = tco_df[~tco_df.index.duplicated(keep="first")]

        tco_df = _apply_exclusions(tco_df, EXCLUDED_TIME_INTERVALS)

        return beam_df, tco_df

    merged_beam_df, merged_tco_df = load_merged_inputs()

    def load_detector_series(grouping: DetectorGrouping) -> pd.Series:
        return _combined_detector_current_a(merged_beam_df, merged_tco_df, grouping)

    def analyze(detector_name: str, current_a: pd.Series, other_current_a: pd.Series | None) -> None:
        times = pd.DatetimeIndex(current_a.index).to_pydatetime()
        currents_uA = current_a.to_numpy(dtype=float) * 1e6

        exp_fit_diag_min_mag = args.exp_fit_diagnostics_min_mag_uA
        if exp_fit_diag_min_mag is None:
            exp_fit_diag_min_mag = float(args.threshold_uA)

        exp_fit_diag_dir: Path | None = None
        if args.exp_fit_diagnostics == "save":
            exp_fit_diag_dir = (
                Path(args.exp_fit_diagnostics_dir)
                if args.exp_fit_diagnostics_dir is not None
                else (figures_root() / "analysis" / "hv_ramping_spike_study" / "plots_exp_fit")
            )

        if args.exp_fit_diagnostics == "none":
            exp_fit_diag_budget = None
        elif args.exp_fit_diagnostics_max_per_detector is not None:
            exp_fit_diag_budget = {"remaining": int(args.exp_fit_diagnostics_max_per_detector), "created": 0}
        else:
            exp_fit_diag_budget = global_exp_fit_diag_budget

        created_before = int(exp_fit_diag_budget.get("created", 0)) if exp_fit_diag_budget is not None else 0

        diag_on_full = args.exp_fit_diagnostics_scope in ("full", "both")
        diag_on_day = args.exp_fit_diagnostics_scope in ("day", "both")
        diag_on_plateau = args.exp_fit_diagnostics_scope in ("plateau", "both")

        spikes, _baseline = _detect_spikes_with_savgol_segmented(
            times,
            currents_uA,
            threshold_uA=args.threshold_uA,
            window_length=args.window_length,
            polyorder=args.polyorder,
            charge_method=args.charge_method,
            tau_fixed_s=args.tau_fixed_s,
            exp_fit_min_points=args.exp_fit_min_points,
            exp_fit_max_window_s=args.exp_fit_max_window_s,
            exp_fit_frac_min=args.exp_fit_frac_min,
            exp_fit_tau_min_s=args.exp_fit_tau_min_s,
            exp_fit_tau_max_s=args.exp_fit_tau_max_s,
            exp_fit_diagnostics=(args.exp_fit_diagnostics if diag_on_full else "none"),
            exp_fit_diag_dir=exp_fit_diag_dir,
            exp_fit_diag_budget=exp_fit_diag_budget,
            exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
            exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
            exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
            exp_fit_diag_include=args.exp_fit_diagnostics_include,
            detector_name=detector_name,
            segment_label=f"{args.tag}_full",
            baseline_method=args.baseline_method,
            median_kernel=args.median_kernel,
            asls_lam=args.asls_lam,
            asls_p=args.asls_p,
            asls_niter=args.asls_niter,
        )

        if args.exclusive_spikes and other_current_a is not None:
            other_times_full = pd.DatetimeIndex(other_current_a.index).to_pydatetime()
            other_currents_uA_full = other_current_a.to_numpy(dtype=float) * 1e6
            other_spikes_full, _ = _detect_spikes_with_savgol_segmented(
                other_times_full,
                other_currents_uA_full,
                threshold_uA=args.threshold_uA,
                window_length=args.window_length,
                polyorder=args.polyorder,
                charge_method=args.charge_method,
                tau_fixed_s=args.tau_fixed_s,
                exp_fit_min_points=args.exp_fit_min_points,
                exp_fit_max_window_s=args.exp_fit_max_window_s,
                exp_fit_frac_min=args.exp_fit_frac_min,
                exp_fit_tau_min_s=args.exp_fit_tau_min_s,
                exp_fit_tau_max_s=args.exp_fit_tau_max_s,
                exp_fit_diagnostics="none",
                exp_fit_diag_dir=None,
                exp_fit_diag_budget=None,
                exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
                exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
                exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
                exp_fit_diag_include=args.exp_fit_diagnostics_include,
                detector_name="Other",
                segment_label=f"{args.tag}_full_other",
                baseline_method=args.baseline_method,
                median_kernel=args.median_kernel,
                asls_lam=args.asls_lam,
                asls_p=args.asls_p,
                asls_niter=args.asls_niter,
            )
            spikes, _other_only, coincident = _exclusive_spikes_by_time(
                spikes,
                other_spikes_full,
                coincidence_window_s=float(args.coincidence_window_s),
            )
            print(
                f"{detector_name}: removed {len(coincident)} coincident spikes; keeping {len(spikes)} exclusive spikes"
            )

        if not args.exclusive_spikes:
            print(f"{detector_name}: detected {len(spikes)} spikes")

        stable_path = Path(args.stable_summary_csv)
        stable_intervals = _read_stable_voltage_summary(stable_path)
        data_start = current_a.index.min()
        data_end = current_a.index.max()
        stable_intervals = stable_intervals[(stable_intervals["end"] > data_start) & (stable_intervals["start"] < data_end)].copy()

        if args.baseline_plots == "day":
            for day_start in pd.date_range(current_a.index.min().normalize(), current_a.index.max().normalize(), freq="1D"):
                day_end = day_start + pd.Timedelta(days=1)
                day_series = current_a[(current_a.index >= day_start) & (current_a.index < day_end)]
                if len(day_series) < 10:
                    continue

                day_times = pd.DatetimeIndex(day_series.index).to_pydatetime()
                day_currents_uA = day_series.to_numpy(dtype=float) * 1e6

                # Detect spikes on the cropped day window (this is what gets plotted).
                day_spikes, day_baseline = _detect_spikes_with_savgol(
                    day_times,
                    day_currents_uA,
                    threshold_uA=args.threshold_uA,
                    window_length=args.window_length,
                    polyorder=args.polyorder,
                    baseline_method=args.baseline_method,
                    median_kernel=args.median_kernel,
                    asls_lam=args.asls_lam,
                    asls_p=args.asls_p,
                    asls_niter=args.asls_niter,
                    charge_method=args.charge_method,
                    tau_fixed_s=args.tau_fixed_s,
                    exp_fit_min_points=args.exp_fit_min_points,
                    exp_fit_max_window_s=args.exp_fit_max_window_s,
                    exp_fit_frac_min=args.exp_fit_frac_min,
                    exp_fit_tau_min_s=args.exp_fit_tau_min_s,
                    exp_fit_tau_max_s=args.exp_fit_tau_max_s,
                    exp_fit_diagnostics=(args.exp_fit_diagnostics if diag_on_day else "none"),
                    exp_fit_diag_dir=exp_fit_diag_dir,
                    exp_fit_diag_budget=exp_fit_diag_budget,
                    exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
                    exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
                    exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
                    exp_fit_diag_include=args.exp_fit_diagnostics_include,
                    detector_name=detector_name,
                    segment_label=f"{args.tag}_{day_start.strftime('%Y%m%d')}",
                )

                if args.exclusive_spikes and other_current_a is not None:
                    other_day = other_current_a[(other_current_a.index >= day_start) & (other_current_a.index < day_end)]
                    other_day_times = pd.DatetimeIndex(other_day.index).to_pydatetime()
                    other_day_currents_uA = other_day.to_numpy(dtype=float) * 1e6
                    other_day_spikes, _ = _detect_spikes_with_savgol(
                        other_day_times,
                        other_day_currents_uA,
                        threshold_uA=args.threshold_uA,
                        window_length=args.window_length,
                        polyorder=args.polyorder,
                        baseline_method=args.baseline_method,
                        median_kernel=args.median_kernel,
                        asls_lam=args.asls_lam,
                        asls_p=args.asls_p,
                        asls_niter=args.asls_niter,
                        charge_method=args.charge_method,
                        tau_fixed_s=args.tau_fixed_s,
                        exp_fit_min_points=args.exp_fit_min_points,
                        exp_fit_max_window_s=args.exp_fit_max_window_s,
                        exp_fit_frac_min=args.exp_fit_frac_min,
                        exp_fit_tau_min_s=args.exp_fit_tau_min_s,
                        exp_fit_tau_max_s=args.exp_fit_tau_max_s,
                        exp_fit_diagnostics="none",
                        exp_fit_diag_dir=None,
                        exp_fit_diag_budget=None,
                        exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
                        exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
                        exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
                        exp_fit_diag_include=args.exp_fit_diagnostics_include,
                        detector_name="Other",
                        segment_label=f"{args.tag}_{day_start.strftime('%Y%m%d')}_other",
                    )
                    day_spikes, _other_only, _ = _exclusive_spikes_by_time(
                        day_spikes,
                        other_day_spikes,
                        coincidence_window_s=float(args.coincidence_window_s),
                    )

                vertical_lines = []
                annotations = []
                if stable_intervals is not None and not stable_intervals.empty:
                    day_intervals = stable_intervals[(stable_intervals["end"] > day_start) & (stable_intervals["start"] < day_end)].copy()
                    for s_raw, e_raw, mean_E_raw in day_intervals[["start", "end", "mean_E_V_per_cm"]].itertuples(index=False, name=None):
                        s = pd.Timestamp(s_raw)
                        e = pd.Timestamp(e_raw)
                        if day_start <= s <= day_end:
                            vertical_lines.append(s)
                        if day_start <= e <= day_end:
                            vertical_lines.append(e)
                        left = max(s, day_start)
                        right = min(e, day_end)
                        if right > left:
                            annotations.append((left + (right - left) / 2, f"{float(mean_E_raw):.0f} V/cm"))
                    vertical_lines = sorted(set(vertical_lines))

                save_path = None
                if args.save_plots:
                    out_dir = figures_root() / "analysis" / "hv_ramping_spike_study" / "plots_baseline_day"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    suffix = "_only" if args.exclusive_spikes else ""
                    filename = f"{args.tag}_{detector_name.lower()}{suffix}_baseline_{day_start.strftime('%Y%m%d')}.png"
                    save_path = out_dir / filename

                _plot_spikes_with_baseline(
                    day_times,
                    day_currents_uA,
                    day_spikes,
                    day_baseline,
                    title=f"Detected Spikes with Rolling Baseline ({detector_name}; {day_start.strftime('%Y-%m-%d')})",
                    save_path=save_path,
                    show_plot=args.show_baseline_plots,
                    vertical_lines=vertical_lines,
                    top_annotations=annotations,
                )

        if args.baseline_plots == "plateau" and stable_intervals is not None and not stable_intervals.empty:
            out_dir = figures_root() / "analysis" / "hv_ramping_spike_study" / "plots_baseline_plateau"
            if args.save_plots:
                out_dir.mkdir(parents=True, exist_ok=True)

            for left_raw, right_raw, mean_E_raw in stable_intervals[["start", "end", "mean_E_V_per_cm"]].itertuples(index=False, name=None):
                left = pd.Timestamp(left_raw)
                right = pd.Timestamp(right_raw)
                seg = current_a[(current_a.index >= left) & (current_a.index < right)]
                if len(seg) < 10:
                    continue

                seg_times = pd.DatetimeIndex(seg.index).to_pydatetime()
                seg_currents_uA = seg.to_numpy(dtype=float) * 1e6

                # Detect spikes on the cropped plateau window (this is what gets plotted).
                seg_spikes, seg_baseline = _detect_spikes_with_savgol_segmented(
                    seg_times,
                    seg_currents_uA,
                    threshold_uA=args.threshold_uA,
                    window_length=args.window_length,
                    polyorder=args.polyorder,
                    charge_method=args.charge_method,
                    tau_fixed_s=args.tau_fixed_s,
                    exp_fit_min_points=args.exp_fit_min_points,
                    exp_fit_max_window_s=args.exp_fit_max_window_s,
                    exp_fit_frac_min=args.exp_fit_frac_min,
                    exp_fit_tau_min_s=args.exp_fit_tau_min_s,
                    exp_fit_tau_max_s=args.exp_fit_tau_max_s,
                    exp_fit_diagnostics=(args.exp_fit_diagnostics if diag_on_plateau else "none"),
                    exp_fit_diag_dir=exp_fit_diag_dir,
                    exp_fit_diag_budget=exp_fit_diag_budget,
                    exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
                    exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
                    exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
                    exp_fit_diag_include=args.exp_fit_diagnostics_include,
                    detector_name=detector_name,
                    segment_label=f"{args.tag}_plateau_{left.strftime('%Y%m%d_%H%M')}",
                    baseline_method=args.baseline_method,
                    median_kernel=args.median_kernel,
                    asls_lam=args.asls_lam,
                    asls_p=args.asls_p,
                    asls_niter=args.asls_niter,
                )

                if args.exclusive_spikes and other_current_a is not None:
                    other_seg = other_current_a[(other_current_a.index >= left) & (other_current_a.index < right)]
                    other_seg_times = pd.DatetimeIndex(other_seg.index).to_pydatetime()
                    other_seg_currents_uA = other_seg.to_numpy(dtype=float) * 1e6
                    other_seg_spikes, _ = _detect_spikes_with_savgol_segmented(
                        other_seg_times,
                        other_seg_currents_uA,
                        threshold_uA=args.threshold_uA,
                        window_length=args.window_length,
                        polyorder=args.polyorder,
                        charge_method=args.charge_method,
                        tau_fixed_s=args.tau_fixed_s,
                        exp_fit_min_points=args.exp_fit_min_points,
                        exp_fit_max_window_s=args.exp_fit_max_window_s,
                        exp_fit_frac_min=args.exp_fit_frac_min,
                        exp_fit_tau_min_s=args.exp_fit_tau_min_s,
                        exp_fit_tau_max_s=args.exp_fit_tau_max_s,
                        exp_fit_diagnostics="none",
                        exp_fit_diag_dir=None,
                        exp_fit_diag_budget=None,
                        exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
                        exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
                        exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
                        exp_fit_diag_include=args.exp_fit_diagnostics_include,
                        detector_name="Other",
                        segment_label=f"{args.tag}_plateau_{left.strftime('%Y%m%d_%H%M')}_other",
                        baseline_method=args.baseline_method,
                        median_kernel=args.median_kernel,
                        asls_lam=args.asls_lam,
                        asls_p=args.asls_p,
                        asls_niter=args.asls_niter,
                    )
                    seg_spikes, _other_only, _ = _exclusive_spikes_by_time(
                        seg_spikes,
                        other_seg_spikes,
                        coincidence_window_s=float(args.coincidence_window_s),
                    )

                mean_E_val = float(mean_E_raw)
                vertical_lines = [left, right]
                annotations = [(left + (right - left) / 2, f"{mean_E_val:.0f} V/cm")]

                save_path = None
                if args.save_plots:
                    e_tag = f"E{mean_E_val:.0f}"
                    suffix = "_only" if args.exclusive_spikes else ""
                    filename = f"{args.tag}_{detector_name.lower()}{suffix}_baseline_{e_tag}_{left.strftime('%Y%m%d_%H%M')}.png"
                    save_path = out_dir / filename

                _plot_spikes_with_baseline(
                    seg_times,
                    seg_currents_uA,
                    seg_spikes,
                    seg_baseline,
                    title=(
                        f"Detected Spikes with Rolling Baseline ({detector_name}; {mean_E_val:.0f} V/cm)"
                    ),
                    save_path=save_path,
                    show_plot=args.show_baseline_plots,
                    vertical_lines=vertical_lines,
                    top_annotations=annotations,
                )

        vertical_lines: list[pd.Timestamp] | None = None
        vertical_lines_colored: list[tuple[pd.Timestamp, str]] | None = None
        spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] | None = None
        span_labels: list[tuple[pd.Timestamp, str, str]] | None = None
        rate = None
        rate_ylabel = "Spikes / hour"
        mean_E: np.ndarray | None = None
        avg_voltage_kV: np.ndarray | None = None
        err_rate: np.ndarray | None = None

        intervals = stable_intervals
        diag_on_plateau_for_summary = (args.exp_fit_diagnostics != "none") and (
            args.exp_fit_diagnostics_scope in {"plateau", "both"}
        )
        (
            times_mid,
            n_spikes_plateau,
            avg_charge_uC,
            err_charge_uC,
            rate_per_hour,
            err_rate,
            mean_E,
            avg_voltage_kV,
        ) = _summarize_plateaus_from_window_detection(
            current_a,
            other_current_a,
            intervals,
            exclusive=bool(args.exclusive_spikes),
            coincidence_window_s=float(args.coincidence_window_s),
            threshold_uA=args.threshold_uA,
            window_length=args.window_length,
            polyorder=args.polyorder,
            baseline_method=args.baseline_method,
            median_kernel=args.median_kernel,
            asls_lam=args.asls_lam,
            asls_p=args.asls_p,
            asls_niter=args.asls_niter,
            charge_method=args.charge_method,
            tau_fixed_s=args.tau_fixed_s,
            exp_fit_min_points=args.exp_fit_min_points,
            exp_fit_max_window_s=args.exp_fit_max_window_s,
            exp_fit_frac_min=args.exp_fit_frac_min,
            exp_fit_tau_min_s=args.exp_fit_tau_min_s,
            exp_fit_tau_max_s=args.exp_fit_tau_max_s,
            exp_fit_diagnostics=(args.exp_fit_diagnostics if diag_on_plateau_for_summary else "none"),
            exp_fit_diag_dir=exp_fit_diag_dir,
            exp_fit_diag_budget=exp_fit_diag_budget,
            exp_fit_diag_min_mag_uA=float(exp_fit_diag_min_mag),
            exp_fit_diag_pre_s=args.exp_fit_diagnostics_pre_s,
            exp_fit_diag_post_s=args.exp_fit_diagnostics_post_s,
            exp_fit_diag_include=args.exp_fit_diagnostics_include,
            detector_name=detector_name,
            tag=args.tag,
        )

        rate = rate_per_hour
        err_rate_out = err_rate
        rate_ylabel = "Spikes / hour"
        # Draw a vline at each plateau boundary (start times), like the October study regime markers.
        vertical_lines = [pd.Timestamp(t) for t in intervals["start"].to_list()]
        # Also include the final end boundary for visual closure.
        if not intervals.empty:
            vertical_lines.append(pd.Timestamp(intervals["end"].iloc[-1]))

        # Color-code plateau regions and annotate avg HV + mean E between boundaries.
        spans = []
        span_labels = []
        vertical_lines_colored = []
        for i, (s_raw, e_raw, mean_e_raw, v_raw) in enumerate(
            intervals[["start", "end", "mean_E_V_per_cm", "avg_voltage_kV"]].itertuples(index=False, name=None)
        ):
            s = pd.Timestamp(s_raw)
            e = pd.Timestamp(e_raw)
            if e <= s:
                continue
            color = "tab:blue" if (i % 2 == 0) else "tab:orange"
            spans.append((s, e, color))
            vertical_lines_colored.append((s, color))
            mid = s + (e - s) / 2
            label = f"{float(v_raw):.1f} kV\n{float(mean_e_raw):.0f} V/cm"
            span_labels.append((mid, label, color))
        if not intervals.empty:
            last_end = pd.Timestamp(intervals["end"].iloc[-1])
            last_color = "tab:blue" if ((len(intervals) - 1) % 2 == 0) else "tab:orange"
            vertical_lines_colored.append((last_end, last_color))

        title_suffix = "; binned by HV plateau (mean E shown in HV summary CSV)"

        analysis_dir = figures_root() / "analysis" / "hv_ramping_spike_study"
        # Always produce a time-axis summary plot.
        suffix = "" if not args.exclusive_spikes else "_only"
        save_path_time = (
            (analysis_dir / f"summary_time_{args.tag}_{detector_name.lower()}{suffix}.png") if args.save_plots else None
        )
        _plot_summary(
            times_mid,
            avg_charge_uC,
            err_charge_uC,
            rate,
            err_rate_out,
            rate_ylabel=rate_ylabel,
            title=(
                f"HV ramping spike summary ({detector_name}{' only' if args.exclusive_spikes else ''} detector; {args.tag})"
                f"{title_suffix}"
            ),
            save_path=save_path_time,
            show_plot=args.show_plots,
            vertical_lines=vertical_lines_colored if vertical_lines_colored else vertical_lines,
            spans=spans,
            span_labels=span_labels,
        )

        # And a second summary vs calibrated HV value (avg_voltage_kV from the plateau summary CSV).
        if avg_voltage_kV is None:
            raise RuntimeError("Internal error: avg_voltage_kV not computed for plateau binning")
        save_path_v = (
            (analysis_dir / f"summary_voltage_{args.tag}_{detector_name.lower()}{suffix}.png") if args.save_plots else None
        )
        _plot_summary_vs_voltage_kV(
            avg_voltage_kV,
            avg_charge_uC,
            err_charge_uC,
            rate,
            err_rate_out,
            title=f"HV ramping spike summary vs avg HV ({detector_name}{' only' if args.exclusive_spikes else ''}; {args.tag})",
            save_path=save_path_v,
            show_plot=args.show_plots,
        )

        if args.write_timestamps:
            ts_suffix = "" if not args.exclusive_spikes else "_only"
            ts_path = spike_timestamps_path(f"spike_timestamps_{args.tag}_{detector_name.lower()}{ts_suffix}.txt")
            _write_spike_timestamps(
                ts_path,
                spikes,
                charge_method=args.charge_method,
                tau_fixed_s=args.tau_fixed_s,
                exp_fit_min_points=args.exp_fit_min_points,
                exp_fit_max_window_s=args.exp_fit_max_window_s,
                exp_fit_frac_min=args.exp_fit_frac_min,
                exp_fit_tau_min_s=args.exp_fit_tau_min_s,
                exp_fit_tau_max_s=args.exp_fit_tau_max_s,
                threshold_uA=args.threshold_uA,
                window_length=args.window_length,
                polyorder=args.polyorder,
                baseline_method=args.baseline_method,
                median_kernel=args.median_kernel,
                asls_lam=args.asls_lam,
                asls_p=args.asls_p,
                asls_niter=args.asls_niter,
            )

        # If diagnostics were requested but none were produced, call out the most common cause:
        # accepted-only + tau window can filter everything out.
        if args.exp_fit_diagnostics != "none" and exp_fit_diag_budget is not None:
            created_after = int(exp_fit_diag_budget.get("created", 0))
            created_here = created_after - created_before
            if created_here <= 0:
                print(
                    "NOTE: exp-fit diagnostics produced 0 plots for this detector. "
                    "Most commonly this happens because --exp-fit-diagnostics-include=accepted filters out all fits "
                    "(e.g., tau outside [--exp-fit-tau-min-s, --exp-fit-tau-max-s] or too few fit points). "
                    "Try: --exp-fit-diagnostics-include all"
                )

    top_current_a = load_detector_series(TOP)
    bottom_current_a = load_detector_series(BOTTOM)

    if args.plot_cathode_overlay:
        if args.hv_ramp is None:
            hv_paths = default_hv_paths
        else:
            hv_paths = [Path(p).expanduser() for p in args.hv_ramp if str(p).strip() != ""]

        missing = [p for p in hv_paths if not p.exists()]
        if missing:
            print("Skipping missing HV ramp files for cathode overlay:")
            for p in missing:
                print(f"  - {p}")
        hv_paths = [p for p in hv_paths if p.exists()]

        if not hv_paths:
            print("No HV ramp CSVs provided for cathode overlay plot.")
        else:
            voltage_kv = _read_hv_ramp_voltage_kv(hv_paths)
            if not voltage_kv.empty:
                voltage_df = _apply_exclusions(voltage_kv.to_frame("voltage_kv"), EXCLUDED_TIME_INTERVALS)
                voltage_kv = voltage_df["voltage_kv"]

            top_current_uA = (top_current_a * 1e6).dropna()
            bottom_current_uA = (bottom_current_a * 1e6).dropna()

            analysis_dir = figures_root() / "analysis" / "hv_ramping_spike_study"
            tag_suffix = f"_{args.tag}" if args.tag else ""
            save_path = (analysis_dir / f"cathode_voltage_top_bottom_current{tag_suffix}.png") if args.save_plots else None
            title = (
                f"Cathode voltage and termination currents"
                f"{(' (' + args.tag + ')') if args.tag else ''}"
            )
            _plot_cathode_voltage_and_termination_currents(
                voltage_kv,
                top_current_uA,
                bottom_current_uA,
                title=title,
                save_path=save_path,
                show_plot=args.show_plots,
            )

    analyze("Top", top_current_a, (bottom_current_a if args.exclusive_spikes else None))
    analyze("Bottom", bottom_current_a, (top_current_a if args.exclusive_spikes else None))


if __name__ == "__main__":
    main()
