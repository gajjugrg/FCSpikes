"""Find transient HV voltage drops ("spikes") during HV ramping.

Goal
----
Produce timestamp windows like:
    (pd.Timestamp("2025-12-05 11:28"), pd.Timestamp("2025-12-05 11:30")),
for transient HV voltage drops of at least some threshold (default: 50 kV).

Input
-----
HV ramp CSVs exported by DCS, with two header lines and at least 3 columns:
  time, voltage[V], current[µA]

By default this script scans:
  <data-root>/csv/ramping/HV_ramp_*.csv

Outputs
-------
- Prints Python-ready window tuples to stdout.
- Optionally writes a CSV report and/or saves diagnostic plots.

Notes
-----
Detection approach:
- Build a smooth baseline voltage (Savitzky–Golay on time-sorted data)
- Compute residual = baseline - measured_voltage
- Find peaks in residual with height >= threshold_kV
- Convert each peak to a (start, end) window by expanding to where residual
  drops below a fraction of the threshold.

This catches fast negative excursions in voltage, even on a slow ramp.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # When run as: python src/find_hv_ramping_voltage_drops.py
    from utilities import fcspikes_root, figures_root
except ModuleNotFoundError:  # pragma: no cover
    # When run as: python -m src.find_hv_ramping_voltage_drops
    from src.utilities import fcspikes_root, figures_root


TIME_FMT = "%Y/%m/%d %H:%M:%S.%f"


@dataclass(frozen=True)
class DropEvent:
    peak_time: datetime
    start_time: datetime
    end_time: datetime
    drop_kv: float
    baseline_kv: float
    min_voltage_kv: float


def _configure_matplotlib_backend(*, show: bool) -> None:
    # Local-only import to avoid forcing matplotlib dependency for non-plot usage.
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
        return

    # Try interactive backends.
    for backend in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue

    matplotlib.use("Agg", force=True)


def read_hv_ramp_csv(path: Path, *, stride: int = 1) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Read HV ramp CSV exported by DCS (two header lines).

    Returns:
      times: list[datetime]
      voltage_kv: numpy array [kV]
      current_uA: numpy array [µA]
    """

    if stride < 1:
        raise ValueError("stride must be >= 1")

    # Fast path: vectorized parsing via pandas (significantly faster than per-row strptime).
    try:
        df = pd.read_csv(
            path,
            skiprows=2,
            header=None,
            usecols=[0, 1, 2],
            names=["time", "voltage_v", "current_uA"],
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
        if stride > 1:
            df = df.iloc[:: int(stride)].copy()

        t = pd.to_datetime(df["time"], format=TIME_FMT, errors="coerce")
        v = pd.to_numeric(df["voltage_v"], errors="coerce")
        c = pd.to_numeric(df["current_uA"], errors="coerce")
        mask = t.notna() & v.notna() & c.notna()

        times = [x.to_pydatetime() for x in t[mask].tolist()]
        voltage_kv = (v[mask].to_numpy(dtype=float) / 1000.0).astype(float)
        current_uA = c[mask].to_numpy(dtype=float).astype(float)
        return times, voltage_kv, current_uA
    except Exception:
        # Fallback: permissive csv.reader (slower but robust).
        times: list[datetime] = []
        voltage_v: list[float] = []
        current_ua: list[float] = []

        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            import csv

            reader = csv.reader(f)
            next(reader, None)
            next(reader, None)

            for i, row in enumerate(reader):
                if not row or len(row) < 3:
                    continue
                if stride > 1 and (i % stride) != 0:
                    continue
                try:
                    tt = datetime.strptime(row[0].strip(), TIME_FMT)
                    vv = float(row[1])
                    cc = float(row[2])
                except Exception:
                    continue
                times.append(tt)
                voltage_v.append(vv)
                current_ua.append(cc)

        voltage_kv = np.asarray(voltage_v, dtype=float) / 1000.0
        current_uA = np.asarray(current_ua, dtype=float)
        return times, voltage_kv, current_uA


def read_multiple_hv_ramps(paths: list[Path], *, stride: int = 1) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    all_times: list[datetime] = []
    all_v: list[float] = []
    all_i: list[float] = []

    for path in paths:
        times, voltage_kv, current_uA = read_hv_ramp_csv(path, stride=stride)
        all_times.extend(times)
        all_v.extend(voltage_kv.tolist())
        all_i.extend(current_uA.tolist())

    if not all_times:
        return [], np.array([]), np.array([])

    order = np.argsort(np.asarray([t.timestamp() for t in all_times], dtype=float))
    sorted_times = [all_times[int(k)] for k in order]
    sorted_v = np.asarray([all_v[int(k)] for k in order], dtype=float)
    sorted_i = np.asarray([all_i[int(k)] for k in order], dtype=float)

    # Drop duplicates / non-monotonic points (keep first occurrence).
    ts = np.asarray([t.timestamp() for t in sorted_times], dtype=float)
    keep = np.ones_like(ts, dtype=bool)
    keep[1:] = ts[1:] > ts[:-1]
    sorted_times = [t for t, k in zip(sorted_times, keep.tolist(), strict=False) if k]
    sorted_v = sorted_v[keep]
    sorted_i = sorted_i[keep]
    return sorted_times, sorted_v, sorted_i


def _median_dt_seconds(times: list[datetime]) -> float:
    if len(times) < 3:
        return float("nan")
    dt = np.diff(np.asarray([t.timestamp() for t in times], dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return float(np.median(dt))


def _savgol_window_points(*, dt_s: float, window_s: float, n: int, max_points: int | None = None) -> int:
    if not np.isfinite(dt_s) or dt_s <= 0:
        # Fallback: 101-point window.
        win = 101
    else:
        win = int(round(float(window_s) / float(dt_s)))
        win = max(5, win)
    if win % 2 == 0:
        win += 1
    if max_points is not None:
        cap = int(max(5, max_points))
        if cap % 2 == 0:
            cap -= 1
        win = min(win, cap)
    win = min(win, n if (n % 2 == 1) else (n - 1))
    return max(5, win)


def find_transient_voltage_drops(
    times: list[datetime],
    voltage_kv: np.ndarray,
    *,
    threshold_kv: float,
    baseline_window_s: float = 60.0,
    baseline_max_points: int = 5001,
    baseline_polyorder: int = 2,
    edge_fraction: float = 0.25,
    max_duration_s: float = 20 * 60.0,
    min_separation_s: float = 30.0,
) -> tuple[list[DropEvent], np.ndarray, np.ndarray]:
    """Detect transient drops and return (events, baseline_kv, residual_kv)."""

    v = np.asarray(voltage_kv, dtype=float)
    if len(times) != v.size:
        raise ValueError("times and voltage_kv must have the same length")
    if v.size < 10:
        return [], np.array([]), np.array([])
    if threshold_kv <= 0:
        raise ValueError("threshold_kv must be > 0")

    from scipy.signal import savgol_filter, find_peaks

    dt_s = _median_dt_seconds(times)
    window_pts = _savgol_window_points(
        dt_s=dt_s,
        window_s=baseline_window_s,
        n=v.size,
        max_points=int(baseline_max_points),
    )
    poly = int(max(1, min(int(baseline_polyorder), window_pts - 2)))

    baseline = savgol_filter(v, window_length=window_pts, polyorder=poly, mode="interp")
    residual = baseline - v  # positive when v drops below baseline

    if not np.any(np.isfinite(residual)):
        return [], baseline, residual

    min_sep_pts = 1
    if np.isfinite(dt_s) and dt_s > 0:
        min_sep_pts = max(1, int(round(float(min_separation_s) / float(dt_s))))

    peaks, props = find_peaks(residual, height=float(threshold_kv), distance=min_sep_pts)
    if peaks.size == 0:
        return [], baseline, residual

    thr_edge = float(threshold_kv) * float(edge_fraction)
    max_pts = None
    if np.isfinite(dt_s) and dt_s > 0:
        max_pts = int(round(float(max_duration_s) / float(dt_s)))

    events: list[DropEvent] = []
    for p in peaks.tolist():
        p = int(p)
        peak_r = float(residual[p])
        if not np.isfinite(peak_r):
            continue

        left = p
        steps = 0
        while left > 0 and float(residual[left]) >= thr_edge:
            left -= 1
            steps += 1
            if max_pts is not None and steps > max_pts:
                break

        right = p
        steps = 0
        while right < (v.size - 1) and float(residual[right]) >= thr_edge:
            right += 1
            steps += 1
            if max_pts is not None and steps > max_pts:
                break

        # Duration filter: ensure it recovers reasonably fast.
        start_t = times[left]
        end_t = times[right]
        duration_s = (end_t - start_t).total_seconds()
        if duration_s <= 0:
            continue
        if duration_s > float(max_duration_s):
            continue

        seg_v = v[left : right + 1]
        seg_b = baseline[left : right + 1]
        if seg_v.size == 0 or seg_b.size == 0:
            continue

        min_v = float(np.nanmin(seg_v))
        # Baseline at the minimum voltage location (more stable than peak index).
        min_idx = int(np.nanargmin(seg_v))
        base_at_min = float(seg_b[min_idx])
        drop_kv = float(base_at_min - min_v)
        if not np.isfinite(drop_kv) or drop_kv < float(threshold_kv):
            continue

        events.append(
            DropEvent(
                peak_time=times[p],
                start_time=start_t,
                end_time=end_t,
                drop_kv=drop_kv,
                baseline_kv=base_at_min,
                min_voltage_kv=min_v,
            )
        )

    # Sort + merge overlapping/nearby windows.
    events.sort(key=lambda e: e.start_time)
    merged: list[DropEvent] = []
    merge_gap = timedelta(seconds=max(1.0, float(min_separation_s)))
    for ev in events:
        if not merged:
            merged.append(ev)
            continue
        last = merged[-1]
        if ev.start_time <= (last.end_time + merge_gap):
            # Merge: expand window, keep max drop.
            new_start = min(last.start_time, ev.start_time)
            new_end = max(last.end_time, ev.end_time)
            if ev.drop_kv >= last.drop_kv:
                merged[-1] = DropEvent(
                    peak_time=ev.peak_time,
                    start_time=new_start,
                    end_time=new_end,
                    drop_kv=ev.drop_kv,
                    baseline_kv=ev.baseline_kv,
                    min_voltage_kv=ev.min_voltage_kv,
                )
            else:
                merged[-1] = DropEvent(
                    peak_time=last.peak_time,
                    start_time=new_start,
                    end_time=new_end,
                    drop_kv=last.drop_kv,
                    baseline_kv=last.baseline_kv,
                    min_voltage_kv=last.min_voltage_kv,
                )
        else:
            merged.append(ev)

    return merged, baseline, residual


def _format_windows_python(events: list[DropEvent]) -> str:
    lines = []
    for ev in events:
        # Match your existing style: minute precision is usually enough.
        s = pd.Timestamp(ev.start_time).strftime("%Y-%m-%d %H:%M")
        e = pd.Timestamp(ev.end_time).strftime("%Y-%m-%d %H:%M")
        lines.append(f"(pd.Timestamp(\"{s}\"), pd.Timestamp(\"{e}\")),")
    return "\n".join(lines)


def _pad_and_round_window(
    start: datetime,
    end: datetime,
    *,
    pad_s: float,
    min_window_s: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    pad = pd.Timedelta(seconds=float(max(0.0, pad_s)))
    s = pd.Timestamp(start) - pad
    e = pd.Timestamp(end) + pad

    # Round outward to full minutes so pasteable windows look like the existing lists.
    s = s.floor("min")
    e = e.ceil("min")

    min_window = pd.Timedelta(seconds=float(max(0.0, min_window_s)))
    if e <= s:
        e = s + pd.Timedelta(minutes=1)
    if (e - s) < min_window:
        e = s + min_window
    return s, e


def _merge_windows(windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: (w[0].value, w[1].value))
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [windows[0]]
    for s, e in windows[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _write_csv(events: list[DropEvent], out_path: Path, *, pad_s: float, min_window_s: float) -> None:
    padded = [_pad_and_round_window(e.start_time, e.end_time, pad_s=pad_s, min_window_s=min_window_s) for e in events]
    merged = _merge_windows(padded)
    df = pd.DataFrame(
        {
            "start": [p[0] for p in padded],
            "end": [p[1] for p in padded],
            "start_merged": [m[0] for m in merged] + [pd.NaT] * max(0, len(padded) - len(merged)),
            "end_merged": [m[1] for m in merged] + [pd.NaT] * max(0, len(padded) - len(merged)),
            "start_raw": [pd.Timestamp(e.start_time) for e in events],
            "end_raw": [pd.Timestamp(e.end_time) for e in events],
            "peak": [pd.Timestamp(e.peak_time) for e in events],
            "drop_kV": [e.drop_kv for e in events],
            "baseline_kV": [e.baseline_kv for e in events],
            "min_voltage_kV": [e.min_voltage_kv for e in events],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _plot_events(
    *,
    times: list[datetime],
    voltage_kv: np.ndarray,
    baseline_kv: np.ndarray,
    residual_kv: np.ndarray,
    events: list[DropEvent],
    out_dir: Path,
    show: bool,
    tag: str,
) -> None:
    _configure_matplotlib_backend(show=show)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax_v.plot(times, voltage_kv, lw=1, color="tab:blue", label="Voltage")
    ax_v.plot(times, baseline_kv, lw=1, color="tab:gray", alpha=0.8, label="Baseline")
    ax_r.plot(times, residual_kv, lw=1, color="tab:orange", label="Residual (baseline - voltage)")

    for ev in events:
        ax_v.axvspan(ev.start_time, ev.end_time, color="tab:red", alpha=0.15)
        ax_r.axvspan(ev.start_time, ev.end_time, color="tab:red", alpha=0.15)
        ax_v.axvline(ev.peak_time, color="tab:red", lw=0.8, alpha=0.7)

    ax_v.set_ylabel("Voltage [kV]")
    ax_r.set_ylabel("Residual [kV]")
    ax_r.set_xlabel("Time")

    for ax in (ax_v, ax_r):
        ax.grid(True, alpha=0.25)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax_r.xaxis.set_major_locator(locator)
    ax_r.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=30)

    ax_v.legend(loc="best")
    ax_r.legend(loc="best")

    fig.tight_layout()
    out_path = out_dir / f"hv_voltage_drops_{tag}.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _plot_overlay_events(
    *,
    times: list[datetime],
    voltage_kv: np.ndarray,
    baseline_kv: np.ndarray,
    residual_kv: np.ndarray,
    events: list[DropEvent],
    out_dir: Path,
    show: bool,
    tag: str,
    pre_s: float,
    post_s: float,
) -> None:
    """Overlay all drop shapes on a common relative-time axis.

    Uses the drop peak time as t=0 and plots data in [t0-pre_s, t0+post_s].
    """

    if not events:
        return

    _configure_matplotlib_backend(show=show)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    tsec = np.asarray([t.timestamp() for t in times], dtype=float)
    v = np.asarray(voltage_kv, dtype=float)
    b = np.asarray(baseline_kv, dtype=float)
    r = np.asarray(residual_kv, dtype=float)

    # Common relative-time grid (seconds) for clean overlays.
    dt_s = _median_dt_seconds(times)
    if not np.isfinite(dt_s) or dt_s <= 0:
        dt_s = 1.0
    grid_s = np.arange(-float(pre_s), float(post_s) + 1e-9, float(dt_s))
    grid_min = grid_s / 60.0

    fig, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ev in events:
        t0 = float(ev.peak_time.timestamp())
        lo = t0 - float(pre_s)
        hi = t0 + float(post_s)
        i0 = int(np.searchsorted(tsec, lo, side="left"))
        i1 = int(np.searchsorted(tsec, hi, side="right"))
        if i1 <= i0 + 2:
            continue

        seg_t = tsec[i0:i1] - t0
        seg_r = r[i0:i1]
        seg_v = v[i0:i1]
        seg_b = b[i0:i1]

        # Interpolate onto common grid; mask outside interpolation range.
        r_grid = np.interp(grid_s, seg_t, seg_r, left=np.nan, right=np.nan)
        v_grid = np.interp(grid_s, seg_t, seg_v, left=np.nan, right=np.nan)
        b_grid = np.interp(grid_s, seg_t, seg_b, left=np.nan, right=np.nan)

        label = pd.Timestamp(ev.peak_time).strftime("%Y-%m-%d %H:%M")
        ax_r.plot(grid_min, r_grid, lw=1, alpha=0.8, label=label)
        ax_v.plot(grid_min, v_grid, lw=1, alpha=0.8)
        ax_v.plot(grid_min, b_grid, lw=1, alpha=0.35, color="black")

    ax_r.axvline(0.0, color="black", lw=0.8, alpha=0.6)
    ax_v.axvline(0.0, color="black", lw=0.8, alpha=0.6)

    ax_r.set_ylabel("Residual [kV] (baseline - voltage)")
    ax_v.set_ylabel("Voltage [kV]")
    ax_v.set_xlabel("Time relative to drop peak [min]")

    for ax in (ax_r, ax_v):
        ax.grid(True, alpha=0.25)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    # Keep legend readable.
    if len(events) <= 12:
        ax_r.legend(loc="best", fontsize=8)
    else:
        ax_r.legend(loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    out_path = out_dir / f"hv_voltage_drops_overlay_{tag}_m{int(round(pre_s/60))}_p{int(round(post_s/60))}.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _plot_snips(
    *,
    times: list[datetime],
    voltage_kv: np.ndarray,
    events: list[DropEvent],
    out_dir: Path,
    show: bool,
    tag: str,
    pre_s: float,
    post_s: float,
    separate_files: bool,
) -> None:
    """Plot voltage-only zoomed views around each transient drop (no overlay, no residual).

    Each snip is shown on a relative-time axis with t=0 at the drop peak.
    """

    if not events:
        return

    _configure_matplotlib_backend(show=show)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    tsec = np.asarray([t.timestamp() for t in times], dtype=float)
    v = np.asarray(voltage_kv, dtype=float)

    def _plot_one(ax, *, ev: DropEvent) -> None:
        t0 = float(ev.peak_time.timestamp())
        lo = t0 - float(pre_s)
        hi = t0 + float(post_s)
        i0 = int(np.searchsorted(tsec, lo, side="left"))
        i1 = int(np.searchsorted(tsec, hi, side="right"))
        if i1 <= i0 + 2:
            return
        seg_t_min = (tsec[i0:i1] - t0) / 60.0
        seg_v = v[i0:i1]

        ax.plot(seg_t_min, seg_v, lw=1, color="tab:blue")
        ax.axvline(0.0, color="black", lw=0.8, alpha=0.6)
        label = pd.Timestamp(ev.peak_time).strftime("%Y-%m-%d %H:%M")
        ax.set_title(f"{label}  (drop \u2248 {ev.drop_kv:.1f} kV)", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    if separate_files:
        for ev in events:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            _plot_one(ax, ev=ev)
            ax.set_xlabel("Time relative to drop peak [min]")
            ax.set_ylabel("Voltage [kV]")
            fig.tight_layout()
            stamp = pd.Timestamp(ev.peak_time).strftime("%Y%m%d_%H%M")
            out_path = out_dir / f"hv_voltage_drop_snip_{tag}_{stamp}_m{int(round(pre_s/60))}_p{int(round(post_s/60))}.png"
            fig.savefig(out_path, dpi=200)
            if show:
                plt.show()
            plt.close(fig)
        return

    # Single stacked figure (one subplot per event).
    n = len(events)
    fig_h = max(5.0, 2.0 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ev in zip(axes, events, strict=False):
        _plot_one(ax, ev=ev)
        ax.set_ylabel("kV")

    axes[-1].set_xlabel("Time relative to drop peak [min]")
    fig.suptitle(f"Transient HV voltage drops (threshold {tag})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = out_dir / f"hv_voltage_drop_snips_{tag}_m{int(round(pre_s/60))}_p{int(round(post_s/60))}.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=str,
        default=None,
        help="HV ramp CSV(s). Default: data/csv/ramping/HV_ramp_*.csv",
    )
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth row (speed).")
    parser.add_argument("--threshold-kv", type=float, default=50.0, help="Min drop size to report [kV].")
    parser.add_argument("--baseline-window-s", type=float, default=60.0, help="Baseline smoothing window [s].")
    parser.add_argument(
        "--baseline-max-points",
        type=int,
        default=5001,
        help="Cap Savitzky–Golay window length in points (prevents very slow filters).",
    )
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.25,
        help="Window edges defined where residual falls below threshold*edge_fraction.",
    )
    parser.add_argument("--max-duration-s", type=float, default=20 * 60.0, help="Max transient duration [s].")
    parser.add_argument("--min-separation-s", type=float, default=30.0, help="Min separation between events [s].")
    parser.add_argument(
        "--pad-s",
        type=float,
        default=60.0,
        help="Pad each detected window by this many seconds on both sides before printing/writing.",
    )
    parser.add_argument(
        "--min-window-s",
        type=float,
        default=60.0,
        help="Ensure output window duration is at least this long [s].",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional CSV output path. Default: none.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a diagnostic plot under figures/analysis/ramping_hv/.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Save an overlay plot of all drops aligned at t=0 (peak).",
    )
    parser.add_argument(
        "--snips",
        action="store_true",
        help="Save non-overlay voltage-only zoom plots around each drop peak.",
    )
    parser.add_argument(
        "--snips-separate",
        action="store_true",
        help="With --snips, write one PNG per event instead of a stacked figure.",
    )
    parser.add_argument(
        "--overlay-pre-s",
        type=float,
        default=60.0,
        help="Seconds before peak for overlay/snips (default: 60 = 1 minute).",
    )
    parser.add_argument(
        "--overlay-post-s",
        type=float,
        default=300.0,
        help="Seconds after peak for overlay/snips (default: 300 = 5 minutes).",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window (implies --plot).")
    parser.add_argument(
        "--tag",
        type=str,
        default="auto",
        help="Filename tag for plots (default: auto).",
    )

    args = parser.parse_args()

    if args.inputs is None or len(args.inputs) == 0:
        ramp_dir = fcspikes_root() / "csv" / "ramping"
        paths = sorted(ramp_dir.glob("HV_ramp_*.csv"))
    else:
        paths = [Path(p).expanduser().resolve() for p in args.inputs]

    if not paths:
        raise SystemExit("No input files found.")

    times, voltage_kv, _current_uA = read_multiple_hv_ramps(paths, stride=int(args.stride))
    if not times:
        raise SystemExit("No rows parsed from inputs.")

    events, baseline, residual = find_transient_voltage_drops(
        times,
        voltage_kv,
        threshold_kv=float(args.threshold_kv),
        baseline_window_s=float(args.baseline_window_s),
        baseline_max_points=int(args.baseline_max_points),
        edge_fraction=float(args.edge_fraction),
        max_duration_s=float(args.max_duration_s),
        min_separation_s=float(args.min_separation_s),
    )

    print("# Detected transient HV drops (pasteable Python):")
    print("import pandas as pd")
    padded_windows = [
        _pad_and_round_window(
            ev.start_time,
            ev.end_time,
            pad_s=float(args.pad_s),
            min_window_s=float(args.min_window_s),
        )
        for ev in events
    ]
    for s, e in _merge_windows(padded_windows):
        linestr_s = s.strftime("%Y-%m-%d %H:%M")
        linestr_e = e.strftime("%Y-%m-%d %H:%M")
        print(f"(pd.Timestamp(\"{linestr_s}\"), pd.Timestamp(\"{linestr_e}\")),")

    if args.out_csv:
        _write_csv(
            events,
            Path(args.out_csv).expanduser().resolve(),
            pad_s=float(args.pad_s),
            min_window_s=float(args.min_window_s),
        )

    if args.plot or args.overlay or args.snips or args.show:
        out_dir = figures_root() / "analysis" / "ramping_hv"
        tag = str(args.tag)
        if tag == "auto":
            # Use date range from data.
            start = pd.Timestamp(times[0]).strftime("%Y%m%d")
            end = pd.Timestamp(times[-1]).strftime("%Y%m%d")
            tag = f"{start}_{end}_thr{int(round(float(args.threshold_kv)))}kV"
        if args.plot or args.show:
            _plot_events(
                times=times,
                voltage_kv=voltage_kv,
                baseline_kv=baseline,
                residual_kv=residual,
                events=events,
                out_dir=out_dir,
                show=bool(args.show),
                tag=tag,
            )
        if args.overlay or args.show:
            _plot_overlay_events(
                times=times,
                voltage_kv=voltage_kv,
                baseline_kv=baseline,
                residual_kv=residual,
                events=events,
                out_dir=out_dir,
                show=bool(args.show),
                tag=tag,
                pre_s=float(args.overlay_pre_s),
                post_s=float(args.overlay_post_s),
            )
        if args.snips:
            snip_dir = out_dir / "transient_drop_snips"
            _plot_snips(
                times=times,
                voltage_kv=voltage_kv,
                events=events,
                out_dir=snip_dir,
                show=bool(args.show),
                tag=tag,
                pre_s=float(args.overlay_pre_s),
                post_s=float(args.overlay_post_s),
                separate_files=bool(args.snips_separate),
            )


if __name__ == "__main__":
    main()
