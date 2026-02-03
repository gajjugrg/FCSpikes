from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utilities import fcspikes_root, figures_root

LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
FONT_WEIGHT = "bold"


def _read_conicalft_stability_csv(path: Path) -> pd.DataFrame:
    # This CSV has 2 header rows:
    #  - Row 0: EPICS-ish PV names
    #  - Row 1: Human-friendly column names with units
    # Data begins at row 2.
    df = pd.read_csv(path, skiprows=[0])

    # Trailing commas often create an extra unnamed empty column.
    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]].copy()

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns, got {df.shape[1]}: {list(df.columns)}")

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.rename(columns={time_col: "time"})
    return df


def _downsample_evenly(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_minutes = int(round(seconds / 60.0))
    days, rem_minutes = divmod(total_minutes, 60 * 24)
    hours, minutes = divmod(rem_minutes, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours:02d}h")
    parts.append(f"{minutes:02d}m")
    return " ".join(parts)


def _find_longest_voltage_plateau(
    df: pd.DataFrame,
    voltage_column: str,
    dv_threshold: float,
    min_duration_seconds: float,
) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    if df.empty:
        return None

    d = df.loc[:, ["time", voltage_column]].dropna().sort_values("time").reset_index(drop=True)
    if len(d) < 2:
        return None

    # Use a numpy datetime64 array to avoid pandas Scalar typing issues.
    t = d["time"].to_numpy(dtype="datetime64[ns]")
    v = pd.to_numeric(d[voltage_column], errors="coerce").to_numpy()
    if np.all(~np.isfinite(v)):
        return None

    # Define "stable" steps based on voltage change between adjacent points.
    dv = np.abs(np.diff(v))

    # Break plateaus across large sampling gaps.
    dt_seconds = (np.diff(t) / np.timedelta64(1, "s")).astype(np.float64)
    median_dt = float(np.nanmedian(dt_seconds)) if len(dt_seconds) else 0.0
    gap_break = max(3600.0, 10.0 * median_dt) if median_dt > 0 else 3600.0

    stable_step = (dv <= dv_threshold) & (dt_seconds <= gap_break)

    # Find contiguous runs of stable_step==True.
    best: tuple[int, int, float] | None = None
    i = 0
    while i < len(stable_step):
        if not stable_step[i]:
            i += 1
            continue
        start_edge = i
        end_edge = i
        while end_edge + 1 < len(stable_step) and stable_step[end_edge + 1]:
            end_edge += 1

        start_idx = start_edge
        end_idx = end_edge + 1
        duration_seconds = float((t[end_idx] - t[start_idx]) / np.timedelta64(1, "s"))
        if duration_seconds >= min_duration_seconds:
            if best is None or duration_seconds > best[2]:
                best = (start_idx, end_idx, duration_seconds)

        i = end_edge + 1

    if best is None:
        return None

    start_idx, end_idx, duration_seconds = best
    return pd.Timestamp(t[start_idx]), pd.Timestamp(t[end_idx]), float(duration_seconds)


def main() -> None:
    default_csvs = [
        fcspikes_root() / "csv" / "conicalFT" / "firsthalf.csv",
        fcspikes_root() / "csv" / "conicalFT" / "secondhalf.csv",
    ]

    parser = argparse.ArgumentParser(description="Plot ConicalFT stability variables vs time")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "(Legacy) Path to a single ConicalFT stability CSV. "
            "Prefer --csvs to pass multiple files."
        ),
    )
    parser.add_argument(
        "--csvs",
        type=str,
        nargs="+",
        default=[str(p) for p in default_csvs],
        help=(
            "One or more CSV paths to read and concatenate (default: "
            "<data-root>/csv/conicalFT/firsthalf.csv <data-root>/csv/conicalFT/secondhalf.csv)"
        ),
    )
    parser.add_argument(
        "--current-column",
        type=str,
        default="Current Filtered [uA]",
        help="Current column to plot (default: 'Current Filtered [uA]').",
    )
    parser.add_argument(
        "--voltage-column",
        type=str,
        default="Voltage [V]",
        help="Voltage column to plot (default: 'Voltage [V]').",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="If >0, downsample to at most this many points (useful for very large CSVs).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-05 12:00:00",
        help="Drop data after this timestamp (default: '2026-01-05 12:00:00'). Set to '' to disable.",
    )
    parser.add_argument(
        "--plateau-dv",
        type=float,
        default=1.0,
        help="Voltage change threshold [V] between adjacent points to consider HV stable (default: 1.0).",
    )
    parser.add_argument(
        "--plateau-min-hours",
        type=float,
        default=24.0,
        help="Minimum plateau duration [hours] to report (default: 24).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Default: figures/analysis/conicalFT/conicalft_stability_<column>.png",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the plot interactively (default: False).",
    )

    args = parser.parse_args()

    csv_paths: list[Path]
    if args.csv is not None:
        csv_paths = [Path(args.csv)]
    else:
        csv_paths = [Path(p) for p in (args.csvs or [])]

    if not csv_paths:
        raise ValueError("No CSV inputs provided. Use --csvs <file1> <file2> ...")

    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"CSV not found: {missing_str}")

    dfs = [_read_conicalft_stability_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True, sort=False)
    if df.empty:
        raise ValueError("No data rows found after parsing input CSVs")
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    end_str = (args.end or "").strip()
    if end_str:
        end_dt = pd.to_datetime(end_str, errors="raise")
        df = df.loc[df["time"] <= end_dt].copy()
        if df.empty:
            raise ValueError(f"No data remaining after applying --end {end_dt!s}")

    if args.current_column not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise KeyError(f"Current column '{args.current_column}' not found. Available columns: {cols}")

    if args.voltage_column not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise KeyError(f"Voltage column '{args.voltage_column}' not found. Available columns: {cols}")

    df[args.current_column] = pd.to_numeric(df[args.current_column], errors="coerce")
    df[args.voltage_column] = pd.to_numeric(df[args.voltage_column], errors="coerce")
    df = df.dropna(subset=[args.current_column, args.voltage_column]).copy()

    plateau = _find_longest_voltage_plateau(
        df,
        voltage_column=args.voltage_column,
        dv_threshold=float(args.plateau_dv),
        min_duration_seconds=float(args.plateau_min_hours) * 3600.0,
    )

    if args.max_points and args.max_points > 0:
        df = _downsample_evenly(df, int(args.max_points))

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()

    voltage_label_kv = f"{args.voltage_column} (kV)"
    ax.plot(df["time"], df[args.voltage_column] / 1000.0, color="tab:blue", linewidth=0.9, label=voltage_label_kv)
    ax2.plot(df["time"], df[args.current_column], color="tab:red", linewidth=0.9, label=args.current_column)
    ax.set_xlabel("Time", fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_ylabel(voltage_label_kv, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax2.set_ylabel(args.current_column, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_title("Conical Feedthrough stability", fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.grid(True, alpha=0.3)

    plateau_label: str | None = None
    if plateau is not None:
        _start, _end, duration_seconds = plateau
        plateau_df = df.loc[(df["time"] >= _start) & (df["time"] <= _end), ["time", args.voltage_column, args.current_column]].copy()
        plateau_df[args.voltage_column] = pd.to_numeric(plateau_df[args.voltage_column], errors="coerce")
        plateau_df[args.current_column] = pd.to_numeric(plateau_df[args.current_column], errors="coerce")
        mean_voltage = float(np.nanmean(plateau_df[args.voltage_column].to_numpy()))
        mean_current = float(np.nanmean(plateau_df[args.current_column].to_numpy()))
        plateau_label = "\n".join(
            [
                f"HV Stability duration: {_format_duration(duration_seconds)}",
                f"Mean Current: {mean_current:.4f} uA",
            ]
        )
    else:
        print(
            "No HV plateau found with current settings: "
            f"--plateau-dv {args.plateau_dv:g}, --plateau-min-hours {args.plateau_min_hours:g}"
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    if plateau_label is not None:
        from matplotlib.lines import Line2D

        handles.append(Line2D([], [], linestyle="none"))
        labels.append(plateau_label)

    if handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            framealpha=0.9,
            prop={"size": LEGEND_FONT_SIZE, "weight": FONT_WEIGHT},
        )

    fig.autofmt_xdate()
    plt.tight_layout()

    out_path: Path
    if args.out is None:
        out_path = figures_root() / "analysis" / "conicalFT" / "conicalft_stability_voltage_and_current.png"
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
