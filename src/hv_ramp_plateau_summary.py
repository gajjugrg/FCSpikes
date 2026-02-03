import argparse
import csv
from datetime import datetime
from pathlib import Path
import platform
import subprocess

import matplotlib
import matplotlib.dates as mdates
import numpy as np


TIME_FMT = "%Y/%m/%d %H:%M:%S.%f"


DRIFT_DISTANCE_CM = 338.6
FIELD_SCALE = (1000.0 / DRIFT_DISTANCE_CM) * 0.9976287  # V/cm per kV
AXIS_LABEL_FONTSIZE = 14


def termination_voltage_kv(t: datetime, *, change_time: datetime) -> float:
	"""Return the termination voltage (kV) for the given time.

	- 1.0 kV before Dec 12 10:39
	- 1.2 kV at/after Dec 12 10:39
	"""
	return 1.2 if t >= change_time else 1.0


def read_calibration_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
	"""Load calibration table and return (slow_control_kV, heinzinger_kV)."""
	data = np.genfromtxt(
		path,
		dtype=float,
		delimiter=None,  # any whitespace or tabs
		names=True,
		encoding="utf-8",
	)
	col_names = data.dtype.names or ()
	required = ("slow_control_readout_kV", "Heinzinger_readout_kV")
	for name in required:
		if name not in col_names:
			raise ValueError(f"Missing column '{name}' in {path}. Found: {col_names}")
	slow = np.asarray(data["slow_control_readout_kV"], dtype=float)
	heinz = np.asarray(data["Heinzinger_readout_kV"], dtype=float)
	mask = np.isfinite(slow) & np.isfinite(heinz)
	return slow[mask], heinz[mask]


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
	"""Fit y = a*x + b (least squares)."""
	A = np.column_stack([x, np.ones_like(x)])
	a, b = np.linalg.lstsq(A, y, rcond=None)[0]
	return float(a), float(b)


def _style_axes(ax) -> None:
	"""Use inward ticks on all sides."""
	ax.tick_params(which="both", direction="in", top=True, right=True)
	ax.xaxis.set_ticks_position("both")
	ax.yaxis.set_ticks_position("both")


def read_hv_ramp_csv(path: Path, *, stride: int = 1) -> tuple[list[datetime], np.ndarray, np.ndarray]:
	"""Read HV ramp CSV exported by DCS (two header lines).

	Returns:
	- times: list of datetime
	- voltage_kv: numpy array [kV]
	- current_uA: numpy array [µA]
	"""
	if stride < 1:
		raise ValueError("stride must be >= 1")

	times: list[datetime] = []
	voltage_v: list[float] = []
	current_ua: list[float] = []

	with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
		reader = csv.reader(f)
		# DCS export has two header lines.
		next(reader, None)
		next(reader, None)

		for i, row in enumerate(reader):
			if not row or len(row) < 3:
				continue
			if stride > 1 and (i % stride) != 0:
				continue
			try:
				t = datetime.strptime(row[0].strip(), TIME_FMT)
				v = float(row[1])
				c = float(row[2])
			except Exception:
				continue
			times.append(t)
			voltage_v.append(v)
			current_ua.append(c)

	voltage_kv = np.asarray(voltage_v, dtype=float) / 1000.0
	current_uA = np.asarray(current_ua, dtype=float)
	return times, voltage_kv, current_uA


def plot_voltage_and_current(
	times: list[datetime],
	voltage_kv: np.ndarray,
	current_uA: np.ndarray,
	*,
	stable_segments: list[tuple[int, int]] | None = None,
	include_mask: np.ndarray | None = None,
	out_path: Path,
	show: bool,
	voltage_ylabel: str = "Voltage [kV]",
) -> None:
	"""Make a single plot: Voltage and Current vs time (dual y-axis)."""
	import matplotlib.pyplot as plt

	fig, ax_v = plt.subplots(figsize=(12, 5))
	ax_i = ax_v.twinx()

	line_v = ax_v.plot(times, voltage_kv, color="tab:blue", lw=1, label="Voltage")
	line_i = ax_i.plot(times, current_uA, color="tab:red", lw=1, label="Current")

	# Annotate stable HV plateaus with mean electric field value.
	if stable_segments:
		if include_mask is None:
			include_mask = np.ones(voltage_kv.size, dtype=bool)

		# Termination voltage changed on Dec 12 at 10:39 (local time in the CSVs).
		# Use the data year if available (common case is 2025).
		years = {t.year for t in times if isinstance(t, datetime)}
		year = 2025 if 2025 in years else times[0].year
		change_time = datetime(year, 12, 12, 10, 39)

		vmin = float(np.nanmin(voltage_kv))
		vmax = float(np.nanmax(voltage_kv))
		voff = max(0.5, 0.02 * max(1e-9, (vmax - vmin)))
		for s, e in stable_segments:
			if e <= s:
				continue
			mid = s + (e - s) // 2
			x_mid = mdates.date2num(times[mid])
			seg_mask = include_mask[s : e + 1]
			v_used = voltage_kv[s : e + 1][seg_mask]
			if v_used.size == 0:
				continue
			avg_v = float(np.mean(v_used))
			seg_times = [t for t, keep in zip(times[s : e + 1], seg_mask) if bool(keep)]
			vterm = np.asarray([termination_voltage_kv(t, change_time=change_time) for t in seg_times], dtype=float)
			mean_e = float(np.mean(np.asarray(v_used, dtype=float) - vterm) * FIELD_SCALE)
			duration_h = (times[e] - times[s]).total_seconds() / 3600.0
			ax_v.text(
				x_mid,
				avg_v + voff,
				f"{mean_e:.1f} V/cm\n{duration_h:.1f} h",
				ha="center",
				va="bottom",
				fontsize=14,
				color="black",
			)

	ax_v.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
	ax_v.set_ylabel(voltage_ylabel, color="tab:blue", fontsize=AXIS_LABEL_FONTSIZE)
	ax_i.set_ylabel("Current [µA]", color="tab:red", fontsize=AXIS_LABEL_FONTSIZE)

	# Ticks on both sides.
	_style_axes(ax_v)
	_style_axes(ax_i)

	# Time formatting.
	locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
	ax_v.xaxis.set_major_locator(locator)
	ax_v.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
	fig.autofmt_xdate(rotation=30)

	# Single legend for both axes.
	lines = line_v + line_i
	labels = [l.get_label() for l in lines]
	ax_v.legend(lines, labels, loc="best")

	ax_v.grid(True, alpha=0.25)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200)
	if show:
		# This blocks until the window is closed.
		plt.show()
	plt.close(fig)


def open_path_in_default_app(path: Path) -> None:
	"""Open a file in the OS default application (best-effort)."""
	system = platform.system()
	try:
		if system == "Darwin":
			subprocess.run(["open", str(path)], check=False)
		elif system == "Windows":
			# Uses file associations.
			subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
		else:
			# Most Linux desktops.
			subprocess.run(["xdg-open", str(path)], check=False)
	except Exception as e:
		print(f"Could not open {path}: {e}")


def _configure_matplotlib_backend(*, show: bool) -> bool:
	"""Configure Matplotlib backend.

	Returns True if an interactive backend was selected (so plt.show() can open a window).
	"""
	if not show:
		# Reliable, non-interactive backend for saving PNGs.
		matplotlib.use("Agg", force=True)
		return False

	# Try to select an interactive backend.
	# NOTE: must happen before importing matplotlib.pyplot.
	backend_candidates: list[str]
	if platform.system() == "Darwin":
		backend_candidates = ["MacOSX", "TkAgg"]
	else:
		backend_candidates = ["TkAgg", "QtAgg"]

	for backend in backend_candidates:
		try:
			matplotlib.use(backend, force=True)
			return True
		except Exception:
			continue

	# If we can't get an interactive backend, fall back to Agg.
	matplotlib.use("Agg", force=True)
	return False


def read_multiple_hv_ramps(paths: list[Path], *, stride: int = 1) -> tuple[list[datetime], np.ndarray, np.ndarray]:
	"""Read multiple HV ramp CSVs and combine them into one time series."""
	all_times: list[datetime] = []
	all_v: list[float] = []
	all_i: list[float] = []
	for path in paths:
		times, voltage_kv, current_uA = read_hv_ramp_csv(path, stride=stride)
		all_times.extend(times)
		all_v.extend(voltage_kv.tolist())
		all_i.extend(current_uA.tolist())

	if len(all_times) == 0:
		return [], np.array([]), np.array([])

	# Sort by time (files may overlap or be out of order).
	order = np.argsort(np.asarray([t.timestamp() for t in all_times], dtype=float))
	sorted_times = [all_times[int(k)] for k in order]
	sorted_v = np.asarray([all_v[int(k)] for k in order], dtype=float)
	sorted_i = np.asarray([all_i[int(k)] for k in order], dtype=float)
	return sorted_times, sorted_v, sorted_i


def find_stable_voltage_periods(
	times: list[datetime],
	voltage_kv: np.ndarray,
	*,
	tol_kv: float,
	min_stable_hours: float,
	max_unstable_minutes: float,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
	"""Find long stable-voltage periods.

	We build contiguous segments where voltage stays within `tol_kv` of the running
	segment mean.

	If voltage deviates from the segment mean but returns within `tol_kv` within
	`max_unstable_minutes`, we do NOT treat that as a voltage change; instead we
	mark that short excursion as unstable and exclude it from average/std
	calculations.

	Finally, we keep only segments longer than `min_stable_hours`.
	"""
	if len(times) == 0:
		return [], []
	if tol_kv <= 0:
		raise ValueError("tol_kv must be > 0")
	if min_stable_hours < 0:
		raise ValueError("min_stable_hours must be >= 0")
	if max_unstable_minutes < 0:
		raise ValueError("max_unstable_minutes must be >= 0")

	# Build segments, allowing short unstable excursions that recover.
	segments_with_exclusions: list[tuple[int, int, list[tuple[int, int]]]] = []
	start = 0
	running_sum = float(voltage_kv[0])
	running_n = 1
	exclusions: list[tuple[int, int]] = []

	i = 1
	while i < len(times):
		mean_v = running_sum / running_n
		if abs(float(voltage_kv[i]) - mean_v) <= tol_kv:
			running_sum += float(voltage_kv[i])
			running_n += 1
			i += 1
			continue

		# Potential short unstable excursion: look ahead for recovery.
		gap_start = i
		recovered_at: int | None = None
		j = i + 1
		while j < len(times):
			gap_minutes = (times[j] - times[gap_start]).total_seconds() / 60.0
			if gap_minutes > max_unstable_minutes:
				break
			if abs(float(voltage_kv[j]) - mean_v) <= tol_kv:
				recovered_at = j
				break
			j += 1

		if recovered_at is not None:
			# Exclude the unstable run [gap_start, recovered_at-1] but keep the segment.
			exclusions.append((gap_start, recovered_at - 1))
			i = recovered_at
			# Include the recovered point in the running mean.
			running_sum += float(voltage_kv[i])
			running_n += 1
			i += 1
			continue

		# No recovery within the allowed window: treat as a real change.
		end = gap_start - 1
		if end >= start:
			segments_with_exclusions.append((start, end, exclusions))
		start = gap_start
		running_sum = float(voltage_kv[start])
		running_n = 1
		exclusions = []
		i = start + 1

	# Final segment.
	if len(times) - 1 >= start:
		segments_with_exclusions.append((start, len(times) - 1, exclusions))

	stable: list[tuple[int, int]] = []
	stable_exclusions: list[list[tuple[int, int]]] = []
	for s, e, exc in segments_with_exclusions:
		duration_h = (times[e] - times[s]).total_seconds() / 3600.0
		if duration_h >= min_stable_hours:
			stable.append((s, e))
			stable_exclusions.append(exc)
	return stable, stable_exclusions


def build_stable_include_mask(
	n: int,
	stable_segments: list[tuple[int, int]],
	exclusions_by_segment: list[list[tuple[int, int]]],
) -> np.ndarray:
	"""Create a boolean mask of points to include in plateau statistics."""
	include = np.zeros(n, dtype=bool)
	for (s, e), exclusions in zip(stable_segments, exclusions_by_segment, strict=False):
		include[s : e + 1] = True
		for xs, xe in exclusions:
			if xe >= xs:
				include[xs : xe + 1] = False
	return include


def write_stable_summary_csv(
	path: Path,
	times: list[datetime],
	voltage_kv: np.ndarray,
	segments: list[tuple[int, int]],
	include_mask: np.ndarray,
) -> None:
	"""Write stable-period summary (start/end/avg voltage)."""
	if not times:
		return
	# Termination voltage changed on Dec 12 at 10:39 (local time in the CSVs).
	# Use the data year if available (common case is 2025).
	years = {t.year for t in times if isinstance(t, datetime)}
	year = 2025 if 2025 in years else times[0].year
	change_time = datetime(year, 12, 12, 10, 39)

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		w = csv.writer(f)
		w.writerow(
			[
				"start",
				"end",
				"duration_hours",
				"avg_voltage_kV",
				"mean_E_V_per_cm",
				"std_voltage_kV",
				"n_points",
			]
		)
		for s, e in segments:
			seg_mask = include_mask[s : e + 1]
			v_used = voltage_kv[s : e + 1][seg_mask]
			if v_used.size == 0:
				continue
			seg_times = [t for t, keep in zip(times[s : e + 1], seg_mask) if bool(keep)]
			vterm = np.asarray([termination_voltage_kv(t, change_time=change_time) for t in seg_times], dtype=float)
			# Mean E uses per-sample termination to correctly handle a segment that crosses the change time.
			dv = np.asarray(v_used, dtype=float) - vterm
			mean_e = float(np.mean(dv) * FIELD_SCALE)
			avg_vterm = float(np.mean(vterm))
			duration_h = (times[e] - times[s]).total_seconds() / 3600.0
			w.writerow(
				[
					times[s].isoformat(sep=" ", timespec="seconds"),
					times[e].isoformat(sep=" ", timespec="seconds"),
					f"{duration_h:.3f}",
					f"{float(np.mean(v_used)):.6f}",
					f"{mean_e:.6f}",
					f"{float(np.std(v_used)):.6f}",
					int(v_used.size),
				]
			)


def main() -> None:
	root = Path(__file__).resolve().parents[1]
	default_input = root / "data" / "csv" / "ramping" / "HV_ramp_Nov27.csv"
	default_input2 = root / "data" / "csv" / "ramping" / "HV_ramp_Dec6.csv"
	default_input3 = root / "data" / "csv" / "ramping" / "HV_ramp_Dec15.csv"
	default_out_dir = root / "figures" / "analysis" / "ramping_hv"
	default_calibration = root / "data" / "calibration_data.txt"

	default_inputs: list[Path] = [default_input]
	if default_input2.exists():
		default_inputs.append(default_input2)
	if default_input3.exists():
		default_inputs.append(default_input3)

	p = argparse.ArgumentParser(description="Plot HV ramp: Voltage and Current on the same plot.")
	p.add_argument("--input", default=str(default_input), help="Path to a single HV_ramp_*.csv (chunk)")
	p.add_argument(
		"--inputs",
		nargs="*",
		default=None,
		help="Optional list of HV_ramp_*.csv chunk files to combine. If omitted, uses Nov27/Dec6/Dec15 if present.",
	)
	p.add_argument("--out-dir", default=str(default_out_dir), help="Output directory for PNGs")
	p.add_argument("--stride", type=int, default=1, help="Read every Nth row (use >1 if file is huge)")
	p.add_argument(
		"--no-calibration",
		action="store_true",
		help="Do not apply SlowControl→Heinzinger calibration to the voltage readout",
	)
	p.add_argument(
		"--calibration-table",
		default=str(default_calibration),
		help="Calibration table used to correct voltage (default: data/calibration_data.txt)",
	)
	p.add_argument(
		"--stable-tol-kv",
		type=float,
		default=0.3, 
		help="Voltage tolerance (kV) used to define a stable period",
	)
	p.add_argument(
		"--min-stable-hours",
		type=float,
		default=6.0,
		help="Minimum duration (hours) to report a stable period",
	)
	p.add_argument(
		"--max-unstable-minutes",
		type=float,
		default=15.0,
		help="If voltage deviates but recovers within this window, exclude the excursion from averages instead of splitting the plateau",
	)
	p.add_argument(
		"--force-last-plateau-end",
		default="2025-12-16 15:00",
		help="Force-cap the end timestamp of the last stable plateau (e.g. '2025-12-16 14:40').",
	)
	p.add_argument(
		"--force-plateau-end",
		action="append",
		default=None,
		help="Force-cap a specific plateau end time by plateau start: 'START=END' (e.g. '2025-12-02 12:04:14=2025-12-03 12:27'). Can be repeated.",
	)
	p.add_argument(
		"--show",
		action="store_true",
		help="Open an interactive plot window (falls back to opening the saved PNG if GUI backend is unavailable)",
	)
	args = p.parse_args()

	interactive = _configure_matplotlib_backend(show=bool(args.show))

	out_dir = Path(args.out_dir).expanduser()
	out_dir.mkdir(parents=True, exist_ok=True)

	input_paths: list[Path]
	if args.inputs is not None and len(args.inputs) > 0:
		input_paths = [Path(p).expanduser() for p in args.inputs]
	else:
		# Default: plot Nov27 and also Dec6 if it exists.
		input_paths = default_inputs

	# Back-compat: if user explicitly set --input, plot that single file.
	# (Unless --inputs was provided.)
	if args.inputs is None and args.input and str(args.input) != str(default_input):
		input_paths = [Path(args.input).expanduser()]

	times, voltage_kv, current_uA = read_multiple_hv_ramps(input_paths, stride=args.stride)
	if len(times) == 0:
		raise RuntimeError("No valid rows read from the provided input files")

	# Apply calibration so plotted/read-back voltage is corrected.
	voltage_ylabel = "Voltage [kV]"
	if not args.no_calibration:
		cal_path = Path(args.calibration_table).expanduser()
		if cal_path.exists():
			slow_kv, heinz_kv = read_calibration_table(cal_path)
			if slow_kv.size >= 2:
				a, b = fit_linear(slow_kv, heinz_kv)
				voltage_kv = a * voltage_kv + b
				voltage_ylabel = "Voltage [kV] (calibrated)"
				print(f"Applied calibration: Heinzinger = {a:.6g} * SlowControl + {b:.6g} kV")
			else:
				print(f"Calibration table has too few rows, skipping: {cal_path}")
		else:
			print(f"Calibration table not found, skipping: {cal_path}")

	# Find stable plateaus and summarize them.
	stable, stable_exclusions = find_stable_voltage_periods(
		times,
		voltage_kv,
		tol_kv=float(args.stable_tol_kv),
		min_stable_hours=float(args.min_stable_hours),
		max_unstable_minutes=float(args.max_unstable_minutes),
	)
	if args.force_plateau_end and len(stable) > 0:
		# Parse START=END rules.
		rules: dict[datetime, datetime] = {}
		for raw in args.force_plateau_end:
			if raw is None:
				continue
			text = str(raw)
			if "=" not in text:
				raise ValueError(f"--force-plateau-end must be 'START=END', got: {text!r}")
			start_s, end_s = text.split("=", 1)
			rules[datetime.fromisoformat(start_s.strip())] = datetime.fromisoformat(end_s.strip())

		# Apply to matching stable segments (match by start time to the nearest second).
		times_ts = np.asarray([t.timestamp() for t in times], dtype=float)
		for seg_idx, (s_idx, e_idx) in enumerate(list(stable)):
			seg_start = times[s_idx].replace(microsecond=0)
			forced_end = rules.get(seg_start)
			if forced_end is None:
				continue
			if not (times[s_idx] < forced_end < times[e_idx]):
				continue
			forced_idx = int(np.searchsorted(times_ts, forced_end.timestamp(), side="right") - 1)
			forced_idx = max(s_idx, min(forced_idx, e_idx))
			stable[seg_idx] = (s_idx, forced_idx)
			new_exc: list[tuple[int, int]] = []
			for xs, xe in stable_exclusions[seg_idx]:
				if xs > forced_idx:
					continue
				new_exc.append((xs, min(xe, forced_idx)))
			stable_exclusions[seg_idx] = new_exc
	if args.force_last_plateau_end and len(stable) > 0:
		forced_end = datetime.fromisoformat(str(args.force_last_plateau_end))
		last_s, last_e = stable[-1]
		# Only apply if the forced end lies within the last plateau range.
		if times[last_s] < forced_end < times[last_e]:
			# Find the last index <= forced_end within the full time array.
			forced_idx = int(np.searchsorted(np.asarray([t.timestamp() for t in times], dtype=float), forced_end.timestamp(), side="right") - 1)
			forced_idx = max(last_s, min(forced_idx, last_e))
			stable[-1] = (last_s, forced_idx)
			# Drop any exclusions that are now out-of-range.
			new_exc = []
			for xs, xe in stable_exclusions[-1]:
				if xs > forced_idx:
					continue
				new_exc.append((xs, min(xe, forced_idx)))
			stable_exclusions[-1] = new_exc
	include_mask = build_stable_include_mask(len(times), stable, stable_exclusions)

	out_path = out_dir / "HV_ramp_voltage_current.png"
	plot_voltage_and_current(
		times,
		voltage_kv,
		current_uA,
		stable_segments=stable,
		include_mask=include_mask,
		out_path=out_path,
		show=bool(args.show) and interactive,
		voltage_ylabel=voltage_ylabel,
	)
	print(f"Wrote: {out_path}")
	if args.show and not interactive:
		print("Matplotlib GUI backend not available; opening saved PNG instead.")
		open_path_in_default_app(out_path)

	summary_path = out_dir / "HV_ramp_stable_voltage_summary.csv"
	write_stable_summary_csv(summary_path, times, voltage_kv, stable, include_mask)
	print(f"Wrote: {summary_path}")

	if len(stable) == 0:
		print("No stable periods found with current thresholds.")
		print("Try increasing --stable-tol-kv or decreasing --min-stable-hours.")
		return

	print("\nStable periods (avg readback voltage):")
	for s, e in stable:
		seg_mask = include_mask[s : e + 1]
		v_used = voltage_kv[s : e + 1][seg_mask]
		if v_used.size == 0:
			continue
		avg_v = float(np.mean(v_used))
		# Compute mean E for the printed summary as well.
		years = {t.year for t in times if isinstance(t, datetime)}
		year = 2025 if 2025 in years else times[0].year
		change_time = datetime(year, 12, 12, 10, 39)
		seg_times = [t for t, keep in zip(times[s : e + 1], seg_mask) if bool(keep)]
		vterm = np.asarray([termination_voltage_kv(t, change_time=change_time) for t in seg_times], dtype=float)
		mean_e = float(np.mean(np.asarray(v_used, dtype=float) - vterm) * FIELD_SCALE)
		duration_h = (times[e] - times[s]).total_seconds() / 3600.0
		print(
			f"  {times[s]}  ->  {times[e]}  "
			f"({duration_h:.1f} h), E_mean={mean_e:.3f} V/cm, avg={avg_v:.3f} kV"
		)

	# (Intentionally not printing "approx change times"; the plot + CSV are the outputs.)


if __name__ == "__main__":
	main()
