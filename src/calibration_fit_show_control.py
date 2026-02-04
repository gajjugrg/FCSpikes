import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def style_axes(ax) -> None:
	"""Style helper: put tick marks on all four sides."""
	ax.tick_params(which="both", direction="in", top=True, right=True)
	ax.xaxis.set_ticks_position("both")
	ax.yaxis.set_ticks_position("both")


def read_calibration_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load calibration table (whitespace/tab separated).

	Expected columns:
	- set_voltage_kV
	- slow_control_readout_kV
	- Heinzinger_readout_kV
	"""
	data = np.genfromtxt(
		path,
		dtype=float,
		delimiter=None,  # any whitespace
		names=True,
		encoding="utf-8",
	)
	col_names = data.dtype.names or ()
	required = ("set_voltage_kV", "slow_control_readout_kV", "Heinzinger_readout_kV")
	for name in required:
		if name not in col_names:
			raise ValueError(f"Missing column '{name}' in {path}. Found: {col_names}")
	set_v = np.asarray(data["set_voltage_kV"], dtype=float)
	slow = np.asarray(data["slow_control_readout_kV"], dtype=float)
	heinz = np.asarray(data["Heinzinger_readout_kV"], dtype=float)
	mask = np.isfinite(set_v) & np.isfinite(slow) & np.isfinite(heinz)
	return set_v[mask], slow[mask], heinz[mask]


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
	"""Fit y = a*x + b (least squares). Returns (a, b, r2, rmse)."""
	A = np.column_stack([x, np.ones_like(x)])
	a, b = np.linalg.lstsq(A, y, rcond=None)[0]
	yhat = a * x + b
	ss_res = float(np.sum((y - yhat) ** 2))
	ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
	r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
	rmse = float(np.sqrt(ss_res / max(1, x.size)))
	return float(a), float(b), r2, rmse


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
	"""Fit y = a*x with b=0. Returns (a, r2, rmse)."""
	den = float(np.dot(x, x))
	if den == 0.0:
		raise ValueError("Cannot fit through origin: all x are zero")
	a = float(np.dot(x, y) / den)
	yhat = a * x
	ss_res = float(np.sum((y - yhat) ** 2))
	ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
	r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
	rmse = float(np.sqrt(ss_res / max(1, x.size)))
	return a, r2, rmse


def ensure_out_dir(out_dir: Path) -> Path:
	"""Create output directory if needed."""
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir


def main() -> None:
	"""CLI entrypoint."""
	root = Path(__file__).resolve().parents[1]
	default_input = root / "data" / "calibration_data.txt"
	default_out = root / "results" / "analysis" / "calibration_show_control"

	p = argparse.ArgumentParser(description="Fit calibration: Heinzinger (true) vs Slow Control (monitored).")
	p.add_argument("--input", default=str(default_input), help="Path to calibration_data.txt (default: data/calibration_data.txt)")
	p.add_argument("--out-dir", default=str(default_out), help="Directory to write plots + summary")
	p.add_argument("--through-origin", action="store_true", help="Also compute a fit forced through origin")
	p.add_argument("--show", action="store_true", help="Show plots interactively")
	args = p.parse_args()

	input_path = Path(args.input).expanduser()
	out_dir = ensure_out_dir(Path(args.out_dir).expanduser())

	set_v, slow, heinz = read_calibration_table(input_path)
	if slow.size < 2:
		raise RuntimeError("Not enough rows to fit.")

	a, b, r2, rmse = fit_linear(slow, heinz)
	fit0 = None
	if args.through_origin:
		a0, r20, rmse0 = fit_through_origin(slow, heinz)
		fit0 = (a0, r20, rmse0)

	print("Calibration fit (Heinzinger is reference):")
	print(f"  Heinzinger = {a:.6g} * SlowControl + {b:.6g} kV")
	print(f"  N={slow.size}, R²={r2:.4f}, RMSE={rmse:.4g} kV")
	if fit0 is not None:
		print("Through-origin fit:")
		print(f"  Heinzinger = {fit0[0]:.6g} * SlowControl")
		print(f"  N={slow.size}, R²={fit0[1]:.4f}, RMSE={fit0[2]:.4g} kV")

	# Plot 1: scatter + fit
	plt.figure(figsize=(6.5, 6))
	plt.scatter(slow, heinz, s=25, alpha=0.8)
	xx = np.linspace(float(np.min(slow)), float(np.max(slow)), 200)
	plt.plot(xx, a * xx + b, color="black", lw=2, label=f"fit: y={a:.6g}x+{b:.6g}")
	if fit0 is not None:
		plt.plot(xx, fit0[0] * xx, color="tab:orange", lw=2, ls="--", label=f"origin: y={fit0[0]:.6g}x")
	plt.xlabel("Slow Control [kV]")
	plt.ylabel("Heinzinger [kV]")
	plt.title("Calibration fit")
	with plt.rc_context({"legend.fontsize": 9}):
		plt.legend()
	style_axes(plt.gca())
	plt.tight_layout()
	plt.savefig(out_dir / "scatter_fit.png", dpi=200)
	if args.show:
		plt.show()
	plt.close()

	# Plot 2: residual vs setpoint
	resid = heinz - (a * slow + b)
	plt.figure(figsize=(9, 4))
	plt.plot(slow, resid, "o", lw=1)
	plt.axhline(0.0, color="black", lw=1)
	plt.xlabel("Slow Control voltage [kV]")
	plt.ylabel("Residual (Heinz - fit) [kV]")
	plt.title("Residual vs Slow Control voltage")
	style_axes(plt.gca())
	plt.tight_layout()
	plt.savefig(out_dir / "residual_vs_slow_control_voltage.png", dpi=200)
	if args.show:
		plt.show()
	plt.close()

	summary = {
		"input": str(input_path.resolve()),
		"n": int(slow.size),
		"fit": {"slope": a, "intercept": b, "r2": r2, "rmse_kV": rmse},
	}
	if fit0 is not None:
		summary["fit_through_origin"] = {"slope": fit0[0], "r2": fit0[1], "rmse_kV": fit0[2]}
	(out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
	main()




















