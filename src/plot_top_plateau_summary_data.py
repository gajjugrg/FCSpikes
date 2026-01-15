"""Plot per-plateau summary variables with the requested fit models.

Reads a tab-separated summary table like:
  figures/analysis/hv_ramping_spike_study/top_plateau_summary_data_<tag>.txt

Implements ONLY the models you provided:

- Spike rate vs HV (kV):
    λ(V) = λ0(V) / (1 + τ_h λ0(V))
    λ0(V) = λ_ref exp(m (V - Vref))

- Mean charge vs HV (kV): logistic saturation
    q(V) = qmin + (qmax-qmin)/(1 + exp(-k (V - Vc)))

Usage:
  python src/plot_top_plateau_summary_data.py --tag hvramping --show
  python src/plot_top_plateau_summary_data.py --tag hvramping --save
  python src/plot_top_plateau_summary_data.py --input /abs/path/to/file.txt --save
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utilities import figures_root


VREF_KV_DEFAULT = 154.0


def _default_input_path(tag: str) -> Path:
    return figures_root() / "analysis" / "hv_ramping_spike_study" / f"top_plateau_summary_data_{tag}.txt"


def lambda_model(V: np.ndarray, lam_ref: float, m: float, tau_h: float, Vref: float) -> np.ndarray:
    """Frequency model: λ = λ0/(1+τ*λ0), λ0 = λ_ref*exp(m*(V-Vref))."""
    lam0 = lam_ref * np.exp(m * (V - Vref))
    return lam0 / (1.0 + tau_h * lam0)


def q_logistic(V: np.ndarray, qmin: float, qmax: float, k: float, Vc: float) -> np.ndarray:
    """Charge model: logistic saturation (captures linear->saturation over limited range)."""
    return qmin + (qmax - qmin) / (1.0 + np.exp(-k * (V - Vc)))


def _finite_sigma(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    return None if not np.isfinite(values).any() else values


def _add_equation_box(ax: plt.Axes, lines: list[str]) -> None:
    if not lines:
        return
    txt = "\n".join(lines)
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )


def _add_param_box(ax: plt.Axes, lines: list[str]) -> None:
    if not lines:
        return
    txt = "\n".join(lines)
    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="hvramping", help="Tag used in the exported summary filename")
    p.add_argument("--input", help="Path to the per-plateau summary .txt file (TSV)")
    p.add_argument("--save", action="store_true", help="Save plots under figures/")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for saved plots (default: figures/analysis/hv_ramping_spike_study/summary_data_plots)",
    )
    p.add_argument(
        "--vref-kv",
        type=float,
        default=VREF_KV_DEFAULT,
        help="Reference voltage Vref (kV) used in the rate model exponent.",
    )
    args = p.parse_args()

    if not args.show:
        plt.switch_backend("Agg")

    input_path = Path(args.input) if args.input else _default_input_path(str(args.tag))
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path, sep="\t")

    needed = [
        "avg_voltage_kV",
        "mean_E_V_per_cm",
        "spike_rate_per_h",
        "avg_charge_uC",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    # Coerce numeric columns used for plotting/fitting.
    for c in [
        "avg_voltage_kV",
        "mean_E_V_per_cm",
        "spike_rate_per_h",
        "spike_rate_err_per_h",
        "avg_charge_uC",
        "avg_charge_err_uC",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    V = df["avg_voltage_kV"].to_numpy(dtype=float)
    E = df["mean_E_V_per_cm"].to_numpy(dtype=float)
    rate = df["spike_rate_per_h"].to_numpy(dtype=float)
    q = df["avg_charge_uC"].to_numpy(dtype=float)

    rate_err = df["spike_rate_err_per_h"].to_numpy(dtype=float) if "spike_rate_err_per_h" in df.columns else None
    q_err = df["avg_charge_err_uC"].to_numpy(dtype=float) if "avg_charge_err_uC" in df.columns else None

    rate_err = _finite_sigma(rate_err)
    q_err = _finite_sigma(q_err)

    ok = np.isfinite(V) & np.isfinite(E) & np.isfinite(rate) & np.isfinite(q)
    if rate_err is not None:
        ok = ok & np.isfinite(rate_err)
    if q_err is not None:
        ok = ok & np.isfinite(q_err)

    V = V[ok]
    E = E[ok]
    rate = rate[ok]
    q = q[ok]
    if rate_err is not None:
        rate_err = rate_err[ok]
    if q_err is not None:
        q_err = q_err[ok]

    if V.size < 3:
        raise ValueError(f"Not enough finite points to fit/plot (n={V.size}).")

    order = np.argsort(V)
    V = V[order]
    E = E[order]
    rate = rate[order]
    q = q[order]
    if rate_err is not None:
        rate_err = rate_err[order]
    if q_err is not None:
        q_err = q_err[order]

    Vref = float(args.vref_kv)

    # Rate model fit.
    p0_rate = [2.0, 0.03, 0.25]
    bounds_rate = ([0.0, -1.0, 0.0], [100.0, 1.0, 100.0])
    popt_rate, pcov_rate = curve_fit(
        lambda V_, lam_ref, m, tau_h: lambda_model(V_, lam_ref, m, tau_h, Vref),
        V,
        rate,
        p0=p0_rate,
        sigma=rate_err,
        absolute_sigma=True if rate_err is not None else False,
        bounds=bounds_rate,
        maxfev=200000,
    )
    perr_rate = np.sqrt(np.diag(pcov_rate))

    # Charge model fit.
    p0_q = [1.5, 4.2, 0.05, 220.0]
    bounds_q = ([-10.0, 0.0, 0.0, 0.0], [10.0, 50.0, 1.0, 1000.0])
    popt_q, pcov_q = curve_fit(
        q_logistic,
        V,
        q,
        p0=p0_q,
        sigma=q_err,
        absolute_sigma=True if q_err is not None else False,
        bounds=bounds_q,
        maxfev=200000,
    )
    perr_q = np.sqrt(np.diag(pcov_q))

    lam_ref, m, tau_h = popt_rate
    lam_sat = 1.0 / tau_h

    Leff_cm = (V * 1e3) / E
    Leff_mean_cm = float(np.nanmean(Leff_cm))

    Vgrid = np.linspace(float(np.nanmin(V) - 5.0), float(np.nanmax(V) + 5.0), 400)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = figures_root() / "analysis" / "hv_ramping_spike_study" / "summary_data_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem

    # Plot fits: rate.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if rate_err is None:
        ax.plot(V, rate, "o", label="data")
    else:
        ax.errorbar(V, rate, yerr=rate_err, fmt="o", capsize=3, label="data")

    # NOTE: don't use a raw string (r"...") here, because then "\n" is not a newline.
    rate_fit_label = (
        "fit: $\\lambda(V)=\\lambda_0/(1+\\tau_h\\,\\lambda_0)$"
        "\n"
        f"$\\lambda_0=\\lambda_\\mathrm{{ref}}\\,\\exp\\left(m\\,(V-{Vref:.0f})\\right)$"
    )
    ax.plot(Vgrid, lambda_model(Vgrid, lam_ref, m, tau_h, Vref), label=rate_fit_label)
    ax.set_xlabel("HV voltage V [kV]")
    ax.set_ylabel("Spike rate λ [1/h]")
    ax.set_title("Spike rate vs HV (fit)")

    _add_param_box(
        ax,
        [
            rf"$V_\mathrm{{ref}}={Vref:.0f}\;\mathrm{{kV}}$",
            rf"$\lambda_\mathrm{{ref}}={lam_ref:.3f}\pm{perr_rate[0]:.3f}\;\mathrm{{h^{{-1}}}}$",
            rf"$m={m:.5f}\pm{perr_rate[1]:.5f}\;\mathrm{{kV^{{-1}}}}$",
            rf"$\tau_h={tau_h:.4f}\pm{perr_rate[2]:.4f}\;\mathrm{{h}}$",
            rf"$\lambda_\mathrm{{sat}}=1/\tau_h={lam_sat:.3f}\;\mathrm{{h^{{-1}}}}$",
        ],
    )

    ax.legend()
    fig.tight_layout()
    if args.save:
        out_path = out_dir / f"{stem}__spike_rate_vs_voltage_kV.png"
        fig.savefig(out_path, dpi=160)
        print(f"Saved {out_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # Plot fits: charge.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if q_err is None:
        ax.plot(V, q, "o", label="data")
    else:
        ax.errorbar(V, q, yerr=q_err, fmt="o", capsize=3, label="data")

    charge_fit_label = (
        r"fit: $q(V)=q_{\min}+\dfrac{q_{\max}-q_{\min}}{1+\exp\left(-k\,(V-V_c)\right)}$"
    )
    ax.plot(Vgrid, q_logistic(Vgrid, *popt_q), label=charge_fit_label)
    ax.set_xlabel("HV [kV]")
    ax.set_ylabel("Mean spike charge ⟨q⟩ [µC]")
    ax.set_title("Mean spike charge vs HV (fit)")

    _add_param_box(
        ax,
        [
            rf"$q_\min={popt_q[0]:.3f}\pm{perr_q[0]:.3f}\;\mu\mathrm{{C}}$",
            rf"$q_\max={popt_q[1]:.3f}\pm{perr_q[1]:.3f}\;\mu\mathrm{{C}}$",
            rf"$k={popt_q[2]:.4f}\pm{perr_q[2]:.4f}\;\mathrm{{kV^{{-1}}}}$",
            rf"$V_c={popt_q[3]:.2f}\pm{perr_q[3]:.2f}\;\mathrm{{kV}}$",
        ],
    )

    ax.legend()
    fig.tight_layout()
    if args.save:
        out_path = out_dir / f"{stem}__avg_charge_vs_voltage_kV.png"
        fig.savefig(out_path, dpi=160)
        print(f"Saved {out_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # Print fitted parameters.
    print("Effective length from E=V/L:", f"{Leff_mean_cm/100:.3f} m (mean)")
    print("\nRate model: λ(V)=λ0/(1+τ λ0),  λ0=λ_ref exp(m (V-{} kV))".format(Vref))
    print(f"λ_ref = {lam_ref:.3f} ± {perr_rate[0]:.3f}  1/h")
    print(f"m     = {m:.5f} ± {perr_rate[1]:.5f}  1/kV")
    print(f"τ     = {tau_h:.4f} ± {perr_rate[2]:.4f}  h  (=> λ_sat = {lam_sat:.3f} 1/h)")

    print("\nCharge model: q(V)=qmin+(qmax-qmin)/(1+exp(-k (V-Vc)))")
    print(f"qmin = {popt_q[0]:.3f} ± {perr_q[0]:.3f}  µC")
    print(f"qmax = {popt_q[1]:.3f} ± {perr_q[1]:.3f}  µC")
    print(f"k    = {popt_q[2]:.4f} ± {perr_q[2]:.4f}  1/kV")
    print(f"Vc   = {popt_q[3]:.2f} ± {perr_q[3]:.2f}  kV")


if __name__ == "__main__":
    main()
