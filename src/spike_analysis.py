import argparse
import glob
import os
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import AutoDateLocator, DateFormatter
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks

from utilities import fcspikes_root, figures_root, spike_timestamps_path

PRESSURE_CHANGE_START = datetime(2026, 1, 29, 14, 0)
PRESSURE_CHANGE_END = datetime(2026, 1, 29, 16, 0)

# Optional cable-switch exclusion window (set to None to disable).
CABLE_SWITCH_START = None
CABLE_SWITCH_END = None

def _adaptive_savgol(y, window_length=51, polyorder=3):
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
        return savgol_filter(y, wl, polyorder, mode='nearest')
    except Exception:
        return y.copy()

def _dequote_array(a):
    a = a.astype(str)
    a = np.char.strip(a)
    a = np.char.strip(a, '"')
    a = np.char.strip(a, "'")
    return a

def _dequote_str(x):
    s = str(x).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def load_combined_data(beam_file, tco_file):
    beam_data = np.loadtxt(beam_file, delimiter=',', skiprows=2, dtype=str)
    tco_data = np.loadtxt(tco_file, delimiter=',', skiprows=2, dtype=str)
    beam_data = _dequote_array(beam_data)
    tco_data  = _dequote_array(tco_data)
    beam_times = [datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f') for row in beam_data]
    tco_times = [datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f') for row in tco_data]
    def safe_float(x):
        s = _dequote_str(x)
        return float(s) if s else 0.0
    beam_dict = {
        t: safe_float(row[1]) + safe_float(row[2]) + safe_float(row[3])
        for t, row in zip(beam_times, beam_data)
    }
    tco_dict = {
        t: safe_float(row[1]) + safe_float(row[2]) + safe_float(row[3])
        for t, row in zip(tco_times, tco_data)
    }
    common_times = sorted(set(beam_dict.keys()) & set(tco_dict.keys()))
    dates = common_times
    total_currents = np.array([beam_dict[t] + tco_dict[t] for t in common_times])
    return dates, total_currents

def detect_spikes_with_savgol(dates, currents_uA, threshold_uA=1e-3, window_length=101, polyorder=2):
    baseline = _adaptive_savgol(currents_uA, window_length=window_length, polyorder=polyorder)
    residual = baseline - currents_uA
    peaks, props = find_peaks(residual, prominence=threshold_uA, width=1)
    results = []
    for idx, mag in zip(peaks, props["prominences"]):
        ts = dates[idx]
        spike_val = currents_uA[idx]
        baseline_val = baseline[idx]
        magnitude = mag
        results.append((ts, magnitude, spike_val, baseline_val))
    return results, baseline

def plot_spike_distribution(spikes, input_file, threshold_uA, bins=30, value="magnitude", save_path=None, show_plot=True):
    if not spikes:
        return
    if value == "magnitude":
        spike_values = [mag for _, mag, _, _ in spikes]
        x_label = "Magnitude (baseline - spike) [uA]"
        title = "Distribution of Spike Magnitudes"
    elif value == "spike":
        spike_values = [val for _, _, val, _ in spikes]
        x_label = "Spike Current (uA)"
        title = "Distribution of Raw Spike Values"
    else:
        raise ValueError("value must be 'magnitude' or 'spike'")
    fname = os.path.basename(input_file)
    plt.figure(figsize=(8, 5))
    plt.hist(spike_values, bins=bins, histtype='step', facecolor='lightblue', edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    text = f"File: {fname}\nThreshold: {threshold_uA:.3f} uA\nNo. Spikes: {len(spikes)}"
    plt.text(0.98, 0.95, text, transform=plt.gca().transAxes,
             fontsize=9, va='top', ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_spikes_with_baseline(dates, currents_uA, spikes_info, baseline_uA, save_path=None, show_plot=True):
    plt.figure(figsize=(14, 5))
    plt.plot(dates, currents_uA, label='Current', alpha=0.7)
    plt.plot(dates, baseline_uA, label='Baseline (rolling)', linestyle='--', color='black')
    for ts, mag, spike_val, baseline_val in spikes_info:
        plt.scatter(ts, spike_val, color='red', zorder=5, label='Spike' if 'Spike' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.vlines(ts, spike_val, baseline_val, color='orange', linestyle=':', label='Magnitude' if 'Magnitude' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.xlabel("Time")
    plt.ylabel("Current (uA)")
    plt.title("Detected Spikes with Rolling Baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close()

def exponential_pdf(t, tau):
    return (1/tau) * np.exp(-t/tau)

def plot_spike_intervals_and_fit(spikes, bins=30, time_unit='s', save_path=None, show_plot=True):
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
    plt.figure(figsize=(8, 5))
    plt.hist(delta_ts, bins=bins, density=True, alpha=0.5, histtype='step', label="Data", color="gray", edgecolor='black')
    t_fit = np.linspace(min(delta_ts), max(delta_ts), 300)
    plt.plot(t_fit, exponential_pdf(t_fit, *popt), 'r--', label=fr"Fit: $\tau$ = {tau_fit:.3f} ± {tau_err:.3f} ({time_unit})")
    plt.xlabel(f"Interval between spikes ({time_unit})")
    plt.ylabel("Probability Density")
    plt.title("Spike Interval Distribution with Exponential Fit")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_overall_intervals(times, bins=30, time_unit='min', title=None, save_path=None, show_plot=True, x_min=0, x_max=500):
    if times is None or len(times) < 2:
        print("Not enough timestamps to compute overall intervals.")
        return None, None
    times_sorted = sorted(times)
    deltas = np.diff([t.timestamp() for t in times_sorted])
    factor = {'s': 1, 'min': 60, 'h': 3600}[time_unit]
    deltas = deltas / factor
    in_range = (deltas >= x_min) & (deltas <= x_max)
    deltas_plot = deltas[in_range] if np.any(in_range) else deltas
    hist_vals, bin_edges = np.histogram(deltas_plot, bins=bins, range=(x_min, x_max), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_dt = np.mean(deltas_plot) if len(deltas_plot) else np.mean(deltas)
    try:
        popt, pcov = curve_fit(exponential_pdf, bin_centers, hist_vals, p0=[mean_dt])
        tau_fit = popt[0]
        tau_err = np.sqrt(np.diag(pcov))[0]
    except Exception as exc:
        print(f"Exponential fit failed: {exc}")
        tau_fit, tau_err = np.nan, np.nan
    plt.figure(figsize=(8, 5))
    plt.hist(deltas_plot, bins=bins, range=(x_min, x_max), density=True, alpha=0.5, histtype='step', color='gray', edgecolor='black', label='Data')
    if np.isfinite(tau_fit):
        t_fit = np.linspace(x_min, x_max, 300)
        plt.plot(t_fit, exponential_pdf(t_fit, tau_fit), 'r--', label=f"Fit: tau = {tau_fit:.3f} ± {tau_err:.3f} {time_unit}")
    plt.xlabel(f"Interval between spikes ({time_unit})")
    plt.ylabel("Probability Density")
    if not title:
        title = "Overall Inter-spike Interval Distribution"
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved overall interval plot to {save_path}")
    if show_plot:
        plt.show()
    plt.close()
    print(f"Overall mean interval: {mean_dt:.3f} {time_unit}")
    if np.isfinite(tau_fit):
        print(f"Overall fitted tau: {tau_fit:.4f} ± {tau_err:.4f} {time_unit}")
    return tau_fit, tau_err

def compute_total_discharge_charge(spikes, tau_fixed=6.6):
    total_charge = sum(mag * tau_fixed for _, mag, _, _ in spikes)
    if spikes:
        avg_charge = total_charge / len(spikes)
        magnitudes = np.array([mag for _, mag, _, _ in spikes])
        rms_magnitude = np.sqrt(np.mean(magnitudes**2))
        error_estimate = rms_magnitude * tau_fixed / np.sqrt(len(spikes))

        # Frequency calculation
        times = [ts for ts, *_ in spikes]
        if len(times) >= 2:
            duration_sec = (max(times) - min(times)).total_seconds()
            freq_hz = len(spikes) / duration_sec if duration_sec > 0 else float('nan')
            freq_per_hr = freq_hz * 3600 if duration_sec > 0 else float('nan')
            duration_hr = duration_sec / 3600.0 if duration_sec > 0 else float('nan')
            n_spikes = len(spikes)
            freq_per_hr = n_spikes / duration_hr if duration_hr > 0 else float('nan')
                # Poisson error: sqrt(N)/duration
            freq_per_hr_err = np.sqrt(n_spikes) / duration_hr if duration_hr > 0 else float('nan')
        else:
            pass
    else:
        pass
    return total_charge

def analyze_single_pair(beam_file, tco_file, threshold_uA=30e-3, tau_fixed=6.6, window_length=2500, polyorder=2, save_plots=False, show_plots=True, out_dir=None):
    dates, currents = load_combined_data(beam_file, tco_file)
    if not dates or len(dates) < 2:
        print("Not enough data in the provided files.")
        return
    currents_uA = currents * 1e6
    spikes, baseline = detect_spikes_with_savgol(
        dates,
        currents_uA,
        threshold_uA=threshold_uA,
        window_length=window_length,
        polyorder=polyorder
    )
    label = os.path.basename(beam_file) + " + " + os.path.basename(tco_file)
    dist_path = None
    if save_plots and out_dir:
        dist_path = os.path.join(out_dir, f"{os.path.basename(beam_file)}_{os.path.basename(tco_file)}_dist.png")
    plot_spike_distribution(spikes, label, threshold_uA, bins=20, value="magnitude", save_path=dist_path, show_plot=show_plots)
    base_path = None
    if save_plots and out_dir:
        base_path = os.path.join(out_dir, f"{os.path.basename(beam_file)}_{os.path.basename(tco_file)}_baseline.png")
    plot_spikes_with_baseline(dates, currents_uA, spikes, baseline, save_path=base_path, show_plot=show_plots)
    int_path = None
    if save_plots and out_dir:
        int_path = os.path.join(out_dir, f"{os.path.basename(beam_file)}_{os.path.basename(tco_file)}_intervals.png")
    plot_spike_intervals_and_fit(spikes, bins=25, time_unit='min', save_path=int_path, show_plot=show_plots)
    compute_total_discharge_charge(spikes, tau_fixed=tau_fixed)


def _extract_date_from_path(path, default_year=None):
    base = os.path.basename(path)
    name_match = re.search(r"_(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{1,2})", base)
    if not name_match:
        return None
    month_str, day_str = name_match.group(1), name_match.group(2)

    year = None
    for part in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"20\d{2}", part):
            year = int(part)
            break
    if year is None:
        year = default_year
    if year is None:
        return None
    try:
        return datetime.strptime(f"{year}-{month_str}-{day_str}", "%Y-%b-%d")
    except ValueError:
        return None

def _collect_by_date(file_paths, default_year):
    by_date = {}
    for path in file_paths:
        dt = _extract_date_from_path(path, default_year=default_year)
        if dt is not None:
            by_date[dt.date()] = path
    return by_date

def _filter_dates(dates, start_date=None, end_date=None):
    filtered = list(dates)
    if start_date:
        filtered = [d for d in filtered if d >= start_date.date()]
    if end_date:
        filtered = [d for d in filtered if d <= end_date.date()]
    return sorted(filtered)

def find_beam_tco_pairs(start_date=None, end_date=None):
    data_root = str(fcspikes_root())
    default_year = start_date.year if start_date else datetime.now().year
    beam_files = glob.glob(os.path.join(data_root, "csv", "beam", "**", "BEAM_*.csv"), recursive=True)
    tco_files = glob.glob(os.path.join(data_root, "csv", "tco", "**", "TCO_*.csv"), recursive=True)

    beam_by_date = _collect_by_date(beam_files, default_year)
    tco_by_date = _collect_by_date(tco_files, default_year)

    common_dates = sorted(set(beam_by_date) & set(tco_by_date))
    common_dates = _filter_dates(common_dates, start_date=start_date, end_date=end_date)

    return [(datetime.combine(d, datetime.min.time()), beam_by_date[d], tco_by_date[d]) for d in common_dates]

def list_available_dates(start_date=None, end_date=None):
    data_root = str(fcspikes_root())
    default_year = start_date.year if start_date else datetime.now().year
    beam_files = glob.glob(os.path.join(data_root, "csv", "beam", "**", "BEAM_*.csv"), recursive=True)
    tco_files = glob.glob(os.path.join(data_root, "csv", "tco", "**", "TCO_*.csv"), recursive=True)
    beam_by_date = _collect_by_date(beam_files, default_year)
    tco_by_date = _collect_by_date(tco_files, default_year)
    beam_dates = _filter_dates(beam_by_date.keys(), start_date=start_date, end_date=end_date)
    tco_dates = _filter_dates(tco_by_date.keys(), start_date=start_date, end_date=end_date)
    return beam_dates, tco_dates

def process_range_with_summary(pairs, data_dir, tau_fixed=6.6, threshold_uA=30e-3, window_length=2500, polyorder=2, save_plots=False, show_plots=True, summary_only=False):
    if not pairs:
        print("No BEAM/TCO pairs provided for range processing.")
        return

    all_spike_times = []
    segment_spike_counts = {}
    segment_spike_rates = {}
    segment_rate_errors = {}

    avg_charges = []
    charge_errors = []
    labels = []
    segment_mid_times = []

    data_span_start = None
    data_span_end = None

    start_day = pairs[0][0].strftime("%b%d")
    end_day = pairs[-1][0].strftime("%b%d")
    all_ts_filename = f"spike_timestamps_{start_day}_to_{end_day}_halfday.txt"
    all_ts_file = str(spike_timestamps_path(all_ts_filename))
    os.makedirs(os.path.dirname(all_ts_file), exist_ok=True)
    fout_all = open(all_ts_file, "w")

    for _, beam_file, tco_file in pairs:
        try:
            dates, currents = load_combined_data(beam_file, tco_file)
            if not dates:
                continue

            currents_uA = currents * 1e6

            if dates and CABLE_SWITCH_START is not None and CABLE_SWITCH_END is not None:
                mask = [not (CABLE_SWITCH_START <= ts < CABLE_SWITCH_END) for ts in dates]
                if not all(mask):
                    mask_arr = np.array(mask, dtype=bool)
                    dates = [ts for ts, keep in zip(dates, mask_arr) if keep]
                    currents_uA = currents_uA[mask_arr]
            if not dates:
                continue

            if data_span_start is None or dates[0] < data_span_start:
                data_span_start = dates[0]
            if data_span_end is None or dates[-1] > data_span_end:
                data_span_end = dates[-1]

            unique_dates = sorted({ts.date() for ts in dates})
            for actual_date in unique_dates:
                day_start_dt = datetime.combine(actual_date, datetime.min.time())
                segment_bounds = [
                    ("AM", day_start_dt, day_start_dt + timedelta(hours=12)),
                    ("PM", day_start_dt + timedelta(hours=12), day_start_dt + timedelta(days=1)),
                ]

                for suffix, seg_start, seg_end in segment_bounds:
                    seg_indices = [i for i, ts in enumerate(dates) if seg_start <= ts < seg_end]
                    if not seg_indices:
                        continue

                    seg_dates = [dates[i] for i in seg_indices]
                    seg_currents_uA = currents_uA[seg_indices]
                    segment_label = f"{actual_date.strftime('%b%d')}_{suffix}"
                    segment_mid = seg_start + (seg_end - seg_start) / 2

                    spikes, baseline = detect_spikes_with_savgol(
                        seg_dates,
                        seg_currents_uA,
                        threshold_uA=threshold_uA,
                        window_length=window_length,
                        polyorder=polyorder,
                    )

                    times = [entry[0] for entry in spikes]
                    segment_spike_counts[segment_label] = len(spikes)
                    if seg_dates:
                        duration_hr = (seg_dates[-1] - seg_dates[0]).total_seconds() / 3600.0
                        rate = len(spikes) / duration_hr if duration_hr > 0 else float('nan')
                        rate_err = np.sqrt(len(spikes)) / duration_hr if duration_hr > 0 else float('nan')
                        segment_spike_rates[segment_label] = rate
                        segment_rate_errors[segment_label] = rate_err

                    for ts, mag, _, _ in spikes:
                        charge = mag * tau_fixed
                        fout_all.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')}  {charge:.6f}\n")
                        all_spike_times.append(ts)

                    if not spikes:
                        continue

                    if not summary_only:
                        combined_label = f"{os.path.basename(beam_file)} + {os.path.basename(tco_file)} ({segment_label})"
                        safe_label = segment_label.replace(" ", "")

                        p_dist = os.path.join(data_dir, f"{safe_label}_dist.png") if save_plots else None
                        plot_spike_distribution(
                            spikes, combined_label, threshold_uA, bins=20, value="magnitude",
                            save_path=p_dist, show_plot=show_plots
                        )

                        p_base = os.path.join(data_dir, f"{safe_label}_baseline.png") if save_plots else None
                        plot_spikes_with_baseline(
                            seg_dates, seg_currents_uA, spikes, baseline, save_path=p_base, show_plot=show_plots
                        )

                        p_int = os.path.join(data_dir, f"{safe_label}_intervals.png") if save_plots else None
                        plot_spike_intervals_and_fit(
                            spikes, bins=25, time_unit='min', save_path=p_int, show_plot=show_plots
                        )

                        compute_total_discharge_charge(spikes, tau_fixed=tau_fixed)

                    magnitudes = np.array([mag for _, mag, _, _ in spikes])
                    avg_q = np.mean(magnitudes) * tau_fixed
                    err_q = np.std(magnitudes, ddof=1) * tau_fixed / np.sqrt(len(spikes))

                    segment_mid_times.append(segment_mid)
                    labels.append(segment_label)
                    avg_charges.append(avg_q)
                    charge_errors.append(err_q)

        except Exception as exc:
            print(f"Error processing {os.path.basename(beam_file)}: {exc}")
            continue

    fout_all.close()
    if not labels:
        print("No half-day segments with sufficient spikes for summary.")
        return

    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    n_spikes_per_segment = [segment_spike_counts.get(label, 0) for label in labels]
    spike_rates_per_segment = [segment_spike_rates.get(label, float('nan')) for label in labels]
    spike_rate_errors = [segment_rate_errors.get(label, float('nan')) for label in labels]

    ax[0].errorbar(
        segment_mid_times,
        avg_charges,
        yerr=charge_errors,
        fmt='o',
        color='black',
        capsize=5,
        label='Avg Charge/spike',
    )
    ax[0].set_ylabel("Charge (uC)")
    ax[0].set_title("Average Charge per Spike per Half-Day")
    ax[0].tick_params(axis='x', labelbottom=False)

    ax[1].errorbar(
        segment_mid_times,
        spike_rates_per_segment,
        yerr=spike_rate_errors,
        fmt='s',
        color='black',
        capsize=5,
        label='Spike Rate',
    )
    ax[1].set_ylabel("Spike Rate (per hour)")
    ax[1].set_title("Spike Rate per Half-Day")
    ax[1].set_xlabel("Time")

    locator = AutoDateLocator(minticks=3, maxticks=12)
    formatter = DateFormatter('%b %d\n%H:%M')
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    if data_span_start is not None and data_span_end is not None:
        span_start = max(data_span_start, PRESSURE_CHANGE_START)
        span_end = min(data_span_end, PRESSURE_CHANGE_END)
        if span_start <= span_end:
            ax[0].axvspan(span_start, span_end, color="tab:red", alpha=0.15, label="Pressure change")
            ax[1].axvspan(span_start, span_end, color="tab:red", alpha=0.15, label="Pressure change")

    ax[0].legend()
    ax[1].legend()

    fig.autofmt_xdate()
    plt.tight_layout()
    if save_plots:
        summary_filename = f"summary_{start_day}_to_{end_day}_halfday.png"
        summary_path = os.path.join(data_dir, summary_filename)
        fig.savefig(summary_path)
        print(f"Saved summary plot to {summary_path}")
    if show_plots:
        plt.show()
    plt.close()

    if not summary_only and len(all_spike_times) >= 2:
        interval_title = f"Inter-spike Intervals: {start_day} to {end_day} (half-day segments)"
        interval_path = os.path.join(data_dir, f"intervals_{start_day}_to_{end_day}_halfday.png") if save_plots else None
        plot_overall_intervals(
            all_spike_times,
            bins=200,
            time_unit='min',
            title=interval_title,
            save_path=interval_path,
            show_plot=show_plots,
        )
    elif not summary_only:
        print("Not enough spikes across period to plot overall intervals.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike analysis for BEAM/TCO data.")
    parser.add_argument("--mode", choices=["single", "range"], default="range", help="Process a single pair or a date range.")
    parser.add_argument("--beam-file", help="Path to a single BEAM csv (required for single mode).")
    parser.add_argument("--tco-file", help="Path to a single TCO csv (required for single mode).")
    parser.add_argument("--start-date", default="2026-01-28", help="Start date (YYYY-MM-DD) for range mode.")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD) for range mode.")
    parser.add_argument("--threshold-ua", type=float, default=30e-3, help="Spike threshold in uA.")
    parser.add_argument("--tau-fixed", type=float, default=6.6, help="Fixed tau in seconds.")
    parser.add_argument("--window-length", type=int, default=2500, help="Savitzky-Golay window length.")
    parser.add_argument("--polyorder", type=int, default=2, help="Savitzky-Golay polynomial order.")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to output directory.")
    parser.add_argument("--show-plots", "--show", action="store_true", help="Show plots interactively.")
    parser.add_argument("--summary-only", action="store_true", help="Only generate the final summary plot (avg charge and spike rate).")
    parser.add_argument("--out-dir", default=str(fcspikes_root()), help="Output directory for plots.")

    args = parser.parse_args()

    if args.mode == "single":
        if not args.beam_file or not args.tco_file:
            raise SystemExit("--beam-file and --tco-file are required in single mode.")
        analyze_single_pair(
            args.beam_file,
            args.tco_file,
            threshold_uA=args.threshold_ua,
            tau_fixed=args.tau_fixed,
            window_length=args.window_length,
            polyorder=args.polyorder,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            out_dir=args.out_dir,
        )
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None
        pairs = find_beam_tco_pairs(start_date=start_date, end_date=end_date)
        if not pairs:
            print("No BEAM/TCO pairs found for the given range.")
            beam_dates, tco_dates = list_available_dates(start_date=start_date, end_date=end_date)
            print("Available BEAM dates:", ", ".join(d.strftime("%Y-%m-%d") for d in beam_dates) or "None")
            print("Available TCO dates:", ", ".join(d.strftime("%Y-%m-%d") for d in tco_dates) or "None")
            raise SystemExit(0)
        process_range_with_summary(
            pairs,
            data_dir=args.out_dir,
            tau_fixed=args.tau_fixed,
            threshold_uA=args.threshold_ua,
            window_length=args.window_length,
            polyorder=args.polyorder,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            summary_only=args.summary_only,
        )
