from __future__ import annotations

import os
import re
from pathlib import Path


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
    # New layout: <root>/csv/beam/<Month>/BEAM_*.csv
    return str(fcspikes_root() / "csv" / "beam" / "*" / "BEAM_*.csv")


def tco_csv_glob() -> str:
    return str(fcspikes_root() / "csv" / "tco" / "*" / "TCO_*.csv")


def find_beam_csv(day_tag: str) -> Path:
    """Find a BEAM csv by tag, e.g. 'Oct29' -> .../BEAM_Oct29.csv."""
    pattern = f"BEAM_{day_tag}.csv"
    matches = list((fcspikes_root() / "csv" / "beam").glob(f"*/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {fcspikes_root()}/csv/beam")
    return sorted(matches)[0]


def find_tco_csv(day_tag: str) -> Path:
    pattern = f"TCO_{day_tag}.csv"
    matches = list((fcspikes_root() / "csv" / "tco").glob(f"*/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {fcspikes_root()}/csv/tco")
    return sorted(matches)[0]


def spike_timestamps_path(filename: str) -> Path:
    return fcspikes_root() / "txt" / "spike_timestamps" / filename


def figures_root() -> Path:
    return (project_root() / "figures").resolve()


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
