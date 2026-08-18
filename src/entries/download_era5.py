#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Download an ERA5 pressure-level dataset and lay it out per the reanalysis_data
convention: everything for one dataset lives under its own directory, named
after the dataset (e.g. "1980_2025_odd_month"), holding

    reanalysis_data/<dataset_name>/data.nc            the ERA5 netCDF itself
    reanalysis_data/<dataset_name>/h_stats.npz         per-time h_avg/h_amp
    reanalysis_data/<dataset_name>/h_stats_plot.png    (both from src/analyze/statistics.py)

Downstream tasks (run_solver.py, batch_simulation.py, inference.py) then refer
to the dataset by this dataset_name rather than a raw .nc path.

Configuration is read from the ``download_era5`` section of config.yml (pass an
alternative path as the first CLI argument).

@author Haoyu Tang hytang2@illinois.edu
"""

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

from types import SimpleNamespace

import cdsapi
import numpy as np
import xarray as xr
import yaml

from src.helpers.print import print_in_box
from src.analyze.statistics import plot_h_stats_ic


DEFAULT_CONFIG = {
    "dataset_name": None,       # required, e.g. "1980_2025_odd_month" -> reanalysis_data/<name>/
    "year_start": None,         # required
    "year_end": None,           # required
    "months": None,             # required, e.g. [1, 3, 5, 7, 9, 11]
    "day_start": None,
    "day_end": None,
    "time_option": None,  # midnight | all
    "pressure_level_start": None,   # required
    "pressure_level_end": None,     # required
    "vertical_integration": False,
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}

    raw = DEFAULT_CONFIG | config.get("download_era5", {})
    cfg = SimpleNamespace(**raw)

    if not cfg.dataset_name:
        raise ValueError("download_era5.dataset_name is required, e.g. '1980_2025_odd_month'.")
    if cfg.year_start is None or cfg.year_end is None:
        raise ValueError("download_era5.year_start / year_end are required.")
    if not cfg.months:
        raise ValueError("download_era5.months is required.")
    if cfg.pressure_level_start is None or cfg.pressure_level_end is None:
        raise ValueError("download_era5.pressure_level_start / pressure_level_end are required.")
    return cfg


def main():
    cfg = load_config()

    dataset_dir = Path("reanalysis_data") / cfg.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_path = dataset_dir / "data.nc"

    years = np.arange(int(cfg.year_start), int(cfg.year_end) + 1, step=1).astype('str').tolist()
    months = cfg.months
    if cfg.day_start is not None and cfg.day_end is not None:
        days = [f"{day:02d}" for day in range(cfg.day_start, cfg.day_end + 1, 1)]
    elif cfg.days is not None:
        days = cfg.days 
    else: 
        raise ValueError("Need either days or day_start, day_end to specify dates to download")
    pressure_levels = np.arange(
        int(cfg.pressure_level_start), int(cfg.pressure_level_end) + 1, step=25
    ).astype('str').tolist()

    if cfg.time_option in ["all", "whole_day", "everything", "every_hour"]:
        times = [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ]
    elif cfg.time_option in ["0000_only", "only_0000", "mid_night", "00:00", "midnight", "midNight"]:
        times = ["00:00"]
    else:
        raise RuntimeError("❌❌ The time_option in download_era5 in config.yml is invalid, either all or mid_night ❌❌")

    vertical_integration = cfg.vertical_integration if len(pressure_levels) >= 2 else False

    # Cute loggings
    years_log = f"Years={years[0]}...{years[-1]}" if len(years) >= 2 else f"Years={years[0]}"
    months_log = f"Months={months}"
    days_log = f"Days={days[0]}...{days[-1]}" if len(days) >= 2 else f"Days={days[0]}"
    times_log = f"Times={times[0]}...{times[-1]}" if len(times) >= 2 else f"Days={times[0]}"
    pressure_log = f'Pressue Levels={pressure_levels[0]}...{pressure_levels[-1]}' if len(pressure_levels) >= 2 else f'Pressure Levles={pressure_levels[0]}'
    print_in_box({
        "title": "Download ERA5 Pressure-Level Dataset",
        "lines": [
            f"dataset_name = {cfg.dataset_name}",
            years_log,
            months_log,
            days_log,
            times_log,
            pressure_log,
            f"File Format = NetCDF",
            f"Vertical Integration ⨜ₕ is {vertical_integration}",
            f"Saving path = {save_path}"
        ],
    })

    if save_path.is_file():
        print(f"✅ {save_path} already exists; skipping download.")
    else:
        dataset =  "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind"
            ],
            "year": years,
            "month": months,
            "day": days,
            "time": times,
            "pressure_level": pressure_levels,
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download(str(save_path))
        print(f" Finished Downloading ERA5 Dataset -> {save_path}")

        if vertical_integration is True:
            # xr.open_dataset() is lazy and keeps the file open for reading; writing
            # the averaged result back to that SAME path while it's still open would
            # collide with its own read handle (netCDF4/HDF5 file locking rejects it,
            # surfacing as a PermissionError). Load fully into memory and close the
            # read handle inside the `with` block before writing back to the same path.
            with xr.open_dataset(save_path) as raw_ds:
                ds = raw_ds.mean(dim='pressure_level').load()
            ds.to_netcdf(save_path)
            print(f" Finished Vertical Integration -> {save_path}")

    # ------------------------------------------------------------------ #
    # h_avg/h_amp diagnostics for this dataset (feeds the solver's
    # non-dimensionalization downstream) -> h_stats.npz + h_stats_plot.png
    # ------------------------------------------------------------------ #
    plot_h_stats_ic(cfg.dataset_name)


if __name__ == "__main__":
    main()
