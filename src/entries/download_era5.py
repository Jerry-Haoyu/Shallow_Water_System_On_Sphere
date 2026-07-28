import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

import cdsapi 
import yaml 
import time
from types import SimpleNamespace 

import numpy as np
import xarray as xr

from src.helpers.print import print_in_box

config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

era5_config = SimpleNamespace(config['download_era5']) 

years = np.arange(int(era5_config.year_start), 
                  int(era5_config.year_end) + 1, 
                  step=1).astype('str').tolist()

months = era5_config.months

days = [f"{day:02d}"for day in range(era5_config.day_start, era5_config.day_end + 1, 1)]

pressure_levels = np.arange(int(era5_config.pressure_level_start), 
                  int(era5_config.pressure_level_end) + 1, 
                  step=25).astype('str').tolist()

if era5_config.time_option in ["all", "whole_day", "everything", "every_hour"]:
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
elif era5_config.time_option in ["0000_only", "only_0000", "mid_night", "00:00", "midnight", "midNight"]:
    times = [
        "00:00"
    ]
else:
    raise RuntimeError("❌❌ The time_option in download_era5 in config.yml is invalid, either all or mid_night ❌❌")

vertical_integration = era5_config.vertical_integration if len(pressure_levels) >= 2 else False
    
# Cute loggings
years_log = f"Years={years[0]}...{years[-1]}" if len(years) >= 2 else f"Years={years[0]}"
months_log = f"Months={months}"
days_log = f"Days={days[0]}...{days[-1]}" if len(days) >= 2 else f"Days={days[0]}"
times_log = f"Times={times[0]}...{times[-1]}" if len(times) >= 2 else f"Days={times[0]}"
pressure_log = f'Pressue Levels={pressure_levels[0]}...{pressure_levels[-1]}' if len(pressure_levels) >= 2 else f'Pressure Levles={pressure_levels[0]}'
content = {
    "title" : "Download ERA5 Pressure-Level Dataset",
    "lines" : [
        years_log,
        months_log,
        days_log,
        times_log,
        pressure_log,
        f"File Format = NetCDF",
        f"Vertical Integration ⨜ₕ is {vertical_integration}",
        f"Saving path = {era5_config.save_path}"
    ]
}

print_in_box(content)
# Finish Loggins

start_time = time.perf_counter()

dataset = "reanalysis-era5-pressure-levels"
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
client.retrieve(dataset, request).download(era5_config.save_path)

end_time = time.perf_counter()
print(f" Finished Downloading ERA5 Dataset in {end_time - start_time:.3f} seconds")
if era5_config.vertical_integration is True:
    start_time = time.perf_counter()
    # xr.open_dataset() is lazy and keeps the file open for reading; writing
    # the averaged result back to that SAME path while it's still open would
    # collide with its own read handle (netCDF4/HDF5 file locking rejects it,
    # surfacing as a PermissionError). Load fully into memory and close the
    # read handle inside the `with` block before writing back to the same path.
    with xr.open_dataset(era5_config.save_path) as raw_ds:
        ds = raw_ds.mean(dim='pressure_level').load()
    ds.to_netcdf(era5_config.save_path)
    end_time = time.perf_counter()
    print(f" Finished Vertical Integration in {end_time - start_time:.3f} seconds")
    