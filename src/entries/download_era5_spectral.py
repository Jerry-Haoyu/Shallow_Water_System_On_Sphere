#!/data/gzhang13/a/hytang2/envs/swe/bin/python
"""
Download raw spectral (spherical-harmonic) ERA5 pressure-level fields from
ECMWF's MARS tape archive via the ``reanalysis-era5-complete`` CDS dataset -
distinct from download_era5.py, which pulls GRIDDED u/v from the regular
``reanalysis-era5-pressure-levels`` dataset for the solver's initial
conditions.

ERA5 archives several pressure-level fields (geopotential, temperature, u, v,
vertical velocity, vorticity, divergence, relative humidity) NATIVELY as
spherical-harmonic coefficients at T639 triangular truncation - before any
interpolation to a lat-lon grid happens. Leaving the MARS 'grid'/'area' keys
unset keeps the retrieval in that native spectral representation. This script
exists to pull that raw spectral data directly (e.g. as ground truth for
validating torch_harmonics' SHT, or as an alternative spectral IC source),
rather than deriving (Phi, zeta, delta) locally via a forward SHT on gridded
(u, v) the way src/numerical_solver/initial_condition.py does.

Output is GRIB only - spectral coefficients have no netCDF representation.
Because a single request can take hours-to-days to fulfill from ECMWF's tape
archive, the download is split into per-month (or per-year) chunks tracked in
a manifest.json, so a multi-day download can be safely resumed across many
separate `make download_era5_spectral` invocations.

Layout (mirrors the reanalysis_data/<dataset_name>/ convention):

    reanalysis_data/<dataset_name>/chunks/<label>.grib   per-chunk GRIB
    reanalysis_data/<dataset_name>/manifest.json         resume state
    reanalysis_data/<dataset_name>/data.grib             merged chunks

Configuration is read from the ``download_era5_spectral`` section of
config.yml (pass an alternative path as the first CLI argument).

@author Haoyu Tang hytang2@illinois.edu
"""

import sys
import json
import calendar
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

from types import SimpleNamespace

import cdsapi
import yaml

from src.helpers.print import print_in_box


# Native spherical-harmonic pressure-level parameters in ERA5-complete (MARS
# param codes) - see ECMWF's ERA5 data documentation. u_wind/v_wind/
# relative_humidity are included for completeness even though this repo
# currently only needs vorticity/divergence/geopotential/temperature.
VARIABLE_PARAM_CODES = {
    "geopotential": "129",
    "temperature": "130",
    "u_wind": "131",
    "v_wind": "132",
    "vertical_velocity": "135",
    "vorticity": "138",
    "divergence": "155",
    "relative_humidity": "157",
}

# ERA5's finest native temporal resolution is hourly (no 30-minute analyses
# exist) - "all"/"hourly" both mean every hour.
TIME_PRESETS = {
    "midnight": ["00:00"],
    "6hourly": ["00:00", "06:00", "12:00", "18:00"],
    "hourly": [f"{h:02d}:00" for h in range(24)],
    "all": [f"{h:02d}:00" for h in range(24)],
}

MARS_DATASET = "reanalysis-era5-complete"

DEFAULT_CONFIG = {
    "dataset_name": None,        # required, e.g. "1970_2025_sparse_500_vo_d_z_t128"
    "year_start": None,          # required
    "year_end": None,            # required
    "months": None,              # required, e.g. [1, 2, ..., 12]
    "day_start": 1,
    "day_end": 31,               # clipped per-month to the actual number of days
    "time_option": "6hourly",    # midnight | 6hourly | hourly | all
    "hours": None,               # explicit override, e.g. ["00:00", "12:00"]
    "pressure_levels": None,     # required, e.g. [500]
    "variables": None,           # required, e.g. ["vorticity", "divergence", "geopotential"]
    "truncation": None,          # spectral truncation (e.g. 128); None/"none" = native T639
    "truncation_fallback": "local",  # "local" (cdo sp2sp) | "error"
    "chunk_by": "month",         # month | year
    "max_concurrent_chunks": 1,  # >1 submits that many chunk requests in flight at once
    "dry_run": False,
    "max_total_size_gb": None,   # required safety guard
    "merge_chunks": True,
    "overwrite": False,
    "keep_native_chunks": False,
}


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}

    raw = DEFAULT_CONFIG | config.get("download_era5_spectral", {})
    cfg = SimpleNamespace(**raw)

    if not cfg.dataset_name:
        raise ValueError("download_era5_spectral.dataset_name is required.")
    if cfg.year_start is None or cfg.year_end is None:
        raise ValueError("download_era5_spectral.year_start / year_end are required.")
    if not cfg.months:
        raise ValueError("download_era5_spectral.months is required.")
    if not cfg.pressure_levels:
        raise ValueError("download_era5_spectral.pressure_levels is required, e.g. [500].")
    if not cfg.variables:
        raise ValueError("download_era5_spectral.variables is required, e.g. ['vorticity'].")
    unknown = [v for v in cfg.variables if v not in VARIABLE_PARAM_CODES]
    if unknown:
        raise ValueError(f"Unknown variable(s) {unknown}; known: {sorted(VARIABLE_PARAM_CODES)}")
    if cfg.truncation in ("none", "None", 0):
        cfg.truncation = None
    if cfg.chunk_by not in ("month", "year"):
        raise ValueError("download_era5_spectral.chunk_by must be 'month' or 'year'.")
    if cfg.max_total_size_gb is None:
        raise ValueError("download_era5_spectral.max_total_size_gb is required (safety guard).")
    if cfg.max_concurrent_chunks < 1:
        raise ValueError("download_era5_spectral.max_concurrent_chunks must be >= 1.")
    return cfg


def resolve_hours(cfg):
    if cfg.hours:
        hours = cfg.hours
    elif cfg.time_option in TIME_PRESETS:
        hours = TIME_PRESETS[cfg.time_option]
    else:
        raise ValueError(
            f"download_era5_spectral.time_option={cfg.time_option!r} is invalid; "
            f"use one of {list(TIME_PRESETS)} or set 'hours' explicitly."
        )
    return [h.split(":")[0] for h in hours]  # MARS wants "HH", not "HH:MM"


def month_dates(year, month, day_start, day_end):
    last_day = calendar.monthrange(year, month)[1]
    end = min(day_end, last_day)
    return [f"{year:04d}-{month:02d}-{day:02d}" for day in range(day_start, end + 1)]


def build_chunks(cfg):
    """One chunk per (year, month), or per year if chunk_by == 'year'; each
    chunk carries the explicit list of MARS 'date' strings it covers (not a
    contiguous 'to' range, since day_start/day_end restricts each month)."""
    years = range(int(cfg.year_start), int(cfg.year_end) + 1)
    chunks = []
    if cfg.chunk_by == "month":
        for year in years:
            for month in cfg.months:
                dates = month_dates(year, month, cfg.day_start, cfg.day_end)
                if dates:
                    chunks.append({"label": f"{year:04d}_{month:02d}", "dates": dates})
    else:  # year
        for year in years:
            dates = []
            for month in cfg.months:
                dates.extend(month_dates(year, month, cfg.day_start, cfg.day_end))
            if dates:
                chunks.append({"label": f"{year:04d}", "dates": dates})
    return chunks


def build_request(cfg, dates, hours, param_codes, truncation):
    request = {
        "class": "ea",
        "stream": "oper",
        "type": "an",
        "expver": "1",
        "levtype": "pl",
        "levelist": "/".join(str(p) for p in cfg.pressure_levels),
        "param": "/".join(param_codes),
        "date": "/".join(dates),
        "time": "/".join(hours),
        # deliberately NO 'grid' / 'area' keys -> stays in spherical-harmonic
        # representation instead of being interpolated to a lat-lon grid.
    }
    if truncation:
        request["truncation"] = str(truncation)
    return request


def load_manifest(manifest_path):
    if manifest_path.is_file():
        with open(manifest_path) as f:
            return json.load(f)
    return {"truncation_mode": None, "chunks": {}}


def save_manifest(manifest_path, manifest):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def check_grib_truncation(path, expected_truncation):
    """Inspect the first GRIB message's gridType/truncation via eccodes."""
    import eccodes
    with open(path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
            raise RuntimeError(f"{path} contains no GRIB messages.")
        try:
            grid_type = eccodes.codes_get(gid, "gridType")
            truncation = eccodes.codes_get(gid, "pentagonalResolutionParameterJ")
        finally:
            eccodes.codes_release(gid)
    is_spectral = grid_type == "sh"
    matches = truncation == int(expected_truncation)
    return is_spectral, matches, grid_type, truncation


def download_result(cfg, result, label, truncation_mode, chunks_dir):
    """Materialize an already-submitted cdsapi Result to chunks_dir/<label>.grib,
    applying the local cdo truncation fallback if truncation_mode == 'local'.
    Shared by the real download loop and the dry-run/probe paths so there is
    exactly one place that turns a Result into a stored (possibly truncated)
    chunk file."""
    out_path = chunks_dir / f"{label}.grib"
    if truncation_mode == "local":
        native_path = chunks_dir / f"{label}_native.grib"
        result.download(str(native_path))
        local_truncate(native_path, out_path, cfg.truncation)
        if not cfg.keep_native_chunks:
            native_path.unlink()
    else:
        result.download(str(out_path))
    return out_path


def local_truncate(native_path, out_path, truncation):
    subprocess.run(
        ["cdo", f"sp2sp,{truncation}", str(native_path), str(out_path)],
        check=True,
    )


def probe_truncation(cfg, client, hours, manifest, manifest_path, dataset_dir):
    """One-time, cached-in-manifest check of whether MARS honors server-side
    'truncation' while keeping output spectral. Skipped entirely when no
    truncation is requested (native T639 needs no verification)."""
    if cfg.truncation is None:
        manifest["truncation_mode"] = "native"
        save_manifest(manifest_path, manifest)
        return "native"
    if manifest.get("truncation_mode"):
        return manifest["truncation_mode"]

    probe_dates = month_dates(cfg.year_start, cfg.months[0], cfg.day_start, cfg.day_end)[:1]
    param_code = VARIABLE_PARAM_CODES[cfg.variables[0]]
    request = build_request(cfg, probe_dates, hours[:1], [param_code], cfg.truncation)
    print_in_box({"title": "Probing MARS server-side spectral truncation", "lines": [str(request)]})

    result = client.retrieve(MARS_DATASET, request)
    probe_path = dataset_dir / "_truncation_probe.grib"
    result.download(str(probe_path))
    is_spectral, matches, grid_type, truncation = check_grib_truncation(probe_path, cfg.truncation)
    probe_path.unlink()

    if is_spectral and matches:
        mode = "server"
    else:
        mode = "local"
        print_in_box({
            "title": "Server-side truncation NOT confirmed",
            "lines": [
                f"gridType={grid_type}, truncation={truncation} (wanted sh @ T{cfg.truncation})",
                f"Falling back to truncation_fallback={cfg.truncation_fallback!r}",
            ],
        })
        if cfg.truncation_fallback == "error":
            raise RuntimeError(
                "MARS did not honor server-side truncation and "
                "truncation_fallback='error'; set it to 'local' to auto-truncate "
                "downloaded native-resolution chunks with cdo instead."
            )
    manifest["truncation_mode"] = mode
    save_manifest(manifest_path, manifest)
    return mode


def _reserve_budget(lock, state, size, max_total_bytes):
    """Atomically check-and-reserve `size` bytes against the shared running
    total; the first caller to push the total over budget flips
    state["stopped"] so every other concurrent worker also backs off instead
    of each independently racing past the limit."""
    with lock:
        if state["stopped"]:
            return False
        if state["running_total"] + size > max_total_bytes:
            state["stopped"] = True
            return False
        state["running_total"] += size
        return True


def _download_one_chunk(cfg, chunk, hours, param_codes, request_truncation,
                         truncation_mode, chunks_dir, max_total_bytes,
                         lock, state, manifest, manifest_path):
    """One (year, month) chunk's full request/quota-check/download/manifest-
    update cycle, safe to run concurrently across chunks - each call opens
    its own cdsapi.Client() (a Client's requests.Session isn't meant to be
    shared across threads) and only touches shared state inside `lock`.
    Returns (label, bytes_written) or (label, None) if skipped over budget.
    """
    client = cdsapi.Client()
    request = build_request(cfg, chunk["dates"], hours, param_codes, request_truncation)
    result = client.retrieve(MARS_DATASET, request)

    # Reserve against the *pre-download* content_length first (the only size
    # known before transferring anything); corrected to the real on-disk size
    # once download_result() returns, since local truncation can shrink it.
    if not _reserve_budget(lock, state, result.content_length, max_total_bytes):
        return chunk["label"], None

    out_path = download_result(cfg, result, chunk["label"], truncation_mode, chunks_dir)
    chunk_bytes = out_path.stat().st_size
    with lock:
        state["running_total"] += chunk_bytes - result.content_length
        manifest["chunks"][chunk["label"]] = {"status": "done", "bytes": chunk_bytes}
        save_manifest(manifest_path, manifest)
    return chunk["label"], chunk_bytes


def merge_chunks(dataset_dir, manifest):
    chunk_paths = [
        dataset_dir / "chunks" / f"{label}.grib"
        for label, info in sorted(manifest["chunks"].items())
        if info["status"] == "done"
    ]
    if not chunk_paths:
        return None
    merged_path = dataset_dir / "data.grib"
    with open(merged_path, "wb") as out:
        for path in chunk_paths:
            with open(path, "rb") as f:
                out.write(f.read())
    return merged_path


def main():
    cfg = load_config()
    hours = resolve_hours(cfg)
    param_codes = [VARIABLE_PARAM_CODES[v] for v in cfg.variables]

    dataset_dir = Path("reanalysis_data") / cfg.dataset_name
    chunks_dir = dataset_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    chunks = build_chunks(cfg)
    max_total_bytes = cfg.max_total_size_gb * (1024 ** 3)

    print_in_box({
        "title": "Download ERA5-complete Raw Spectral Dataset",
        "lines": [
            f"dataset_name = {cfg.dataset_name}",
            f"Years={cfg.year_start}...{cfg.year_end}, Months={cfg.months}",
            f"Days={cfg.day_start}...{cfg.day_end} (per month), Hours={hours}",
            f"Pressure Levels={cfg.pressure_levels} hPa",
            f"Variables={cfg.variables}",
            f"Truncation={'T' + str(cfg.truncation) if cfg.truncation else 'native (T639)'}",
            f"Chunks={len(chunks)} ({cfg.chunk_by}), max_total_size={cfg.max_total_size_gb} GB",
            f"dry_run={cfg.dry_run}",
        ],
    })

    client = cdsapi.Client()
    truncation_mode = probe_truncation(cfg, client, hours, manifest, manifest_path, dataset_dir)
    print(f"Truncation mode: {truncation_mode}")

    pending = [
        c for c in chunks
        if cfg.overwrite or manifest["chunks"].get(c["label"], {}).get("status") != "done"
    ]
    if not pending:
        print("Nothing pending; all chunks already done.")
    request_truncation = cfg.truncation if truncation_mode == "server" else None

    if cfg.dry_run:
        if not pending:
            return
        chunk = pending[0]
        request = build_request(cfg, chunk["dates"], hours, param_codes, request_truncation)
        result = client.retrieve(MARS_DATASET, request)
        content_length = result.content_length
        out_path = download_result(cfg, result, chunk["label"], truncation_mode, chunks_dir)
        actual_bytes = out_path.stat().st_size
        out_path.unlink()
        est_total_bytes = content_length * len(chunks)
        print_in_box({
            "title": "Dry-run size estimate",
            "lines": [
                f"Sample chunk '{chunk['label']}' downloaded = {content_length / 1024**2:.1f} MB"
                + (f" (stored after local truncation = {actual_bytes / 1024**2:.1f} MB)"
                   if truncation_mode == "local" else ""),
                f"Extrapolated total over {len(chunks)} chunks (pre-truncation-fallback size) "
                f"= {est_total_bytes / 1024**3:.2f} GB",
                f"Budget (max_total_size_gb) = {cfg.max_total_size_gb} GB",
            ],
        })
        return

    # Concurrency is opt-in (max_concurrent_chunks defaults to 1, i.e. today's
    # sequential behavior): the CDS/MARS backend runs many users' requests
    # concurrently already, so submitting a *few* chunks at once overlaps
    # their queue-wait time, but requests can't be steered onto different
    # tapes from the client side, so it's not a guaranteed speedup, and going
    # wide risks tripping MARS's "inefficient request" policy - keep this
    # modest (e.g. 2-4), not maxed out.
    lock = threading.Lock()
    state = {
        "running_total": sum(
            info.get("bytes", 0) for info in manifest["chunks"].values() if info["status"] == "done"
        ),
        "stopped": False,
    }
    stopped_reported = False

    with ThreadPoolExecutor(max_workers=cfg.max_concurrent_chunks) as pool:
        futures = {
            pool.submit(_download_one_chunk, cfg, chunk, hours, param_codes, request_truncation,
                        truncation_mode, chunks_dir, max_total_bytes, lock, state,
                        manifest, manifest_path): chunk
            for chunk in pending
        }
        for future in as_completed(futures):
            label, chunk_bytes = future.result()  # re-raises any worker exception here
            if chunk_bytes is None:
                if not stopped_reported:
                    print_in_box({
                        "title": "Stopping: max_total_size_gb would be exceeded",
                        "lines": [
                            f"Downloaded so far = {state['running_total'] / 1024**3:.2f} GB",
                            f"Skipped chunk '{label}' and any not yet started.",
                            f"Budget = {cfg.max_total_size_gb} GB",
                            "Completed chunks and manifest.json are preserved; raise the "
                            "budget or narrow the config to continue.",
                        ],
                    })
                    stopped_reported = True
                continue
            print(f"Finished chunk {label} ({chunk_bytes / 1024**2:.1f} MB); "
                  f"running total = {state['running_total'] / 1024**3:.2f} GB")

    if cfg.merge_chunks:
        merged_path = merge_chunks(dataset_dir, manifest)
        if merged_path:
            print(f"Finished merging chunks -> {merged_path}")


if __name__ == "__main__":
    main()
