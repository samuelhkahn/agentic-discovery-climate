#!/usr/bin/env python3
"""
Download CERES EBAF TOA Ed4.2 monthly data and convert to parquet format.

This downloads the NetCDF files from the CERES data portal and creates
parquet files matching the schema expected by the existing notebooks.

Data source: https://ceres.larc.nasa.gov/data/
Product: CERES_EBAF-TOA_Ed4.2 (monthly, 1° × 1°)
Period: March 2000 - present
"""

import os
import sys
from pathlib import Path
import subprocess

import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "CERES_2022-11-09_25176"
CERES_FILENAME = "CERES_EBAF-TOA_Edition4.2_200003-202407.nc"
CERES_URL = f"https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2/{CERES_FILENAME}"
# Requires NASA Earthdata login: https://urs.earthdata.nasa.gov/
CERES_ALT_URL = "https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp"


def download_ceres_netcdf():
    """Download CERES EBAF TOA Ed4.2 NetCDF file.

    Requires NASA Earthdata credentials. Set up a ~/.netrc file with:
        machine urs.earthdata.nasa.gov
        login YOUR_USERNAME
        password YOUR_PASSWORD

    Or register at: https://urs.earthdata.nasa.gov/
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nc_path = DATA_DIR / CERES_FILENAME

    if nc_path.exists() and nc_path.stat().st_size > 1_000_000:
        print(f"CERES NetCDF already exists: {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")
        return nc_path

    # Check for .netrc credentials
    netrc_path = Path.home() / ".netrc"
    has_netrc = netrc_path.exists() and "urs.earthdata.nasa.gov" in netrc_path.read_text()

    print(f"Downloading CERES EBAF TOA Ed4.2...")
    print(f"File: {CERES_FILENAME}")
    print(f"URL: {CERES_URL}")
    print(f"Destination: {nc_path}")
    print()

    if not has_netrc:
        print("NASA Earthdata credentials needed. Two options:")
        print()
        print("OPTION 1: Create ~/.netrc file:")
        print("  1. Register at https://urs.earthdata.nasa.gov/")
        print("  2. Create ~/.netrc with:")
        print("     machine urs.earthdata.nasa.gov")
        print("     login YOUR_USERNAME")
        print("     password YOUR_PASSWORD")
        print("  3. Run this script again")
        print()
        print("OPTION 2: Manual download:")
        print(f"  1. Go to: {CERES_URL}")
        print(f"  2. Log in with Earthdata credentials")
        print(f"  3. Save the file to: {nc_path}")
        print()
        print(f"OPTION 3: Use CERES ordering tool:")
        print(f"  {CERES_ALT_URL}")
        print(f"  Select: EBAF-TOA Ed4.2, All months, All variables, NetCDF")
        print(f"  Save to: {nc_path}")
        return None

    try:
        # Use curl with .netrc for Earthdata auth
        subprocess.run(
            ["curl", "-n", "-L", "-c", "/tmp/cookies.txt", "-b", "/tmp/cookies.txt",
             "-o", str(nc_path), CERES_URL],
            check=True, timeout=600
        )
        if nc_path.stat().st_size > 1_000_000:
            print(f"Downloaded: {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")
            return nc_path
        else:
            print("Download resulted in a small file (possibly auth error).")
            nc_path.unlink(missing_ok=True)
            return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Download failed: {e}")
        print("Please download manually.")
        return None


def netcdf_to_parquet(nc_path):
    """Convert CERES NetCDF to parquet format matching notebook expectations."""
    import xarray as xr
    import pandas as pd

    print(f"Converting {nc_path} to parquet...")

    ds = xr.open_dataset(nc_path)
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")

    # Flatten to dataframe with (time, lat, lon) as columns
    df = ds.to_dataframe().reset_index()

    # Rename columns to match existing notebook conventions
    # CERES EBAF variable names → notebook names
    rename_map = {
        "toa_sw_all_mon": "toa_sw_all_mon",
        "toa_sw_clr_t_mon": "toa_sw_clr_mon",
        "solar_mon": "solar_mon",
        "cldarea_total_mon": "cldarea_total_mon",
        "cldtau_total_mon": "cldtau_total_mon",
        "cldtau_lin_total_mon": "cldtau_lin_total_mon",
        "lwp_total_mon": "lwp_total_mon",
        "iwp_total_mon": "iwp_total_mon",
        "cldwatrad_total_mon": "cldwatrad_total_mon",
        "cldicerad_total_mon": "cldicerad_total_mon",
        "ini_aod55_mon": "ini_aod55_mon",
        "ini_precip_mon": "ini_precip_mon",
        "ini_albedo_mon": "ini_albedo_mon",
        "aux_snow_mon": "aux_snow_mon",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Compute derived columns
    if "solar_mon" in df.columns and "toa_sw_all_mon" in df.columns:
        df["toa_alb_all_mon"] = df["toa_sw_all_mon"] / df["solar_mon"].replace(0, np.nan)

    # Add convenience columns
    if "time" in df.columns:
        df["year"] = pd.to_datetime(df["time"]).dt.year
        df["month"] = pd.to_datetime(df["time"]).dt.month

    if "lat" in df.columns:
        df["northern_hemisphere"] = (df["lat"] > 0).astype(int)

    # Drop rows with NaN in target
    if "toa_alb_all_mon" in df.columns:
        df = df.dropna(subset=["toa_alb_all_mon"])

    # Split into two parquet files (matching existing convention)
    midpoint = len(df) // 2
    p1 = DATA_DIR / "ceres.parquet.gzip"
    p2 = DATA_DIR / "ceres1.parquet.gzip"

    df.iloc[:midpoint].to_parquet(p1, compression="gzip")
    df.iloc[midpoint:].to_parquet(p2, compression="gzip")

    print(f"Saved {len(df)} rows total:")
    print(f"  {p1} ({p1.stat().st_size / 1e6:.1f} MB)")
    print(f"  {p2} ({p2.stat().st_size / 1e6:.1f} MB)")
    print(f"Columns: {list(df.columns)}")

    return df


def main():
    nc_path = download_ceres_netcdf()
    if nc_path and nc_path.exists():
        try:
            df = netcdf_to_parquet(nc_path)
            print(f"\nDone! {len(df)} rows ready for analysis.")
        except ImportError as e:
            print(f"\nMissing dependency: {e}")
            print("Install with: pip install xarray netCDF4")
    else:
        print("\nTo proceed, place the CERES EBAF NetCDF file at:")
        print(f"  {DATA_DIR}/CERES_EBAF-TOA_Ed4.2_Subset_200003-202312.nc")


if __name__ == "__main__":
    main()
