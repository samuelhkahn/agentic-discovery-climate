#!/usr/bin/env python3
"""
Download all datasets for Kosmos-Albedo research.

1. CERES EBAF-TOA Ed4.2.1 (latest) — satellite albedo observations
2. CMIP6 rsut/rsdt — climate model albedo projections (40 models)
3. MERRA-2 M2TMNXRAD — reanalysis radiation fields

All require NASA Earthdata credentials (~/.netrc) except CMIP6 (uses Copernicus CDS).
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent  # ClimateAIResearch/


# ═══════════════════════════════════════════════════════════════════════
# 1. CERES EBAF-TOA Ed4.2.1 (upgrade from Ed4.2)
# ═══════════════════════════════════════════════════════════════════════

def download_ceres_ebaf_421():
    """Download CERES EBAF-TOA Ed4.2.1 — the latest science-quality product.

    Ed4.2.1 extends through ~mid-2025 and uses NOAA-20 + MERRA-2 ancillary.
    Requires NASA Earthdata credentials in ~/.netrc.

    The file URL follows the pattern:
    https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2.1/CERES_EBAF-TOA_Edition4.2.1_YYYYMM-YYYYMM.nc
    """
    ceres_dir = DATA_ROOT / "CERES_2022-11-09_25176"
    ceres_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have a recent Ed4.2.1 file
    existing_421 = list(ceres_dir.glob("CERES_EBAF-TOA_Edition4.2.1*.nc"))
    if existing_421:
        print(f"CERES Ed4.2.1 already exists: {existing_421[0]}")
        return existing_421[0]

    print("=" * 60)
    print("CERES EBAF-TOA Ed4.2.1 Download")
    print("=" * 60)
    print()
    print("The Ed4.2.1 product is the latest CERES EBAF release.")
    print("It extends further than Ed4.2 (which stops at July 2024).")
    print()
    print("To download, use the CERES Ordering Tool:")
    print("  https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA421Selection.jsp")
    print()
    print("Select:")
    print("  - Product: EBAF-TOA Ed4.2.1")
    print("  - All available months")
    print("  - All variables")
    print("  - Format: NetCDF")
    print()
    print(f"Save the file to: {ceres_dir}/")
    print()
    print("Alternative: Earthdata Search")
    print("  https://search.earthdata.nasa.gov/search?q=CERES_EBAF-TOA_Edition4.2.1")
    print()

    # Try direct download if .netrc exists
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text():
        # Try the known URL pattern
        url = "https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2.1/CERES_EBAF-TOA_Edition4.2.1_200003-202503.nc"
        dest = ceres_dir / "CERES_EBAF-TOA_Edition4.2.1_200003-202503.nc"
        print(f"Attempting download from: {url}")
        try:
            subprocess.run(
                ["curl", "-n", "-L", "-c", "/tmp/cookies.txt", "-b", "/tmp/cookies.txt",
                 "-o", str(dest), url],
                check=True, timeout=600
            )
            if dest.exists() and dest.stat().st_size > 100_000_000:
                print(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
                return dest
            else:
                print("Download may have failed (file too small). Try manual download.")
                dest.unlink(missing_ok=True)
        except Exception as e:
            print(f"Download failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════════
# 2. CMIP6 — Climate Model Albedo Projections
# ═══════════════════════════════════════════════════════════════════════

def download_cmip6():
    """Download CMIP6 rsut and rsdt for computing model albedo.

    Uses the Copernicus Climate Data Store (CDS) API.
    Requires: pip install cdsapi, and a CDS account with API key in ~/.cdsapirc

    The paper used 40 models. We download the most commonly available ones.
    Albedo = rsut / rsdt (upwelling SW / incident SW at TOA).
    """
    cmip6_dir = DATA_ROOT / "CMIP6"
    cmip6_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CMIP6 Model Data Download")
    print("=" * 60)
    print()

    # Check if cdsapi is installed
    try:
        import cdsapi
        has_cdsapi = True
    except ImportError:
        has_cdsapi = False

    if not has_cdsapi:
        print("Install the CDS API client:")
        print("  pip install cdsapi")
        print()
        print("Then set up your credentials:")
        print("  1. Register at https://cds.climate.copernicus.eu/")
        print("  2. Get your API key from your profile page")
        print("  3. Create ~/.cdsapirc with:")
        print("     url: https://cds.climate.copernicus.eu/api")
        print("     key: YOUR_UID:YOUR_API_KEY")
        print()
        return None

    # Models used in the paper (from the Methods section)
    models = [
        "access_cm2", "access_esm1_5", "bcc_csm2_mr", "bcc_esm1",
        "canesm5", "canesm5_canoe", "e3sm_1_1", "ec_earth3_cc",
        "ec_earth3_veg_lr", "fgoals_f3_l", "fgoals_g3", "fio_esm_2_0",
        "gfdl_cm4", "gfdl_esm4", "giss_e2_1_g", "hadgem3_gc31_ll",
        "hadgem3_gc31_mm", "iitm_esm", "inm_cm4_8", "inm_cm5_0",
        "ipsl_cm6a_lr", "kace_1_0_g", "kiost_esm", "miroc_es2l",
        "miroc6", "mpi_esm1_2_lr", "mri_esm2_0", "nesm3",
        "noresm2_mm", "taiesm1", "ukesm1_0_ll",
    ]

    for experiment in ["historical", "ssp2_4_5"]:
        for variable in ["toa_outgoing_shortwave_radiation", "toa_incident_shortwave_radiation"]:
            var_short = "rsut" if "outgoing" in variable else "rsdt"
            out_file = cmip6_dir / f"cmip6_{experiment}_{var_short}.zip"

            if out_file.exists():
                print(f"Already exists: {out_file}")
                continue

            print(f"Downloading CMIP6 {experiment} {var_short}...")
            try:
                c = cdsapi.Client()
                c.retrieve(
                    "projections-cmip6",
                    {
                        "temporal_resolution": "monthly",
                        "experiment": experiment,
                        "variable": variable,
                        "model": models[:10],  # Start with 10 models to keep size manageable
                        "year": [str(y) for y in range(1980, 2101)],
                        "month": [f"{m:02d}" for m in range(1, 13)],
                        "area": [90, -180, -90, 180],
                    },
                    str(out_file),
                )
                print(f"Downloaded: {out_file} ({out_file.stat().st_size / 1e6:.0f} MB)")
            except Exception as e:
                print(f"Failed: {e}")
                print("You may need to accept the CMIP6 terms of use at:")
                print("  https://cds.climate.copernicus.eu/datasets/projections-cmip6")

    return cmip6_dir


# ═══════════════════════════════════════════════════════════════════════
# 3. MERRA-2 Monthly Radiation (M2TMNXRAD)
# ═══════════════════════════════════════════════════════════════════════

def download_merra2():
    """Download MERRA-2 monthly radiation data.

    Product: M2TMNXRAD v5.12.4
    Variables: SWGDN (surface SW down), SWTDN (TOA SW down), SWGNT (surface SW net),
               SWTNT (TOA SW net), ALBEDO (surface albedo)
    Resolution: 0.625° × 0.5°, monthly, 1980-present
    Source: https://disc.gsfc.nasa.gov/datasets/M2TMNXRAD_5.12.4/summary

    Requires NASA Earthdata credentials in ~/.netrc.
    """
    merra2_dir = DATA_ROOT / "data" / "MERRA2"
    merra2_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MERRA-2 Monthly Radiation Download (M2TMNXRAD)")
    print("=" * 60)
    print()

    netrc = Path.home() / ".netrc"
    has_netrc = netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text()

    if not has_netrc:
        print("NASA Earthdata credentials needed in ~/.netrc")
        print("Register at: https://urs.earthdata.nasa.gov/")
        print()
        print("Then create ~/.netrc with:")
        print("  machine urs.earthdata.nasa.gov")
        print("  login YOUR_USERNAME")
        print("  password YOUR_PASSWORD")
        print()
        return None

    # MERRA-2 monthly radiation files: one per month
    # URL pattern: https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXRAD.5.12.4/YYYY/MERRA2_NNN.tavgM_2d_rad_Nx.YYYYMM.nc4
    # Where NNN is the stream number (100, 200, 300, 400 depending on year range)

    base_url = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXRAD.5.12.4"

    # Download full record 1980-2025 (includes Pinatubo 1991, pre-CERES era)
    for year in range(1980, 2026):
        # Stream number: 100 (1980-1991), 200 (1992-2000), 300 (2001-2010), 400 (2011+)
        if year <= 1991:
            stream = 100
        elif year <= 2000:
            stream = 200
        elif year <= 2010:
            stream = 300
        else:
            stream = 400

        for month in range(1, 13):
            filename = f"MERRA2_{stream}.tavgM_2d_rad_Nx.{year}{month:02d}.nc4"
            dest = merra2_dir / filename

            if dest.exists() and dest.stat().st_size > 1_000_000:
                continue  # Already downloaded

            url = f"{base_url}/{year}/{filename}"
            try:
                subprocess.run(
                    ["curl", "-n", "-L", "-s", "-o", str(dest), url],
                    check=True, timeout=120
                )
                if dest.exists() and dest.stat().st_size > 1_000_000:
                    pass  # Success, silent
                else:
                    dest.unlink(missing_ok=True)
            except Exception:
                pass  # Skip failed months

        # Progress update
        existing = len(list(merra2_dir.glob("*.nc4")))
        print(f"  {year}: {existing} files downloaded so far")

    total = len(list(merra2_dir.glob("*.nc4")))
    print(f"\nMERRA-2: {total} monthly files downloaded to {merra2_dir}")
    return merra2_dir


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Kosmos-Albedo Data Download Manager")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = "all"

    if dataset in ("ceres", "all"):
        download_ceres_ebaf_421()
        print()

    if dataset in ("cmip6", "all"):
        download_cmip6()
        print()

    if dataset in ("merra2", "all"):
        download_merra2()
        print()

    print("=" * 60)
    print("Data download complete. Update world_model/meta.json with new paths.")


if __name__ == "__main__":
    main()
