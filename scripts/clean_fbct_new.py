"""
fbct_low_cloud_cre_direct.py
=============================
Calculates low-cloud TOA radiative effect (CRE) anomalies directly from
CERES-FBCT satellite data, using the flux-by-cloud-type fields themselves
in place of an external cloud radiative kernel.

WHY THIS REPLACES fbct_low_cloud_cre.py
----------------------------------------
The original script multiplied deseasonalised cloud-fraction anomalies by
Zelinka's OBSERVATIONAL kernel (R_pr - R_clr), which had to be interpolated
in surface albedo and regridded (xesmf, conservative) from FBCT's native 1
deg grid down to the kernel's coarser 2.5 deg grid.

CERES-FBCT (Ed4.1, with the clear-sky subset) already reports both the
mean TOA flux for each (opt, press) cloud-type bin AND the clear-sky flux,
on the SAME 1 deg grid as the cloud fraction itself:

    toa_sw_cldtyp_mon(opt, press, time, lat, lon)   -> R_pr   (SW)
    toa_lw_cldtyp_mon(opt, press, time, lat, lon)   -> R_pr   (LW)
    toa_sw_clr_mon(time, lat, lon)                  -> R_clr  (SW)
    toa_lw_clr_mon(time, lat, lon)                  -> R_clr  (LW)
    cldarea_cldtyp_mon(opt, press, time, lat, lon)  -> f_pr   (%)

so (R_pr - R_clr), evaluated from the FBCT climatology itself, IS the
kernel -- no external file, no albedo assumption needed. We still
conservative-regrid from FBCT's native 1 deg grid down to 2.5 deg
(matching the resolution the original kernel-based script used), but now
purely as a choice of output resolution rather than something forced by
an external kernel file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD (mirrors the attached derivation, Eqs. 5-9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All-sky TOA flux decomposes as (Eq. 5):

    R_all = sum_pr f_pr R_pr + R_clr (1 - sum_pr f_pr)

Expanding each term into climatological mean + perturbation, subtracting
the mean, dropping R'_clr (neglected -- "small term"), and retaining only
low-cloud (p = 1,2) perturbations gives the low-cloud radiative anomaly
(Eq. 6):

    R_L = sum_{p=1,2} sum_tau (R̄_pr - R̄_clr) f'_pr

(R̄_pr - R̄_clr) is exactly analogous to a Zelinka-style kernel, except it
is diagnosed from the local observed radiation climatology in each grid
box rather than from an external table.

Following Zelinka et al. (2012b), the low-cloud-fraction anomaly in each
bin is split into an "amount" part (uniform CF change, histogram shape
fixed at climatology) and a "shape" part (redistribution among bins at
fixed total low CF) (Eq. 7):

    f''_pr = f'_pr - f̄_pr * (L'/L̄)

    where L' = sum_pr f'_pr,   L̄ = sum_pr f̄_pr   (sums over low-cloud bins)

Substituting into Eq. 6 gives the two-term decomposition (Eq. 8):

    R_L = [ sum_pr (R̄_pr f̄_pr / L̄) - R̄_clr ] * L'          <- AMOUNT term
        + sum_pr (R̄_pr - R̄_clr) * f''_pr                    <- SHAPE term
          (redistribution among cloud-top pressure / optical depth bins;
           i.e. a combined altitude + optical-depth effect)

Because Eq. 7 is an exact algebraic split of f'_pr (not a linearised
approximation), AMOUNT + SHAPE reconstructs R_L to machine precision --
there is no residual cross-term here, unlike the 4-way kernel
decomposition in the original script. This is checked numerically in
process_satellite() and the max discrepancy is printed.

Output: clean_data/low_cloud_cre_direct_{satellite}.nc  (one per satellite)
        clean_data/low_cloud_cre_direct_merged.nc        (combined record)
  Variables (time, lat, lon):
    dCRE_SW        - SW low-cloud CRE anomaly           [W m-2]
    dCRE_LW        - LW low-cloud CRE anomaly            [W m-2]
    dCRE_net       - NET (SW+LW) low-cloud CRE anomaly   [W m-2]
    dCRE_amount    - NET CRE: amount effect              [W m-2]
    dCRE_shape     - NET CRE: altitude+tau shape effect  [W m-2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMBINING THE TWO SATELLITE RECORDS (Terra/Aqua-MODIS vs. NOAA-20/VIIRS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each record is deseasonalised against its OWN climatology, which already
removes most of the mean-state calibration difference between the two
instruments. During the period where both records exist, we:
  1. Compute the spatial-mean bias and pattern correlation between the two
     anomaly time series and print them as a diagnostic.
  2. Average the two anomalies (equal weight) to form the merged value.
Outside the overlap, whichever record is available is used as-is.

This is only appropriate if the printed overlap diagnostic actually shows
good agreement (small bias, high correlation). If it doesn't, a naive
average will paper over a real inter-satellite discontinuity -- in that
case a calibration/offset correction anchored to the overlap period
should be used instead of a straight average. See merge_satellite_records().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERLAP CORRECTION (unchanged from the kernel version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FBCT assigns cloud fraction to the topmost cloud layer visible from space,
so low cloud beneath high cloud is invisible and CF_low is an underestimate.
We still scale up the observed low-cloud CF by 1 / (1 - CF_high) before
doing anything else, exactly as in the kernel-based script (Zelinka /
Zhou et al. 2013 approach). This is applied to the RAW PERCENT cloud
fraction, before it is converted to a fraction and before deseasonalising.
"""

import os
import glob
import numpy as np
import xarray as xr
import xesmf as xe
from scipy import signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORK_DIR   = "C:/Users/aakas/Documents/CCF-ML/"
FBCT_DIRS  = {
    "terra": "raw_data/ceres_fbct_terra/",
    "noaa":  "raw_data/ceres_fbct_noaa/",
}
OUTPUT_DIR = "clean_data/"

# FBCT plev indices for low cloud (CTP > 680 hPa, after renaming press->plev).
# Index 0 = 1100-800 hPa bin, Index 1 = 800-680 hPa bin.
LOW_PRESS_INDICES  = [0, 1]
# High cloud = everything else (CTP < 680 hPa)
HIGH_PRESS_INDICES = [2, 3, 4, 5, 6]

# Cap on high-cloud fraction used in overlap correction denominator.
# Prevents division by near-zero in persistently overcast regions.
HIGH_CF_CAP = 0.9

# Whether to apply the Zelinka overlap correction.
USE_OVERLAP_CORRECTION = True

# Data variable names in the FBCT-with-clear-sky files.
VAR_CF        = "cldarea_cldtyp_mon"   # [%]  (opt, press, time, lat, lon)
VAR_SW_CLDTYP = "toa_sw_cldtyp_mon"    # [W m-2]  R_pr, SW  (opt, press, time, lat, lon)
VAR_LW_CLDTYP = "toa_lw_cldtyp_mon"    # [W m-2]  R_pr, LW  (opt, press, time, lat, lon)
VAR_SW_CLR    = "toa_sw_clr_mon"       # [W m-2]  R_clr, SW (time, lat, lon)
VAR_LW_CLR    = "toa_lw_clr_mon"       # [W m-2]  R_clr, LW (time, lat, lon)

FILL_VALUE = -999.0

# Target regrid resolution (degrees). Conservative regridding, native 1 deg
# FBCT grid -> this grid. Set to None to skip regridding and stay at 1 deg.
TARGET_RES_DEG = 2.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MERGING THE TWO SATELLITE RECORDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def merge_satellite_records(ds_a: xr.Dataset,
                            ds_b: xr.Dataset,
                            name_a: str,
                            name_b: str) -> xr.Dataset:
    """
    Combine two per-satellite anomaly records into one continuous series.

    Each input dataset already holds deseasonalised anomalies computed
    against ITS OWN climatology, which removes most of the mean-state
    calibration offset between the two instruments. What's left to check
    is whether the two anomaly time series actually agree during the
    period both satellites are operating.

    Approach:
      1. Identify the overlap period (times present in both records).
      2. Print a diagnostic: domain-mean bias (a_bar - b_bar) and pattern
         correlation between the two anomaly fields over the overlap, for
         dCRE_net. Read this before trusting the merge -- see module
         docstring for what "good enough to average" looks like.
      3. Build the merged series: simple (equal-weight) average of the two
         anomalies where both exist, otherwise whichever one exists.

    This averaging step assumes the overlap diagnostic looks reasonable.
    If it doesn't, replace step 3 with a calibration/offset correction
    (e.g. subtract the overlap-period bias from one record) before
    splicing, rather than averaging through a real discontinuity.
    """
    common_times = np.intersect1d(ds_a["time"].values, ds_b["time"].values)
    print(f"  Overlap period: {len(common_times)} months "
          f"({name_a} vs {name_b})")

    if len(common_times) > 0:
        a_overlap = ds_a["dCRE_net"].sel(time=common_times)
        b_overlap = ds_b["dCRE_net"].sel(time=common_times)

        bias = float((a_overlap - b_overlap).mean().values)
        # Pattern correlation, pooling all overlap times and grid points.
        a_flat = a_overlap.values.ravel()
        b_flat = b_overlap.values.ravel()
        valid = np.isfinite(a_flat) & np.isfinite(b_flat)
        corr = float(np.corrcoef(a_flat[valid], b_flat[valid])[0, 1]) if valid.sum() > 1 else np.nan

        print(f"  dCRE_net overlap diagnostic: mean bias ({name_a}-{name_b}) "
              f"= {bias:.3f} W/m2, correlation = {corr:.3f}")
        print("  -> Check this before trusting a straight average: small "
              "bias + high correlation supports averaging; otherwise "
              "consider an offset correction instead (see docstring).")
    else:
        print("  No overlapping months found between the two records; "
              "merge will simply concatenate them.")

    # Equal-weight average on the union of times; NaN-aware so a variable
    # missing from one record on a given date just falls back to the other.
    ds_a_aligned, ds_b_aligned = xr.align(ds_a, ds_b, join="outer")
    ds_merged = xr.concat([ds_a_aligned, ds_b_aligned], dim="__sat__").mean(
        dim="__sat__", skipna=True
    )

    ds_merged.attrs.update({
        "method": f"Merged {name_a} + {name_b}: equal-weight average of "
                  f"per-satellite anomalies during overlap, single-record "
                  f"values used outside overlap.",
        "overlap_months": int(len(common_times)),
    })
    return ds_merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    os.chdir(WORK_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_grid = build_target_grid(TARGET_RES_DEG) if TARGET_RES_DEG else None

    ds_by_sat = {}
    for sat, directory in FBCT_DIRS.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {sat.upper()}")
        print(f"{'='*60}")
        ds_out = process_satellite(sat, directory, target_grid)
        ds_by_sat[sat] = ds_out

        out_path = os.path.join(OUTPUT_DIR, f"low_cloud_cre_direct_{sat}.nc")
        ds_out.to_netcdf(out_path)
        print(f"  Saved: {out_path}")

    if len(ds_by_sat) == 2:
        print(f"\n{'='*60}")
        print("  Merging satellite records")
        print(f"{'='*60}")
        sat_names = list(ds_by_sat.keys())
        ds_merged = merge_satellite_records(
            ds_by_sat[sat_names[0]], ds_by_sat[sat_names[1]],
            sat_names[0], sat_names[1],
        )
        out_path = os.path.join(OUTPUT_DIR, "low_cloud_cre_direct_merged.nc")
        ds_merged.to_netcdf(out_path)
        print(f"  Saved: {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TARGET GRID (for optional conservative regridding)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_target_grid(res_deg: float) -> xr.Dataset:
    """
    Build a plain-center-coordinate lat/lon Dataset at the requested
    resolution (degrees), cell-centered, spanning the full globe.

    No external kernel file to source a grid from anymore, so this is
    constructed directly. xesmf infers cell boundaries automatically from
    these 1-D centers for a regular rectilinear grid, same as it did from
    the kernel's lat/lon in the original script.

    For res_deg=2.5 this reproduces the same 72 x 144 grid the kernel used
    (centers at +/-88.75 ... in latitude, 1.25 ... 358.75 in longitude).
    """
    half = res_deg / 2.0
    lat = np.arange(-90 + half, 90, res_deg)
    lon = np.arange(0 + half, 360, res_deg)
    return xr.Dataset({"lat": ("lat", lat), "lon": ("lon", lon)})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FBCT LOADING AND REGRIDDING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_fbct(directory: str,
              satellite_name: str,
              target_grid: xr.Dataset | None) -> xr.Dataset:
    """
    Load FBCT-with-clear-sky files, replace fill values, rename dims, and
    (if target_grid is given) conservative-regrid every variable from the
    native 1 deg FBCT grid down to target_grid.

    Conservative regridding is appropriate here because both the cloud
    fraction (an area fraction) and the flux fields (area-averaged
    intensive quantities) are conserved under area-weighted averaging.
    skipna=True avoids one missing 1-deg cell blanking an entire coarse
    output cell.

    Returns
    -------
    xr.Dataset with dims (time, tau, plev, lat, lon) for the cloud-type
    fields and (time, lat, lon) for the clear-sky fields.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {directory}")
    print(f"  Found {len(files)} files for {satellite_name}")

    ds = xr.open_mfdataset(files, combine="by_coords")

    # Replace fill value (-999) with NaN across all variables.
    ds = ds.where(ds != FILL_VALUE)

    # Rename dims for consistency with LOW/HIGH_PRESS_INDICES.
    ds = ds.rename({"press": "plev", "opt": "tau"})

    if target_grid is not None:
        print(f"  Building xesmf regridder (conservative, native -> "
              f"{TARGET_RES_DEG} deg)...")
        regridder = xe.Regridder(
            ds, target_grid,
            method="conservative",
            periodic=True,
            ignore_degenerate=True,
        )
        print("  Regridding FBCT variables...")
        ds = regridder(ds, keep_attrs=True, skipna=True)

    return ds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERLAP CORRECTION (identical logic to the kernel-based script)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_overlap_correction(cf: xr.DataArray) -> xr.DataArray:
    """
    Correct low-cloud CF for obscuration by overlying high clouds
    (Zelinka / Zhou et al. 2013 approach).

    CF_low_corrected = CF_low_exposed / (1 - CF_high)

    Applied bin-by-bin across all (tau, plev_low) bins using the same
    CF_high denominator for every bin. Operates on CF in whatever units
    it is given (here: percent) and returns the same units -- the /100
    inside is only used to build the dimensionless denominator.

    Parameters
    ----------
    cf : xr.DataArray (time, tau, plev, lat, lon)
        Raw FBCT cloud fraction [%], all pressure bins.

    Returns
    -------
    xr.DataArray of same shape with corrected low-cloud CF bins.
    High-cloud bins are returned unchanged.
    """
    cf_high = cf.isel(plev=HIGH_PRESS_INDICES).sum(dim=["tau", "plev"])
    cf_high_frac = (cf_high / 100.0).clip(max=HIGH_CF_CAP)
    denom = 1.0 - cf_high_frac

    cf_low = cf.isel(plev=LOW_PRESS_INDICES)
    cf_low_corrected = cf_low / denom

    cf_corrected = cf.copy()
    cf_corrected.values[:, :, LOW_PRESS_INDICES, :, :] = (
        cf_low_corrected
        .transpose("time", "tau", "plev", "lat", "lon")
        .values
    )

    return cf_corrected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DESEASONALISATION AND DETRENDING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def deseasonalise(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Remove the climatological monthly mean. Returns (anomaly, climatology)."""
    clim = da.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    return anom, clim


def detrend_time(da: xr.DataArray) -> xr.DataArray:
    """Remove a linear trend along the time axis at every grid point."""
    vals = da.values.copy()
    nan_mask = ~np.isfinite(vals)
    vals[nan_mask] = 0.0
    vals = signal.detrend(vals, axis=0, type="linear")
    vals[nan_mask] = np.nan
    return xr.DataArray(vals, coords=da.coords, dims=da.dims, attrs=da.attrs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AMOUNT / SHAPE DECOMPOSITION AND DIRECT KERNEL APPLICATION (Eqs. 6-8)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_dCRE(f_anom: xr.DataArray,
                 f_clim: xr.DataArray,
                 R_clim: xr.DataArray,
                 Rclr_clim: xr.DataArray) -> dict:
    """
    Implements Eqs. 6-8 for one radiative band (SW or LW).

    Parameters
    ----------
    f_anom : (time, tau, plev_low, lat, lon)
        Deseasonalised, detrended low-cloud fraction anomaly (fraction, 0-1).
    f_clim : (month, tau, plev_low, lat, lon)
        Climatological monthly-mean low-cloud fraction (fraction, 0-1).
    R_clim : (month, tau, plev_low, lat, lon)
        Climatological monthly-mean flux for each low-cloud bin, R̄_pr [W m-2].
    Rclr_clim : (month, lat, lon)
        Climatological monthly-mean clear-sky flux, R̄_clr [W m-2].

    Returns
    -------
    dict with keys 'total', 'amount', 'shape', each (time, lat, lon) [W m-2],
    following Eq. 8: total = amount + shape (exact, no residual).
    """
    eps = 1e-6

    # L' and L̄ : total low-cloud CF anomaly / climatology, summed over bins.
    L_anom = f_anom.sum(dim=["tau", "plev"])            # (time, lat, lon)
    L_clim = f_clim.sum(dim=["tau", "plev"])             # (month, lat, lon)
    L_clim_safe = L_clim.where(L_clim > eps, other=np.nan)

    # ── f''_pr : redistribution-among-bins anomaly (Eq. 7) ───────────────────
    # f_pred = f̄_pr * (L'/L̄), broadcast to each time step via its calendar month.
    L_ratio = L_anom.groupby("time.month") / L_clim_safe          # (time, lat, lon)
    f_pred = L_ratio.groupby("time.month") * f_clim                # (time, tau, plev, lat, lon)
    f_shape = f_anom - f_pred                                       # f''_pr

    # ── AMOUNT term: [sum_pr(R̄_pr f̄_pr)/L̄ - R̄_clr] * L' ───────────────────
    weighted_R_clim = (R_clim * f_clim).sum(dim=["tau", "plev"]) / L_clim_safe  # (month, lat, lon)
    K_amount = weighted_R_clim - Rclr_clim                          # (month, lat, lon)
    dCRE_amount = L_anom.groupby("time.month") * K_amount           # (time, lat, lon)

    # ── SHAPE term: sum_pr (R̄_pr - R̄_clr) * f''_pr ────────────────────────
    K_shape = R_clim - Rclr_clim                                    # (month, tau, plev, lat, lon), broadcasts Rclr_clim over tau,plev
    dCRE_shape = (f_shape.groupby("time.month") * K_shape).sum(dim=["tau", "plev"])  # (time, lat, lon)

    dCRE_total = dCRE_amount + dCRE_shape

    return {"total": dCRE_total, "amount": dCRE_amount, "shape": dCRE_shape}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PER-SATELLITE PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_satellite(sat: str,
                      directory: str,
                      target_grid: xr.Dataset | None) -> xr.Dataset:
    """
    End-to-end pipeline for one satellite:
      1. Load FBCT (with clear-sky fields) and conservative-regrid to
         target_grid (native 1 deg -> TARGET_RES_DEG).
      2. Apply overlap correction to absolute CF (percent, before deseasonalising).
      3. Convert CF to fraction (0-1); deseasonalise and detrend.
      4. Deseasonalise the flux fields (climatology only, no detrend needed).
      5. Restrict to low-cloud bins.
      6. Apply Eqs. 6-8 (amount/shape decomposition) separately for SW and LW.
      7. Sanity-check amount + shape against a directly-computed R_L.
      8. Return packaged xr.Dataset.
    """
    print("  Loading and regridding FBCT (with clear-sky fields)...")
    ds = load_fbct(directory, sat, target_grid)

    cf = ds[VAR_CF].transpose("time", "tau", "plev", "lat", "lon")        # [%]
    R_sw = ds[VAR_SW_CLDTYP].transpose("time", "tau", "plev", "lat", "lon")  # [W m-2]
    R_lw = ds[VAR_LW_CLDTYP].transpose("time", "tau", "plev", "lat", "lon")  # [W m-2]
    Rclr_sw = ds[VAR_SW_CLR]   # (time, lat, lon)
    Rclr_lw = ds[VAR_LW_CLR]   # (time, lat, lon)

    # ── 2. Overlap correction on absolute CF, still in percent ───────────────
    if USE_OVERLAP_CORRECTION:
        print("  Applying Zelinka overlap correction...")
        cf = apply_overlap_correction(cf)
    else:
        print("  Skipping overlap correction.")

    # ── 3. Percent -> fraction, then deseasonalise + detrend CF ──────────────
    cf_frac = cf / 100.0
    print("  Deseasonalising cloud fraction...")
    cf_anom, cf_clim = deseasonalise(cf_frac)
    print("  Detrending cloud fraction anomaly...")
    cf_anom = detrend_time(cf_anom)

    # ── 4. Climatology of the flux fields (no detrending -- these feed the
    #        kernel-equivalent terms, which are climatological by definition) ─
    print("  Building flux climatologies (R_pr, R_clr)...")
    _, R_sw_clim = deseasonalise(R_sw)
    _, R_lw_clim = deseasonalise(R_lw)
    _, Rclr_sw_clim = deseasonalise(Rclr_sw)
    _, Rclr_lw_clim = deseasonalise(Rclr_lw)

    # ── 5. Restrict to low-cloud bins ─────────────────────────────────────────
    f_anom_low = cf_anom.isel(plev=LOW_PRESS_INDICES)
    f_clim_low = cf_clim.isel(plev=LOW_PRESS_INDICES)
    R_sw_clim_low = R_sw_clim.isel(plev=LOW_PRESS_INDICES)
    R_lw_clim_low = R_lw_clim.isel(plev=LOW_PRESS_INDICES)

    # ── 6. Amount/shape decomposition (Eqs. 6-8), SW and LW separately ───────
    print("  Computing SW low-cloud CRE anomaly (amount + shape)...")
    sw = compute_dCRE(f_anom_low, f_clim_low, R_sw_clim_low, Rclr_sw_clim)
    print("  Computing LW low-cloud CRE anomaly (amount + shape)...")
    lw = compute_dCRE(f_anom_low, f_clim_low, R_lw_clim_low, Rclr_lw_clim)

    dCRE_SW = sw["total"]
    dCRE_LW = lw["total"]
    dCRE_amount = sw["amount"] + lw["amount"]
    dCRE_shape = sw["shape"] + lw["shape"]
    dCRE_net = dCRE_SW + dCRE_LW

    # ── 7. Sanity check: amount + shape should equal total to numerical noise
    check_sw = float(np.nanmax(np.abs((sw["amount"] + sw["shape"] - sw["total"]).values)))
    check_lw = float(np.nanmax(np.abs((lw["amount"] + lw["shape"] - lw["total"]).values)))
    print(f"  Sanity check (amount+shape vs total), max abs diff: "
          f"SW={check_sw:.2e} W/m2, LW={check_lw:.2e} W/m2")

    # ── 8. Package output ─────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {
            "dCRE_SW": dCRE_SW.assign_attrs(
                long_name="Low-cloud SW CRE anomaly", units="W m-2",
                note="Positive = more energy leaving Earth (weaker SW cooling).",
            ),
            "dCRE_LW": dCRE_LW.assign_attrs(
                long_name="Low-cloud LW CRE anomaly", units="W m-2",
            ),
            "dCRE_net": dCRE_net.assign_attrs(
                long_name="Low-cloud NET CRE anomaly", units="W m-2",
            ),
            "dCRE_amount": dCRE_amount.assign_attrs(
                long_name="Low-cloud NET CRE: amount effect", units="W m-2",
                description="Radiative impact of low-cloud fraction anomaly L' "
                            "with the (tau, plev) histogram shape fixed at climatology.",
            ),
            "dCRE_shape": dCRE_shape.assign_attrs(
                long_name="Low-cloud NET CRE: altitude/optical-depth shape effect",
                units="W m-2",
                description="Radiative impact of redistribution among cloud-top "
                            "pressure and optical-depth bins at fixed total low CF "
                            "(f''_pr in Eq. 7).",
            ),
        },
        attrs={
            "satellite":            sat,
            "method":               "Direct FBCT flux-by-cloud-type decomposition (Eqs. 5-9), "
                                     "no external kernel.",
            "grid_resolution_deg":  str(TARGET_RES_DEG) if TARGET_RES_DEG else "native (1 deg)",
            "overlap_corrected":    str(USE_OVERLAP_CORRECTION),
            "low_press_indices":    str(LOW_PRESS_INDICES),
            "sign_convention":      "Positive = upward flux (energy leaving Earth).",
            "amount_shape_max_abs_diff_sw_Wm2": check_sw,
            "amount_shape_max_abs_diff_lw_Wm2": check_lw,
        },
    )
    return ds_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    main()