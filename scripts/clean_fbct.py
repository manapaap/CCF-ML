"""
fbct_low_cloud_cre.py
=====================
Calculates low-cloud TOA radiative effect (CRE) anomalies from CERES-FBCT
satellite data using the Zelinka et al. observational cloud radiative kernels.

Strategy: regrid FBCT from 1 deg down to the kernel's native 2.5 deg grid
using xesmf conservative regridding, then apply the kernel directly without
any spatial interpolation of the kernel itself.

Output: clean_data/low_cloud_cre_{satellite}.nc  (one file per satellite)
  Variables (time, lat, lon):
    dCRE_SW       - SW low-cloud CRE anomaly              [W m-2]
    dCRE_LW       - LW low-cloud CRE anomaly              [W m-2]
    dCRE_net      - NET (SW+LW) low-cloud CRE anomaly     [W m-2]
    dCRE_amount   - NET CRE: amount effect                [W m-2]
    dCRE_altitude - NET CRE: altitude effect              [W m-2]
    dCRE_tau      - NET CRE: optical depth effect         [W m-2]
    dCRE_residual - NET CRE: residual cross-term          [W m-2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERLAP CORRECTION (Zelinka / Zhou et al. 2013 approach)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FBCT assigns cloud fraction to the topmost cloud layer visible from space.
A low cloud beneath a high cloud is invisible: its fraction is attributed
to the high-cloud bin, so low-cloud CF is underestimated.

Zelinka's correction accounts for this by scaling up the observed low-cloud
CF by the fraction of sky that is NOT blocked by overlying high cloud:

    CF_low_corrected(t, tau, p) = CF_low_exposed(t, tau, p) / (1 - CF_high(t))

where CF_high(t) is the total cloud fraction in all HIGH-cloud pressure bins
(plev indices 2-6, i.e. CTP < 680 hPa), summed over all tau bins.

Intuition: if 30% of the sky is covered by high cloud, the remaining 70% is
what the satellite could see. The low-cloud CF we observe is only over that
70%. Dividing by (1 - 0.30) = 0.70 scales it back to a full-sky estimate.

This correction is applied to the ABSOLUTE cloud fraction BEFORE
deseasonalisation, so that the climatology and anomalies are both corrected.

Note: CF_high is capped at 0.9 to avoid division by near-zero in persistently
overcast regions (ITCZ etc.), consistent with Zelinka's implementation.
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

WORK_DIR    = "C:/Users/aakas/Documents/CCF-ML/"
KERNEL_PATH = "kernels/obs_cloud_kernels.nc"
FBCT_DIRS   = {
    "terra": "raw_data/ceres_fbct_terra/",
    "noaa":  "raw_data/ceres_fbct_noaa/",
}
OUTPUT_DIR = "clean_data/"

# Surface albedo for SW kernel interpolation.
# 0.07 is standard for open ocean. Kernel albcs values are [0.0, 0.5, 1.0].
OCEAN_ALBEDO = 0.07

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    os.chdir(WORK_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare the kernel once; reuse for both satellites.
    # The kernel stays on its native 2.5 deg grid. FBCT will be regridded
    # down to match it.
    K_SW, K_LW = load_and_prepare_kernel(KERNEL_PATH, OCEAN_ALBEDO)

    for sat, directory in FBCT_DIRS.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {sat.upper()}")
        print(f"{'='*60}")
        ds_out = process_satellite(sat, directory, K_SW, K_LW)

        out_path = os.path.join(OUTPUT_DIR, f"low_cloud_cre_{sat}.nc")
        ds_out.to_netcdf(out_path)
        print(f"  Saved: {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KERNEL LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_and_prepare_kernel(kernel_path: str,
                            albedo: float) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load the Zelinka observational kernel and prepare it for direct
    multiplication against regridded FBCT cloud fraction anomalies.

    The kernel stays on its native 2.5 deg (72 lat x 144 lon) grid.
    FBCT is regridded to match it in load_and_regrid_fbct().

    Steps here:
      1. Interpolate SW kernel along the albcs dimension to OCEAN_ALBEDO.
      2. Drop kernel tau index 0 (sub-visible cirrus, tau < 0.3, not in FBCT).
      3. Re-index tau (0-5) and plev (0-6) to plain integers matching FBCT's
         renamed opt and press coordinate values.

    Returns
    -------
    K_SW : (time=12, tau=6, plev=7, lat=72, lon=144)  [W m-2 %-1]
    K_LW : (time=12, tau=6, plev=7, lat=72)           [W m-2 %-1]
        K_LW has no lon dimension (zonal mean only); xarray will broadcast
        it over lon automatically when multiplied against FBCT.
    """
    print("Loading kernel...")
    kern = xr.open_dataset(kernel_path)

    # ── 1. Interpolate SW kernel to ocean surface albedo ─────────────────────
    # The SW radiative effect of a cloud depends on the clear-sky albedo of
    # the surface beneath it — a brighter surface reduces the effective cloud
    # cooling. The kernel provides 3 albedo values [0.0, 0.5, 1.0]; we
    # interpolate linearly to 0.07 (open ocean).
    K_SW = kern["SWkernel"].interp(
        albcs=albedo,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    # Shape after: (time=12, tau=7, plev=7, lat=72, lon=144)
    K_LW = kern["LWkernel"]
    # Shape: (time=12, tau=7, plev=7, lat=72)   — no lon dim, it's zonal mean

    # ── 2. Drop kernel tau index 0 ────────────────────────────────────────────
    # The kernel's 7 tau bins span [0.01, 380]. The first bin (centre ~0.055)
    # covers tau < 0.3 — optically thin cirrus that MODIS/VIIRS cannot detect.
    # FBCT therefore has only 6 tau bins (all tau > 0.3).
    # We drop kernel tau[0] so the remaining 6 kernel bins correspond 1-to-1
    # with FBCT's 6 opt bins.
    K_SW = K_SW.isel(tau=slice(1, None))   # shape: (12, 6, 7, 72, 144)
    K_LW = K_LW.isel(tau=slice(1, None))   # shape: (12, 6, 7, 72)

    # ── 3. Re-index tau and plev to plain 0-based integers ───────────────────
    # Kernel coordinates after the isel are physical values (hPa, unitless tau
    # midpoints). FBCT's opt and press coordinates are plain integers 0-5 / 0-6.
    # Assigning matching integer values lets xarray align them by coordinate
    # value during multiplication without any manual indexing.
    K_SW = K_SW.assign_coords(tau=np.arange(6, dtype=float),
                              plev=np.arange(7, dtype=float))
    K_LW = K_LW.assign_coords(tau=np.arange(6, dtype=float),
                              plev=np.arange(7, dtype=float))

    print(f"  Kernel ready. SW: {K_SW.shape}, LW: {K_LW.shape}")
    return K_SW, K_LW


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FBCT LOADING AND REGRIDDING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_and_regrid_fbct(directory: str,
                         satellite_name: str,
                         target_grid: xr.Dataset) -> xr.Dataset:
    """
    Load FBCT files, replace fill values, rename dims, and regrid from
    the native 1 deg grid to the kernel's 2.5 deg grid using xesmf.

    We use conservative regridding, which is correct for intensive quantities
    like cloud fraction: it preserves the area-weighted mean, so total cloud
    area is conserved across the regridding step.

    Parameters
    ----------
    directory : str
        Path to folder of monthly FBCT .nc files.
    satellite_name : str
        Label for log messages.
    target_grid : xr.Dataset
        Dataset with 'lat' and 'lon' coordinates matching the kernel grid
        (2.5 deg, 72 x 144). Used to define the xesmf output grid.

    Returns
    -------
    xr.Dataset with dims (time, tau, plev, lat=72, lon=144).
    """
    files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {directory}")
    print(f"  Found {len(files)} files for {satellite_name}")

    ds = xr.open_mfdataset(files, combine="by_coords")

    # Replace fill value (-999) with NaN
    ds = ds.where(ds != -999.0)

    # Rename dims to match kernel coordinate names so alignment is automatic:
    #   press -> plev  (7 pressure bins, same ISCCP order, index 0 = lowest)
    #   opt   -> tau   (6 tau bins; after dropping kernel tau[0] these align 1:1)
    ds = ds.rename({"press": "plev", "opt": "tau"})

    # ── Regrid with xesmf ─────────────────────────────────────────────────────
    # xesmf expects lat/lon as the spatial dimensions. Our data also has
    # tau and plev as leading dims, which xesmf handles by looping over them.
    # We build the regridder from a single-variable 2D slice to keep it light,
    # then apply it to each variable.
    print("  Building xesmf regridder (conservative, 1 deg -> 2.5 deg)...")
    regridder = xe.Regridder(
        ds,           # source grid: FBCT at 1 deg
        target_grid,  # target grid: kernel at 2.5 deg
        method="conservative",
        periodic=True,           # longitude is periodic
        ignore_degenerate=True,  # skip degenerate cells at poles
    )

    print("  Regridding FBCT variables...")
    ds_regrid = regridder(ds, keep_attrs=True)

    return ds_regrid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERLAP CORRECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_overlap_correction(cf: xr.DataArray) -> xr.DataArray:
    """
    Correct low-cloud CF for obscuration by overlying high clouds, following
    the approach used in Zelinka et al. / Zhou et al. (2013).

    PROBLEM:
      FBCT reports the CF of the topmost cloud layer exposed to space.
      A low cloud hidden beneath a high cloud is invisible to the satellite
      and its CF is attributed to the high-cloud bin. The raw FBCT low-cloud
      CF is therefore an UNDERESTIMATE of the true low-cloud fraction,
      because it is only observed over the fraction of sky not blocked by
      high cloud.

    CORRECTION:
      If CF_high is the fraction of sky covered by high cloud, then low clouds
      can only be observed over the remaining (1 - CF_high) fraction. The true
      (full-sky) low-cloud CF is therefore:

        CF_low_corrected = CF_low_exposed / (1 - CF_high)

      This is applied bin-by-bin across all (tau, plev_low) bins, using the
      same CF_high denominator for every bin (since the obscuration is a column
      effect that does not depend on the low cloud's optical depth or pressure).

    Applied to ABSOLUTE cloud fraction (before deseasonalisation), so that
    both the climatology and anomalies are consistently corrected.

    Parameters
    ----------
    cf : xr.DataArray (time, tau, plev, lat, lon)
        Raw FBCT cloud fraction [%], all pressure bins.

    Returns
    -------
    xr.DataArray of same shape with corrected low-cloud CF bins.
    High-cloud bins are returned unchanged.
    """
    # Total high-cloud CF at each time step: sum over all high-cloud bins
    # (plev indices 2-6) and all tau bins. Units: %.
    # Shape: (time, lat, lon)
    cf_high = cf.isel(plev=HIGH_PRESS_INDICES).sum(dim=["tau", "plev"])

    # Convert to fractional units [0-1] for the denominator calculation,
    # then cap at HIGH_CF_CAP to prevent division blowing up where the sky
    # is persistently overcast with high cloud (e.g. deep tropics).
    cf_high_frac = (cf_high / 100.0).clip(max=HIGH_CF_CAP)

    # Denominator: the observable sky fraction (1 - CF_high).
    # Shape: (time, lat, lon) — will broadcast over (tau, plev_low) automatically.
    denom = 1.0 - cf_high_frac

    # Extract just the low-cloud bins and correct them
    cf_low = cf.isel(plev=LOW_PRESS_INDICES)       # (time, tau, plev_low, lat, lon)
    cf_low_corrected = cf_low / denom              # broadcast over tau, plev

    # Reassemble: copy the full cf array and overwrite the low-cloud plev
    # slots with the corrected values. We work on .values (numpy) to avoid
    # xarray alignment checks that trip on the mismatched plev sizes between
    # cf_low_corrected (2 bins) and cf (7 bins).
    cf_corrected = cf.copy()
    # cf dims are (time, tau, plev, lat, lon); LOW_PRESS_INDICES picks along axis 2
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
    """
    Remove the climatological monthly mean to isolate interannual anomalies.

    The kernel method multiplies cloud fraction ANOMALIES by the kernel to get
    CRE ANOMALIES. We must deseasonalise first so we're working with dCF, not
    absolute CF.

    Returns (anomaly, climatology). The climatology is needed by decompose_dcf()
    to define the reference histogram shape for the amount/altitude/tau split.
    """
    clim = da.groupby("time.month").mean("time")   # (month=12, tau, plev, lat, lon)
    anom = da.groupby("time.month") - clim         # (time, tau, plev, lat, lon)
    return anom, clim


def detrend_time(da: xr.DataArray) -> xr.DataArray:
    """
    Remove a linear trend along the time axis at every grid point.

    Satellite records can carry long-term drifts from instrument aging or
    orbital changes. Detrending isolates interannual variability for the CCF
    analysis. NaN locations are filled with 0 before detrending (safe because
    detrend only removes a slope, not the level) and then restored.
    """
    vals = da.values.copy()
    nan_mask = ~np.isfinite(vals)
    vals[nan_mask] = 0.0
    vals = signal.detrend(vals, axis=0, type="linear")
    vals[nan_mask] = np.nan
    return xr.DataArray(vals, coords=da.coords, dims=da.dims, attrs=da.attrs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AMOUNT / ALTITUDE / TAU DECOMPOSITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decompose_dcf(dcf_low: xr.DataArray,
                  cf_clim_low: xr.DataArray) -> dict:
    """
    Decompose the low-cloud CF anomaly into amount, altitude, and tau
    components following Zelinka et al. (2012b).

    The cloud fraction anomaly dC(t, p, tau) is a 2D perturbation to the
    (pressure, optical depth) histogram. We split it into three parts:

    AMOUNT: total cloud fraction changes, histogram shape held at climatology.
        dC_amt(t,p,tau) = dC_tot(t) * [C_bar(p,tau) / C_bar_tot]

    ALTITUDE: pressure marginal shifts (clouds move up/down), total CF and
        tau distribution fixed. We find the excess in the pressure marginal
        beyond what amount predicts, then redistribute it across tau bins
        using the climatological tau-given-p shape.
        excess_p = dC_p(t,p) - dC_tot(t) * [C_bar_p(p) / C_bar_tot]
        dC_alt(t,p,tau) = excess_p * [C_bar(p,tau) / C_bar_p(p)]

    TAU: tau marginal shifts (clouds thicken/thin), total CF and pressure
        distribution fixed. Analogous to altitude but in the tau direction.
        excess_tau = dC_tau(t,tau) - dC_tot(t) * [C_bar_tau(tau) / C_bar_tot]
        dC_tau_comp(t,p,tau) = excess_tau * [C_bar(p,tau) / C_bar_tau(tau)]

    RESIDUAL: dC - dC_amt - dC_alt - dC_tau  (small cross-term, ~5%)

    Parameters
    ----------
    dcf_low : (time, tau, plev, lat, lon)
        Low-cloud CF anomaly, deseasonalised and detrended.
    cf_clim_low : (month=12, tau, plev, lat, lon)
        Climatological monthly-mean low-cloud CF.

    Returns dict with keys 'amount', 'altitude', 'tau', 'residual'.
    """
    eps = 1e-6
    C_bar = cf_clim_low.mean("month")                          # (tau, plev, lat, lon)

    C_bar_tot = C_bar.sum(dim=["tau", "plev"])                 # (lat, lon)
    C_bar_p   = C_bar.sum(dim="tau")                          # (plev, lat, lon)
    C_bar_tau = C_bar.sum(dim="plev")                         # (tau, lat, lon)

    C_bar_tot_s = C_bar_tot.where(C_bar_tot > eps, other=np.nan)
    C_bar_p_s   = C_bar_p.where(C_bar_p > eps, other=np.nan)
    C_bar_tau_s = C_bar_tau.where(C_bar_tau > eps, other=np.nan)

    shape_norm  = C_bar / C_bar_tot_s          # (tau, plev, lat, lon)
    p_norm      = C_bar_p / C_bar_tot_s        # (plev, lat, lon)
    tau_norm    = C_bar_tau / C_bar_tot_s      # (tau, lat, lon)
    tau_given_p = C_bar / C_bar_p_s            # conditional tau shape within each p bin
    p_given_tau = C_bar / C_bar_tau_s          # conditional p shape within each tau bin

    dC_tot      = dcf_low.sum(dim=["tau", "plev"])   # (time, lat, lon)
    dC_p_marg   = dcf_low.sum(dim="tau")             # (time, plev, lat, lon)
    dC_tau_marg = dcf_low.sum(dim="plev")            # (time, tau, lat, lon)

    dC_amt = dC_tot * shape_norm

    dC_p_excess  = dC_p_marg - dC_tot * p_norm
    dC_alt       = dC_p_excess * tau_given_p

    dC_tau_excess = dC_tau_marg - dC_tot * tau_norm
    dC_tau_comp   = dC_tau_excess * p_given_tau

    dC_resid = dcf_low - dC_amt - dC_alt - dC_tau_comp

    return {"amount": dC_amt, "altitude": dC_alt,
            "tau": dC_tau_comp, "residual": dC_resid}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KERNEL APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_kernel(dcf: xr.DataArray,
                 K_SW: xr.DataArray,
                 K_LW: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Multiply cloud fraction anomalies by the kernel and sum over (tau, plev).

    dCRE(t, lat, lon) = sum_{tau,p}  K(month(t), tau, p, lat)  x  dCF(t, tau, p, lat, lon)

    The kernel time dimension holds calendar months (1.0-12.0). We select
    the correct month for each FBCT time step, then reassign the time
    coordinate to FBCT's datetime values before multiplying.

    K_LW has no lon dimension; xarray broadcasts it over lon automatically.
    K_SW has lon (from the kernel file); it aligns with FBCT lon by coordinate.
    """
    cal_months = dcf["time"].dt.month.values.astype(float)
    month_da = xr.DataArray(cal_months, dims="time")

    K_SW_t = K_SW.sel(time=month_da).assign_coords(time=dcf["time"])
    K_LW_t = K_LW.sel(time=month_da).assign_coords(time=dcf["time"])

    dCRE_SW = (K_SW_t * dcf).sum(dim=["tau", "plev"])
    dCRE_LW = (K_LW_t * dcf).sum(dim=["tau", "plev"])

    return dCRE_SW, dCRE_LW


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PER-SATELLITE PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_satellite(sat: str,
                      directory: str,
                      K_SW: xr.DataArray,
                      K_LW: xr.DataArray) -> xr.Dataset:
    """
    End-to-end pipeline for one satellite:
      1. Load and regrid FBCT to 2.5 deg (xesmf conservative).
      2. Apply overlap correction to absolute CF (before deseasonalising).
      3. Deseasonalise and detrend CF.
      4. Select low-cloud bins; restrict kernel likewise.
      5. Decompose dCF into amount/altitude/tau.
      6. Apply kernel to get SW, LW CRE anomalies (total + per component).
      7. Return packaged xr.Dataset.
    """
    # ── 1. Load and regrid ────────────────────────────────────────────────────
    # Build a minimal target-grid dataset from the kernel's lat/lon coords.
    # xesmf needs a Dataset with 'lat' and 'lon' to define the output grid.
    kern_ds = xr.open_dataset(KERNEL_PATH)
    target_grid = xr.Dataset({"lat": kern_ds["lat"], "lon": kern_ds["lon"]})

    print("  Loading and regridding FBCT...")
    ds = load_and_regrid_fbct(directory, sat, target_grid)

    # Cloud fraction: transpose to (time, tau, plev, lat, lon)
    cf = ds["cldarea_cldtyp_mon"].transpose("time", "tau", "plev", "lat", "lon")

    # ── 2. Overlap correction (on absolute CF, before deseasonalising) ────────
    # Correction must be applied to absolute values so that the seasonal cycle
    # and the anomalies are both corrected consistently.
    if USE_OVERLAP_CORRECTION:
        print("  Applying Zelinka overlap correction...")
        cf = apply_overlap_correction(cf)
    else:
        print("  Skipping overlap correction.")

    # ── 3. Deseasonalise and detrend ──────────────────────────────────────────
    print("  Deseasonalising...")
    cf_anom, cf_clim = deseasonalise(cf)
    print("  Detrending...")
    cf_anom = detrend_time(cf_anom)

    # ── 4. Select low-cloud pressure bins ─────────────────────────────────────
    # plev 0-1 = CTP > 680 hPa (low cloud). Restrict both CF anomaly and kernel.
    dcf_low     = cf_anom.isel(plev=LOW_PRESS_INDICES)
    cf_clim_low = cf_clim.isel(plev=LOW_PRESS_INDICES)
    K_SW_low    = K_SW.isel(plev=LOW_PRESS_INDICES)
    K_LW_low    = K_LW.isel(plev=LOW_PRESS_INDICES)

    # ── 5. Decompose dCF ──────────────────────────────────────────────────────
    print("  Decomposing dCF into amount/altitude/tau...")
    comps = decompose_dcf(dcf_low, cf_clim_low)

    # ── 6. Apply kernel ───────────────────────────────────────────────────────
    print("  Applying kernel...")
    dCRE_SW, dCRE_LW = apply_kernel(dcf_low, K_SW_low, K_LW_low)

    dCRE_comps = {}
    for name, dcf_comp in comps.items():
        sw, lw = apply_kernel(dcf_comp, K_SW_low, K_LW_low)
        dCRE_comps[name] = sw + lw   # NET for each component (standard convention)

    # ── 7. Package output ─────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {
            "dCRE_SW": dCRE_SW.assign_attrs(
                long_name="Low-cloud SW CRE anomaly", units="W m-2",
                note="Positive = more energy leaving Earth (weaker SW cooling).",
            ),
            "dCRE_LW": dCRE_LW.assign_attrs(
                long_name="Low-cloud LW CRE anomaly", units="W m-2",
            ),
            "dCRE_net": (dCRE_SW + dCRE_LW).assign_attrs(
                long_name="Low-cloud NET CRE anomaly", units="W m-2",
            ),
            "dCRE_amount": dCRE_comps["amount"].assign_attrs(
                long_name="Low-cloud NET CRE: amount effect", units="W m-2",
                description="Changes in total low CF; histogram shape fixed at climatology.",
            ),
            "dCRE_altitude": dCRE_comps["altitude"].assign_attrs(
                long_name="Low-cloud NET CRE: altitude effect", units="W m-2",
                description="Shifts in cloud-top pressure; total CF and tau fixed.",
            ),
            "dCRE_tau": dCRE_comps["tau"].assign_attrs(
                long_name="Low-cloud NET CRE: optical depth effect", units="W m-2",
                description="Shifts in cloud optical depth; total CF and pressure fixed.",
            ),
            "dCRE_residual": dCRE_comps["residual"].assign_attrs(
                long_name="Low-cloud NET CRE: residual cross-term", units="W m-2",
            ),
        },
        attrs={
            "satellite":         sat,
            "overlap_corrected": str(USE_OVERLAP_CORRECTION),
            "ocean_albedo":      str(OCEAN_ALBEDO),
            "low_press_indices": str(LOW_PRESS_INDICES),
            "sign_convention":   "Positive = upward flux (energy leaving Earth).",
        },
    )
    return ds_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    main()