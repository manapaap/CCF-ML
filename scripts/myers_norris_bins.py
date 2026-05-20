# -*- coding: utf-8 -*-
"""
myers_norris_binning.py

Replicates the Myers & Norris (2013) binning approach for estimating
partial derivatives of cloud variables with respect to a cloud controlling
factor (CCF), holding another CCF approximately constant via binning.

For each of the five stratocumulus regions (plus an ALL-region aggregate):
    1. Mask land, flatten (lat, lon, time) -> (cell,)
    2. Compute temporal ESDOF correction ratio via Bretherton (1999)
    3. Bin by var_bin into equal-frequency bins
    4. Within each bin, split on within-bin median of var_del
    5. Estimate slope and uncertainty via t-test with N_eff

Produces two figures:
    - plot_myers_norris : 2x3 panel of slopes vs var_bin (one per region + ALL)
    - plot_2d_histogram : 2x3 panel of 2D binned heatmaps (mean target_var)
                          with box area scaled to observation count

Usage (from CCF-ML working directory):
    import xarray as xr
    import scripts.utils as utils
    from myers_norris_binning import plot_myers_norris, plot_2d_histogram

    ceres_clean = xr.open_dataset('clean_data/ccf_cre_clean.nc')
    sc_regions  = utils.get_stratocumulus_regions()

    fig1, axes1 = plot_myers_norris(
        ds=ceres_clean, sc_regions=sc_regions,
        var_bin='eis', var_del='w_700', target_var='cldarea_low_adj',
    )
    fig2, axes2 = plot_2d_histogram(
        ds=ceres_clean, sc_regions=sc_regions,
        var_bin='eis', var_del='w_700', target_var='cldarea_low_adj',
    )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import xarray as xr
from scipy import stats

os.chdir('C:/Users/aakas/Documents/CCF-ML/')
import scripts.utils as utils


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Unit conversion: Pa/s -> hPa/day
# 1 Pa/s * (1 hPa / 100 Pa) * (86400 s / day) = 864 hPa/day
# The 2D histogram uses hPa/day; slopes divide by 10 locally for 10 hPa/day units.
PA_S_TO_HPA_DAY = 864.0

# Variables that need unit conversion on extraction
CONVERT_TO_HPA_DAY = {'w_700'}

UNITS = {
    'sst':             'SST (°C)',
    'eis':             'EIS (K)',
    'speed':           '10m Windspeed (m/s)',
    'cold_adv':        'Cold Advection (K/day)',
    'w_700':           'Subsidence (hPa/day)',
    'ln_AOD':          'ln(AOD)',
    'rh_700':          '700 hPa RH (%)',
    'cldarea_high':    'Cirrus Cover (%)',
    'cldarea_low_adj': 'Low Cloud Cover (%)',
    'dCRE_net':        'Low CRE Net (W/m²)',
    'dCRE_amt':        'Low CRE Amount (W/m²)',
    'dCRE_tau':        'Low CRE Tau (W/m²)',
    'dCRE_alt':        'Low CRE Alt (W/m²)',
    'lwp_low':         'Low LWP (g/m²)',
}


# ─────────────────────────────────────────────────────────────────────────────
#  ESDOF  (Bretherton 1999)
# ─────────────────────────────────────────────────────────────────────────────

def compute_esdof_ratio(da_region):
    """
    Estimate the temporal ESDOF correction ratio for a region.

    Builds the n_time x n_time temporal covariance matrix C from the
    (n_time, n_valid_cells) matrix of unit-normalised anomalies, then:
        N_eff = tr(C)^2 / tr(C^2)
        esdof_ratio = N_eff / n_time

    Parameters
    ----------
    da_region : xr.DataArray, shape (time, lat, lon), land already masked.

    Returns
    -------
    esdof_ratio : float in [0, 1]
    """
    vals   = da_region.values                    # (time, lat, lon)
    n_time = vals.shape[0]
    flat   = vals.reshape(n_time, -1)            # (time, n_cells)

    # Retain only cells valid across all timesteps
    valid = np.all(np.isfinite(flat), axis=0)
    flat  = flat[:, valid]

    if flat.shape[1] == 0:
        return 1.0

    # Anomalies, unit-normalised per cell
    flat = flat - flat.mean(axis=0, keepdims=True)
    std  = flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    flat = flat / std

    # Temporal covariance matrix (n_time x n_time), averaged over cells
    C    = (flat @ flat.T) / flat.shape[1]
    trC  = np.trace(C)
    trC2 = np.trace(C @ C)

    if trC2 == 0:
        return 1.0

    n_eff = (trC ** 2) / trC2
    return float(np.clip(n_eff / n_time, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-region data extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_region_flat(ds, region_dict, var_bin, var_del, target_var):
    """
    Subset to region, mask land, flatten (time, lat, lon) -> (n_obs,).

    Applies Pa/s -> hPa/day conversion for w_700 variables.

    Returns
    -------
    arr_bin, arr_del, arr_target : np.ndarray (n_obs,)
    esdof_ratio                  : float
    """
    # Spatial subset — handles lon wrapping (e.g. SEA crosses 360°)
    ds_reg = utils.region_sel(ds, region_dict)

    # Land mask
    ds_reg = ds_reg.where(ds_reg['sst'].notnull())

    # ESDOF on var_del (temporal autocorrelation of the split variable)
    esdof_ratio = compute_esdof_ratio(ds_reg[var_del])

    def flatten(da, varname):
        arr = da.values.reshape(-1).astype(np.float64)
        if varname in CONVERT_TO_HPA_DAY:
            arr = arr * PA_S_TO_HPA_DAY
        return arr

    arr_bin    = flatten(ds_reg[var_bin],    var_bin)
    arr_del    = flatten(ds_reg[var_del],    var_del)
    arr_target = flatten(ds_reg[target_var], target_var)

    mask = np.isfinite(arr_bin) & np.isfinite(arr_del) & np.isfinite(arr_target)
    arr_bin    = arr_bin[mask]
    arr_del    = arr_del[mask]
    arr_target = arr_target[mask]

    # Clip to 2.5–97.5th percentile on both CCF axes to remove extremes
    # that inflate error bars and sparse histogram cells
    bin_lo, bin_hi = np.percentile(arr_bin, [2.5, 97.5])
    del_lo, del_hi = np.percentile(arr_del, [2.5, 97.5])
    pct_mask = ((arr_bin >= bin_lo) & (arr_bin <= bin_hi) &
                (arr_del >= del_lo) & (arr_del <= del_hi))

    return arr_bin[pct_mask], arr_del[pct_mask], arr_target[pct_mask], esdof_ratio


# ─────────────────────────────────────────────────────────────────────────────
#  Slope estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_slopes(arr_bin, arr_del, arr_target, esdof_ratio, n_bins=9,
                   var_del=None, bin_edges=None):
    """
    Equal-width bins on arr_bin. Within each bin, split on within-bin
    median of arr_del (high vs low). Estimate finite-difference slope and
    95% CI half-width using ESDOF-adjusted sample size.

    Parameters
    ----------
    bin_edges : np.ndarray, optional
        Pre-computed bin edges for arr_bin. If provided, n_bins is ignored.
        Allows multiple target_vars to share identical bin positions.

    Returns
    -------
    bin_centers : (n_bins,)  median of var_bin per bin
    slopes      : (n_bins,)
    errors      : (n_bins,)  95% CI half-width
    n_bins_obs  : (n_bins,)  raw observation count
    bin_edges   : (n_bins+1,) bin edges used (for reuse by caller)
    """
    # Rescale arr_del from hPa/day to 10 hPa/day for slope units only,
    # but only for variables that were converted to hPa/day on extraction.
    # The 2D histogram keeps hPa/day; this keeps slope y-axis consistent with
    # Myers & Norris who report slopes in % / (10 hPa/day).
    if var_del in CONVERT_TO_HPA_DAY:
        arr_del = arr_del / 10.0

    # Equal-width bins — if bin_edges provided externally (multi-target plots),
    # reuse them so all target_vars share identical x-positions.
    if bin_edges is None:
        bin_edges = np.linspace(arr_bin.min(), arr_bin.max(), n_bins + 1)
    actual_n_bins = len(bin_edges) - 1

    bin_centers = np.full(actual_n_bins, np.nan)
    slopes      = np.full(actual_n_bins, np.nan)
    errors      = np.full(actual_n_bins, np.nan)
    n_bins_obs  = np.zeros(actual_n_bins, dtype=int)

    for i in range(actual_n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = ((arr_bin >= lo) & (arr_bin < hi) if i < actual_n_bins - 1
                  else (arr_bin >= lo) & (arr_bin <= hi))

        del_bin    = arr_del[in_bin]
        target_bin = arr_target[in_bin]
        n_bin      = in_bin.sum()

        if n_bin < 10:
            continue

        bin_centers[i] = np.median(arr_bin[in_bin])
        n_bins_obs[i]  = n_bin

        del_median = np.median(del_bin)
        high = del_bin >  del_median
        low  = del_bin <= del_median

        if high.sum() < 2 or low.sum() < 2:
            continue

        mean_target_high = target_bin[high].mean()
        mean_target_low  = target_bin[low].mean()
        mean_del_high    = del_bin[high].mean()
        mean_del_low     = del_bin[low].mean()

        delta_del    = mean_del_high - mean_del_low
        delta_target = mean_target_high - mean_target_low

        if np.abs(delta_del) < 1e-10:
            continue

        slopes[i] = delta_target / delta_del

        # Pooled std of target across both halves
        pooled_std = np.sqrt(
            (target_bin[high].var() * high.sum() +
             target_bin[low].var()  * low.sum()) /
            (high.sum() + low.sum())
        )

        # ESDOF-adjusted bin sample size
        n_eff_bin = max(n_bin * esdof_ratio, 2.0)
        t_crit    = stats.t.ppf(0.975, df=max(n_eff_bin - 2, 1.0))

        errors[i] = t_crit * (pooled_std / np.sqrt(n_eff_bin)) / np.abs(delta_del)

    return bin_centers, slopes, errors, n_bins_obs, bin_edges


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: collect per-region arrays + build ALL aggregate
# ─────────────────────────────────────────────────────────────────────────────

def _collect_region_data(ds, sc_regions, var_bin, var_del, target_var):
    """
    Extract flat arrays for each region and build an ALL aggregate.

    For the ALL region the ESDOF ratio is a weighted average of per-region
    ratios (weighted by n_obs). This is an approximation — if this analysis
    reaches review, consider recomputing ESDOF directly from the combined
    spatial field.

    Returns
    -------
    region_data : dict  {region_name: (arr_bin, arr_del, arr_target, esdof_ratio)}
                  Includes an 'ALL' entry as the last key.
    """
    region_data = {}

    all_bin, all_del, all_target = [], [], []
    total_obs      = 0
    weighted_esdof = 0.0

    for region, region_dict in sc_regions.items():
        print(f'  Extracting {region} ...', flush=True)
        arr_bin, arr_del, arr_target, esdof_ratio = extract_region_flat(
            ds, region_dict, var_bin, var_del, target_var
        )
        n = len(arr_bin)
        print(f'    n_obs={n:,}  esdof_ratio={esdof_ratio:.3f}')

        region_data[region] = (arr_bin, arr_del, arr_target, esdof_ratio)

        all_bin.append(arr_bin)
        all_del.append(arr_del)
        all_target.append(arr_target)
        weighted_esdof += esdof_ratio * n
        total_obs      += n

    # ALL aggregate
    all_esdof = weighted_esdof / total_obs if total_obs > 0 else 1.0
    region_data['ALL'] = (
        np.concatenate(all_bin),
        np.concatenate(all_del),
        np.concatenate(all_target),
        all_esdof,
    )
    print(f'  ALL: n_obs={total_obs:,}  esdof_ratio={all_esdof:.3f}')

    return region_data


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: draw one slope panel
# ─────────────────────────────────────────────────────────────────────────────

def _plot_slope_panel(ax, arr_bin, arr_del, arr_target, esdof_ratio,
                      region_label, xlabel, ylabel, n_bins, show_ylabel,
                      var_del=None):
    bin_centers, slopes, errors, n_obs, _ = compute_slopes(
        arr_bin, arr_del, arr_target,
        esdof_ratio=esdof_ratio,
        n_bins=n_bins,
        var_del=var_del,
    )

    valid   = np.isfinite(slopes) & np.isfinite(bin_centers)
    n_valid = n_obs[valid].astype(float)

    if n_valid.max() > 0:
        marker_sizes = 30 + 120 * (np.sqrt(n_valid) / np.sqrt(n_valid.max()))
    else:
        marker_sizes = np.full(valid.sum(), 50.0)

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.errorbar(
        bin_centers[valid], slopes[valid],
        yerr=errors[valid],
        fmt='none', ecolor='k', elinewidth=1.2, capsize=3, alpha=0.8,
        zorder=2,
    )
    ax.scatter(
        bin_centers[valid], slopes[valid],
        s=marker_sizes, color='grey', edgecolors='k', linewidths=0.5,
        zorder=3,
    )

    ax.set_title(region_label, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.5, ls=':')


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: draw one 2D histogram panel
# ─────────────────────────────────────────────────────────────────────────────

def _plot_2d_hist_panel(ax, arr_bin, arr_del, arr_target,
                        xlabel, ylabel, target_label, region_label,
                        n_bins_x, n_bins_y, cmap, norm, show_ylabel):
    """
    2D binned heatmap: color = mean(target_var) in each bin,
    box area scaled proportionally to sqrt(N / N_max) a la Myers & Norris.
    Median lines for each axis are drawn as in Fig 3a.
    """
    # Equal-width bin edges on each axis (consistent with Myers & Norris;
    # ensures all cells have identical dimensions before area scaling)
    x_edges = np.linspace(arr_bin.min(), arr_bin.max(), n_bins_x + 1)
    y_edges = np.linspace(arr_del.min(), arr_del.max(), n_bins_y + 1)

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    mean_target = np.full((ny, nx), np.nan)
    counts      = np.zeros((ny, nx), dtype=int)

    for ix in range(nx):
        x_lo, x_hi = x_edges[ix], x_edges[ix + 1]
        x_mask = ((arr_bin >= x_lo) & (arr_bin < x_hi) if ix < nx - 1
                  else (arr_bin >= x_lo) & (arr_bin <= x_hi))
        for iy in range(ny):
            y_lo, y_hi = y_edges[iy], y_edges[iy + 1]
            y_mask = ((arr_del >= y_lo) & (arr_del < y_hi) if iy < ny - 1
                      else (arr_del >= y_lo) & (arr_del <= y_hi))
            in_cell = x_mask & y_mask
            n = in_cell.sum()
            if n > 0:
                mean_target[iy, ix] = arr_target[in_cell].mean()
                counts[iy, ix]      = n

    # Max count for scaling box sizes
    max_count = counts.max() if counts.max() > 0 else 1

    # Bin half-widths for box sizing
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers  = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_widths   = x_edges[1:] - x_edges[:-1]
    y_widths   = y_edges[1:] - y_edges[:-1]

    for ix in range(nx):
        for iy in range(ny):
            n = counts[iy, ix]
            if n == 0:
                continue
            val = mean_target[iy, ix]
            if not np.isfinite(val):
                continue

            # Scale factor: area ∝ N, so side length ∝ sqrt(N)
            scale     = np.sqrt(n / max_count)
            box_w     = x_widths[ix] * scale
            box_h     = y_widths[iy] * scale
            x_corner  = x_centers[ix] - box_w / 2
            y_corner  = y_centers[iy]  - box_h / 2

            color = cmap(norm(val))
            rect  = mpatches.Rectangle(
                (x_corner, y_corner), box_w, box_h,
                facecolor=color, edgecolor='none',
            )
            ax.add_patch(rect)

    # Median reference lines (as in Myers & Norris Fig 3a)
    ax.axvline(np.median(arr_bin), color='k', lw=1.0, ls='-')
    ax.axhline(np.median(arr_del), color='k', lw=1.0, ls='-')

    ax.set_xlim(x_edges[0],  x_edges[-1])
    ax.set_ylim(y_edges[0],  y_edges[-1])
    ax.set_title(region_label, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.5, ls=':')


# ─────────────────────────────────────────────────────────────────────────────
#  Public figure: slope panels (2 x 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_myers_norris(ds, sc_regions, var_bin, var_del, target_var,
                      n_bins=9, figsize=(14, 8)):
    """
    2x3 panel of Myers & Norris style slope estimates, one per region + ALL.

    Parameters
    ----------
    ds          : xr.Dataset
    sc_regions  : dict from utils.get_stratocumulus_regions()
    var_bin     : str  CCF to bin by (x-axis)
    var_del     : str  CCF to differentiate over
    target_var  : str  cloud variable
    n_bins      : int  equal-frequency bins along var_bin (default 9)
    figsize     : tuple

    Returns
    -------
    fig, axes  (2x3 array)
    """
    region_data = _collect_region_data(ds, sc_regions, var_bin, var_del, target_var)

    xlabel = UNITS.get(var_bin, var_bin)
    # For slope ylabel, use 10 hPa/day for w_700 (slopes are computed in
    # those units); the 2D histogram keeps plain hPa/day.
    del_unit = '10 hPa/day' if var_del in CONVERT_TO_HPA_DAY else UNITS.get(var_del, var_del)
    ylabel = (f'∂({UNITS.get(target_var, target_var)}) / '
              f'∂({del_unit})')

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)

    for ax, (region, (arr_bin, arr_del, arr_target, esdof_ratio)) in zip(
            axes.flat, region_data.items()):
        show_ylabel = ax in axes[:, 0]
        _plot_slope_panel(
            ax, arr_bin, arr_del, arr_target, esdof_ratio,
            region_label=region, xlabel=xlabel, ylabel=ylabel,
            n_bins=n_bins, show_ylabel=show_ylabel,
            var_del=var_del,
        )

    # Hide the unused sixth panel if fewer than 6 entries
    for ax in axes.flat[len(region_data):]:
        ax.set_visible(False)

    fig.suptitle(
        f'∂({UNITS.get(target_var, target_var)}) / '
        f'∂({UNITS.get(var_del, var_del)})  '
        f'| binned by {UNITS.get(var_bin, var_bin)}',
        fontsize=12,
    )
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
#  Public figure: 2D histogram panels (2 x 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d_histogram(ds, sc_regions, var_bin, var_del, target_var,
                      n_bins_x=10, n_bins_y=10, figsize=(14, 8),
                      cmap='RdBu_r'):
    """
    2x3 panel of 2D binned heatmaps (mean target_var as color,
    box area scaled to observation count), one per region + ALL.

    Parameters
    ----------
    ds          : xr.Dataset
    sc_regions  : dict from utils.get_stratocumulus_regions()
    var_bin     : str  x-axis CCF
    var_del     : str  y-axis CCF
    target_var  : str  cloud variable (color fill)
    n_bins_x    : int  bins along var_bin (default 10)
    n_bins_y    : int  bins along var_del (default 10)
    figsize     : tuple
    cmap        : str  matplotlib colormap name

    Returns
    -------
    fig, axes  (2x3 array)
    """
    region_data = _collect_region_data(ds, sc_regions, var_bin, var_del, target_var)

    # Compute a shared color norm across all regions for comparability
    all_targets = np.concatenate([rd[2] for rd in region_data.values()])
    vmin, vmax  = np.nanpercentile(all_targets, [2, 98])
    norm        = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj    = plt.get_cmap(cmap)

    xlabel       = UNITS.get(var_bin,    var_bin)
    ylabel       = UNITS.get(var_del,    var_del)
    target_label = UNITS.get(target_var, target_var)

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    for ax, (region, (arr_bin, arr_del, arr_target, _)) in zip(
            axes.flat, region_data.items()):
        show_ylabel = ax in axes[:, 0]
        _plot_2d_hist_panel(
            ax, arr_bin, arr_del, arr_target,
            xlabel=xlabel, ylabel=ylabel,
            target_label=target_label, region_label=region,
            n_bins_x=n_bins_x, n_bins_y=n_bins_y,
            cmap=cmap_obj, norm=norm, show_ylabel=show_ylabel,
        )

    for ax in axes.flat[len(region_data):]:
        ax.set_visible(False)

    # Shared colorbar
    sm  = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.04)
    cbar.set_label(target_label, fontsize=9)

    fig.suptitle(
        f'Mean {target_label} | '
        f'{xlabel} vs {ylabel}',
        fontsize=12,
    )
    return fig, axes



# ─────────────────────────────────────────────────────────────────────────────
#  Multi-target: collect region data for several target_vars at once
# ─────────────────────────────────────────────────────────────────────────────

def _collect_region_data_multi(ds, sc_regions, var_bin, var_del, target_vars):
    """
    Like _collect_region_data but for multiple target_vars sharing the same
    var_bin / var_del.  Extraction of arr_bin, arr_del, and ESDOF is done once
    per region (using the first target_var); arr_target is extracted separately
    for each target_var but with the same NaN/percentile mask.

    Returns
    -------
    region_data : dict
        {region: {
            'arr_bin'     : np.ndarray,
            'arr_del'     : np.ndarray,
            'esdof_ratio' : float,
            'targets'     : {target_var: np.ndarray},
        }}
        Includes an ALL entry as the last key.
    """
    region_data = {}

    all_bin, all_del = [], []
    all_targets  = {tv: [] for tv in target_vars}
    total_obs      = 0
    weighted_esdof = 0.0

    for region, region_dict in sc_regions.items():
        print(f'  Extracting {region} ...', flush=True)

        # Spatial subset and land mask
        ds_reg = utils.region_sel(ds, region_dict)
        ds_reg = ds_reg.where(ds_reg['sst'].notnull())

        # ESDOF once per region on var_del
        esdof_ratio = compute_esdof_ratio(ds_reg[var_del])

        def flatten(da, varname):
            arr = da.values.reshape(-1).astype(np.float64)
            if varname in CONVERT_TO_HPA_DAY:
                arr = arr * PA_S_TO_HPA_DAY
            return arr

        arr_bin_raw = flatten(ds_reg[var_bin], var_bin)
        arr_del_raw = flatten(ds_reg[var_del], var_del)

        # Build a joint finite mask across all target_vars + bin + del
        finite_mask = np.isfinite(arr_bin_raw) & np.isfinite(arr_del_raw)
        target_arrs_raw = {}
        for tv in target_vars:
            tv_arr = flatten(ds_reg[tv], tv)
            finite_mask = finite_mask & np.isfinite(tv_arr)
            target_arrs_raw[tv] = tv_arr

        arr_bin_f = arr_bin_raw[finite_mask]
        arr_del_f = arr_del_raw[finite_mask]

        # Percentile clipping on bin and del axes (shared mask)
        bin_lo, bin_hi = np.percentile(arr_bin_f, [2.5, 97.5])
        del_lo, del_hi = np.percentile(arr_del_f, [2.5, 97.5])
        pct_mask = ((arr_bin_f >= bin_lo) & (arr_bin_f <= bin_hi) &
                    (arr_del_f >= del_lo) & (arr_del_f <= del_hi))

        arr_bin_out = arr_bin_f[pct_mask]
        arr_del_out = arr_del_f[pct_mask]
        targets_out = {tv: target_arrs_raw[tv][finite_mask][pct_mask]
                       for tv in target_vars}

        n = len(arr_bin_out)
        print(f'    n_obs={n:,}  esdof_ratio={esdof_ratio:.3f}')

        region_data[region] = {
            'arr_bin':     arr_bin_out,
            'arr_del':     arr_del_out,
            'esdof_ratio': esdof_ratio,
            'targets':     targets_out,
        }

        all_bin.append(arr_bin_out)
        all_del.append(arr_del_out)
        for tv in target_vars:
            all_targets[tv].append(targets_out[tv])
        weighted_esdof += esdof_ratio * n
        total_obs      += n

    # ALL aggregate — weighted-average ESDOF (see note in _collect_region_data)
    all_esdof = weighted_esdof / total_obs if total_obs > 0 else 1.0
    region_data['ALL'] = {
        'arr_bin':     np.concatenate(all_bin),
        'arr_del':     np.concatenate(all_del),
        'esdof_ratio': all_esdof,
        'targets':     {tv: np.concatenate(all_targets[tv]) for tv in target_vars},
    }
    print(f'  ALL: n_obs={total_obs:,}  esdof_ratio={all_esdof:.3f}')

    return region_data


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: draw one multi-target slope panel
# ─────────────────────────────────────────────────────────────────────────────

def _plot_multi_slope_panel(ax, region_entry, target_vars, var_del,
                            colors, n_bins, xlabel, ylabel,
                            show_ylabel, region_label):
    """
    Plot slopes for multiple target_vars on a single axis, with shared bin
    edges and horizontal offsets so error bars are visually distinct.
    Bin center ticks are drawn as small marks below the x-axis.
    """
    arr_bin = region_entry['arr_bin']
    arr_del = region_entry['arr_del']
    esdof   = region_entry['esdof_ratio']

    # Compute shared bin edges from arr_bin range
    bin_edges = np.linspace(arr_bin.min(), arr_bin.max(), n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    # Horizontal offsets: spread target_vars evenly across ±20% of bin width
    n_tv = len(target_vars)
    offsets = np.linspace(-0.2, 0.2, n_tv) * bin_width

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)

    # Track bin centers (same for all target_vars; draw ticks once)
    shared_centers = None

    for tv, color, offset in zip(target_vars, colors, offsets):
        arr_target = region_entry['targets'][tv]

        bin_centers, slopes, errors, n_obs, _ = compute_slopes(
            arr_bin, arr_del, arr_target,
            esdof_ratio=esdof,
            n_bins=n_bins,
            var_del=var_del,
            bin_edges=bin_edges,
        )

        valid   = np.isfinite(slopes) & np.isfinite(bin_centers)
        n_valid = n_obs[valid].astype(float)

        if n_valid.max() > 0:
            marker_sizes = 30 + 100 * (np.sqrt(n_valid) / np.sqrt(n_valid.max()))
        else:
            marker_sizes = np.full(valid.sum(), 40.0)

        x_plot = bin_centers[valid] + offset

        ax.errorbar(
            x_plot, slopes[valid],
            yerr=errors[valid],
            fmt='none', ecolor=color, elinewidth=1.2, capsize=3, alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            x_plot, slopes[valid],
            s=marker_sizes, color=color, edgecolors='k', linewidths=0.4,
            zorder=3, label=UNITS.get(tv, tv),
        )

        if shared_centers is None:
            shared_centers = bin_centers

    # Draw small bin-center ticks just below the x-axis
    if shared_centers is not None:
        ax_ymin = ax.get_ylim()[0]
        tick_y  = ax_ymin
        for bc in shared_centers[np.isfinite(shared_centers)]:
            ax.annotate(
                '', xy=(bc, tick_y),
                xytext=(bc, tick_y),
                annotation_clip=False,
                arrowprops=None,
            )
            ax.axvline(bc, color='k', lw=0.5, alpha=0.25, ls=':',
                       ymin=0, ymax=0.03, zorder=1)

    ax.set_title(region_label, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.5, ls=':')


# ─────────────────────────────────────────────────────────────────────────────
#  Public figure: multi-target slope panels (2 x 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_myers_norris_multi(ds, sc_regions, var_bin, var_del, target_vars,
                            n_bins=9, figsize=(14, 8),
                            colors=None):
    """
    2x3 panel of Myers & Norris style slopes for multiple target_vars
    overlaid on each panel, one panel per region + ALL.

    All target_vars share the same bin edges (computed from their common
    var_bin range) so x-positions are consistent. Points are offset
    horizontally within each bin to keep error bars legible.

    Parameters
    ----------
    ds          : xr.Dataset
    sc_regions  : dict from utils.get_stratocumulus_regions()
    var_bin     : str   CCF to bin by (x-axis)
    var_del     : str   CCF to differentiate over
    target_vars : list  cloud variables to overlay, e.g.
                        ['dCRE_net', 'dCRE_amt', 'dCRE_tau']
    n_bins      : int   equal-width bins (default 9)
    figsize     : tuple
    colors      : list  one color per target_var; defaults to tab10

    Returns
    -------
    fig, axes  (2x3 array)
    """
    if colors is None:
        cmap10 = plt.get_cmap('tab10')
        colors = [cmap10(i) for i in range(len(target_vars))]

    region_data = _collect_region_data_multi(
        ds, sc_regions, var_bin, var_del, target_vars
    )

    del_unit = ('10 hPa/day' if var_del in CONVERT_TO_HPA_DAY
                else UNITS.get(var_del, var_del))
    # ylabel uses first target_var units as representative (all same units here)
    tv_unit  = UNITS.get(target_vars[0], target_vars[0])
    ylabel   = f'∂(CRE) / ∂({del_unit})'
    xlabel   = UNITS.get(var_bin, var_bin)

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)

    for ax, (region, region_entry) in zip(axes.flat, region_data.items()):
        show_ylabel = ax in axes[:, 0]
        _plot_multi_slope_panel(
            ax, region_entry, target_vars, var_del,
            colors=colors, n_bins=n_bins,
            xlabel=xlabel, ylabel=ylabel,
            show_ylabel=show_ylabel, region_label=region,
        )

    for ax in axes.flat[len(region_data):]:
        ax.set_visible(False)

    # Shared legend on the figure
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markeredgecolor='k',
                   markersize=8, label=UNITS.get(tv, tv))
        for tv, c in zip(target_vars, colors)
    ]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.98, 0.02), fontsize=9, framealpha=0.8)

    fig.suptitle(
        f'∂(CRE) / ∂({del_unit})  |  binned by {UNITS.get(var_bin, var_bin)}',
        fontsize=12,
    )
    return fig, axes

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ceres_clean = xr.open_dataset('clean_data/ccf_cre_clean.nc')
    sc_regions  = utils.get_stratocumulus_regions()

    fig1, _ = plot_myers_norris(
        ds=ceres_clean, sc_regions=sc_regions,
        var_bin='eis', var_del='w_700', target_var='cldarea_low_adj',
        n_bins=9,
    )
    fig2, _ = plot_2d_histogram(
        ds=ceres_clean, sc_regions=sc_regions,
        var_bin='eis', var_del='w_700', target_var='cldarea_low_adj',
        n_bins_x=10, n_bins_y=10,
    )
    fig3, _ = plot_myers_norris_multi(
        ds=ceres_clean, sc_regions=sc_regions,
        var_bin='eis', var_del='w_700',
        target_vars=['dCRE_net', 'dCRE_amt', 'dCRE_tau'],
        n_bins=9,
    )

    os.makedirs('figures', exist_ok=True)
    fig1.savefig('figures/myers_norris_slopes.png',      dpi=150, bbox_inches='tight')
    fig2.savefig('figures/myers_norris_2d_hist.png',     dpi=150, bbox_inches='tight')
    fig3.savefig('figures/myers_norris_cre_multi.png',   dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()