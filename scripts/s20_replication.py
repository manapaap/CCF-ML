# -*- coding: utf-8 -*-
"""
s20_replication.py

Replicates Scott et al. (2020, J. Climate 33, 7717-7734) "model 1" maps of
low-cloud radiative sensitivity to meteorology, as closely as our data
allow, at 2.5° instead of S20's 5° resolution:

    Fig. 4a : R² of model 1 (total low-cloud radiative perturbation)
    Fig. 5  : total nonobscured low-cloud radiative perturbation
    Fig. 6  : amount component       (first term of S20 Eq. 8)
    Fig. 7  : altitude + optical depth component (second term of Eq. 8)
    Fig. S1 analogue : nonobscured low-cloud fraction

Method (S20 sections 2-3 and appendix)
--------------------------------------
- CERES-FBCT Terra/Aqua MODIS only, July 2002 - December 2018
  (RECORD = 'terra'). RECORD = 'merged' extends to December 2025 with
  NOAA-20 VIIRS, merged as in clean_fbct_new.py.
- Extra target not in S20: CERES-SYN overlap-adjusted low cloud
  (cldarea_low_syn, low / (1 - mid - high)), for comparison with FBCT L_n.
- Six predictors: SST, EIS, Tadv, RH700, ω700, WS. No cirrus or aerosol.
- EIS uses SST as the surface temperature (from regime_masks.py),
  matching Wood & Bretherton (2006) as used by S20.
- Deseasonalized, linearly detrended monthly anomalies over 2002-2018;
  predictors standardized by their local standard deviation.
- Nonobscured low cloud: S20 replace L' by L_n'(1 - Ū) with
  L_n = L / (1 - U). clean_fbct_new.py applies this (1 - Ū) weighting,
  with Ū the monthly climatology of FBCT upper-level cloud fraction
  (USE_VISIBLE_WEIGHT); this script checks for it.
- Model 1 (Eq. 10): R' = sum_i β_i x_i' + ε, no intercept, at each ocean
  grid box between 60°S and 60°N.
- Significance (Eq. A1): 95% CI  β ± t·sqrt(C_ii · N_nom / N_eff), with
  C = σ̂²(XᵀX)⁻¹, N_nom/N_eff = (1 + r_t)/(1 - r_t) where r_t is the lag-1
  autocorrelation of the cloud-radiative anomaly (the target), and
  df = N_eff - (n + 1). No multiple-testing correction. All coefficients
  are shaded; significant grid boxes are stippled.

Remaining differences from S20 (data, not method)
-------------------------------------------------
- 2.5° grid instead of 5°
- ERA5 SST instead of NOAA OISST v2 (S20 report insensitivity)
- WS is the speed of the monthly-mean wind, not the monthly mean of the
  scalar speed (our ERA5 download has monthly-mean u10/v10 only)
- Released FBCT Ed4.1 instead of the beta version S20 used
- EIS LCL from Bolton (1980) instead of Georgakakos & Bras (1984)

Run from the CCF-ML root (after regime_masks.py and clean_fbct_new.py):
    python -m scripts.s20_replication
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe
from scipy import stats

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

from scripts.clean_fbct_new import (
    build_target_grid, LOW_PRESS_INDICES, HIGH_PRESS_INDICES,
    HIGH_CF_CAP, FILL_VALUE, VAR_CF,
)


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

# 'terra' : Terra/Aqua MODIS only over S20's 2002-2018 period
# 'merged': Terra/Aqua + NOAA-20 VIIRS through the end of the ERA5 record,
#           merged like clean_fbct_new.py (per-satellite anomalies averaged
#           where both satellites exist)
RECORD = 'merged'

SATELLITES = {
    'terra': {'fbct_dir': 'raw_data/ceres_fbct_terra',
              'cre_file': 'clean_data/low_cloud_cre_direct_terra.nc'},
    'noaa':  {'fbct_dir': 'raw_data/ceres_fbct_noaa',
              'cre_file': 'clean_data/low_cloud_cre_direct_noaa.nc'},
}
RECORD_SATS = {'terra': ['terra'], 'merged': ['terra', 'noaa']}[RECORD]
PERIOD      = {'terra':  ('2002-07', '2018-12'),
               'merged': ('2002-07', '2025-12')}[RECORD]
SUFFIX      = '' if RECORD == 'terra' else f'_{RECORD}'

CCF_FILE   = 'clean_data/ccf_cre_clean.nc'           # SST, Tadv, RH, ω, WS, SYN CF
EIS_FILE   = 'clean_data/regime_masks.nc'            # SST-based EIS
FBCT_CACHE = 'clean_data/fbct_{sat}_low_high_2p5.nc'
OUT_FILE   = f'clean_data/s20_regression{SUFFIX}.nc'
FIG_DIR    = f'figures/s20_replication{SUFFIX}'

LAT_MAX = 60.0
ALPHA   = 0.05

PREDICTORS = ['sst', 'eis', 'Tadv', 'rh_700', 'w_700', 'speed']
PRED_LABELS = {
    'sst': 'SST', 'eis': 'EIS', 'Tadv': 'Tadv',
    'rh_700': 'RH$_{700}$', 'w_700': 'ω$_{700}$', 'speed': 'WS',
}

# target: (S20 figure, suptitle, panel letters?, colorbar limit, ticks, symbol, unit)
TARGETS = {
    'R_total':  ('fig5_total', 'Total Low-Cloud Radiative Perturbation',
                 True, 5.0, [-4, -2, 0, 2, 4], 'R', 'W m$^{-2}$ σ$^{-1}$'),
    'R_amount': ('fig6_amount', 'Low-Cloud Amount Component',
                 False, 3.0, [-3, -1.5, 0, 1.5, 3], 'R', 'W m$^{-2}$ σ$^{-1}$'),
    'R_shape':  ('fig7_altitude_tau', 'Low-Cloud Altitude + Optical Depth Component',
                 False, 3.0, [-3, -1.5, 0, 1.5, 3], 'R', 'W m$^{-2}$ σ$^{-1}$'),
    'L_n':      ('figS1_low_cloud', 'Nonobscured Low-Cloud Fraction',
                 True, 5.0, [-4, -2, 0, 2, 4], 'L$_n$', '% σ$^{-1}$'),
    # Not in S20: CERES-SYN overlap-adjusted low cloud (the PDP / MN13
    # target), regressed the same way for a like-for-like comparison
    'L_syn':    ('figS1b_low_cloud_syn', 'CERES-SYN Overlap-Adjusted Low-Cloud Fraction',
                 True, 5.0, [-4, -2, 0, 2, 4], 'L$_{SYN}$', '% σ$^{-1}$'),
}

# ═════════════════════════════════════════════


def by_month_string(ds):
    """Re-index time to 'YYYY-MM' strings so files with different day stamps align."""
    return ds.assign_coords(time=ds['time'].dt.strftime('%Y-%m').values)


def in_period(ds):
    t = ds['time'].values
    return ds.isel(time=(t >= PERIOD[0]) & (t <= PERIOD[1]))


def lag1_autocorr(resid, valid):
    """
    Lag-1 autocorrelation along time for every column, using only
    consecutive month pairs where both months are valid.

    Parameters
    ----------
    resid : (n_time, n_cells), zero where invalid
    valid : (n_time, n_cells) bool
    """
    pair = valid[1:] & valid[:-1]
    a = np.where(pair, resid[1:], 0.0)
    b = np.where(pair, resid[:-1], 0.0)
    n = pair.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        a_m = a.sum(0) / n
        b_m = b.sum(0) / n
        cov = (np.where(pair, (a - a_m) * (b - b_m), 0.0)).sum(0)
        var = np.sqrt((np.where(pair, (a - a_m) ** 2, 0.0)).sum(0) *
                      (np.where(pair, (b - b_m) ** 2, 0.0)).sum(0))
        return cov / var


def anomalies(da):
    """
    Deseasonalize (subtract monthly-mean climatology) and remove a linear
    trend at every grid box, over the analysis period only.
    """
    month = xr.DataArray([int(t[5:]) for t in da['time'].values],
                         dims='time', coords={'time': da['time']},
                         name='month')
    anom = da.groupby(month) - da.groupby(month).mean('time')
    anom = anom.drop_vars('month', errors='ignore')

    t   = xr.DataArray(np.arange(da.sizes['time'], dtype=float),
                       dims='time', coords={'time': da['time']})
    ok  = anom.notnull()
    tm  = t.where(ok).mean('time')
    am  = anom.mean('time')
    slope = (((t - tm) * (anom - am)).where(ok).sum('time')
             / ((t - tm) ** 2).where(ok).sum('time'))
    return anom - am - slope * (t - tm)


def load_fbct_low_high(sat):
    """
    FBCT low-level (L) and upper-level (U) cloud fraction [%] for one
    satellite, summed over (tau, plev) bins and conservatively regridded
    to 2.5°. Cached because the raw files are several GB.
    """
    cache = FBCT_CACHE.format(sat=sat)
    if os.path.isfile(cache):
        return xr.open_dataset(cache)

    print(f'  Summing {sat} FBCT cloud fraction bins (first run only) ...',
          flush=True)
    files = sorted(glob.glob(os.path.join(SATELLITES[sat]['fbct_dir'], '*.nc')))
    cf = xr.open_mfdataset(files, combine='by_coords')[VAR_CF]
    cf = cf.where(cf != FILL_VALUE).rename({'press': 'plev', 'opt': 'tau'})

    lh = xr.Dataset({
        'L': cf.isel(plev=LOW_PRESS_INDICES).sum(['tau', 'plev'], min_count=1),
        'U': cf.isel(plev=HIGH_PRESS_INDICES).sum(['tau', 'plev'], min_count=1),
    }).load()

    regridder = xe.Regridder(lh, build_target_grid(2.5), method='conservative',
                             periodic=True, ignore_degenerate=True)
    lh = regridder(lh, skipna=True)
    lh.to_netcdf(cache)
    return lh


def satellite_targets(sat):
    """
    FBCT targets for one satellite, as anomalies against that satellite's
    own climatology (as in clean_fbct_new.py).
    """
    cre = in_period(by_month_string(xr.open_dataset(SATELLITES[sat]['cre_file'])))
    lh  = in_period(by_month_string(load_fbct_low_high(sat)))
    times = sorted(set(cre['time'].values) & set(lh['time'].values))
    print(f'  {sat}: {len(times)} months, {times[0]} to {times[-1]}')
    sel = dict(time=times)

    if cre.attrs.get('visible_weight_1_minus_Ubar') != 'True':
        raise ValueError(f"{SATELLITES[sat]['cre_file']} lacks the (1 - Ū) "
                         'weighting; rerun clean_fbct_new.py')
    U = (lh['U'].sel(**sel) / 100).clip(max=HIGH_CF_CAP)

    # CRE files (already weighted by 1 - Ū in clean_fbct_new.py):
    # positive = more upward flux (cooling); flip to S20's positive =
    # energy gained by the climate system.
    raw = {
        'R_total':  -cre['dCRE_net'].sel(**sel),
        'R_amount': -cre['dCRE_amount'].sel(**sel),
        'R_shape':  -cre['dCRE_shape'].sel(**sel),
        'L_n':      lh['L'].sel(**sel) / (1 - U),
    }
    return xr.Dataset({k: anomalies(v) for k, v in raw.items()})


def build_dataset():
    """Predictor and target anomalies over PERIOD on the 2.5° grid."""
    ccf = in_period(by_month_string(xr.open_dataset(CCF_FILE)))
    eis = in_period(by_month_string(xr.open_dataset(EIS_FILE)))['eis']

    # Equal-weight average of per-satellite anomalies where they overlap
    per_sat = xr.align(*[satellite_targets(s) for s in RECORD_SATS],
                       join='outer')
    fbct = xr.concat(per_sat, dim='sat').mean('sat', skipna=True)

    times = sorted(set(ccf['time'].values) & set(eis['time'].values)
                   & set(fbct['time'].values))
    print(f'  Analysis: {len(times)} months, {times[0]} to {times[-1]}')
    sel = dict(time=times)

    raw = {k: fbct[k].sel(**sel) for k in fbct.data_vars}
    raw['L_syn'] = ccf['cldarea_low_syn'].sel(**sel)
    for p in PREDICTORS:
        raw[p] = (eis if p == 'eis' else ccf[p]).sel(**sel)

    ds = xr.Dataset({k: anomalies(v) for k, v in raw.items()})

    ocean = (ccf['sst'].notnull().all('time')
             & (np.abs(ccf['lat']) <= LAT_MAX))
    return ds.where(ocean), ocean


def s20_regression(X, y):
    """
    S20 model 1 at every grid box (vectorized over cells): no intercept,
    standardized predictors, appendix Eq. A1 confidence intervals.

    Parameters
    ----------
    X : (n_time, n_cells, n) predictor anomalies
    y : (n_time, n_cells)    target anomalies

    Returns
    -------
    dict: 'beta', 'delta' (95% CI half-width), 'sig' (n_cells, n);
          'r_t', 'n_eff', 'r2' (n_cells,)
    """
    n_time, n_cells, n = X.shape
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=2)
    N     = valid.sum(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        Xv = np.where(valid[..., None], X, 0.0)
        mu = Xv.sum(0) / N[:, None]
        sd = np.sqrt((valid[..., None] * (Xv - mu) ** 2).sum(0) / (N[:, None] - 1))
        Xz = np.where(valid[..., None], (Xv - mu) / sd, 0.0)
    yv = np.where(valid, y, 0.0)

    XtX = np.einsum('tci,tcj->cij', Xz, Xz)
    Xty = np.einsum('tci,tc->ci', Xz, yv)
    fit = (N > 3 * n) & np.all(sd > 0, axis=1)

    beta = np.full((n_cells, n), np.nan)
    inv  = np.full((n_cells, n, n), np.nan)
    inv[fit]  = np.linalg.inv(XtX[fit])
    beta[fit] = np.einsum('cij,cj->ci', inv[fit], Xty[fit])

    resid = np.where(valid, yv - np.einsum('tci,ci->tc', Xz, np.nan_to_num(beta)), 0.0)
    ssr   = (resid ** 2).sum(0)

    # r_t: lag-1 autocorrelation of the cloud-radiative anomaly itself
    r_t    = np.clip(lag1_autocorr(yv, valid), 0.0, 0.99)
    n_eff  = N * (1 - r_t) / (1 + r_t)
    df     = n_eff - (n + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        sigma2 = ssr / (N - n)                                  # mean squared error
        C_ii   = sigma2[:, None] * np.diagonal(inv, axis1=1, axis2=2)
        t_crit = stats.t.ppf(1 - ALPHA / 2, df)
        delta  = t_crit[:, None] * np.sqrt(C_ii * (N / n_eff)[:, None])
        r2     = 1 - ssr / (yv ** 2).sum(0)

    bad = ~fit | (df <= 0)
    beta[bad] = np.nan
    delta[bad] = np.nan
    return {
        'beta': beta, 'delta': delta, 'sig': np.abs(beta) > delta,
        'r_t': np.where(bad, np.nan, r_t),
        'n_eff': np.where(bad, np.nan, n_eff),
        'r2': np.where(bad, np.nan, r2),
    }


def regress_target(ds, target):
    stacked = ds[PREDICTORS + [target]].stack(cell=('lat', 'lon'))
    X = np.stack([stacked[p].transpose('time', 'cell').values
                  for p in PREDICTORS], axis=2).astype(np.float64)
    y = stacked[target].transpose('time', 'cell').values.astype(np.float64)
    res = s20_regression(X, y)

    return xr.Dataset(
        {
            'beta':  (('cell', 'predictor'), res['beta']),
            'delta': (('cell', 'predictor'), res['delta']),
            'sig':   (('cell', 'predictor'), res['sig']),
            'r_t':   ('cell', res['r_t']),
            'n_eff': ('cell', res['n_eff']),
            'r2':    ('cell', res['r2']),
        },
        coords={'cell': stacked['cell'], 'predictor': PREDICTORS},
    ).unstack('cell').transpose('predictor', 'lat', 'lon')


# ─────────────────────────────────────────────
#  Plotting (S20 style)
# ─────────────────────────────────────────────

PROJ     = ccrs.PlateCarree(central_longitude=205)
DATA_CRS = ccrs.PlateCarree()


def _edges(c):
    d = c[1] - c[0]
    return np.append(c - d / 2, c[-1] + d / 2)


def _style_map(ax):
    ax.add_feature(cfeature.LAND, facecolor='black', zorder=3)
    ax.set_extent([-180, 180, -LAT_MAX, LAT_MAX], crs=PROJ)
    gl = ax.gridlines(xlocs=np.arange(-180, 181, 30),
                      ylocs=np.arange(-60, 61, 15),
                      color='k', linestyle=':', linewidth=0.6, zorder=4)
    gl.top_labels = gl.right_labels = False
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_coefficients(res, target):
    fname, suptitle, letters, vmax, ticks, sym, unit = TARGETS[target]
    lat, lon = res['lat'].values, res['lon'].values
    lon_e, lat_e = _edges(lon), _edges(lat)
    lon2, lat2 = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(3, 2, figsize=(12, 7.6),
                             subplot_kw={'projection': PROJ})
    fig.subplots_adjust(wspace=0.05, hspace=0.28)

    for j, (ax, p) in enumerate(zip(axes.flat, PREDICTORS)):
        beta = res['beta'].sel(predictor=p).values
        sig  = res['sig'].sel(predictor=p).values.astype(bool)

        pm = ax.pcolormesh(lon_e, lat_e, beta, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, transform=DATA_CRS,
                           zorder=1)
        ax.scatter(lon2[sig], lat2[sig], s=0.6, c='k', marker='.',
                   linewidths=0, transform=DATA_CRS, zorder=2)
        _style_map(ax)

        prefix = f'{"abcdef"[j]}) ' if letters else ''
        ax.set_title(f'{prefix}∂{sym}/∂{PRED_LABELS[p]}', fontsize=14)

    cbar = fig.colorbar(pm, ax=axes, orientation='horizontal',
                        fraction=0.035, pad=0.04, aspect=45, ticks=ticks,
                        extend='both')
    cbar.set_label(unit, fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    fig.suptitle(suptitle, fontsize=15, y=0.955)
    return fig, fname


def plot_r2(res):
    lat, lon = res['lat'].values, res['lon'].values
    fig, ax = plt.subplots(figsize=(8, 3.6), subplot_kw={'projection': PROJ})
    pm = ax.pcolormesh(_edges(lon), _edges(lat), res['r2'].values,
                       cmap='YlGnBu', vmin=0, vmax=1, transform=DATA_CRS)
    _style_map(ax)
    ax.set_title('a) Total R$^2$ Model 1', fontsize=14)
    cbar = fig.colorbar(pm, ax=ax, orientation='horizontal', fraction=0.06,
                        pad=0.06, aspect=40)
    cbar.ax.tick_params(labelsize=11)
    return fig


def main():
    print('Building anomalies:', flush=True)
    ds, ocean = build_dataset()

    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}
    for target in TARGETS:
        res = regress_target(ds, target)
        results[target] = res

        w = np.cos(np.deg2rad(res['lat'])) * ocean
        r2_mean = float((res['r2'] * w).sum() / w.where(res['r2'].notnull()).sum())
        print(f'\n{target}: area-mean R² = {r2_mean:.2f}, '
              f'median r_t = {float(res["r_t"].median()):.2f}, '
              f'median N_eff = {float(res["n_eff"].median()):.0f}')
        for p in PREDICTORS:
            frac = float(res['sig'].sel(predictor=p).where(res['beta'].sel(predictor=p).notnull()).mean())
            print(f'  {p:7s} significant at {100 * frac:4.1f}% of grid boxes')

        fig, fname = plot_coefficients(res, target)
        fig.savefig(f'{FIG_DIR}/{fname}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig = plot_r2(results['R_total'])
    fig.savefig(f'{FIG_DIR}/fig4a_r2.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    out = xr.concat([results[t] for t in TARGETS],
                    dim=xr.DataArray(list(TARGETS), dims='target', name='target'))
    out['sig'] = out['sig'].astype('int8')
    out.attrs['description'] = (
        'Replication of Scott et al. (2020) model 1 at 2.5 deg, '
        f'{PERIOD[0]} to {PERIOD[1]}, FBCT record: {RECORD}. beta: target units '
        'per 1 sigma of predictor; R targets positive = energy gain.')
    out.to_netcdf(OUT_FILE)
    print(f'\nSaved {OUT_FILE} and figures in {FIG_DIR}/')


if __name__ == '__main__':
    main()
