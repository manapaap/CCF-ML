# -*- coding: utf-8 -*-
"""
mn13_cf_definitions.py

MN13 Fig. 4c replication, ∂(CF)/∂(ω700) at fixed EIS, over the MN13
subsidence regime for several definitions of low cloud fraction, all on
one panel:

    1) SYN  low / (1 - mid - high)   nonobscured: corrected for all cloud above
    2) FBCT low / (1 - mid - high)   nonobscured (S20 L_n)
    3) SYN  low / (1 - high)         corrected for high cloud only
    4) FBCT low / (1 - high)         corrected for high cloud only
    5) FBCT low + mid                as seen from above (MN13 used ISCCP low+mid)
    6) FBCT (low + mid) / (1 - high) low + mid corrected for high cloud
    7) SYN  low                      uncorrected (as seen from above)
    8) FBCT low                      uncorrected (as seen from above)

Layers
------
CERES-SYN : low sfc-700, mid 700-300 (mid-low + mid-high), high above 300 hPa
CERES-FBCT: low 1000-680, mid 680-440, high above 440 hPa (ISCCP bins)

Each field is built on the native 1° grid, bilinearly regridded to the 2.5°
grid of ccf_cre_clean.nc (as in clean_data.py), and deseasonalized and
linearly detrended over the ccf_cre_clean.nc period. Binning, ESDOF, and
90% CIs follow mn13_fig4.py.

Run from the CCF-ML root (xesmf needs ESMFMKFILE if the env is not active):
    python -m scripts.mn13_cf_definitions
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xesmf as xe

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

import scripts.utils as utils
from scripts.myers_norris_bins import compute_esdof_ratio, compute_slopes, PA_S_TO_HPA_DAY
from scripts.regime_mn_slopes import align_masks, weighted_mean_slope
from scripts.mn13_fig4 import plot_d_dw, EIS_EDGES_SLOPE, CONF


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

DATA_FILE = 'clean_data/ccf_cre_clean.nc'
MASK_FILE = 'clean_data/regime_masks.nc'
SYN_FILE  = 'raw_data/ceres_syn_new.nc'
FIG_PATH  = 'figures/mn13_cf_definitions.png'

FBCT_LOW, FBCT_MID, FBCT_HIGH = [0, 1], [2, 3], [4, 5, 6]
COVER_CAP = 90.0          # cap on the obscuring cloud in each denominator (%)

DEFINITIONS = {
    'syn_nonobs':  'SYN low / (1 − mid − high)',
    'fbct_nonobs': 'FBCT low / (1 − mid − high)',
    'syn_hi':      'SYN low / (1 − high)',
    'fbct_hi':     'FBCT low / (1 − high)',
    'fbct_lm':     'FBCT low + mid',
    'fbct_lm_hi':  'FBCT (low + mid) / (1 − high)',
    'syn_raw':     'SYN low (uncorrected)',
    'fbct_raw':    'FBCT low (uncorrected)',
}
COLORS = ['tab:blue', 'tab:orange', 'tab:cyan', 'tab:red', 'tab:gray', 'k',
          'tab:purple', 'tab:green']

# ═════════════════════════════════════════════


def corrected(num, cover):
    """Random-overlap correction num / (1 - cover), both in %."""
    return (100 * num / (100 - cover.clip(max=COVER_CAP))).clip(max=100)


def build_definitions():
    """The six cloud fractions [%] on the native 1° grid."""
    syn = xr.open_dataset(SYN_FILE)[['cldarea_low_mon', 'cldarea_mid_low_mon',
                                     'cldarea_mid_high_mon', 'cldarea_high_mon']]
    s_low  = syn['cldarea_low_mon']
    s_mid  = syn['cldarea_mid_low_mon'] + syn['cldarea_mid_high_mon']
    s_high = syn['cldarea_high_mon']

    cf = utils.fbct_press_profile()
    f_low  = cf.isel(press=FBCT_LOW).sum('press', min_count=1)
    f_mid  = cf.isel(press=FBCT_MID).sum('press', min_count=1)
    f_high = cf.isel(press=FBCT_HIGH).sum('press', min_count=1)

    syn_defs = xr.Dataset({
        'syn_nonobs': corrected(s_low, s_mid + s_high),
        'syn_hi':     corrected(s_low, s_high),
        'syn_raw':    s_low,
    })
    fbct_defs = xr.Dataset({
        'fbct_nonobs': corrected(f_low, f_mid + f_high),
        'fbct_hi':     corrected(f_low, f_high),
        'fbct_lm':     f_low + f_mid,
        'fbct_lm_hi':  corrected(f_low + f_mid, f_high),
        'fbct_raw':    f_low,
    })
    return syn_defs, fbct_defs


def to_anomalies(ds_1deg, target):
    """Regrid 1° -> target grid (bilinear), deseasonalize and detrend."""
    ds_1deg = ds_1deg.assign_coords(
        time=ds_1deg['time'] - pd.Timedelta(days=14))
    ds_1deg = ds_1deg.reindex(time=target['time'])
    regridder = xe.Regridder(ds_1deg[['lat', 'lon']], target[['lat', 'lon']],
                             'bilinear', periodic=True)
    ds = regridder(ds_1deg)

    anom = ds.groupby('time.month') - ds.groupby('time.month').mean('time')
    anom = anom.drop_vars('month')
    t  = xr.DataArray(np.arange(anom.sizes['time'], dtype=float), dims='time',
                      coords={'time': anom['time']})
    ok = anom.notnull()
    tm = t.where(ok).mean('time')
    am = anom.mean('time')
    slope = (((t - tm) * (anom - am)).where(ok).sum('time')
             / ((t - tm) ** 2).where(ok).sum('time'))
    return anom - am - slope * (t - tm)


def main():
    ds    = xr.open_dataset(DATA_FILE)
    masks = align_masks(xr.open_dataset(MASK_FILE), ds)

    print('Building cloud fraction definitions ...', flush=True)
    syn_defs, fbct_defs = build_definitions()
    cf = xr.merge([to_anomalies(syn_defs, ds), to_anomalies(fbct_defs, ds)])

    # Sanity check: definition 2 should reproduce cldarea_low_adj
    a, b = cf['fbct_nonobs'].values.ravel(), ds['cldarea_low_adj'].values.ravel()
    k = np.isfinite(a) & np.isfinite(b)
    print(f'corr(fbct_nonobs, cldarea_low_adj) = {np.corrcoef(a[k], b[k])[0, 1]:.4f}')

    region = masks['mask_mn13'].astype(bool) & ds['sst'].notnull()
    active = masks['mn13_active'].astype(bool) & region

    def flat(da):
        return da.where(active).values.ravel().astype(float)

    d = {'eis': flat(ds['eis']), 'w': flat(ds['w_700']) * PA_S_TO_HPA_DAY}
    for name in DEFINITIONS:
        d[name] = flat(cf[name])
    ok = np.all([np.isfinite(x) for x in d.values()], axis=0)
    d = {k: x[ok] for k, x in d.items()}
    d['esdof_w'] = compute_esdof_ratio(ds['w_700'].where(region))
    print(f'MN13 regime: {int(region.any("time").sum())} grid boxes, '
          f'n_obs={ok.sum():,}, ESDOF ratio w700={d["esdof_w"]:.3f}')

    print(f'\n{"definition":32s} {"dCF/dw700":>10s} {"90% CI":>7s} {"CF std":>7s}')
    for name, label in DEFINITIONS.items():
        _, s, e, n, _ = compute_slopes(d['eis'], d['w'], d[name],
                                       esdof_ratio=d['esdof_w'], var_del='w_700',
                                       bin_edges=EIS_EDGES_SLOPE, conf=CONF)
        mean, ci = weighted_mean_slope(s, e, n)
        print(f'{label:32s} {mean:+10.2f} {ci:7.2f} {np.std(d[name]):7.2f}')

    fig, ax = plt.subplots(figsize=(11, 7))
    plot_d_dw(ax, d, list(DEFINITIONS), COLORS, list(DEFINITIONS.values()),
              '∂(CF)/∂(ω$_{700}$) (% (10 hPa day$^{-1}$)$^{-1}$)',
              'MN13 regime: ∂(CF)/∂(ω$_{700}$) for different low cloud definitions')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9, ncol=2)

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    print(f'\nSaved {FIG_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
