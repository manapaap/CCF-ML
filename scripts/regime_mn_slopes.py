# -*- coding: utf-8 -*-
"""
regime_mn_slopes.py

Myers & Norris (2013) style finite-difference partial derivatives
∂(cloud)/∂(ω700)|EIS computed over the regime masks from regime_masks.py
rather than the five hand-drawn stratocumulus boxes:

    a) MN13 subsidence regime (30°S-30°N, only months with ω700 > 0)
    b) S20 stratocumulus      (ω700 > 15 hPa/day, EIS > 1 K)
    c) S20 trade cumulus      (ω700 > 0, EIS < 1 K)

Anomalies (deseasonalized, detrended) come from clean_data/ccf_cre_clean.nc.
Within each EIS-anomaly interval, grid-box-months are split at the median
ω700 anomaly and the slope is the difference in mean cloud property over
the difference in mean ω700 (compute_slopes in myers_norris_bins.py).
Error bars use the Bretherton (1999) temporal ESDOF, as before.

Choices matched to MN13 (Fig. 4c and Table 2):
    - fixed 0.8 K EIS-anomaly intervals from -2.4 to 2.4 K
    - 90% confidence intervals
    - no percentile clipping of the CCFs
    - an overall ∂/∂ω700|EIS averaged over intervals weighted by count

Produces two figures (1 x 3, one panel per mask):
    figures/regime_mn_cf.png   : ∂(low cloud fraction)/∂(ω700)
    figures/regime_mn_cre.png  : ∂(CRE)/∂(ω700) for net, amount, tau+alt
and a CSV of the per-interval slopes in misc/.

Run from the CCF-ML root after regime_masks.py:
    python -m scripts.regime_mn_slopes
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from string import ascii_lowercase as lowers

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

from scripts.myers_norris_bins import (
    compute_esdof_ratio,
    compute_slopes,
    PA_S_TO_HPA_DAY,
    UNITS,
)


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

DATA_FILE = 'clean_data/ccf_cre_clean.nc'
MASK_FILE = 'clean_data/regime_masks.nc'

VAR_BIN  = 'eis'
VAR_DEL  = 'w_700'
CF_VARS  = ['cldarea_low_adj']
CRE_VARS = ['dCRE_net', 'dCRE_amt', 'dCRE_tau']

BIN_EDGES = np.round(np.arange(-2.4, 2.41, 0.8), 1)   # MN13 Fig. 4c
CONF      = 0.90                                       # MN13 90% CI

# (mask variable, panel title, restrict to months with ω700 > 0)
MASKS = [
    ('mask_mn13',   'MN13 Subsidence Regime', True),
    ('mask_s20_sc', 'S20 Stratocumulus',      False),
    ('mask_s20_cu', 'S20 Trade Cumulus',      False),
]

CRE_COLORS = ['k', 'tab:blue', 'tab:red']

FIG_DIR = 'figures'
CSV_DIR = 'misc'

# ═════════════════════════════════════════════


def align_masks(masks, ds):
    """Select the mask file's months that match the anomaly file."""
    months = ds['time'].dt.strftime('%Y-%m').values
    masks  = masks.assign_coords(
        time=masks['time'].dt.strftime('%Y-%m').values)
    missing = set(months) - set(masks['time'].values)
    if missing:
        raise ValueError(f'{len(missing)} months missing from {MASK_FILE}, '
                         f'e.g. {sorted(missing)[:3]}')
    return masks.sel(time=months).assign_coords(time=ds['time'])


def extract_mask_flat(ds, masks, mask_name, active_only, target_vars):
    """
    Apply a regime mask and flatten (time, lat, lon) -> (n_obs,).

    The ESDOF ratio is computed on the static mask (all months) so that
    compute_esdof_ratio keeps every masked cell; the MN13 active-month
    filter is applied afterwards.

    Returns
    -------
    entry : dict {'arr_bin', 'arr_del', 'esdof_ratio', 'targets', 'n_cells'}
    """
    region = masks[mask_name].astype(bool) & ds['sst'].notnull()
    ds_m   = ds.where(region)

    esdof_ratio = compute_esdof_ratio(ds_m[VAR_DEL])

    if active_only:
        ds_m = ds_m.where(masks['mn13_active'].astype(bool))

    arr_bin = ds_m[VAR_BIN].values.ravel().astype(np.float64)
    arr_del = ds_m[VAR_DEL].values.ravel().astype(np.float64) * PA_S_TO_HPA_DAY
    targets = {tv: ds_m[tv].values.ravel().astype(np.float64)
               for tv in target_vars}

    finite = np.isfinite(arr_bin) & np.isfinite(arr_del)
    for arr in targets.values():
        finite &= np.isfinite(arr)

    n_cells = int(region.any('time').sum())
    print(f'  {mask_name}: {n_cells} grid boxes, n_obs={finite.sum():,}, '
          f'esdof_ratio={esdof_ratio:.3f}')

    return {
        'arr_bin':     arr_bin[finite],
        'arr_del':     arr_del[finite],
        'esdof_ratio': esdof_ratio,
        'targets':     {tv: arr[finite] for tv, arr in targets.items()},
        'n_cells':     n_cells,
    }


def weighted_mean_slope(slopes, errors, n_obs):
    """
    MN13's overall ∂/∂ω700|EIS: interval slopes averaged with weights
    proportional to count. The CI treats intervals as independent.
    """
    ok = np.isfinite(slopes) & np.isfinite(errors)
    w  = n_obs[ok] / n_obs[ok].sum()
    return (float(np.sum(w * slopes[ok])),
            float(np.sqrt(np.sum((w * errors[ok]) ** 2))))


def slope_table(entry, target_vars, mask_name):
    """Per-interval slopes for all target_vars as a DataFrame."""
    rows = []
    for tv in target_vars:
        centers, slopes, errors, n_obs, _ = compute_slopes(
            entry['arr_bin'], entry['arr_del'], entry['targets'][tv],
            esdof_ratio=entry['esdof_ratio'], var_del=VAR_DEL,
            bin_edges=BIN_EDGES, conf=CONF,
        )
        for i in range(len(BIN_EDGES) - 1):
            rows.append({
                'mask': mask_name, 'target': tv,
                'eis_lo': BIN_EDGES[i], 'eis_hi': BIN_EDGES[i + 1],
                'eis_median': centers[i], 'slope': slopes[i],
                'ci_halfwidth': errors[i], 'n_obs': n_obs[i],
            })
        mean, ci = weighted_mean_slope(slopes, errors, n_obs)
        rows.append({'mask': mask_name, 'target': tv,
                     'eis_lo': BIN_EDGES[0], 'eis_hi': BIN_EDGES[-1],
                     'eis_median': np.nan, 'slope': mean,
                     'ci_halfwidth': ci, 'n_obs': n_obs.sum()})
    return pd.DataFrame(rows)


def plot_panel(ax, entry, target_vars, colors, title):
    """
    MN13 Fig. 4c style panel: one point per EIS interval, marker area
    scaled by count, CI error bars. Multiple target_vars are offset
    horizontally within each interval.
    """
    bin_width = BIN_EDGES[1] - BIN_EDGES[0]
    offsets   = (np.linspace(-0.15, 0.15, len(target_vars)) * bin_width
                 if len(target_vars) > 1 else [0.0])

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    for edge in BIN_EDGES:
        ax.axvline(edge, color='k', lw=0.4, ls=':', alpha=0.4)

    summary = []
    for tv, color, offset in zip(target_vars, colors, offsets):
        centers, slopes, errors, n_obs, _ = compute_slopes(
            entry['arr_bin'], entry['arr_del'], entry['targets'][tv],
            esdof_ratio=entry['esdof_ratio'], var_del=VAR_DEL,
            bin_edges=BIN_EDGES, conf=CONF,
        )
        valid = np.isfinite(slopes) & np.isfinite(centers)
        n_valid = n_obs[valid].astype(float)
        sizes = 30 + 150 * np.sqrt(n_valid / n_valid.max())

        ax.errorbar(centers[valid] + offset, slopes[valid],
                    yerr=errors[valid], fmt='none', ecolor=color,
                    elinewidth=1.2, capsize=3, alpha=0.8, zorder=2)
        ax.scatter(centers[valid] + offset, slopes[valid], s=sizes,
                   color=color if len(target_vars) > 1 else 'grey',
                   edgecolors='k', linewidths=0.5, zorder=3,
                   label=UNITS.get(tv, tv))

        mean, ci = weighted_mean_slope(slopes, errors, n_obs)
        summary.append((color, f'{mean:+.2f} ± {ci:.2f}'))

    # Overall weighted-mean partial derivative, one line per target
    for k, (color, text) in enumerate(summary):
        ax.text(0.03, 0.96 - 0.07 * k, text, transform=ax.transAxes,
                color=color, fontsize=11, va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                          pad=1))

    ax.set_title(f'{title}\n({entry["n_cells"]} grid boxes, '
                 f'ESDOF ratio {entry["esdof_ratio"]:.2f})', fontsize=12)
    ax.set_xlabel('EIS anomaly (K)', fontsize=13)
    ax.set_xlim(BIN_EDGES[0], BIN_EDGES[-1])
    ax.set_xticks(BIN_EDGES)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis='y', alpha=0.3, lw=0.5, ls=':')


def plot_regime_slopes(entries, target_vars, ylabel, colors=None,
                       figsize=(16, 5)):
    colors = colors or ['k'] * len(target_vars)
    fig, axes = plt.subplots(1, len(MASKS), figsize=figsize, sharey=True)

    for j, (ax, (mask_name, title, _)) in enumerate(zip(axes, MASKS)):
        plot_panel(ax, entries[mask_name], target_vars, colors,
                   f'{lowers[j]}) {title}')
    axes[0].set_ylabel(ylabel, fontsize=13)

    if len(target_vars) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(target_vars), fontsize=12, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    return fig, axes


def main():
    ds    = xr.open_dataset(DATA_FILE)
    masks = align_masks(xr.open_dataset(MASK_FILE), ds)

    if not (np.allclose(ds['lat'], masks['lat'])
            and np.allclose(ds['lon'], masks['lon'])):
        raise ValueError(f'{MASK_FILE} is not on the grid of {DATA_FILE}')

    target_vars = CF_VARS + CRE_VARS
    print('Extracting masked anomalies:')
    entries = {name: extract_mask_flat(ds, masks, name, active, target_vars)
               for name, _, active in MASKS}

    table = pd.concat([slope_table(entries[name], target_vars, name)
                       for name, _, _ in MASKS], ignore_index=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    csv_path = f'{CSV_DIR}/regime_mn_slopes.csv'
    table.to_csv(csv_path, index=False)
    print(f'\nWeighted-mean d/d(w700)|EIS (per 10 hPa/day, {CONF:.0%} CI):')
    overall = table[table['eis_median'].isna()]
    print(overall[['mask', 'target', 'slope', 'ci_halfwidth', 'n_obs']]
          .round(3).to_string(index=False))
    print(f'Saved {csv_path}')

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_cf, _ = plot_regime_slopes(
        entries, CF_VARS,
        ylabel='∂(Low CF)/∂(ω$_{700}$)  (% / 10 hPa day$^{-1}$)',
    )
    fig_cf.savefig(f'{FIG_DIR}/regime_mn_cf.png', dpi=150,
                   bbox_inches='tight')

    fig_cre, _ = plot_regime_slopes(
        entries, CRE_VARS, colors=CRE_COLORS,
        ylabel='∂(CRE)/∂(ω$_{700}$)  (W m$^{-2}$ / 10 hPa day$^{-1}$)',
    )
    fig_cre.savefig(f'{FIG_DIR}/regime_mn_cre.png', dpi=150,
                    bbox_inches='tight')
    print(f'Saved {FIG_DIR}/regime_mn_cf.png and {FIG_DIR}/regime_mn_cre.png')
    plt.show()


if __name__ == '__main__':
    main()
