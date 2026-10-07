# -*- coding: utf-8 -*-
"""
mn13_fig4.py

Replicates Myers & Norris (2013, J. Climate 26, 7507-7524) Fig. 4 with
CERES and ERA5, using only the MN13 definition of the low cloud regime,
and adds a CRE panel:

    a) 2D histogram of mean low cloud fraction anomaly in bins of EIS and
       ω700 anomaly; box area proportional to the number of grid-box-months
    b) ∂CF/∂EIS in 8 hPa/day ω700 intervals (split at the median EIS)
    c) ∂CF/∂ω700 in 0.8 K EIS intervals (split at the median ω700)
    d) as (c) for low-cloud CRE: net, amount, and tau + altitude components

MN13 regime: ocean grid boxes in 30°S-30°N with climatological ω700 > 0 in
every calendar month and monthly ω700 > 0 in at least 80% of months; only
months with ω700 > 0 are used (regime_masks.py: mask_mn13, mn13_active).

Cloud fraction is cldarea_low_adj: CERES-FBCT nonobscured low cloud
L/(1 - U), with U all cloud above 680 hPa (S20; utils.fbct_low_high).
Error bars are 90% confidence intervals with the Bretherton (1999)
temporal ESDOF, as in MN13.

Run from the CCF-ML root after regime_masks.py and clean_data.py:
    python -m scripts.mn13_fig4
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

from scripts.myers_norris_bins import (
    compute_esdof_ratio, compute_slopes, PA_S_TO_HPA_DAY,
)
from scripts.regime_mn_slopes import align_masks, weighted_mean_slope


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

DATA_FILE = 'clean_data/ccf_cre_clean.nc'
MASK_FILE = 'clean_data/regime_masks.nc'
FIG_PATH  = 'figures/mn13_fig4.png'

CF_VAR   = 'cldarea_low_adj'
CRE_VARS = ['dCRE_net', 'dCRE_amt', 'dCRE_tau']
CRE_LABELS = {'dCRE_net': 'Net', 'dCRE_amt': 'Amount',
              'dCRE_tau': 'Tau + Altitude'}
CRE_COLORS = ['k', 'tab:blue', 'tab:red']

CONF = 0.90                                           # MN13 90% CI

EIS_EDGES_SLOPE = np.round(np.arange(-2.4, 2.41, 0.8), 1)   # panels c, d
W_EDGES_SLOPE   = np.arange(-32, 33, 8)                     # panel b
EIS_EDGES_HIST  = np.round(np.arange(-2.8, 2.81, 0.4), 1)   # panel a
W_EDGES_HIST    = np.arange(-32, 33, 4)
CF_LEVELS       = np.arange(-5, 6, 1)                       # panel a colors

# ═════════════════════════════════════════════


def extract_mn13(ds, masks):
    """
    Flatten the MN13 regime (active months only) to 1-D arrays.

    ESDOF ratios are computed on the static mask over all months, for EIS
    (used when differentiating by EIS) and ω700 (when differentiating by ω).

    Returns
    -------
    dict with 'eis', 'w' [hPa/day], 'cf', CRE arrays, 'esdof_eis',
    'esdof_w', 'n_cells'
    """
    region = masks['mask_mn13'].astype(bool) & ds['sst'].notnull()
    ds_m   = ds.where(region)
    esdof_eis = compute_esdof_ratio(ds_m['eis'])
    esdof_w   = compute_esdof_ratio(ds_m['w_700'])

    ds_a = ds_m.where(masks['mn13_active'].astype(bool))
    out = {
        'eis': ds_a['eis'].values.ravel().astype(float),
        'w':   ds_a['w_700'].values.ravel().astype(float) * PA_S_TO_HPA_DAY,
        'cf':  ds_a[CF_VAR].values.ravel().astype(float),
    }
    for v in CRE_VARS:
        out[v] = ds_a[v].values.ravel().astype(float)

    ok = np.all([np.isfinite(a) for a in out.values()], axis=0)
    out = {k: a[ok] for k, a in out.items()}
    out.update(esdof_eis=esdof_eis, esdof_w=esdof_w,
               n_cells=int(region.any('time').sum()))
    print(f'MN13 regime: {out["n_cells"]} grid boxes, n_obs={ok.sum():,}, '
          f'ESDOF ratio EIS={esdof_eis:.3f}, w700={esdof_w:.3f}')
    return out


def _in_bin(x, lo, hi, last):
    return (x >= lo) & ((x <= hi) if last else (x < hi))


def plot_hist2d(ax, d, cmap, norm):
    """MN13 Fig. 4a: mean CF anomaly per (EIS, ω) bin, box area ∝ count."""
    xe, ye = EIS_EDGES_HIST, W_EDGES_HIST
    nx, ny = len(xe) - 1, len(ye) - 1
    mean = np.full((ny, nx), np.nan)
    count = np.zeros((ny, nx), dtype=int)
    for i in range(nx):
        in_x = _in_bin(d['eis'], xe[i], xe[i + 1], i == nx - 1)
        for j in range(ny):
            cell = in_x & _in_bin(d['w'], ye[j], ye[j + 1], j == ny - 1)
            if cell.any():
                count[j, i] = cell.sum()
                mean[j, i] = d['cf'][cell].mean()

    dx, dy = xe[1] - xe[0], ye[1] - ye[0]
    for i in range(nx):
        for j in range(ny):
            if count[j, i] == 0:
                continue
            f = np.sqrt(count[j, i] / count.max())
            xc, yc = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])
            ax.add_patch(mpatches.Rectangle(
                (xc - f * dx / 2, yc - f * dy / 2), f * dx, f * dy,
                facecolor=cmap(norm(mean[j, i])), edgecolor='none'))

    # Median ω in each EIS interval and median EIS in each ω interval (the
    # splits used in panels c and b), drawn as connected staircases
    for edges, xs, ys, horiz in ((EIS_EDGES_SLOPE, d['eis'], d['w'], True),
                                 (W_EDGES_SLOPE, d['w'], d['eis'], False)):
        meds = []
        for k in range(len(edges) - 1):
            sel = _in_bin(xs, edges[k], edges[k + 1], k == len(edges) - 2)
            meds.append(np.median(ys[sel]) if sel.sum() >= 10 else np.nan)
        along  = np.repeat(edges, 2)[1:-1]        # e0, e1, e1, e2, ..., eK
        across = np.repeat(meds, 2)               # m0, m0, m1, m1, ...
        if horiz:
            ax.plot(along, across, 'k', lw=1.2)
        else:
            ax.plot(across, along, 'k', lw=1.2)

    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xticks(EIS_EDGES_SLOPE)
    ax.set_yticks(W_EDGES_SLOPE)
    ax.set_xlabel('EIS anomaly (K)', fontsize=12)
    ax.set_ylabel('ω$_{700}$ anomaly (hPa day$^{-1}$)', fontsize=12)
    ax.set_title('a) CERES-FBCT nonobscured low cloud (%)', fontsize=13)
    ax.grid(True, ls=':', alpha=0.6)


def _sizes(n):
    n = n.astype(float)
    return 20 + 280 * (n / n.max())


def plot_dcf_deis(ax, d):
    """MN13 Fig. 4b: ∂CF/∂EIS within ω intervals, plotted against ω."""
    centers, slopes, errors, n, _ = compute_slopes(
        d['w'], d['eis'], d['cf'], esdof_ratio=d['esdof_eis'],
        var_del='eis', bin_edges=W_EDGES_SLOPE, conf=CONF)
    ok = np.isfinite(slopes)
    ax.errorbar(slopes[ok], centers[ok], xerr=errors[ok], fmt='none',
                ecolor='k', elinewidth=1, capsize=3)
    ax.scatter(slopes[ok], centers[ok], s=_sizes(n[ok]), c='k', zorder=3)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylim(W_EDGES_SLOPE[0], W_EDGES_SLOPE[-1])
    ax.set_yticks(W_EDGES_SLOPE)
    ax.set_xlabel('∂(CF)/∂(EIS) (% K$^{-1}$)', fontsize=12)
    ax.set_ylabel('ω$_{700}$ anomaly (hPa day$^{-1}$)', fontsize=12)
    ax.set_title('b)', fontsize=13, loc='left')
    ax.grid(True, ls=':', alpha=0.6)
    mean, ci = weighted_mean_slope(slopes, errors, n)
    print(f'  d(CF)/d(EIS): {mean:+.2f} +/- {ci:.2f} %/K')


def plot_d_dw(ax, d, targets, colors, labels, ylabel, title):
    """MN13 Fig. 4c style: ∂/∂ω700 within EIS intervals, offset by target."""
    width   = EIS_EDGES_SLOPE[1] - EIS_EDGES_SLOPE[0]
    offsets = (np.linspace(-0.15, 0.15, len(targets)) * width
               if len(targets) > 1 else [0.0])
    for k, (tv, color, off) in enumerate(zip(targets, colors, offsets)):
        centers, slopes, errors, n, _ = compute_slopes(
            d['eis'], d['w'], d[tv], esdof_ratio=d['esdof_w'],
            var_del='w_700', bin_edges=EIS_EDGES_SLOPE, conf=CONF)
        ok = np.isfinite(slopes)
        ax.errorbar(centers[ok] + off, slopes[ok], yerr=errors[ok],
                    fmt='none', ecolor=color, elinewidth=1, capsize=3)
        ax.scatter(centers[ok] + off, slopes[ok], s=_sizes(n[ok]), c=color,
                   zorder=3, label=labels[k])
        mean, ci = weighted_mean_slope(slopes, errors, n)
        print(f'  d({tv})/d(w700): {mean:+.2f} +/- {ci:.2f} per 10 hPa/day')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlim(EIS_EDGES_SLOPE[0], EIS_EDGES_SLOPE[-1])
    ax.set_xticks(EIS_EDGES_SLOPE)
    ax.set_xlabel('EIS anomaly (K)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, loc='left')
    ax.grid(True, ls=':', alpha=0.6)


def main():
    ds    = xr.open_dataset(DATA_FILE)
    masks = align_masks(xr.open_dataset(MASK_FILE), ds)
    d     = extract_mn13(ds, masks)

    cmap = plt.get_cmap('RdBu_r', len(CF_LEVELS) + 1)   # +2 for the extensions
    norm = BoundaryNorm(CF_LEVELS, cmap.N, extend='both')

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    plot_hist2d(axes[0, 0], d, cmap, norm)
    plot_dcf_deis(axes[0, 1], d)
    plot_d_dw(axes[1, 0], d, ['cf'], ['k'], ['CF'],
              '∂(CF)/∂(ω$_{700}$) (% (10 hPa day$^{-1}$)$^{-1}$)', 'c)')
    plot_d_dw(axes[1, 1], d, CRE_VARS, CRE_COLORS,
              [CRE_LABELS[v] for v in CRE_VARS],
              '∂(CRE)/∂(ω$_{700}$) (W m$^{-2}$ (10 hPa day$^{-1}$)$^{-1}$)',
              'd) Low-cloud CRE (positive = energy gain)')
    axes[1, 1].legend(loc='lower right', fontsize=10, framealpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    pos = axes[0, 0].get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.085, pos.width, 0.015])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal',
                        ticks=CF_LEVELS[::2])
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle('MN13 Fig. 4 with CERES: MN13 subsidence regime, '
                 f'{d["n_cells"]} grid boxes, ω$_{{700}}$ > 0 months',
                 fontsize=14, y=0.95)

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    print(f'Saved {FIG_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
