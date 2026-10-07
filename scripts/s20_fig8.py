# -*- coding: utf-8 -*-
"""
s20_fig8.py

Replicates Scott et al. (2020) Fig. 8: area-weighted mean low-cloud
radiative response coefficients ∂R/∂x_i (W m-2 per 1σ, model 1) in each
S20 regime for the six CCFs. For every CCF and regime the total
coefficient (◇) is flanked by its two Eq. 8 components: nonobscured
low-cloud amount (Δ) and low-cloud altitude + optical depth (○).

Coefficients come from s20_replication.py (clean_data/s20_regression*.nc);
regimes from regime_masks.py. Means are cos(latitude) weighted. 95%
intervals follow S20 Eq. A2,

    ± sqrt(Σ w_j² δ_j²) / Σ w_j × sqrt(N*_nom / N*_eff),

with δ_j the grid-box 95% half-width (Eq. A1), N*_nom the number of grid
boxes in the regime and N*_eff the effective number of spatial degrees of
freedom of the regime's anomaly field (Bretherton et al. 1999, Eq. 5).

Run from the CCF-ML root after s20_replication.py and regime_masks.py:
    python -m scripts.s20_fig8
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

from scripts.myers_norris_bins import compute_esdof_ratio
from scripts.regime_mn_slopes import align_masks


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

COEF_FILE = 'clean_data/s20_regression_merged.nc'   # or s20_regression.nc (2002-18)
DATA_FILE = 'clean_data/ccf_cre_clean.nc'           # anomalies for N*_eff
MASK_FILE = 'clean_data/regime_masks.nc'
FIG_PATH  = 'figures/s20_fig8.png'
CSV_PATH  = 'misc/s20_fig8_regime_means.csv'

PREDICTORS = ['sst', 'eis', 'Tadv', 'rh_700', 'w_700', 'speed']
PRED_LABELS = ['∂R/∂SST', '∂R/∂EIS', '∂R/∂Tadv', '∂R/∂RH$_{700}$',
               '∂R/∂ω$_{700}$', '∂R/∂WS']

REGIMES = {   # mask, label, colour (as S20 Fig. 3/8)
    'mask_s20_sc': ('Stratocumulus',   '#2f4858'),
    'mask_s20_cu': ('Trade Cumulus',   '#b5d99c'),
    'mask_s20_ta': ('Tropical Ascent', '#5e9e7e'),
    'mask_s20_ml': ('Mid-Latitude',    '#33b5e5'),
}

# S20 target in the coefficient file -> anomaly field in DATA_FILE, marker
COMPONENTS = {
    'R_total':  ('dCRE_net', 'D', 'Total'),
    'R_amount': ('dCRE_amt', '^', 'Nonobscured amount'),
    'R_shape':  ('dCRE_tau', 'o', 'Altitude + optical depth'),
}

# ═════════════════════════════════════════════


def regime_mean(beta, delta, region, n_eff):
    """S20 Eq. A2: cos-lat weighted mean and 95% half-width."""
    w = np.cos(np.deg2rad(beta['lat'])) * xr.ones_like(beta)
    ok = region & beta.notnull() & delta.notnull()
    w, b, d = w.where(ok), beta.where(ok), delta.where(ok)
    n_nom = int(ok.sum())
    mean = float((w * b).sum() / w.sum())
    ci = float(np.sqrt((w ** 2 * d ** 2).sum()) / w.sum()
               * np.sqrt(n_nom / n_eff))
    return mean, ci, n_nom


def spatial_n_eff(anom, region):
    """
    Bretherton et al. (1999) Eq. 5, (Σλ)² / Σλ², for the regime's grid
    boxes. compute_esdof_ratio forms the same eigenvalue ratio from the
    time x time matrix (identical nonzero eigenvalues) and divides by
    n_time, so multiply back.
    """
    da = anom.where(region)
    return compute_esdof_ratio(da) * da.sizes['time']


def main():
    coef = xr.open_dataset(COEF_FILE)
    ds   = xr.open_dataset(DATA_FILE)
    masks = align_masks(xr.open_dataset(MASK_FILE), ds)

    rows = []
    for mask, (label, _) in REGIMES.items():
        region = masks[mask].astype(bool) & ds['sst'].notnull().all('time')
        for comp, (field, _, _) in COMPONENTS.items():
            n_eff = spatial_n_eff(ds[field], region)
            for p in PREDICTORS:
                mean, ci, n_nom = regime_mean(
                    coef['beta'].sel(target=comp, predictor=p),
                    coef['delta'].sel(target=comp, predictor=p),
                    region, n_eff)
                rows.append(dict(regime=label, component=comp, predictor=p,
                                 mean=mean, ci95=ci, n_nom=n_nom,
                                 n_eff=round(n_eff, 1)))
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    table.to_csv(CSV_PATH, index=False)

    print('Area-weighted mean dR/dx (W m-2 per sigma) +/- 95% CI, total [amount, shape]')
    for label in table['regime'].unique():
        sub = table[table['regime'] == label]
        print(f"\n{label}: N*_nom = {sub['n_nom'].iloc[0]}, "
              f"N*_eff = {sub.groupby('component')['n_eff'].first().to_dict()}")
        for p in PREDICTORS:
            v = sub[sub['predictor'] == p].set_index('component')
            print(f"  {p:7s} {v.loc['R_total', 'mean']:+6.2f} ± "
                  f"{v.loc['R_total', 'ci95']:.2f}   "
                  f"[{v.loc['R_amount', 'mean']:+6.2f}, "
                  f"{v.loc['R_shape', 'mean']:+6.2f}]")

    # ── Figure: S20 Fig. 8 layout ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_reg = len(REGIMES)
    group_w = 0.8
    reg_off = np.linspace(-group_w / 2, group_w / 2, n_reg + 1)[:-1] + group_w / (2 * n_reg)
    comp_off = {'R_amount': -0.045, 'R_total': 0.0, 'R_shape': 0.045}

    for i, p in enumerate(PREDICTORS):
        for k, (mask, (label, color)) in enumerate(REGIMES.items()):
            for comp, (_, marker, _) in COMPONENTS.items():
                r = table[(table['regime'] == label) & (table['predictor'] == p)
                          & (table['component'] == comp)].iloc[0]
                x = i + reg_off[k] + comp_off[comp]
                total = comp == 'R_total'
                ax.errorbar(x, r['mean'], yerr=r['ci95'], fmt='none',
                            ecolor=color, elinewidth=2.0 if total else 1.0,
                            capsize=0, zorder=2)
                ax.plot(x, r['mean'], marker=marker, linestyle='none',
                        markersize=9 if total else 6,
                        markerfacecolor=color if total else 'white',
                        markeredgecolor=color, markeredgewidth=1.4, zorder=3)
        if i:
            ax.axvline(i - 0.5, color='k', lw=0.6, ls=':', alpha=0.6)

    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(PREDICTORS)))
    ax.set_xticklabels(PRED_LABELS, fontsize=13)
    ax.set_xlim(-0.5, len(PREDICTORS) - 0.5)
    ax.set_ylabel('W m$^{-2}$ σ$^{-1}$', fontsize=13)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', ls=':', alpha=0.6)

    reg_handles = [Line2D([], [], marker='D', ls='none', markersize=8,
                          markerfacecolor=c, markeredgecolor=c, label=l)
                   for l, c in REGIMES.values()]
    comp_handles = [Line2D([], [], marker=m, ls='none', markersize=8,
                           markerfacecolor='white' if c != 'R_total' else 'grey',
                           markeredgecolor='grey', label=lab)
                    for c, (_, m, lab) in COMPONENTS.items()]
    leg = ax.legend(handles=reg_handles, loc='upper right', fontsize=11,
                    framealpha=0.9)
    ax.add_artist(leg)
    ax.legend(handles=comp_handles, loc='lower right', fontsize=11,
              framealpha=0.9)

    period = ('2002-2025, CERES-FBCT Terra/Aqua + NOAA-20'
              if 'merged' in COEF_FILE else '2002-2018, CERES-FBCT Terra/Aqua')
    ax.set_title(f'Regime-mean low-cloud radiative response coefficients ({period})',
                 fontsize=13)

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    print(f'\nSaved {FIG_PATH} and {CSV_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
