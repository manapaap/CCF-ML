# -*- coding: utf-8 -*-
"""
subsidence_cre_pdp.py

Loads the cached final random forests trained by all_regions.py on the
three low cloud CRE targets (net, amount, tau + altitude) and plots the
subsidence PDP for each region as a 1x3 row of subplots:

    a) dCRE_net    b) dCRE_amt    c) dCRE_tau

This isolates the subsidence panel of each run's all_regions_pdp.png so
the amount and thickness contributions to the net CRE can be compared
side by side.

Requires all_regions.py to have been run for CRE_net, CRE_amt, and CRE_tau
(RF) so the final models exist in misc/hyperparams/{RUN_LABEL}_rf/.
"""

import os
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.inspection import partial_dependence
from string import ascii_lowercase as lowers

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

import scripts.utils as utils
from scripts.ccf_ml_shared import preprocess_region, xr_to_df


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

DATA_FILE  = 'clean_data/ccf_cre_clean.nc'
MODEL_TYPE = 'rf'
DROP_VARS  = ['u10', 'v10', 'msl']
TEMP_SPLIT = 0.8

# Must match the FEATURE_COLS (and order) the cached models were trained on
FEATURE_COLS = ['sst', 'eis', 'speed', 'Tadv',
                'w_700', 'ln_AOD', 'rh_700', 'cldarea_high']
PDP_VAR = 'w_700'

# (run label, target column, panel title)
RUNS = [
    ('CRE_net', 'dCRE_net', 'Net'),
    ('CRE_amt', 'dCRE_amt', 'Amount'),
    ('CRE_tau', 'dCRE_tau', 'Optical Depth + Altitude'),
]

SHARE_Y = True   # common y-axis makes net ≈ amount + tau easy to read
OUT_PATH = 'figures/subsidence_cre_pdp.png'

UNITS = {
    'w_700': 'Subsidence (Pa/s)',
}

# ═════════════════════════════════════════════


def load_test_set(ds_norm, temp_split=0.8):
    """
    Rebuilds the held-out test period exactly as load_or_fit_final does.

    Returns
    -------
    X_test : np.ndarray, normalized
    """
    n_split = int(temp_split * len(ds_norm.time))
    test    = xr_to_df(ds_norm.isel(time=slice(n_split, None)))
    return test[FEATURE_COLS].values


def load_model(run_label, region):
    path = os.path.join('misc/hyperparams', f'{run_label}_{MODEL_TYPE}',
                        f'{region.lower()}_{MODEL_TYPE}_final_model.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} not found. Run all_regions.py with '
            f"RUN_LABEL='{run_label}' and MODEL_TYPE='{MODEL_TYPE}' first.")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    if model.n_features_in_ != len(FEATURE_COLS):
        raise ValueError(
            f'{path} expects {model.n_features_in_} features but '
            f'FEATURE_COLS has {len(FEATURE_COLS)}.')
    return model


def pdp_curve(model, X_test, ccf_std, ccf_means, target_col,
              grid_resolution=50):
    """
    One-feature PDP for PDP_VAR, rescaled to physical units.

    Returns
    -------
    x_vals, y_vals : np.ndarray
    """
    i          = FEATURE_COLS.index(PDP_VAR)
    feat_std   = float(ccf_std[PDP_VAR].values)
    feat_mean  = float(ccf_means[PDP_VAR].values)
    target_std = float(ccf_std[target_col].values)

    pd_res = partial_dependence(model, X_test, features=[i],
                                kind='average',
                                grid_resolution=grid_resolution)
    x_vals = pd_res['grid_values'][0] * feat_std + feat_mean
    y_vals = pd_res['average'][0] * target_std
    return x_vals, y_vals


def plot_subsidence_cre_pdp(ccf_data, sc_regions, figsize=(15, 4.5)):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, len(RUNS), figsize=figsize, sharey=SHARE_Y)
    legend_handles = []

    # Preprocess each region once; normalization is shared across targets
    region_data = {region: preprocess_region(ccf_data, region_dict,
                                             drop_vars=DROP_VARS)
                   for region, region_dict in sc_regions.items()}

    for j, (ax, (run_label, target_col, panel_title)) in \
            enumerate(zip(axes, RUNS)):
        print(f'\n{run_label}:')
        for color, (region, (ds_norm, ccf_means, ccf_std)) in \
                zip(colors, region_data.items()):
            print(f'  {region}')
            model  = load_model(run_label, region)
            X_test = load_test_set(ds_norm, TEMP_SPLIT)

            x_vals, y_vals = pdp_curve(model, X_test, ccf_std,
                                       ccf_means, target_col)
            line, = ax.plot(x_vals, y_vals, color=color,
                            linewidth=2, label=region)
            if j == 0:
                legend_handles.append(line)

        ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_title(f'{lowers[j]}) {panel_title}', fontsize=14)
        ax.set_xlabel(UNITS.get(PDP_VAR, PDP_VAR), fontsize=13)
        if j == 0 or not SHARE_Y:
            ax.set_ylabel('Δ Low Cloud CRE (W/m²)', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc='lower center', ncol=len(legend_handles),
               fontsize=12, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    return fig, axes


def main():
    ccf_data   = xr.open_dataset(DATA_FILE)
    sc_regions = utils.get_stratocumulus_regions()

    fig, _ = plot_subsidence_cre_pdp(ccf_data, sc_regions)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
