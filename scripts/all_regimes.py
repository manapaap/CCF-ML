# -*- coding: utf-8 -*-
"""
all_regimes.py

Version of all_regions.py that trains one model per Scott et al. (2020)
cloud regime instead of per hand-drawn stratocumulus box:

    Sc : stratocumulus    (annual-mean ω700 > 15 hPa/day, EIS > 1 K)
    Cu : trade cumulus    (ω700 > 0, EIS < 1 K)
    TA : tropical ascent  (|lat| < 25°, ω700 < 0)
    ML : mid-latitude     (remaining 60°S-60°N ocean)

Masks come from clean_data/regime_masks.nc (regime_masks.py). Upper-level
(cirrus) cloud is not a predictor: the targets are already corrected for
obscuration by it, so it mostly enters through the overlap effect.

Produces, per run:
    1. PDP figure comparing the four regimes, with each regime's held-out
       test R² (random forest) in the legend
    2. Box-and-whisker CV metrics (RF vs linear) and a summary CSV
    3. Per-regime permutation variable importance

The regimes are much larger than the old boxes (up to ~1,850 grid boxes),
so hyperparameters are tuned on a random subset of TUNE_MAX_CELLS grid
boxes per regime (about the size of an old box), while CV metrics and the
final model use the full regime. PDPs and variable importance are computed
on a random subsample of the test set.

Run from the CCF-ML root, optionally naming runs (default: all four):
    python -m scripts.all_regimes [deseasonalized CRE_net CRE_amt CRE_tau]
Completed regimes are cached, so an interrupted run resumes.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

from scripts.all_regions import _make_model, UNITS
from scripts.ccf_ml_shared import (
    load_or_tune,
    load_or_run_cv,
    load_or_fit_final,
    plot_pdp,
    plot_varimp,
    plot_metrics_comparison,
)
from scripts.regime_mn_slopes import align_masks


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

DATA_FILE  = 'clean_data/ccf_cre_clean.nc'
MASK_FILE  = 'clean_data/regime_masks.nc'
MODEL_TYPE = 'rf'

# run label -> target column
RUNS = {
    'deseasonalized': 'cldarea_low_adj',
    'CRE_net':        'dCRE_net',
    'CRE_amt':        'dCRE_amt',
    'CRE_tau':        'dCRE_tau',
}

REGIMES = {   # short name -> (mask variable, legend label)
    'Sc': ('mask_s20_sc', 'Stratocumulus'),
    'Cu': ('mask_s20_cu', 'Trade Cumulus'),
    'TA': ('mask_s20_ta', 'Tropical Ascent'),
    'ML': ('mask_s20_ml', 'Mid-Latitude'),
}

FEATURE_COLS = ['sst', 'eis', 'speed', 'Tadv', 'w_700', 'ln_AOD', 'rh_700']
DROP_VARS    = ['u10', 'v10', 'msl']

TEMP_SPLIT  = 0.8
N_FOLDS     = 5
BLOCK_SIZE  = 5
N_ITER      = 30
INNER_FOLDS = 3

TUNE_MAX_CELLS = 150      # grid boxes used for hyperparameter tuning
PDP_MAX_ROWS   = 10_000   # test rows used for PDPs
VARIMP_ROWS    = 20_000   # test rows used for permutation importance
SEED           = 6767

# ═════════════════════════════════════════════


def preprocess_mask(ccf_data, mask, drop_vars=None):
    """
    As ccf_ml_shared.preprocess_region, but selecting grid boxes with a
    regime mask instead of a lat/lon box. Returns the normalized dataset
    and per-variable means/stds (Tadv means/stds converted to K/day).
    """
    ds = ccf_data.where(mask & ccf_data['sst'].notnull())
    keep = mask.any('lon') if 'time' not in mask.dims else mask.any(('time', 'lon'))
    ds = ds.sel(lat=ds['lat'][keep.values])     # drop all-empty latitude rows

    ds_means = ds.mean()
    ds_std   = ds.std()
    ds_norm  = (ds - ds_means) / ds_std

    ds_std['Tadv']   = ds_std['Tadv']   * 86400
    ds_means['Tadv'] = ds_means['Tadv'] * 86400

    if drop_vars:
        ds_norm = ds_norm.drop_vars([v for v in drop_vars if v in ds_norm])
    return ds_norm, ds_means, ds_std


def tuning_subset(ds_norm, rng, max_cells=TUNE_MAX_CELLS):
    """Random subset of at most max_cells grid boxes for tuning."""
    has_data = ds_norm['sst'].notnull().any('time')
    cells = np.argwhere(has_data.values)
    if len(cells) <= max_cells:
        return ds_norm
    pick = cells[rng.choice(len(cells), max_cells, replace=False)]
    sub = xr.zeros_like(has_data)
    sub.values[pick[:, 0], pick[:, 1]] = True
    return ds_norm.where(sub)


def subsample(rng, n_max, *arrays):
    n = len(arrays[0])
    if n <= n_max:
        return arrays
    idx = rng.choice(n, n_max, replace=False)
    return tuple(a[idx] for a in arrays)


def run(label, target_col, ccf_data, masks):
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    base_model, param_dist, supports_varimp = _make_model(MODEL_TYPE)
    param_dir = f'misc/hyperparams/regimes_{label}_{MODEL_TYPE}'
    fig_dir   = f'figures/regimes_{label}_{MODEL_TYPE}'
    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    pdp_models, fold_metrics_dict, rows = {}, {}, []

    for short, (mask_name, regime_label) in REGIMES.items():
        print(f'\n{"=" * 55}\n  Regime: {regime_label}  |  Run: {label}\n{"=" * 55}',
              flush=True)
        mask = masks[mask_name].astype(bool)
        ds_norm, ccf_means, ccf_std = preprocess_mask(ccf_data, mask, DROP_VARS)
        n_cells = int(ds_norm['sst'].notnull().any('time').sum())
        print(f'  {n_cells} grid boxes')

        # Hyperparameters tuned on a subset of grid boxes (cached)
        best_params, _ = load_or_tune(
            xr_ds=tuning_subset(ds_norm, rng), feature_cols=FEATURE_COLS,
            target_col=target_col, ccf_std=ccf_std,
            param_path=f'{param_dir}/{short.lower()}_{MODEL_TYPE}_params.pkl',
            param_distributions=param_dist, base_model=base_model,
            n_iter=N_ITER, n_folds=N_FOLDS, inner_folds=INNER_FOLDS,
            block_size=BLOCK_SIZE,
        )

        # CV metrics on the full regime (cached)
        ml_cv = load_or_run_cv(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS, target_col=target_col,
            ccf_std=ccf_std,
            metrics_path=f'{param_dir}/{short.lower()}_{MODEL_TYPE}_cv_metrics.csv',
            model=clone(base_model).set_params(**best_params),
            n_folds=N_FOLDS, block_size=BLOCK_SIZE,
        )
        lin_cv = load_or_run_cv(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS, target_col=target_col,
            ccf_std=ccf_std,
            metrics_path=f'{param_dir}/{short.lower()}_lin_cv_metrics.csv',
            model=LinearRegression(), n_folds=N_FOLDS, block_size=BLOCK_SIZE,
        )
        fold_metrics_dict[regime_label] = {
            MODEL_TYPE.upper(): ml_cv[['pearson_r', 'rmse']],
            'lin':              lin_cv[['pearson_r', 'rmse']],
        }

        # Final models on the full regime (cached)
        X_test, y_test, ml_model, n_train, n_test = load_or_fit_final(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS, target_col=target_col,
            best_params=best_params,
            model_path=f'{param_dir}/{short.lower()}_{MODEL_TYPE}_final_model.pkl',
            base_model=clone(base_model), temp_split=TEMP_SPLIT,
        )
        _, _, lin_model, _, _ = load_or_fit_final(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS, target_col=target_col,
            best_params={},
            model_path=f'{param_dir}/{short.lower()}_lin_final_model.pkl',
            base_model=LinearRegression(), temp_split=TEMP_SPLIT,
        )

        r2_ml  = r2_score(y_test, ml_model.predict(X_test))
        r2_lin = r2_score(y_test, lin_model.predict(X_test))
        print(f'  Held-out test R²: {MODEL_TYPE.upper()} {r2_ml:.3f} | '
              f'linear {r2_lin:.3f}')

        rows.append({
            'regime': regime_label, 'n_cells': n_cells,
            'test_r2_ml': r2_ml, 'test_r2_lin': r2_lin,
            'ml_r_median':  ml_cv['pearson_r'].median(),
            'lin_r_median': lin_cv['pearson_r'].median(),
            'ml_rmse_median':  ml_cv['rmse'].median(),
            'lin_rmse_median': lin_cv['rmse'].median(),
            'n_train': n_train, 'n_test': n_test,
        })

        X_pdp, = subsample(rng, PDP_MAX_ROWS, X_test)
        pdp_models[f'{regime_label} (R² = {r2_ml:.2f})'] = (
            ml_model, X_pdp, ccf_std, ccf_means)

        if supports_varimp:
            X_vi, y_vi = subsample(rng, VARIMP_ROWS, X_test, y_test)
            fig_vi, _, _ = plot_varimp(
                model=ml_model, X_test=X_vi, y_test=y_vi,
                feature_cols=FEATURE_COLS,
                title=f'Permutation Importance — {regime_label} ({label})')
            fig_vi.savefig(f'{fig_dir}/{short.lower()}_varimp.png',
                           dpi=150, bbox_inches='tight')
            plt.close(fig_vi)

    summary = pd.DataFrame(rows).set_index('regime')
    print('\nSummary:\n' + summary.round(3).to_string())
    summary.to_csv(f'misc/regime_summary_{label}.csv')

    fig_cmp, _ = plot_metrics_comparison(
        fold_metrics_dict,
        title=f'Model Performance ({MODEL_TYPE.upper()} vs Linear) — {label} '
              '(S20 regimes, distribution across CV folds)')
    fig_cmp.savefig(f'{fig_dir}/metrics_comparison.png', dpi=150,
                    bbox_inches='tight')
    plt.close(fig_cmp)

    fig_pdp, _ = plot_pdp(
        models=pdp_models, feature_cols=FEATURE_COLS, target_var=target_col,
        units=UNITS, title='', xlabel='', n_cols=4, figsize=(16, 6),
        show_ci_ticks=False,
    )
    fig_pdp.savefig(f'{fig_dir}/all_regimes_pdp.png', dpi=150,
                    bbox_inches='tight')
    plt.close(fig_pdp)
    print(f'Saved {fig_dir}/all_regimes_pdp.png')


def main(labels=None):
    ccf_data = xr.open_dataset(DATA_FILE)
    masks    = align_masks(xr.open_dataset(MASK_FILE), ccf_data)
    for label in labels or RUNS:
        run(label, RUNS[label], ccf_data, masks)


if __name__ == '__main__':
    main(sys.argv[1:])
