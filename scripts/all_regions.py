# -*- coding: utf-8 -*-
"""
all_regions.py

Multi-region CCF ML workflow. Loops over all five canonical
stratocumulus regions (NEP, SEP, NEA, SEA, SEI) and produces:

    1. A combined PDP figure comparing all regions (with 95% CI ticks)
    2. A two-panel box-and-whisker metrics comparison figure
       (Pearson R and RMSE, model vs linear, distributions across CV folds)
    3. A summary CSV with per-fold median R and RMSE per region
    4. Per-region permutation variable importance figures (RF/GBT only)
    5. A 3×2 multi-region variable importance comparison figure (RF/GBT only)
    6. Per-region CCF histogram figures
    7. 2D PDP figures for selected variable pairs (raw and interaction-only)

Switch between runs by editing the CONFIG block below.
MODEL_TYPE selects the non-linear estimator: 'rf', 'gbt', or 'mlp'.
Hyperparameter caches are stored in separate subdirectories per run.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import randint, uniform, loguniform

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

import scripts.utils as utils
from scripts.ccf_ml_shared import (
    preprocess_region,
    run_spatial_temporal_cv,
    load_or_tune,
    load_or_run_cv,
    load_or_fit_final,
    plot_pdp,
    plot_pdp_2d,
    plot_varimp,
    plot_varimp_all_regions,
    plot_ccf_histograms,
    plot_metrics_comparison,
)

# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

# ── Input data ────────────────────────────────
# DATA_FILE  = 'clean_data/ccf_clouds_clean.nc'
# RUN_LABEL  = 'deseasonalized'

# Low Cloud CRE Alternative:
DATA_FILE  = 'clean_data/ccf_cre_clean.nc'
RUN_LABEL  = 'mcao'

# ── Model type ────────────────────────────────
# 'rf'  — Random Forest (supports varimp)
# 'gbt' — Gradient Boosted Trees (supports varimp)
# 'mlp' — Multi-layer Perceptron / Neural Network (varimp skipped)
MODEL_TYPE = 'rf'

# ── Hyperparameter cache ──────────────────────
# Caches are keyed by both run label and model type so they never collide.
PARAM_SUBDIR = f'misc/hyperparams/{RUN_LABEL}_{MODEL_TYPE}'

# ── Model / CV settings ───────────────────────
# cldarea_low_adj
TARGET_COL  = 'cldarea_low_adj'
DROP_VARS   = ['u10', 'v10', 'msl']
TEMP_SPLIT  = 0.8
N_FOLDS     = 5
BLOCK_SIZE  = 5
N_ITER      = 30
INNER_FOLDS = 3

FEATURE_COLS = ['sst', 'mcao', 'speed', 'Tadv',
                'w_700', 'ln_AOD', 'rh_700', 'cldarea_high']

# ── 2D PDP pairs to generate ─────────────────
# Each entry is (var1, var2).  Two figures per pair: raw and interaction-only.
PDP_2D_PAIRS = [
    ('sst', 'eis'),
    ('sst', 'cold_adv'),
]
# ── Plot labels ───────────────────────────────
UNITS = {
    'sst':             'SST (°C)',
    'eis':             'EIS (K)',
    'speed':           '10m Windspeed (m/s)',
    'Tadv':        'Temperature Advection (K/day)',
    'w_700':           'Subsidence (Pa/s)',
    'ln_AOD':          'ln(AOD)',
    'rh_700':          '700 hPa RH (%)',
    'cldarea_high':    'Cirrus Cover (%)',
    'dCRE_net':        'Low Cloud CRE- Net (W/m²)',
    'dCRE_amt':        'Low Cloud CRE- Amount (W/m²)',
    'dCRE_tau':        'Low Cloud CRE- Tau (W/m²)',
    'cldarea_low_adj': 'Low Cloud Cover (%)',
}

# ── Per-model hyperparameter search spaces ────
# Edit these to adjust the search range for each model type.
_RF_PARAMS = {
    'n_estimators':     randint(100, 500),
    'max_depth':        [5, 10, 20, None],
    'min_samples_leaf': randint(5, 50),
    'max_features':     uniform(0.2, 0.6),
}

_GBT_PARAMS = {
    'n_estimators':   randint(100, 500),
    'max_depth':      randint(3, 8),
    'learning_rate':  loguniform(0.01, 0.3),
    'subsample':      uniform(0.6, 0.4),       # 0.6–1.0
    'min_samples_leaf': randint(5, 50),
}

_MLP_PARAMS = {
    # Hidden layer sizes: try a range of single and double hidden layer configs
    'hidden_layer_sizes': [(64,), (128,), (256,), (64, 32), (128, 64),
                           (256, 128), (128, 64, 32)],
    'alpha':        loguniform(1e-4, 1e-1),    # L2 regularisation
    'learning_rate_init': loguniform(1e-4, 1e-2),
}

# ── Model factory ─────────────────────────────
# Returns (base_estimator, param_distributions, supports_varimp)
def _make_model(model_type):
    if model_type == 'rf':
        return (RandomForestRegressor(n_jobs=-1),
                _RF_PARAMS, True)
    elif model_type == 'gbt':
        return (GradientBoostingRegressor(),
                _GBT_PARAMS, True)
    elif model_type == 'mlp':
        return (MLPRegressor(max_iter=500, early_stopping=True,
                             validation_fraction=0.1, random_state=42),
                _MLP_PARAMS, False)
    else:
        raise ValueError(f"Unknown MODEL_TYPE '{model_type}'. "
                         f"Choose from: 'rf', 'gbt', 'mlp'.")

# ═════════════════════════════════════════════


def main():
    global all_models, fold_metrics_dict, summary_df

    np.random.seed(6767)

    base_model, param_distributions, supports_varimp = _make_model(MODEL_TYPE)
    model_label = MODEL_TYPE.upper()
    fig_dir = f'figures/{RUN_LABEL}_{MODEL_TYPE}'
    os.makedirs(fig_dir,      exist_ok=True)
    os.makedirs(PARAM_SUBDIR, exist_ok=True)

    ccf_data   = xr.open_dataset(DATA_FILE)
    sc_regions = utils.get_stratocumulus_regions()

    # all_models feeds the PDP and 2D-PDP plots
    # {region: (model, X_test, ccf_std, ccf_means)}
    all_models     = {}
    all_lin_models = {}

    # For the multi-region varimp grid (tree models only)
    # {region: model} and {region: (X_test, y_test)}
    varimp_models  = {}
    varimp_testset = {}

    # fold_metrics_dict feeds the boxplot
    fold_metrics_dict = {}

    summary_rows = []

    for region, region_dict in sc_regions.items():
        print(f'\n{"="*50}')
        print(f'  Region: {region}  |  Run: {RUN_LABEL}')
        print(f'{"="*50}')

        # ── Preprocess ────────────────────────
        ds_norm, ccf_means, ccf_std = preprocess_region(
            ccf_data, region_dict, drop_vars=DROP_VARS
        )

        # ── Linear regression CV ──────────────
        print('\nLinear Regression:')
        lin_cv_results, _ = run_spatial_temporal_cv(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS,
            target_col=TARGET_COL, ccf_std=ccf_std,
            model=LinearRegression(),
            n_folds=N_FOLDS, block_size=BLOCK_SIZE,
        )

        # ── Hyperparameter tuning (cached) ────
        param_path = os.path.join(
            PARAM_SUBDIR, f'{region.lower()}_{MODEL_TYPE}_params.pkl')
        best_params, _ = load_or_tune(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS,
            target_col=TARGET_COL, ccf_std=ccf_std,
            param_path=param_path,
            param_distributions=param_distributions,
            base_model=base_model,
            n_iter=N_ITER, n_folds=N_FOLDS,
            inner_folds=INNER_FOLDS, block_size=BLOCK_SIZE,
        )

        # ── CV metrics (cached per model type and linear) ─────────────────
        from sklearn.base import clone
        ml_metrics_path  = os.path.join(
            PARAM_SUBDIR, f'{region.lower()}_{MODEL_TYPE}_cv_metrics.csv')
        lin_metrics_path = os.path.join(
            PARAM_SUBDIR, f'{region.lower()}_lin_cv_metrics.csv')

        ml_cv_results  = load_or_run_cv(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS,
            target_col=TARGET_COL, ccf_std=ccf_std,
            metrics_path=ml_metrics_path,
            model=clone(base_model).set_params(**best_params),
            n_folds=N_FOLDS, block_size=BLOCK_SIZE,
        )
        lin_cv_results = load_or_run_cv(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS,
            target_col=TARGET_COL, ccf_std=ccf_std,
            metrics_path=lin_metrics_path,
            model=LinearRegression(),
            n_folds=N_FOLDS, block_size=BLOCK_SIZE,
        )

        fold_metrics_dict[region] = {
            model_label: ml_cv_results[['pearson_r', 'rmse']],
            'lin':       lin_cv_results[['pearson_r', 'rmse']],
        }

        # ── Fit final models (cached) ─────────────────────────────────────
        ml_model_path  = os.path.join(
            PARAM_SUBDIR, f'{region.lower()}_{MODEL_TYPE}_final_model.pkl')
        lin_model_path = os.path.join(
            PARAM_SUBDIR, f'{region.lower()}_lin_final_model.pkl')

        X_test, y_test, ml_model, n_train, n_test = load_or_fit_final(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS,
            target_col=TARGET_COL, best_params=best_params,
            model_path=ml_model_path,
            base_model=clone(base_model),
            temp_split=TEMP_SPLIT,
        )
        _, _, lin_model, _, _ = load_or_fit_final(
            xr_ds=ds_norm, feature_cols=FEATURE_COLS,
            target_col=TARGET_COL, best_params={},
            model_path=lin_model_path,
            base_model=LinearRegression(),
            temp_split=TEMP_SPLIT,
        )

        # ── Summary row ───────────────────────
        summary_rows.append({
            'region':          region,
            'ml_r_median':     ml_cv_results['pearson_r'].median(),
            'ml_r_iqr':        ml_cv_results['pearson_r'].quantile(0.75)
                               - ml_cv_results['pearson_r'].quantile(0.25),
            'ml_rmse_median':  ml_cv_results['rmse'].median(),
            'ml_rmse_iqr':     ml_cv_results['rmse'].quantile(0.75)
                               - ml_cv_results['rmse'].quantile(0.25),
            'lin_r_median':    lin_cv_results['pearson_r'].median(),
            'lin_r_iqr':       lin_cv_results['pearson_r'].quantile(0.75)
                               - lin_cv_results['pearson_r'].quantile(0.25),
            'lin_rmse_median': lin_cv_results['rmse'].median(),
            'lin_rmse_iqr':    lin_cv_results['rmse'].quantile(0.75)
                               - lin_cv_results['rmse'].quantile(0.25),
            'n_train':         n_train,
            'n_test':          n_test,
        })

        # ── Per-region variable importance (tree models only) ─────────────
        if supports_varimp:
            fig_vi, _, _ = plot_varimp(
                model=ml_model, X_test=X_test, y_test=y_test,
                feature_cols=FEATURE_COLS,
                title=f'Permutation Importance — {region} ({RUN_LABEL})',
            )
            fig_vi.savefig(f'{fig_dir}/{region.lower()}_varimp.png',
                           dpi=150, bbox_inches='tight')
            plt.close(fig_vi)

        # ── Per-region CCF histograms ─────────
        fig_hist, _ = plot_ccf_histograms(
            X_test=X_test, feature_cols=FEATURE_COLS,
            ccf_std=ccf_std, ccf_means=ccf_means,
            region_label=region, units=UNITS,
        )
        fig_hist.savefig(f'{fig_dir}/{region.lower()}_histograms.png',
                         dpi=150, bbox_inches='tight')
        plt.close(fig_hist)

        # ── Collect for multi-region plots ────
        all_models[region]     = (ml_model,  X_test, ccf_std, ccf_means)
        all_lin_models[region] = (lin_model, X_test, ccf_std, ccf_means)
        if supports_varimp:
            varimp_models[region]  = ml_model
            varimp_testset[region] = (X_test, y_test)

    # ─────────────────────────────────────────
    #  Summary CSV
    # ─────────────────────────────────────────

    summary_df = pd.DataFrame(summary_rows).set_index('region')
    print('\n\nSummary — all regions (median across CV folds):')
    print(summary_df.round(3).to_string())

    os.makedirs('misc', exist_ok=True)
    csv_path = f'misc/region_summary_{RUN_LABEL}.csv'
    summary_df.to_csv(csv_path)
    print(f'\nSaved to {csv_path}')

    # ─────────────────────────────────────────
    #  Metrics comparison boxplot
    # ─────────────────────────────────────────

    fig_cmp, _ = plot_metrics_comparison(
        fold_metrics_dict,
        title=f'Model Performance ({model_label} vs Linear) — '
              f'{RUN_LABEL.title()} (Distribution across CV Folds)',
    )
    fig_cmp.savefig(f'{fig_dir}/metrics_comparison.png',
                    dpi=150, bbox_inches='tight')

    # ─────────────────────────────────────────
    #  Multi-region variable importance grid
    #  (skipped for MLP)
    # ─────────────────────────────────────────

    if supports_varimp and varimp_models:
        fig_vi_all, _, _ = plot_varimp_all_regions(
            models_dict=varimp_models,
            y_test_dict=varimp_testset,
            feature_cols=FEATURE_COLS,
            units=UNITS,
            title=f'Permutation Importance — All Regions '
                  f'({model_label}, {RUN_LABEL})',
        )
        fig_vi_all.savefig(f'{fig_dir}/all_regions_varimp.png',
                           dpi=150, bbox_inches='tight')

    # ─────────────────────────────────────────
    #  Cross-region PDP  (with 95% CI ticks)
    # ─────────────────────────────────────────

    fig_pdp, _ = plot_pdp(
        models=all_models,
        feature_cols=FEATURE_COLS,
        target_var=TARGET_COL,
        units=UNITS,
        title='',
        xlabel='',
        n_cols=4, figsize=(16, 6),
        show_ci_ticks=False,
    )
    fig_pdp.savefig(f'{fig_dir}/all_regions_pdp.png',
                    dpi=150, bbox_inches='tight')

    fig_pdp_lin, _ = plot_pdp(
        models=all_lin_models,
        feature_cols=FEATURE_COLS,
        target_var=TARGET_COL,
        units=UNITS,
        title='',
        xlabel='',
        n_cols=4, figsize=(16, 6),
        show_ci_ticks=False,
        center_curves=True,
    )
    fig_pdp_lin.savefig(f'{fig_dir}/all_regions_pdp_linear.png',
                        dpi=150, bbox_inches='tight')

    # ─────────────────────────────────────────
    #  2D PDP figures (raw + interaction-only)
    # ─────────────────────────────────────────
    skip = """
    for var1, var2 in PDP_2D_PAIRS:
        # Three views per pair: raw, var1 removed, both removed (interaction)
        _pdp2d_modes = [
            (False, False, 'raw'),
            (True,  False, f'minus_{var1}'),
            (False, True,  f'minus_{var2}'),
            (True,  True,  'interaction'),
        ]
        for sub1, sub2, suffix in _pdp2d_modes:
            fig_2d, _ = plot_pdp_2d(
                models=all_models,
                feature_cols=FEATURE_COLS,
                var1=var1, var2=var2,
                target_var=TARGET_COL,
                units=UNITS,
                title=f'2D PDP ({suffix}): {UNITS.get(var1, var1)} × '
                      f'{UNITS.get(var2, var2)} — {RUN_LABEL}',
                shared_colorbar=True,
                subtract_var1=sub1,
                subtract_var2=sub2,
            )
            fname = f'{fig_dir}/pdp_2d_{var1}_{var2}_{suffix}.png'
            fig_2d.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Saved 2D PDP: {fname}')

    plt.show() """


if __name__ == '__main__':
    main()