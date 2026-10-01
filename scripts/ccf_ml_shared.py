# -*- coding: utf-8 -*-
"""
ccf_ml_shared.py

Shared functions for the CCF machine learning pipeline.
Imported by train_model.py and all_regions.py.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import os

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance, partial_dependence
from string import ascii_lowercase as lowers

# ─────────────────────────────────────────────
#  Data formatting
# ─────────────────────────────────────────────

def assign_checkerboard_folds(df, n_folds=4, block_size=5):
    lat_block = (df['lat'] // block_size).astype(int)
    lon_block = (df['lon'] // block_size).astype(int)
    df['fold'] = (lat_block + lon_block) % n_folds
    return xr.broadcast(df)[0]


def xr_to_df(ds):
    stacked = ds.stack(cell=('lat', 'lon', 'time'))
    df = stacked.to_dataframe().dropna()
    df = df.reset_index(drop=True)
    return df


def preprocess_region(ccf_data, region_dict, drop_vars=None):
    """
    Selects a region, masks land, normalizes, and returns the processed
    dataset along with its mean and std for later rescaling.

    Cold advection is rescaled from K/s to K/day in ds_means and ds_std
    (but NOT in ds_norm, which stays normalized).

    Returns
    -------
    ds_norm  : xarray.Dataset, normalized
    ds_means : xarray.Dataset, per-variable means (cold_adv in K/day)
    ds_std   : xarray.Dataset, per-variable stds  (cold_adv in K/day)
    """
    import scripts.utils as utils

    ds = utils.region_sel(ccf_data, region_dict)
    ds = ds.where(ds['sst'].notnull())

    ds_means = ds.mean()
    ds_std   = ds.std()
    ds_norm  = (ds - ds_means) / ds_std

    ds_std['Tadv']   = ds_std['Tadv']   * 86400
    ds_means['Tadv'] = ds_means['Tadv'] * 86400

    if drop_vars:
        ds_norm = ds_norm.drop_vars(
            [v for v in drop_vars if v in ds_norm]
        )

    return ds_norm, ds_means, ds_std


# ─────────────────────────────────────────────
#  Cross-validation infrastructure
# ─────────────────────────────────────────────

def format_xr_cv(xr_ds, temp_split=0.8, n_folds=5, block_size=5):
    xr_ds = assign_checkerboard_folds(xr_ds, n_folds=n_folds,
                                      block_size=block_size)
    n_split = int(temp_split * len(xr_ds.time))
    train   = xr_to_df(xr_ds.isel(time=slice(None, n_split)))
    test    = xr_to_df(xr_ds.isel(time=slice(n_split, None)))

    for n in range(n_folds):
        yield (train.copy().query(f'fold != {n}'),
               test.copy().query(f'fold == {n}'))


def df_spatial_temporal_cv_splits(df, temp_split=0.8, n_folds=5,
                                   block_size=5):
    lat_block   = (df['lat'] // block_size).astype(int)
    lon_block   = (df['lon'] // block_size).astype(int)
    fold_labels = (lat_block + lon_block) % n_folds

    times       = np.sort(df['time'].unique())
    n_train_t   = int(len(times) * temp_split)
    train_times = set(times[:n_train_t])
    val_times   = set(times[n_train_t:])

    for fold_id in range(n_folds):
        is_val   = (fold_labels == fold_id)
        is_train = ~is_val
        train_mask = is_train & df['time'].isin(train_times)
        val_mask   = is_val   & df['time'].isin(val_times)
        yield np.where(train_mask)[0], np.where(val_mask)[0]


# ─────────────────────────────────────────────
#  Per-fold metric helper
# ─────────────────────────────────────────────

def fold_metrics(y_true, y_pred, ccf_std, target_col):
    """
    Compute Pearson R and RMSE for a single fold, rescaled to physical
    units using ccf_std.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, in normalized units
    ccf_std        : xarray.Dataset
    target_col     : str

    Returns
    -------
    r    : float, Pearson correlation coefficient
    rmse : float, RMSE in physical units
    """
    target_std  = float(ccf_std[target_col].values)
    y_true_phys = y_true * target_std
    y_pred_phys = y_pred * target_std

    r, _  = stats.pearsonr(y_true_phys, y_pred_phys)
    rmse  = np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2))
    return r, rmse


# ─────────────────────────────────────────────
#  Model training
# ─────────────────────────────────────────────

def run_spatial_temporal_cv(xr_ds, feature_cols, target_col,
                             ccf_std, model=None,
                             temp_split=0.8, n_folds=5,
                             block_size=5):
    """
    Spatial-temporal CV returning per-fold Pearson R and RMSE
    (in physical units) in addition to R².

    Parameters
    ----------
    xr_ds        : xarray.Dataset, normalized
    feature_cols : list of str
    target_col   : str
    ccf_std      : xarray.Dataset, for rescaling metrics
    model        : sklearn estimator
    temp_split   : float
    n_folds      : int
    block_size   : int

    Returns
    -------
    results_df : pd.DataFrame [fold, val_r2, pearson_r, rmse,
                                n_train, n_val]
    models     : list of fitted estimators
    """
    if model is None:
        model = RandomForestRegressor()

    results, models = [], []

    for fold_idx, (train_df, test_df) in enumerate(
            format_xr_cv(xr_ds, temp_split, n_folds, block_size)):

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target_col].values

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        val_r2       = r2_score(y_test, y_pred)
        r, rmse      = fold_metrics(y_test, y_pred, ccf_std, target_col)

        results.append({
            'fold':      fold_idx,
            'val_r2':    val_r2,
            'pearson_r': r,
            'rmse':      rmse,
            'n_train':   len(y_train),
            'n_val':     len(y_test),
        })
        models.append(fold_model)

    results_df = pd.DataFrame(results)
    # ── FIX: report ± 1 std (fold-to-fold spread), not ± 2 std ──────────
    print(f"\nMean val R²:      {results_df['val_r2'].mean():.3f} "
          f"± {results_df['val_r2'].std():.3f} (1σ across folds)")
    print(f"Mean Pearson R:   {results_df['pearson_r'].mean():.3f} "
          f"± {results_df['pearson_r'].std():.3f} (1σ across folds)")
    print(f"Mean RMSE:        {results_df['rmse'].mean():.3f} "
          f"± {results_df['rmse'].std():.3f} (1σ across folds)")
    return results_df, models


def run_spatial_temporal_cv_tuned(xr_ds, feature_cols, target_col,
                                   ccf_std, model, param_distributions,
                                   temp_split=0.8, n_folds=5,
                                   block_size=5, n_iter=20,
                                   inner_folds=3, random_state=42):
    """
    Nested spatial-temporal CV with inner RandomizedSearchCV.
    Returns per-fold Pearson R and RMSE alongside R².

    Returns
    -------
    results_df : pd.DataFrame [fold, val_r2, pearson_r, rmse,
                                inner_r2, best_params, n_train, n_val]
    models     : list of best fitted estimators per outer fold
    """
    results, models = [], []

    xr_ds_folded = assign_checkerboard_folds(
        xr_ds, n_folds=n_folds, block_size=block_size)
    n_split      = int(temp_split * len(xr_ds_folded.time))
    train_full   = xr_to_df(xr_ds_folded.isel(time=slice(None, n_split)))
    test_full    = xr_to_df(xr_ds_folded.isel(time=slice(n_split, None)))

    for fold_idx in range(n_folds):
        print(f"\nOuter fold {fold_idx} — running RandomizedSearchCV...")

        train_df = train_full.query(f'fold != {fold_idx}')
        test_df  = test_full.query(f'fold == {fold_idx}')

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target_col].values

        inner_splits = list(df_spatial_temporal_cv_splits(
            train_df.reset_index(drop=True),
            temp_split=temp_split,
            n_folds=inner_folds,
            block_size=block_size
        ))

        search = RandomizedSearchCV(
            estimator=clone(model),
            param_distributions=param_distributions,
            n_iter=n_iter, cv=inner_splits,
            scoring='r2', n_jobs=-1,
            random_state=random_state, refit=True
        )
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)

        val_r2  = r2_score(y_test, y_pred)
        r, rmse = fold_metrics(y_test, y_pred, ccf_std, target_col)

        print(f"  Best inner R²={search.best_score_:.3f} | "
              f"params={search.best_params_}")
        print(f"  Outer val R²={val_r2:.3f}  R={r:.3f}  "
              f"RMSE={rmse:.3f}  "
              f"(n_train={len(y_train)}, n_val={len(y_test)})")

        results.append({
            'fold':        fold_idx,
            'val_r2':      val_r2,
            'pearson_r':   r,
            'rmse':        rmse,
            'inner_r2':    search.best_score_,
            'best_params': search.best_params_,
            'n_train':     len(y_train),
            'n_val':       len(y_test),
        })
        models.append(search.best_estimator_)

    results_df = pd.DataFrame(results)
    # ── FIX: report ± 1 std (fold-to-fold spread), not ± 2 std ──────────
    print(f"\nMean outer val R²:    {results_df['val_r2'].mean():.3f} "
          f"± {results_df['val_r2'].std():.3f} (1σ across folds)")
    print(f"Mean outer Pearson R: {results_df['pearson_r'].mean():.3f} "
          f"± {results_df['pearson_r'].std():.3f} (1σ across folds)")
    print(f"Mean outer RMSE:      {results_df['rmse'].mean():.3f} "
          f"± {results_df['rmse'].std():.3f} (1σ across folds)")
    return results_df, models


def select_best_params(results_df):
    best_fold = results_df.loc[results_df['inner_r2'].idxmax()]
    print(f"Selected params from fold {int(best_fold['fold'])} "
          f"(inner R²={best_fold['inner_r2']:.3f})")
    return best_fold['best_params']


def fit_final_model(xr_ds, feature_cols, target_col, best_params,
                    base_model=None, temp_split=0.8):
    """
    Fits a final model on the training period. Returns test data,
    the fitted model, and train/test sample sizes.

    Returns
    -------
    X_test, y_test, final_model, n_train, n_test
    """
    if base_model is None:
        base_model = RandomForestRegressor(n_jobs=-1)

    n_split = int(temp_split * len(xr_ds.time))
    train   = xr_to_df(xr_ds.isel(time=slice(None, n_split)))
    test    = xr_to_df(xr_ds.isel(time=slice(n_split, None)))

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test  = test[feature_cols].values
    y_test  = test[target_col].values

    final_model = clone(base_model).set_params(**best_params)
    final_model.fit(X_train, y_train)

    print(f"Final model fitted on {len(y_train)} training, "
          f"{len(y_test)} test observations")

    return X_test, y_test, final_model, len(y_train), len(y_test)


def load_or_tune(xr_ds, feature_cols, target_col, ccf_std,
                 param_path, param_distributions, base_model,
                 n_iter=30, n_folds=5, inner_folds=3, block_size=5):
    """
    Loads cached hyperparameters if available, otherwise runs nested CV
    tuning and saves to param_path.

    Returns
    -------
    best_params : dict
    cv_results  : pd.DataFrame or None if loaded from cache
    """
    if os.path.isfile(param_path):
        print(f'Loading hyperparameters from {param_path}')
        with open(param_path, 'rb') as f:
            return pickle.load(f), None

    print(f'No cached params at {param_path}. Running tuning...')
    cv_results, _ = run_spatial_temporal_cv_tuned(
        xr_ds=xr_ds, feature_cols=feature_cols,
        target_col=target_col, ccf_std=ccf_std,
        model=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter, n_folds=n_folds,
        inner_folds=inner_folds, block_size=block_size
    )
    best_params = select_best_params(cv_results)

    os.makedirs(os.path.dirname(param_path), exist_ok=True)
    with open(param_path, 'wb') as f:
        pickle.dump(best_params, f)

    return best_params, cv_results


# ─────────────────────────────────────────────
#  Caching helpers
# ─────────────────────────────────────────────

def load_or_run_cv(xr_ds, feature_cols, target_col, ccf_std,
                   metrics_path, model, n_folds=5, block_size=5):
    """
    Load fold-level CV metrics from a CSV if available, otherwise run
    CV, save the results, and return them.

    Parameters
    ----------
    xr_ds        : xarray.Dataset, normalised
    feature_cols : list of str
    target_col   : str
    ccf_std      : xarray.Dataset
    metrics_path : str, path to CSV cache
    model        : sklearn estimator (already has best params set)
    n_folds      : int
    block_size   : int

    Returns
    -------
    cv_results : pd.DataFrame [fold, val_r2, pearson_r, rmse, n_train, n_val]
    """
    if os.path.isfile(metrics_path):
        print(f'Loading CV metrics from {metrics_path}')
        return pd.read_csv(metrics_path)

    print(f'No cached metrics at {metrics_path}. Running CV...')
    cv_results, _ = run_spatial_temporal_cv(
        xr_ds=xr_ds, feature_cols=feature_cols,
        target_col=target_col, ccf_std=ccf_std,
        model=model, n_folds=n_folds, block_size=block_size,
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    cv_results.to_csv(metrics_path, index=False)
    print(f'Saved CV metrics to {metrics_path}')
    return cv_results


def load_or_fit_final(xr_ds, feature_cols, target_col, best_params,
                      model_path, base_model=None, temp_split=0.8):
    """
    Load a fitted final model from disk if available, otherwise fit,
    save, and return it. In both cases the test set is extracted from
    xr_ds so X_test / y_test are always fresh (cheap operation).

    Parameters
    ----------
    xr_ds        : xarray.Dataset, normalised
    feature_cols : list of str
    target_col   : str
    best_params  : dict
    model_path   : str, path to .pkl cache
    base_model   : sklearn estimator
    temp_split   : float

    Returns
    -------
    X_test, y_test, final_model, n_train, n_test
    """
    if base_model is None:
        base_model = RandomForestRegressor(n_jobs=-1)

    # Always extract the test split — it's just an xarray slice + stack
    n_split = int(temp_split * len(xr_ds.time))
    train   = xr_to_df(xr_ds.isel(time=slice(None, n_split)))
    test    = xr_to_df(xr_ds.isel(time=slice(n_split, None)))
    X_test  = test[feature_cols].values
    y_test  = test[target_col].values

    if os.path.isfile(model_path):
        print(f'Loading final model from {model_path}')
        with open(model_path, 'rb') as f:
            final_model = pickle.load(f)
        return X_test, y_test, final_model, len(train), len(test)

    print(f'No cached model at {model_path}. Fitting...')
    X_train = train[feature_cols].values
    y_train = train[target_col].values

    final_model = clone(base_model).set_params(**best_params)
    final_model.fit(X_train, y_train)
    print(f'Final model fitted on {len(y_train)} training, '
          f'{len(y_test)} test observations')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f'Saved final model to {model_path}')

    return X_test, y_test, final_model, len(y_train), len(y_test)


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_ccf_histograms(X_test, feature_cols, ccf_std, ccf_means,
                        region_label=None, units=None,
                        n_cols=4, figsize=(16, 6), bins=40):
    """
    Histogram grid showing the physical-unit distribution of each CCF
    variable in the test set for a single region.  Call once per region.

    Parameters
    ----------
    X_test        : np.ndarray, shape (n_samples, n_features),
                    in normalized units
    feature_cols  : list of str
    ccf_std       : xarray.Dataset, per-variable stds
    ccf_means     : xarray.Dataset, per-variable means
    region_label  : str, used in the suptitle
    units         : dict, variable name -> axis label string
    n_cols        : int
    figsize       : tuple
    bins          : int

    Returns
    -------
    fig, axes
    """
    units      = units or {}
    n_features = len(feature_cols)
    n_rows     = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, (feat, ax) in enumerate(zip(feature_cols, axes_flat)):
        feat_std  = float(ccf_std[feat].values)
        feat_mean = float(ccf_means[feat].values)
        x_phys    = X_test[:, i] * feat_std + feat_mean

        ax.hist(x_phys, bins=bins, color='steelblue',
                edgecolor='white', linewidth=0.4, alpha=0.85)
        ax.set_xlabel(units.get(feat, feat), fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Mark 2.5 / 97.5 percentiles
        p_lo = np.percentile(x_phys, 2.5)
        p_hi = np.percentile(x_phys, 97.5)
        for pv in (p_lo, p_hi):
            ax.axvline(pv, color='firebrick', linewidth=1.2,
                       linestyle='--', alpha=0.8)

    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    title = 'CCF Variable Distributions (test set)'
    if region_label:
        title += f' — {region_label}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    return fig, axes


def plot_pdp(models, feature_cols, target_var, units=None, title=None,
             xlabel=None, n_cols=4, figsize=(16, 6),
             show_ci_ticks=True, center_curves=False):
    """
    Partial dependence plots for one or more models with a shared
    figure legend centred below all subplots.

    The 2.5th/97.5th percentile range of each feature across all
    models' test sets can be marked on the x-axis with small coloured
    tick marks to indicate the data support region.

    Parameters
    ----------
    models        : dict {label: (estimator, X_test, ccf_std, ccf_means)}
    feature_cols  : list of str
    target_var    : str
    units         : dict, variable name -> axis label string
    title         : str, suptitle
    xlabel        : str, optional note below the legend
    n_cols        : int
    figsize       : tuple
    show_ci_ticks : bool
        If True, draw small coloured x-axis ticks at the 2.5th and
        97.5th percentiles of each feature for each model/region.
    center_curves : bool
        If True, subtract the mean of each PDP curve before plotting,
        so all curves are zero-centred. Useful for comparing linear
        regression PDPs (which carry an intercept offset) against RF
        PDPs on the same scale.

    Returns
    -------
    fig, axes
    """
    n_features     = len(feature_cols)
    n_rows         = int(np.ceil(n_features / n_cols))
    units          = units or {}
    colors         = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_handles = []
    legend_labels  = []

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, (feat, ax) in enumerate(zip(feature_cols, axes_flat)):
        for color, (label, (model, X_test, ccf_std, ccf_means)) in \
                zip(colors, models.items()):

            feat_std   = float(ccf_std[feat].values)
            feat_mean  = float(ccf_means[feat].values)
            target_std = float(ccf_std[target_var].values)

            pd_res = partial_dependence(
                model, X_test, features=[i],
                kind='average', grid_resolution=50
            )
            x_vals = pd_res['grid_values'][0] * feat_std + feat_mean
            y_vals = pd_res['average'][0] * target_std

            if center_curves:
                y_vals = y_vals - y_vals.mean()

            line, = ax.plot(x_vals, y_vals, color=color,
                            linewidth=2, label=label)
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(label)

            # ── CI tick marks on x-axis ──────────────────────────────────
            if show_ci_ticks:
                x_phys = X_test[:, i] * feat_std + feat_mean
                p_lo   = np.percentile(x_phys, 2.5)
                p_hi   = np.percentile(x_phys, 97.5)
                # get_xaxis_transform(): x in data coords, y in axes fraction.
                # y=0 always means the x-axis spine, regardless of data limits.
                for pv in (p_lo, p_hi):
                    ax.plot([pv], [0], marker='|', color=color,
                            markersize=10, markeredgewidth=2.5,
                            clip_on=False, zorder=5,
                            transform=ax.get_xaxis_transform())

        ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel(lowers[i] + ') ' + units.get(feat, feat), fontsize=13)
        ax.set_ylabel(f"Δ {units.get(target_var, target_var)}", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    fig.legend(legend_handles, legend_labels,
               loc='lower center', ncol=len(models),
               fontsize=12, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.06))

    if xlabel:
        fig.text(0.5, -0.12, xlabel, ha='center', va='top',
                 fontsize=10, style='italic', color='dimgrey')

    plt.tight_layout()
    return fig, axes


def plot_varimp(model, X_test, y_test, feature_cols,
                title=None, n_repeats=10, figsize=(8, 5),
                random_state=42):
    """
    Permutation feature importance bar chart with 1-sigma error bars.

    Returns
    -------
    fig, ax, importances_df
    """
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, n_jobs=-1,
        random_state=random_state
    )
    importances_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': perm.importances_mean,
        'std':        perm.importances_std
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(importances_df['feature'], importances_df['importance'],
            xerr=importances_df['std'], color='steelblue',
            ecolor='black', capsize=4, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean decrease in R²', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, ax, importances_df


def plot_varimp_all_regions(models_dict, y_test_dict, feature_cols,
                             units=None, title=None,
                             n_repeats=10, figsize=(14, 9),
                             random_state=42):
    """
    3×2 grid of permutation importance bar charts, one panel per region,
    with a consistent x-axis scale across all panels.
    The bottom-right panel is left empty if there are exactly 5 regions.

    Parameters
    ----------
    models_dict   : dict {region: fitted estimator}
    y_test_dict   : dict {region: (X_test, y_test)}
                    both arrays in normalized units
    feature_cols  : list of str
    units         : dict, variable name -> display label
    title         : str, figure suptitle
    n_repeats     : int, permutation repeats
    figsize       : tuple
    random_state  : int

    Returns
    -------
    fig, axes, all_importances   (dict {region: importances_df})
    """
    units   = units or {}
    regions = list(models_dict.keys())

    # ── Compute importances for every region first ────────────────────────
    all_importances = {}
    for region in regions:
        model         = models_dict[region]
        X_test, y_test = y_test_dict[region]
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats, n_jobs=-1,
            random_state=random_state
        )
        all_importances[region] = pd.DataFrame({
            'feature':    feature_cols,
            'importance': perm.importances_mean,
            'std':        perm.importances_std
        })

    # ── Shared x-axis limits ──────────────────────────────────────────────
    all_means = np.concatenate(
        [df['importance'].values for df in all_importances.values()]
    )
    all_stds  = np.concatenate(
        [df['std'].values for df in all_importances.values()]
    )
    x_min = min(0.0, (all_means - all_stds).min()) * 1.15
    x_max = (all_means + all_stds).max() * 1.15

    # ── Feature display labels (consistent ordering by global mean) ───────
    global_mean = np.mean(
        [df.set_index('feature')['importance']
         for df in all_importances.values()], axis=0
    )
    feat_order = pd.Series(global_mean, index=feature_cols)\
                   .sort_values(ascending=True).index.tolist()
    feat_labels = [units.get(f, f) for f in feat_order]

    # ── Layout: 3 columns, 2 rows ─────────────────────────────────────────
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              sharey=True, sharex=True)
    axes_flat = axes.flatten()

    for ax_idx, region in enumerate(regions):
        ax  = axes_flat[ax_idx]
        df  = all_importances[region].set_index('feature')\
                                      .loc[feat_order].reset_index()

        ax.barh(feat_labels, df['importance'],
                xerr=df['std'],
                color='steelblue', ecolor='black',
                capsize=3, edgecolor='white', linewidth=0.4,
                alpha=0.85)
        ax.axvline(0, color='k', linewidth=0.8,
                   linestyle='--', alpha=0.5)
        ax.set_xlim(x_min, x_max)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_title(region, fontsize=11, fontweight='bold', pad=4)

        # Only label y-axis for the left column
        if ax_idx % n_cols != 0:
            ax.tick_params(labelleft=False)

        # Only label x-axis for the bottom row
        if ax_idx // n_cols == n_rows - 1 or \
                ax_idx >= len(regions) - n_cols:
            ax.set_xlabel('Mean decrease in R²', fontsize=9)

    # ── Hide the unused bottom-right panel ───────────────────────────────
    for ax in axes_flat[len(regions):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig, axes, all_importances


def plot_pdp_2d(models, feature_cols, var1, var2,
                target_var, units=None, title=None,
                grid_resolution=30, figsize=(14, 9),
                cmap='RdBu_r', shared_colorbar=True,
                per_panel_colorbar=False,
                subtract_var1=False, subtract_var2=False):
    """
    3×2 grid of 2D partial dependence heatmaps, one panel per model
    (region), showing the joint response surface for var1 × var2.
    The bottom-right panel is left empty if there are exactly 5 regions.

    Parameters
    ----------
    models              : dict {label: (estimator, X_test, ccf_std, ccf_means)}
    feature_cols        : list of str
    var1, var2          : str, names of the two features to cross
    target_var          : str
    units               : dict, variable name -> axis label string
    title               : str, figure suptitle
    grid_resolution     : int, PDP grid points per axis
    figsize             : tuple
    cmap                : str, matplotlib colormap
    shared_colorbar     : bool
        If True a single colorbar is placed in the empty corner panel,
        with limits set to the global min/max across all regions.
    per_panel_colorbar  : bool
        If True each panel gets its own colorbar with independent limits.
        Overrides shared_colorbar when both are True.
    subtract_var1       : bool
        If True, subtract the marginal 1D PDP of var1 from the surface:
            Z[i,j] -= Z_1d_var1[i]
        Removes the main effect of var1, leaving var2's response and
        any interaction structure.
    subtract_var2       : bool
        If True, subtract the marginal 1D PDP of var2 from the surface:
            Z[i,j] -= Z_1d_var2[j]
        Removes the main effect of var2, leaving var1's response and
        any interaction structure.
        Setting both subtract_var1=True and subtract_var2=True isolates
        the pure interaction effect (equivalent to the old interaction_only).

    Returns
    -------
    fig, axes
    """
    units    = units or {}
    regions  = list(models.keys())
    feat_idx = {f: i for i, f in enumerate(feature_cols)}

    i1 = feat_idx[var1]
    i2 = feat_idx[var2]

    xlabel = units.get(var1, var1)
    ylabel = units.get(var2, var2)
    _base_unit = units.get(target_var, target_var)
    # Build a zlabel that reflects which main effects have been removed
    _removed = ([xlabel] if subtract_var1 else []) + \
               ([ylabel] if subtract_var2 else [])
    if _removed:
        zlabel = f"Δ {_base_unit} (− {', '.join(_removed)} main effect)"
    else:
        zlabel = f"Δ {_base_unit}"

    # ── Compute all 2D PDPs first so we can find the global colour range ──
    pd_data = {}
    for label, (model, X_test, ccf_std, ccf_means) in models.items():
        target_std = float(ccf_std[target_var].values)
        feat1_std  = float(ccf_std[var1].values)
        feat1_mean = float(ccf_means[var1].values)
        feat2_std  = float(ccf_std[var2].values)
        feat2_mean = float(ccf_means[var2].values)

        pd_res = partial_dependence(
            model, X_test, features=[(i1, i2)],
            kind='average', grid_resolution=grid_resolution
        )
        x1 = pd_res['grid_values'][0] * feat1_std + feat1_mean
        x2 = pd_res['grid_values'][1] * feat2_std + feat2_mean
        z  = pd_res['average'][0] * target_std   # shape (grid_res, grid_res)

        if subtract_var1 or subtract_var2:
            # Only compute 1D PDPs for whichever axes are being subtracted
            if subtract_var1:
                pd1 = partial_dependence(
                    model, X_test, features=[i1],
                    kind='average', grid_resolution=grid_resolution
                )
                z1 = pd1['average'][0] * target_std  # shape (grid_res,)
                z  = z - z1[:, np.newaxis]
            if subtract_var2:
                pd2 = partial_dependence(
                    model, X_test, features=[i2],
                    kind='average', grid_resolution=grid_resolution
                )
                z2 = pd2['average'][0] * target_std  # shape (grid_res,)
                z  = z - z2[np.newaxis, :]

        pd_data[label] = (x1, x2, z)

    # ── Global colour limits ──────────────────────────────────────────────
    use_shared = shared_colorbar and not per_panel_colorbar
    if use_shared:
        all_z   = np.concatenate([d[2].ravel() for d in pd_data.values()])
        abs_max = np.abs(all_z).max()
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = None, None

    # ── Layout: 3 columns, 2 rows ─────────────────────────────────────────
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.35})

    axes_flat = axes.flatten()
    ims = []   # keep handles for the shared colorbar

    for ax_idx, label in enumerate(regions):
        ax        = axes_flat[ax_idx]
        x1, x2, z = pd_data[label]

        # pcolormesh needs 2-D coordinate arrays
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')

        im = ax.pcolormesh(X1, X2, z, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='auto')
        ims.append(im)

        if per_panel_colorbar:
            fig.colorbar(im, ax=ax, label=zlabel,
                         fraction=0.046, pad=0.04)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(label, fontsize=11, fontweight='bold', pad=4)

    # ── Handle spare (empty) panels and shared colorbar ─────────────────
    spare_axes = axes_flat[len(regions):]
    if use_shared and ims and len(spare_axes) > 0:
        # Repurpose the first spare axes as a thin colorbar strip
        cax = spare_axes[0]
        cax.set_visible(True)
        pos = cax.get_position()
        cax.set_position([
            pos.x0 + pos.width * 0.35,
            pos.y0 + pos.height * 0.05,
            pos.width * 0.25,
            pos.height * 0.90,
        ])
        cbar = fig.colorbar(ims[0], cax=cax, label=zlabel)
        cbar.ax.tick_params(labelsize=9)
        for ax in spare_axes[1:]:
            ax.set_visible(False)
    else:
        for ax in spare_axes:
            ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    return fig, axes


def plot_metrics_comparison(fold_metrics_dict, title=None, figsize=(11, 8)):
    """
    Two-panel box-and-whisker comparison of Pearson R and RMSE across
    regions, for the ML model and linear regression. Each box represents
    the distribution of a metric across CV folds, matching the style of
    Liu et al. (JAMES) model comparison figures.

    Parameters
    ----------
    fold_metrics_dict : dict of {region: {model_label: df, 'lin': df}}
                        where each df has columns [pearson_r, rmse]
                        with one row per CV fold. The non-linear model
                        key is inferred automatically (anything not 'lin').
    title             : str or None
    figsize           : tuple

    Returns
    -------
    fig, (ax_rmse, ax_r)
    """
    regions    = list(fold_metrics_dict.keys())
    n_regions  = len(regions)
    x          = np.arange(n_regions)
    width      = 0.35
    ml_color   = 'steelblue'
    lin_color  = 'coral'

    # Infer the ML model key from the first region's dict
    first_keys = list(fold_metrics_dict[regions[0]].keys())
    ml_key     = next(k for k in first_keys if k != 'lin')

    fig, (ax_rmse, ax_r) = plt.subplots(2, 1, figsize=figsize,
                                         sharex=True)

    bp_props = dict(patch_artist=True, widths=width,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4,
                                    alpha=0.5))

    for ax, metric, ylabel, panel_label in [
        (ax_rmse, 'rmse',      'RMSE (%)',   '(a) Root Mean Square Error'),
        (ax_r,    'pearson_r', 'Pearson R',  '(b) Correlation Coefficient'),
    ]:
        ml_data  = [fold_metrics_dict[r][ml_key][metric].values
                    for r in regions]
        lin_data = [fold_metrics_dict[r]['lin'][metric].values
                    for r in regions]

        bp_ml  = ax.boxplot(ml_data,
                            positions=x - width / 2,
                            **bp_props)
        bp_lin = ax.boxplot(lin_data,
                            positions=x + width / 2,
                            **bp_props)

        for patch in bp_ml['boxes']:
            patch.set_facecolor(ml_color)
            patch.set_alpha(0.8)
        for patch in bp_lin['boxes']:
            patch.set_facecolor(lin_color)
            patch.set_alpha(0.8)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_title(panel_label, fontsize=11, loc='left', pad=6)

        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=ml_color,  alpha=0.8, label=ml_key),
                     Patch(facecolor=lin_color, alpha=0.8,
                           label='Linear Regression')],
            fontsize=10, framealpha=0.7
        )

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(regions, fontsize=11)
    ax_r.set_xlabel('Region', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    return fig, (ax_rmse, ax_r)