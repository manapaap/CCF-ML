# -*- coding: utf-8 -*-
"""
Training the ML model on the isolated boxes

This file is intended to serve as a "workflow" to be later follwoed
in a notebook or other format
"""


import xarray as xr
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


os.chdir('C:/Users/aakas/Documents/CCF-ML/')
import scripts.utils as utils


def assign_checkerboard_folds(df, n_folds=4, block_size=5):
    """
    Assigns fold IDs based on a checkerboard of (block_size x block_size)
    degree blocks. Each block is assigned to a fold by its position modulo
    n_folds. Works with irregular domains.
    
    block_size: size of each block in degrees
    """
    lat_block = (df['lat'] // block_size).astype(int)
    lon_block = (df['lon'] // block_size).astype(int)
    
    df['fold'] = (lat_block + lon_block) % n_folds
    return xr.broadcast(df)[0]


def format_xr_cv(xr_ds, temp_split=0.8, n_folds=5, block_size=5):
    """
    Resizes the xarray dataarray 
    
    Sets spatial-temporal cross validation to be in (train, test) splits
    as arrays of indices. Spatial folds created via checkerboard method
    temporal fold created by training on first 80% of data and testing on
    second half
    """
    xr_ds = assign_checkerboard_folds(xr_ds,
                                      n_folds=n_folds,
                                      block_size=block_size)
    # Temporal Test/Train split
    train_times = xr_ds.time[:int(temp_split*len(xr_ds.time))]
    test_times = xr_ds.time[int(temp_split*len(xr_ds.time)):]
    train = xr_ds.sel({'time': train_times})
    test = xr_ds.sel({'time': test_times})
    # Reshape arrays
    train = xr_to_df(train)
    test = xr_to_df(test)
    # yield test-train pairs
    for n in range(n_folds):
        # Train on everything but n and test on n
        train_n = train.copy().query(f'fold != {n}')
        test_n = test.copy().query(f'fold == {n}')
        yield train_n, test_n
        
        
def xr_to_df(ds):
    """
    Reshapes an xarray Dataset with dims (time, latitude, longitude)
    into a flat pandas DataFrame where each row is a single
    spatial-temporal observation.
    
    Parameters
    ----------
    ds : xarray.Dataset, with dims (time, latitude, longitude)
    feature_vars : list of str, CCF variable names
    target_var : str, cloud fraction variable name
    
    Returns
    -------
    pd.DataFrame with columns for each variable plus time, latitude,
    longitude. NaN rows (land, masked ocean) are dropped.
    """
    # Stacks all dimentions with "cell" as index
    stacked = ds.stack(cell=('lat', 'lon', 'time'))
    # Convert to dataframe - cell as index, and drop ocean cells
    df = stacked.to_dataframe().dropna()
    # Reset index so rows are contiguous integers - required for
    # sklearn's index-based CV interface and drop useless vars
    df = df.reset_index(drop=True)
    return df


def run_spatial_temporal_cv(xr_ds, feature_cols, target_col,
                             model=None,
                             temp_split=0.8, n_folds=5, block_size=5):
    """
    Runs spatial-temporal cross validation manually, iterating over
    folds from format_xr_cv(). Returns a DataFrame of per-fold metrics
    and the fitted models for inspection.

    Parameters
    ----------
    xr_ds        : xarray.Dataset, pre-processed and masked
    feature_cols : list of str, CCF predictor variable names
    target_col   : str, target variable name
    model        : instantiated sklearn estimator, e.g.
                   RandomForestRegressor(n_estimators=200, n_jobs=-1)
                   LinearRegression()
                   Defaults to RandomForestRegressor() if not specified.
    temp_split   : float, fraction of timesteps used for training
    n_folds      : int, number of spatial CV folds
    block_size   : int, spatial block size in degrees

    Returns
    -------
    results_df : pd.DataFrame with columns [fold, val_r2, n_train, n_val]
    models     : list of fitted estimators, one per fold
    """
    if model is None:
        model = RandomForestRegressor()

    results = []
    models  = []

    cv_folds = format_xr_cv(xr_ds, temp_split, n_folds, block_size)

    for fold_idx, (train_df, test_df) in enumerate(cv_folds):
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target_col].values

        # Clone gives a fresh unfitted estimator with the same hyperparameters
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        val_r2 = r2_score(y_test, fold_model.predict(X_test))

        results.append({
            'fold':    fold_idx,
            'val_r2':  val_r2,
            'n_train': len(y_train),
            'n_val':   len(y_test)
        })
        models.append(fold_model)
        
    results_df = pd.DataFrame(results)
    print(f"\nMean val R²: {results_df['val_r2'].mean():.3f} "
          f"± {2 * results_df['val_r2'].std():.3f}")

    return results_df, models


def df_spatial_temporal_cv_splits(df, temp_split=0.8, n_folds=5, block_size=5):
    """
    Yields (train_idx, val_idx) index pairs from a flat DataFrame,
    compatible with sklearn's CV interface.
    Used for the inner loop of nested CV.
    """
    # Assign spatial folds
    lat_block = (df['lat'] // block_size).astype(int)
    lon_block = (df['lon'] // block_size).astype(int)
    fold_labels = (lat_block + lon_block) % n_folds

    # Temporal split
    times = np.sort(df['time'].unique())
    n_train = int(len(times) * temp_split)
    train_times = set(times[:n_train])
    val_times   = set(times[n_train:])

    for fold_id in range(n_folds):
        is_val_space   = (fold_labels == fold_id)
        is_train_space = ~is_val_space

        train_mask = is_train_space & df['time'].isin(train_times)
        val_mask   = is_val_space   & df['time'].isin(val_times)

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        yield train_idx, val_idx


def run_spatial_temporal_cv_tuned(xr_ds, feature_cols, target_col,
                                   model, param_distributions,
                                   temp_split=0.8, n_folds=5, block_size=5,
                                   n_iter=20, inner_folds=3,
                                   random_state=42):
    results = []
    models  = []

    # Need fold column intact for inner splitting, so regenerate here
    xr_ds_folded = assign_checkerboard_folds(xr_ds, n_folds=n_folds,
                                              block_size=block_size)
    train_times = xr_ds_folded.time[:int(temp_split * len(xr_ds_folded.time))]
    test_times  = xr_ds_folded.time[int(temp_split * len(xr_ds_folded.time)):]

    train_full = xr_to_df(xr_ds_folded.sel(time=train_times))
    test_full  = xr_to_df(xr_ds_folded.sel(time=test_times))

    for fold_idx in range(n_folds):
        print(f"\nOuter fold {fold_idx} — running RandomizedSearchCV...")

        # Outer split — fold column dropped after splitting
        train_df = train_full.query(f'fold != {fold_idx}')
        test_df  = test_full.query( f'fold == {fold_idx}')

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target_col].values

        # Inner splits — generated from train_df which still has
        # lat, lon, time columns available
        inner_splits = list(
            df_spatial_temporal_cv_splits(
                train_df.reset_index(drop=True),
                temp_split=temp_split,
                n_folds=inner_folds,
                block_size=block_size
            )
        )

        search = RandomizedSearchCV(
            estimator=clone(model),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_splits,
            scoring='r2',
            n_jobs=-1,
            random_state=random_state,
            refit=True
        )
        search.fit(X_train, y_train)

        val_r2 = r2_score(y_test, search.predict(X_test))

        print(f"  Best inner R²={search.best_score_:.3f} | "
              f"params={search.best_params_}")
        print(f"  Outer val R²={val_r2:.3f}  "
              f"(n_train={len(y_train)}, n_val={len(y_test)})")

        results.append({
            'fold':        fold_idx,
            'val_r2':      val_r2,
            'inner_r2':    search.best_score_,
            'best_params': search.best_params_,
            'n_train':     len(y_train),
            'n_val':       len(y_test)
        })
        models.append(search.best_estimator_)

    results_df = pd.DataFrame(results)
    print(f"\nMean outer val R²: {results_df['val_r2'].mean():.3f} "
          f"± {2 * results_df['val_r2'].std():.3f}")

    return results_df, models


def select_best_params(results_df):
    """
    Selects best hyperparameters from nested CV results.
    Uses the fold with the highest inner validation R².
    """
    best_fold = results_df.loc[results_df['inner_r2'].idxmax()]
    print(f"Selected params from fold {int(best_fold['fold'])} "
          f"(inner R²={best_fold['inner_r2']:.3f})")
    return best_fold['best_params']


def fit_final_model(xr_ds, feature_cols, target_col, best_params,
                    base_model=None, temp_split=0.8):
    """
    Fits a single final model on all available data using
    the best hyperparameters from nested CV.
    
    Returns testing data for model evaluation
    """
    if base_model is None:
        base_model = RandomForestRegressor(n_jobs=-1)
    
    # Temporal Test/Train split
    train_times = xr_ds.time[:int(temp_split*len(xr_ds.time))]
    test_times = xr_ds.time[int(temp_split*len(xr_ds.time)):]
    train = xr_ds.sel({'time': train_times})
    test = xr_ds.sel({'time': test_times})
    # split
    train = xr_to_df(train)
    test = xr_to_df(test)
    X_train  = train[feature_cols].values
    y_train  = train[target_col].values
    X_test  = test[feature_cols].values
    y_test  = test[target_col].values
    
    final_model = clone(base_model).set_params(**best_params)
    final_model.fit(X_train, y_train)
    print(f"Final model fitted on {len(y_train)} observations")

    return X_test, y_test, final_model


def main():
    global X_test, y_test, final_model, ccf_sep
    np.random.seed(6767)
    # load anomaly data
    ccf_data = xr.open_dataset('clean_data/ccf_clouds_clean.nc')
    # get patches
    sc_regions = utils.get_stratocumulus_regions()
    # select region
    ccf_sep = utils.region_sel(ccf_data, sc_regions['SEP'])
    # mask out grid points over land
    ccf_sep = ccf_sep.where(ccf_sep['sst'].notnull())
    # normalize- subtract mean and zero-center
    # retain values in case wanted later
    ccf_means = ccf_sep.mean()
    ccf_std = ccf_sep.std()
    ccf_sep -= ccf_means
    ccf_sep /= ccf_std
    # drop useless vars
    ccf_sep = ccf_sep.drop_vars(['u10', 'v10', 'msl'])
    # generate rf models and summary results
    print('Default Random Forest')
    _, _ = run_spatial_temporal_cv(
                        xr_ds        = ccf_sep,
                        feature_cols = ['sst', 'eis', 'speed',
                                        'cold_adv', 'w_700', 'ln_AOD',
                                        'rh_700', 'cldarea_high'],
                        target_col   = 'cldarea_low_adj',
                        n_folds=5, block_size=5,
                        model=RandomForestRegressor(n_estimators=100,
                                                    max_depth=None,
                                                    min_samples_leaf=1,
                                                    n_jobs=-1)
        )
    # linear regression?
    print('Linear Regression')
    _, _ = run_spatial_temporal_cv(
                        xr_ds        = ccf_sep,
                        feature_cols = ['sst', 'eis', 'speed',
                                        'cold_adv', 'w_700', 'ln_AOD',
                                        'rh_700', 'cldarea_high'],
                        target_col   = 'cldarea_low_adj',
                        n_folds=5, block_size=5,
                        model=LinearRegression(),
        )
    # Random forest has an R2 of 0.26, while linear regression has R2 of 0.31
    # Hyperparameter search!
    
    is_tuned = os.path.isfile('misc/hyperparams/sep_rf_params.csv')
    
    if not is_tuned:
        param_distributions = {
        'n_estimators':     randint(100, 500),
        'max_depth':        [5, 10, 20, None],
        'min_samples_leaf': randint(5, 50),
        'max_features':     uniform(0.2, 0.6)
        }
    
        results, models = run_spatial_temporal_cv_tuned(
            xr_ds               = ccf_sep,
            feature_cols        = ['sst', 'eis', 'speed',
                                    'cold_adv', 'w_700', 'ln_AOD',
                                    'rh_700', 'cldarea_high'],
            target_col          = 'cldarea_low_adj',
            model               = RandomForestRegressor(n_jobs=-1),
            param_distributions = param_distributions,
            n_iter              = 30,
            n_folds             = 5,
            inner_folds         = 3,
            block_size          = 5
        )
        # fit model based on best hyperparameters
        best_params = select_best_params(results)
        best_params.to_csv('misc/hyperparams/sep_rf_params.csv')
    else:
        best_params = pd.read_csv('misc/hyperparams/sep_rf_params.csv')
    
    X_test, y_test, final_model = fit_final_model(
        xr_ds        = ccf_sep,
        feature_cols = ['sst', 'eis', 'speed',
                                'cold_adv', 'w_700', 'ln_AOD',
                                'rh_700', 'cldarea_high'],
        target_col   = 'cldarea_low_adj',
        best_params  = best_params,
        base_model   = RandomForestRegressor(n_jobs=-1)
    )
    
    # variable importance and partial dependence
    # PDP
    fig, ax = plt.subplots(figsize=(12, 8))

    PartialDependenceDisplay.from_estimator(
        final_model,
        X_test,
        features=list(range(8)),
        feature_names=['sst', 'eis', 'speed',
                                'cold_adv', 'w_700', 'ln_AOD',
                                'rh_700', 'cldarea_high'],
        n_jobs=-1,
        ax=ax
    )
    plt.tight_layout()
    
    # VarImp
    perm = permutation_importance(
        final_model, X_test, y_test,
        n_repeats=10,
        n_jobs=-1,
        random_state=42)
    
    perm_importances = pd.Series(
        perm.importances_mean,
        index=['sst', 'eis', 'speed',
                                'cold_adv', 'w_700', 'ln_AOD',
                                'rh_700', 'cldarea_high']
    ).sort_values(ascending=False)

    print(perm_importances)
    
    
if __name__ == '__main__':
    main()
    
