# -*- coding: utf-8 -*-
"""
Utility Functions for CCF-ML Project
"""


import xarray as xr
from scipy.ndimage import gaussian_filter
import sys
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def smooth_data(era5_field, sigma=3):
    """
    Gaussian smooths a given field and returns it. Wraps around to prevent
    artifact at 180 deg
    """
    n_wrap = int(3 * sigma)
    wrapped = xr.concat(
        [era5_field.isel(lon=slice(-n_wrap, None)),
         era5_field,
         era5_field.isel(lon=slice(0, n_wrap))],
        dim="lon")
    smoothed_wrapped = xr.apply_ufunc(
        gaussian_filter, wrapped,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        kwargs={"sigma": sigma},
        vectorize=True)

    # Crop back to original longitude range
    smoothed = smoothed_wrapped.isel(lon=slice(n_wrap, -n_wrap))
    smoothed = smoothed.assign_coords(lon=era5_field.lon)
    return smoothed


def progress_bar(n, max_val, cus_str=''):
    """
    I love progress bars in long loops
    """
    sys.stdout.write('\033[2K\033[1G')
    print(f'Computing...{100 * (n + 1) / max_val:.2f}% complete ' + cus_str,
          end="\r") 
    
    
def low_cloud_adj(ceres_syn, include_mid=True):
    """
    Calculates adjusted low cloud amount (%) assuming random overlap with
    higher clouds.
    """
    high_area = ceres_syn['cldarea_high_mon']

    if include_mid:
        # Not +=: that would modify cldarea_high_mon in ceres_syn in place
        high_area = high_area + ceres_syn['cldarea_mid_low_mon'] +\
            ceres_syn['cldarea_mid_high_mon']

    denom = 100 - high_area

    low_adj = ceres_syn['cldarea_low_mon'] / denom

    return xr.where(denom > 1, 100 * low_adj, np.nan)


# CERES-FBCT cloud-top pressure bins (press index): 0-1 have CTP > 680 hPa
FBCT_LOW_PRESS  = [0, 1]
FBCT_HIGH_PRESS = [2, 3, 4, 5, 6]
FBCT_DIRS = {'terra': 'raw_data/ceres_fbct_terra',
             'noaa':  'raw_data/ceres_fbct_noaa'}


def _fbct_cf(directory):
    """FBCT cloud area [%] (opt, press, time, lat, lon) for one record."""
    import glob, os
    files = sorted(glob.glob(os.path.join(directory, '*.nc')))
    cf = xr.open_mfdataset(files, combine='by_coords')['cldarea_cldtyp_mon']
    return cf.where(cf != -999.0)


def _fbct_sums(directory):
    """Low (L) and upper-level (U) cloud area [%] from one FBCT record."""
    cf = _fbct_cf(directory)
    return xr.Dataset({
        'L': cf.isel(press=FBCT_LOW_PRESS).sum(['opt', 'press'], min_count=1),
        'U': cf.isel(press=FBCT_HIGH_PRESS).sum(['opt', 'press'], min_count=1),
    }).load()


def _merge_fbct_records(terra, noaa):
    """
    Join Terra/Aqua MODIS and NOAA-20 VIIRS: remove the NOAA-minus-Terra
    mean difference for each calendar month over their overlap from NOAA,
    then average the two where both exist. Returns (merged, n_overlap).
    """
    overlap = np.intersect1d(terra['time'], noaa['time'])
    diff = (noaa.sel(time=overlap) - terra.sel(time=overlap))
    offset = diff.groupby('time.month').mean('time')
    noaa = (noaa.groupby('time.month') - offset).drop_vars('month')

    terra, noaa = xr.align(terra, noaa, join='outer')
    out = xr.concat([terra, noaa], dim='sat').mean('sat', skipna=True)
    return out.clip(0, 100), len(overlap)  # offset removal can pass 100


def fbct_press_profile(cache='clean_data/fbct_press_1deg.nc'):
    """
    CERES-FBCT cloud area [%] in each of the 7 cloud-top pressure layers,
    summed over optical depth, on the native 1° grid (time, press, lat, lon).

    Layers (press index): 0: 1000-800, 1: 800-680, 2: 680-560,
    3: 560-440, 4: 440-310, 5: 310-180, 6: 180-10 hPa. Terra and NOAA
    records merged as in fbct_low_high. Cached.
    """
    import os
    if os.path.isfile(cache):
        return xr.open_dataset(cache)['cf']
    recs = [_fbct_cf(FBCT_DIRS[s]).sum('opt', min_count=1).load()
            .to_dataset(name='cf') for s in ('terra', 'noaa')]
    out, _ = _merge_fbct_records(*recs)
    out['cf'].attrs = {'units': '%', 'long_name': 'FBCT cloud area by CTP layer'}
    out.to_netcdf(cache)
    return out['cf']


def fbct_low_high(cache='clean_data/fbct_low_high_1deg.nc', u_cap=90.0):
    """
    Low and high cloud cover [%] on the native 1° grid from CERES-FBCT,
    following Scott et al. (2020):

        U   : upper-level cloud, all cloud with CTP < 680 hPa
        L   : low cloud visible from above, CTP > 680 hPa
        L_n : nonobscured low cloud, L / (1 - U) (random overlap)

    U is capped at u_cap (as HIGH_CF_CAP in clean_fbct_new.py) in the L_n
    denominator only.

    The Terra/Aqua MODIS (2002-07 to 2023-02) and NOAA-20 VIIRS (2018-05
    onward) records are joined by removing the NOAA-minus-Terra mean
    difference for each calendar month over their overlap from NOAA, then
    averaging the two where both exist. Cached because the raw files are
    several GB.

    Returns
    -------
    xr.Dataset (time, lat, lon) with 'L', 'U', 'L_n'
    """
    import os
    if os.path.isfile(cache):
        return xr.open_dataset(cache)

    terra = _fbct_sums(FBCT_DIRS['terra'])
    noaa  = _fbct_sums(FBCT_DIRS['noaa'])

    out, n_overlap = _merge_fbct_records(terra, noaa)
    out['L_n'] = (100 * out['L'] / (100 - out['U'].clip(max=u_cap))).clip(max=100)
    for v, name in (('L', 'low cloud (CTP > 680 hPa), visible'),
                    ('U', 'upper-level cloud (CTP < 680 hPa)'),
                    ('L_n', 'nonobscured low cloud L/(1-U), S20')):
        out[v].attrs = {'units': '%', 'long_name': name}
    out.attrs['overlap_months'] = int(n_overlap)
    out.to_netcdf(cache)
    return out


def plot_scalar_field(data, title='', cbar_lab='',
                      levels=4, to='', cent_lon=180):
    """
    Contour plot of a scalar field by providing the data directly.
    """
    era5 = data.fillna(0).copy()
    proj = ccrs.PlateCarree(central_longitude=cent_lon)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    # Ensure longitude wraps correctly if in 0-360 range
    # if era5.lon.max() > 180:
    #     era5 = era5.assign_coords(lon=(((era5.lon + 180) % 360) - 180))
    #     era5 = era5.sortby('lon')

    lon = era5.lon.values
    lat = era5.lat.values
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Improved color normalization
    vmin, vmax = np.nanpercentile(era5.values, [0.5, 99.5])  # Robust scaling
    if vmin >= 0:
        norm = TwoSlopeNorm(vmin=vmin,
                            vcenter=(vmin+vmax)/2, vmax=vmax)
        cmap = 'Reds'
    elif vmax <= 0:
        norm = TwoSlopeNorm(vmin=vmin,
                            vcenter=(vmin+vmax)/2, vmax=vmax)
        cmap = 'Blues_r'
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'

    fig, ax = plt.subplots(figsize=(10, 5), dpi=600,
                                     subplot_kw={'projection': proj})
    ax.set_global()
    ax.set_title(title)

    # pcolormesh plot
    pcm = ax.pcolormesh(lon2d, lat2d, era5, transform=ccrs.PlateCarree(), 
                        shading='nearest', cmap=cmap, norm=norm)

    # Contour overlay
    levels = np.linspace(vmin, vmax, levels)  # Define contour levels
    #contour = ax.contour(lon2d, lat2d, era5, levels=levels, 
    #                     colors='black', linewidths=0.8, 
    #                     transform=ccrs.PlateCarree())
    lon1d = np.asarray(lon2d).reshape(-1)
    lat1d = np.asarray(lat2d).reshape(-1)
    era1d = np.asarray(era5).reshape(-1)
    contour = ax.tricontour(lon1d, lat1d, era1d, levels=levels, 
                        colors='black', linewidths=0.8, 
                      transform=ccrs.PlateCarree())
    ax.clabel(contour, inline=True, fontsize=8)

    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical',
                        pad=0.05, shrink=0.65, format='%02d')
    cbar.set_label(cbar_lab)

    if to:
        fig.savefig(f'figures\saves\{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)      
    plt.show()


def get_stratocumulus_regions():
    """
    Returns bounding box definitions for five canonical marine low cloud
    regions, defined to match climatologically high low cloud cover. Regions
    were determined by eyeballing areas with high low cloud cover in 
    low_cloud_clim.py
    
    Returns
    -------
    dict of dict, each with keys: min_lon, max_lon, min_lat, max_lat
    Compatible with xarray .sel(latitude=slice(...), longitude=slice(...))
    via dictionary unpacking.
    """
    regions = {
        'NEP': {  # Northeast Pacific
            'min_lon': 210, 'max_lon': 245,
            'min_lat': 15,  'max_lat': 40
        },
        'SEP': {  # Southeast Pacific
            'min_lon': 250, 'max_lon': 290,
            'min_lat': -30, 'max_lat': -5
        },
        'NEA': {  # Northeast Atlantic
            'min_lon': 305, 'max_lon': 340,
            'min_lat': 10,  'max_lat': 35
        },
        'SEA': {  # Southeast Atlantic
            'min_lon': 340, 'max_lon': 375,
            'min_lat': -30, 'max_lat': -5
        },
        'SEI': {  # Southeast Indian
            'min_lon': 75,  'max_lon': 110,
            'min_lat': -45, 'max_lat': -20
        },
    }
    return regions


def region_sel(ds, region_dict, lon_dim='lon', lat_dim='lat'):
    """
    Selects a region from a dataset given a bounding box dictionary with
    keys min_lon, max_lon, min_lat, max_lat. Handles wrapping across the
    0/360 degree longitude boundary automatically.

    Parameters
    ----------
    ds : xarray.Dataset or DataArray
    region_dict : dict with keys min_lon, max_lon, min_lat, max_lat
    lon_dim : str, name of longitude dimension in ds
    lat_dim : str, name of latitude dimension in ds

    Returns
    -------
    xarray.Dataset or DataArray, subset to the requested region
    """
    min_lon = region_dict['min_lon']
    max_lon = region_dict['max_lon']
    min_lat = region_dict['min_lat']
    max_lat = region_dict['max_lat']

    lat_sel = ds.sel({lat_dim: slice(min_lat, max_lat)})

    if max_lon > ds[lon_dim].values.max():
        # Box wraps across 0/360 boundary - split and concatenate
        max_lon_wrapped = max_lon % 360
        left  = lat_sel.sel({lon_dim: slice(min_lon, ds[lon_dim].values.max())})
        right = lat_sel.sel({lon_dim: slice(ds[lon_dim].values.min(), 
                                            max_lon_wrapped)})

        return xr.concat([left, right], dim=lon_dim)
    else:
        return lat_sel.sel({lon_dim: slice(min_lon, max_lon)})
    
