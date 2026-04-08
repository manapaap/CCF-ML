# -*- coding: utf-8 -*-
"""
Deseasonalizes ERA5 and CERES data relevant to the project. 
Regrids the ERA5 variables to the CERES grid for consistency. 
Merges all data into a single dataarray for convenience. 
"""


import xarray as xr
import numpy as np
from os import chdir
import pandas as pd
from climlab.utils.thermo import EIS
from sys import platform
import xesmf as xe


if platform == 'win32':
    chdir('C:/Users/aakas/Documents/CCF-ML/')
else:
    chdir('/mnt/c/Users/aakas/Documents/CCF-ML/')    

import scripts.utils as utils


def cold_adv_periodic(era5_data):
    """
    Calculates cold air advection using u10/v10 and SST gradients,
    using periodic longitude derivatives to avoid 180° artifacts.
    """
    R_earth = 6.371e6
    deg_to_rad = np.pi / 180.0
    lat = era5_data['lat'] * deg_to_rad
    lon = era5_data['lon'] * deg_to_rad
    # Assume regular lat/lon spacing
    dlat = float(lat[1] - lat[0])  # radians
    dlon = float(lon[1] - lon[0])  # radians
    # Grid spacing
    dy = R_earth * dlat  # constant
    dx = R_earth * np.cos(lat) * dlon  # varies with lat, shape (lat,)
    # -- Latitude derivative (standard centered diff) --
    dsst_dy = era5_data['sst'].differentiate('lat') / dy  # shape: (time, lat, lon)
    # -- Longitude derivative: periodic central diff --
    sst = era5_data['sst']
    lon_dim = 'lon'
    lat_dim = 'lat'
    sst_roll_p1 = sst.roll({lon_dim: -1}, roll_coords=False)
    sst_roll_m1 = sst.roll({lon_dim: 1}, roll_coords=False)
    # dx is (lat,) so we need to broadcast to match (lat, lon)
    dx_broadcast = dx.broadcast_like(sst.isel({lon_dim: 0}))  # shape (lat, lon)
    dsst_dx = (sst_roll_p1 - sst_roll_m1) / (2 * dx_broadcast)
    # -- Interpolate wind to match SST gradients --
    u_mid = era5_data['u10'].interp(lat=dsst_dy.lat, lon=dsst_dx.lon)
    v_mid = era5_data['v10'].interp(lat=dsst_dy.lat, lon=dsst_dx.lon)
    # -- Compute cold air advection --
    cold_adv = -(u_mid * dsst_dx + v_mid * dsst_dy)

    return cold_adv


def calc_eis(era5_eis):
    """
    Calculates estimated inversion strength of dataarray and returns the same,
    per Wood, 2006. Uses climlab.utils.thermo.
    """
    t_700 = era5_eis.sel(pressure_level=700)['t']
    t_1000 = era5_eis.sel(pressure_level=1000)['t']
    # climlab EIS
    eis = EIS(t_1000, t_700)
    return eis.drop_vars('pressure_level')


def deseasonalize(xr_ds):
    """
    Removes seasonal cycle by subtracting monthly mean climatology
    """
    clim = xr_ds.groupby('time.month').mean(dim='time')
    # deas
    years = xr_ds.time.dt.year
    months = xr_ds.time.dt.month
    num = len(years)
    # Anomaly time series
    for n, (year, month) in enumerate(zip(years, months)):
        utils.progress_bar(n, num, 
                           f'deseasonalizing...{int(year)}-{int(month)}')     
        xr_ds[{"time": n }] -= clim.sel(month=month)
    return xr_ds


def main():
    # global ceres_syn, era5_sing, era5_1deg
    # Load files
    # ceres_hist = xr.load_dataset('raw_data/ceres_hist.nc')
    ceres_syn = xr.load_dataset('raw_data/ceres_syn.nc')
    era5_pres = xr.load_dataset('raw_data/era5_pres.nc').\
        drop_vars(['expver', 'number']).\
        rename({'valid_time': 'time',
                'latitude': 'lat',
                'longitude': 'lon'})
    era5_sing = xr.load_dataset('raw_data/era5_single.nc').\
        drop_vars(['expver', 'number']).\
        rename({'valid_time': 'time',
                'latitude': 'lat',
                'longitude': 'lon'})
    # Adjust ceres-syn time to start at 0
    ceres_syn['time'] = ceres_syn['time'] - pd.Timedelta(days=14) 
    # Create adjusted low cloud cover variable and ln(AOD)
    ceres_syn['cldarea_low_adj'] = utils.low_cloud_adj(ceres_syn)
    ceres_syn['ln_AOD'] = np.log(ceres_syn['ini_aod55_mon'])
    # Now, for ERA5, calculate cold advection, EIS, and WindSpeed
    era5_sing['eis'] = calc_eis(era5_pres)
    era5_sing['speed'] = np.hypot(era5_sing['u10'], era5_sing['v10'])
    era5_sing['cold_adv'] = cold_adv_periodic(era5_sing)
    era5_sing['w_700'] = era5_pres['w'].sel(pressure_level=700)
    era5_sing['rh_700'] = era5_pres['r'].sel(pressure_level=700)
    era5_sing = era5_sing.drop_vars('pressure_level')
    # deseasonalze ERA5 and CERES-SYN Data
    ceres_syn = deseasonalize(ceres_syn)
    era5_sing = deseasonalize(era5_sing)
    # regrid ERA5 to CERES grid
    regridder = xe.Regridder(era5_sing[['lat', 'lon']],
                             ceres_syn[['lat', 'lon']],
                             "bilinear", periodic=True)
    era5_1deg = regridder(era5_sing.copy())  
    # save data- only retain CCFs and cloud for now
    era5_1deg['cldarea_low_adj'] = ceres_syn['cldarea_low_adj']
    era5_1deg['cldarea_high'] = ceres_syn['cldarea_high_mon']
    era5_1deg['ln_AOD'] = ceres_syn['ln_AOD']
    era5_1deg.to_netcdf('clean_data/ccf_clouds_clean.nc')
    
    
if __name__ == '__main__':
    main()

