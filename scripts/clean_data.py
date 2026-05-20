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
from sys import platform
import xesmf as xe


if platform == 'win32':
    chdir('C:/Users/aakas/Documents/CCF-ML/')
else:
    chdir('/mnt/c/Users/aakas/Documents/CCF-ML/')    

import scripts.utils as utils



# =========================
# Constants (exact structure from climlab.utils.constants)
# =========================
cp = 1004.               # specific heat of dry air at constant pressure (J/kg/K)
Rd = 287.                # gas constant for dry air (J/kg/K)
Rv = 461.5               # gas constant for water vapor (J/kg/K)
g = 9.8                  # gravitational acceleration (m/s^2)
Lhvap = 2.5e6            # latent heat of vaporization (J/kg)
eps = Rd / Rv            # ratio of gas constants

# =========================
# Helper functions (structure preserved)
# =========================

def esat(T):
    """Saturation vapor pressure (Pa)"""
    T0 = 273.15
    es0 = 610.78
    return es0 * np.exp((Lhvap / Rv) * (1. / T0 - 1. / T))


def qsat(T, p):
    """Saturation specific humidity (kg/kg)

    climlab expects p in hPa here
    """
    es = esat(T)
    p_pa = p * 100.
    return eps * es / (p_pa - (1. - eps) * es)


def theta(T, p):
    """Potential temperature (K), p in hPa"""
    p0 = 1000.
    return T * (p0 / p)**(Rd / cp)


def Tlcl(T, RH):
    """LCL temperature (Bolton 1980)"""
    return 1. / (1. / (T - 55.) - np.log(RH) / 2840.) + 55.


def plcl(T, p, Tl):
    """LCL pressure (hPa)"""
    return p * (Tl / T)**(cp / Rd)

# =========================
# EIS (no algebraic changes)
# =========================

def EIS(Ts, T700):

    # Lower tropospheric stability
    LTS = theta(T700, 700) - theta(Ts, 1000)

    # LCL
    T_lcl = Tlcl(Ts, 0.8)
    p_lcl = plcl(Ts, 1000, T_lcl)

    # Heights (hypsometric, same structure)
    z700 = (Rd * T700 / g) * np.log(1000 / 700)
    zlcl = (Rd * Ts / g) * np.log(1000 / 700)

    # Temperature at 850 hPa (same midpoint approximation used in climlab)
    T850 = (Ts + T700) / 2.

    # Moist adiabatic lapse rate at 850 hPa (EXACT expression)
    Gammam = (g / cp * (1.0 - (1.0 + Lhvap * qsat(T850, 850.) / Rd / T850) /
                        (1.0 + Lhvap**2 * qsat(T850, 850.) / cp / Rv / T850**2)))

    # Final EIS
    return LTS - Gammam * (z700 - zlcl)



def cold_adv_periodic(era5_data):
    R_earth = 6.371e6
    deg_to_rad = np.pi / 180.0

    lat_deg = era5_data['lat']   # degrees, as stored
    lon_deg = era5_data['lon']

    dlat_rad = float(lat_deg[1] - lat_deg[0]) * deg_to_rad
    dlon_rad = float(lon_deg[1] - lon_deg[0]) * deg_to_rad

    lat_rad = lat_deg * deg_to_rad
    # dy = R_earth * dlat_rad                        # scalar, metres
    dx = R_earth * np.cos(lat_rad) * dlon_rad      # (lat,), metres
    # dSST/dy: differentiate wrt degrees, then convert
    dsst_dy = era5_data['sst'].differentiate('lat') / (R_earth * deg_to_rad)
    # dSST/dx: periodic centred diff, dx broadcasts over lon automatically
    sst = era5_data['sst']
    dsst_dx = (sst.roll(lon=-1, roll_coords=False)
             - sst.roll(lon=1,  roll_coords=False)) / (2 * dx)
    # No interpolation needed — u10/v10/sst share the same ERA5 grid
    cold_adv = -(era5_data['u10'] * dsst_dx + era5_data['v10'] * dsst_dy)
    cold_adv.attrs['units'] = 'K s-1'
    return cold_adv


def prop_adv_periodic(era5_data, era5_925, var='eis'):
    """
    "Advection" of a propert by the mean wind, reflecting a BL
    adjustment timescale
    """
    R_earth = 6.371e6
    deg_to_rad = np.pi / 180.0

    lat_deg = era5_data['lat']   # degrees, as stored
    lon_deg = era5_data['lon']

    # dlat_rad = float(lat_deg[1] - lat_deg[0]) * deg_to_rad
    dlon_rad = float(lon_deg[1] - lon_deg[0]) * deg_to_rad

    lat_rad = lat_deg * deg_to_rad
    # dy = R_earth * dlat_rad                        # scalar, metres
    dx = R_earth * np.cos(lat_rad) * dlon_rad      # (lat,), metres
    # dSST/dy: differentiate wrt degrees, then convert
    dvar_dy = era5_data[var].differentiate('lat') / (R_earth * deg_to_rad)
    # dSST/dx: periodic centred diff, dx broadcasts over lon automatically
    sst = era5_data[var]
    dvar_dx = (sst.roll(lon=-1, roll_coords=False)
             - sst.roll(lon=1,  roll_coords=False)) / (2 * dx)
    # Drop the extra unit in the pressure-level data
    u_925 = era5_925['u'].mean('pressure_level')
    v_925 = era5_925['v'].mean('pressure_level')
    var_adv = -(u_925 * dvar_dx + v_925 * dvar_dy)
    # var_adv.attrs['units'] =  era5_data[var].attrs['units'] + ' s-1'
    return var_adv


def calc_eis(era5_eis):
    """
    Calculates estimated inversion strength of dataarray and returns the same,
    per Wood, 2006. Uses climlab.utils.thermo.
    """
    t_700 = era5_eis.sel(pressure_level=700)['t']
    t_1000 = era5_eis.sel(pressure_level=1000)['t']
    # climlab EIS
    eis = EIS(t_1000, t_700)
    return eis


def deseasonalize(xr_ds):
    """
    Removes seasonal cycle by subtracting monthly mean climatology
    
    also detrends the data
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


def detrend_dim(da, dim='time', deg=1):
    # detrend along a single dimension
    for var in da.data_vars:
        p = da[var].polyfit(dim=dim, deg=deg)
        fit = xr.polyval(da[dim], p.polyfit_coefficients)
        da[var] -= fit
    return da


def main():
    global ceres_syn, era5_sing, era5_1deg
    # Load files
    # ceres_hist = xr.load_dataset('raw_data/ceres_hist.nc')
    ceres_syn = xr.load_dataset('raw_data/ceres_syn_new.nc')
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
    era5_925 = xr.load_dataset('raw_data/era5_925.nc').\
        drop_vars(['expver', 'number']).\
        rename({'valid_time': 'time',
                'latitude': 'lat',
                'longitude': 'lon'}).\
        sel({'time':era5_sing.time})
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
    # Calculate pseudo-advection terms
    era5_sing['deis_ds'] = prop_adv_periodic(era5_sing, era5_925, var='eis')
    era5_sing['drh_700_ds'] = prop_adv_periodic(era5_sing, era5_925, var='rh_700')
    era5_sing['dspeed_ds'] = prop_adv_periodic(era5_sing, era5_925, var='speed')
    era5_sing['dw_700_ds'] = prop_adv_periodic(era5_sing, era5_925, var='w_700')
    # regrid ERA5 to CERES grid
    regridder = xe.Regridder(era5_sing[['lat', 'lon']],
                             ceres_syn[['lat', 'lon']],
                             "bilinear", periodic=True)
    era5_1deg = regridder(era5_sing.copy())  
    # Transfer CERES variables of intrest
    era5_1deg['cldarea_low_adj'] = ceres_syn['cldarea_low_adj']
    era5_1deg['cldarea_high'] = ceres_syn['cldarea_high_mon']
    era5_1deg['ln_AOD'] = ceres_syn['ln_AOD']
    era5_1deg['lwp_low'] = ceres_syn['lwp_low_mon']
    # save seasonal cycle data too
    era5_1deg.to_netcdf('clean_data/ccf_clouds_raw.nc')
    # deseasonalize 
    era5_1deg = deseasonalize(era5_1deg)
    # Detrend
    era5_1deg = detrend_dim(era5_1deg)
    # Save Data
    era5_1deg.to_netcdf('clean_data/ccf_clouds_clean.nc')
    
    # In clean_fbct, we got cleaned low cloud CRE
    # Will the model results be any different?
    low_cre = xr.open_dataset('clean_data/low_cloud_cre_terra.nc').\
        drop_vars(['albcs', 'month'])
    low_cre['time'] = low_cre['time'] - pd.Timedelta(days=14) 
    # regrid era5 to 2.5 degree and transfer variables
    regridder = xe.Regridder(era5_1deg[['lat', 'lon']],
                             low_cre[['lat', 'lon']],
                             "bilinear", periodic=True)
    era5_25deg = regridder(era5_1deg.copy()) 
    era5_25deg = era5_25deg.sel(time=low_cre.time)
    era5_25deg['dCRE_net'] = low_cre['dCRE_net']
    era5_25deg['dCRE_amt'] = low_cre['dCRE_amount']
    era5_25deg['dCRE_tau'] = low_cre['dCRE_tau']
    era5_25deg['dCRE_alt'] = low_cre['dCRE_altitude']
    # Save final file
    era5_25deg.to_netcdf('clean_data/ccf_cre_clean.nc')
    
    
if __name__ == '__main__':
    main()
