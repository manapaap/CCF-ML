# -*- coding: utf-8 -*-
"""
regime_masks.py

Builds low cloud regime masks on the 2.5° analysis grid from raw ERA5
monthly means, following:

    MN13 : Myers & Norris (2013), J. Climate 26, 7507-7524
           Ocean grid boxes within 30°S-30°N with long-term mean
           ω700 > 0 for every calendar month AND monthly mean ω700 > 0 for
           at least 80% of the record. Only months with ω700 > 0 are later
           analysed (saved here as the time-varying `mn13_active`).

    S20  : Scott et al. (2020), J. Climate 33, 7717-7734
           Ocean grid boxes within 60°S-60°N classified with ANNUAL MEAN
           climatological thresholds:
             stratocumulus : ω700 > 15 hPa/day and EIS > 1 K
             trade cumulus : ω700 > 0  hPa/day and EIS < 1 K

Workflow
--------
1. Load ERA5 SST, 700 hPa temperature and 700 hPa ω one year at a time
   (era5_pres.nc is too large to hold in memory at once)
2. Compute EIS with the EIS function from clean_data.py, but with SST as
   the surface temperature as in Wood & Bretherton (2006), MN13 and S20.
   (clean_data.py uses 1000 hPa air temperature, which is ~2.3 K colder
   than SST over 60S-60N oceans and raises EIS by ~2.6 K. That bias makes
   the absolute S20 threshold EIS > 1 K select far too many grid boxes.)
3. Conservatively regrid EIS and ω700 from 0.25° to the 2.5° grid of
   clean_data/ccf_cre_clean.nc
4. Build the monthly climatology, annual mean, and the three masks
5. Save everything to clean_data/regime_masks.nc and plot the masks

Run from the CCF-ML root (xesmf needs ESMFMKFILE set if the conda env
is not activated):
    python -m scripts.regime_masks
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cutil
import xesmf as xe
from matplotlib.colors import TwoSlopeNorm
from string import ascii_lowercase as lowers

os.chdir('C:/Users/aakas/Documents/CCF-ML/')

from scripts.clean_data import EIS


# ═════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════

SINGLE_FILE = 'raw_data/era5_single.nc'
PRES_FILE   = 'raw_data/era5_pres.nc'
GRID_FILE   = 'clean_data/ccf_cre_clean.nc'   # defines the 2.5° target grid
OUT_FILE    = 'clean_data/regime_masks.nc'
FIG_PATH    = 'figures/regime_masks.png'

CHUNK_MONTHS   = 12      # months of ERA5 processed per pass
OCEAN_FRAC_MIN = 0.5     # min ocean fraction for a 2.5° cell to count as ocean

PA_S_TO_HPA_DAY = 864.0

# MN13
MN13_LAT_MAX     = 30.0
MN13_ACTIVE_FRAC = 0.8   # fraction of months with ω700 > 0

# S20
S20_LAT_MAX  = 60.0
S20_SC_W700  = 15.0      # hPa/day
S20_EIS_THR  = 1.0       # K
S20_CU_W700  = 0.0       # hPa/day

MASK_LABELS = {
    'mask_mn13':   'MN13 subsidence regime',
    'mask_s20_sc': 'S20 stratocumulus',
    'mask_s20_cu': 'S20 trade cumulus',
}

# ═════════════════════════════════════════════


def _era5_rename(ds):
    return (ds.drop_vars([v for v in ('expver', 'number') if v in ds.coords])
              .rename({'valid_time': 'time',
                       'latitude':   'lat',
                       'longitude':  'lon'})
              .sortby('lat'))


def build_regridder(src, target):
    """
    Conservative regridder from the ERA5 0.25° grid to the 2.5° target grid.
    Cell bounds are built explicitly; ERA5 polar edges are clipped to ±90.
    """
    def with_bounds(lat, lon):
        dlat = float(lat[1] - lat[0])
        dlon = float(lon[1] - lon[0])
        lat_b = np.clip(np.append(lat - dlat / 2, lat[-1] + dlat / 2),
                        -90, 90)
        lon_b = np.append(lon - dlon / 2, lon[-1] + dlon / 2)
        return xr.Dataset(coords={'lat': lat, 'lon': lon,
                                  'lat_b': lat_b, 'lon_b': lon_b})

    grid_in  = with_bounds(src['lat'].values,    src['lon'].values)
    grid_out = with_bounds(target['lat'].values, target['lon'].values)
    return xe.Regridder(grid_in, grid_out, 'conservative')


def load_eis_w700(regridder, era5_sing, era5_pres):
    """
    Computes EIS (surface temperature = SST) and ω700 (hPa/day) in chunks
    of months and regrids each chunk to 2.5°. EIS is NaN over land; cells
    that are more than half land stay NaN after regridding.

    Returns
    -------
    ds : xr.Dataset (time, lat, lon) with 'eis' [K] and 'w_700' [hPa/day]
    """
    times  = era5_sing['time'].values
    chunks = []
    for start in range(0, len(times), CHUNK_MONTHS):
        t_sel = times[start:start + CHUNK_MONTHS]
        print(f'  {str(t_sel[0])[:7]} to {str(t_sel[-1])[:7]}', flush=True)

        pres = era5_pres.sel(time=t_sel)
        sst  = era5_sing['sst'].sel(time=t_sel).load()
        t700 = pres['t'].sel(pressure_level=700).load()
        w700 = pres['w'].sel(pressure_level=700).load()

        chunk = xr.Dataset({
            'eis':   EIS(sst, t700),
            'w_700': w700 * PA_S_TO_HPA_DAY,
        }).drop_vars('pressure_level')
        chunks.append(regridder(chunk, skipna=True, na_thres=0.5))

    ds = xr.concat(chunks, dim='time')
    ds['eis'].attrs   = {'units': 'K',       'long_name': 'EIS (Wood & Bretherton 2006), Ts = SST'}
    ds['w_700'].attrs = {'units': 'hPa/day', 'long_name': '700 hPa pressure velocity'}
    return ds


def ocean_fraction(regridder, era5_sing):
    """ERA5 SST is NaN over land; its valid fraction is the ocean fraction."""
    is_ocean = era5_sing['sst'].isel(time=0).notnull().astype('float64')
    return regridder(is_ocean).rename('ocean_frac')


def build_masks(ds, ocean_frac):
    """
    Returns
    -------
    masks : xr.Dataset with static masks (lat, lon) and the time-varying
            MN13 active-month mask (time, lat, lon)
    """
    clim_month = ds.groupby('time.month').mean('time')
    clim_ann   = clim_month.mean('month')

    ocean   = ocean_frac >= OCEAN_FRAC_MIN
    abs_lat = np.abs(ds['lat'])

    # MN13
    subs_every_month = (clim_month['w_700'] > 0).all('month')
    subs_frac        = (ds['w_700'] > 0).mean('time')
    mask_mn13 = (ocean & (abs_lat <= MN13_LAT_MAX) & subs_every_month
                 & (subs_frac >= MN13_ACTIVE_FRAC))
    mn13_active = mask_mn13 & (ds['w_700'] > 0)

    # S20
    in_s20 = ocean & (abs_lat <= S20_LAT_MAX)
    mask_s20_sc = (in_s20 & (clim_ann['w_700'] > S20_SC_W700)
                   & (clim_ann['eis'] > S20_EIS_THR))
    mask_s20_cu = (in_s20 & (clim_ann['w_700'] > S20_CU_W700)
                   & (clim_ann['eis'] < S20_EIS_THR))

    masks = xr.Dataset({
        'mask_mn13':   mask_mn13,
        'mask_s20_sc': mask_s20_sc,
        'mask_s20_cu': mask_s20_cu,
        'mn13_active': mn13_active,
        'ocean_frac':  ocean_frac,
        'eis_clim':    clim_month['eis'],
        'w_700_clim':  clim_month['w_700'],
        'eis_ann':     clim_ann['eis'],
        'w_700_ann':   clim_ann['w_700'],
    })
    for name, label in MASK_LABELS.items():
        masks[name].attrs['long_name'] = label
    masks['mn13_active'].attrs['long_name'] = \
        'MN13 mask AND monthly w_700 > 0 (months MN13 analysed)'
    return masks


def area_fraction(mask):
    """Fraction of global surface area covered by a (lat, lon) mask."""
    w = np.cos(np.deg2rad(mask['lat'])) * xr.ones_like(mask, dtype=float)
    return float((w * mask).sum() / w.sum())


def plot_masks(masks, figsize=(10, 11)):
    """
    One row per mask. Shading is annual mean ω700; selected grid boxes are
    hatched. S20 panels add the EIS = 1 K contour.
    """
    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(len(MASK_LABELS), 1, figsize=figsize,
                             subplot_kw={'projection': proj})

    w_ann, lon_c = cutil.add_cyclic_point(masks['w_700_ann'].values,
                                          coord=masks['lon'].values)
    eis_ann, _   = cutil.add_cyclic_point(masks['eis_ann'].values,
                                          coord=masks['lon'].values)
    lat = masks['lat'].values
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)

    for j, (ax, (name, label)) in enumerate(zip(axes, MASK_LABELS.items())):
        cf = ax.contourf(lon_c, lat, w_ann, levels=np.arange(-60, 61, 10),
                         cmap='RdBu_r', norm=norm, extend='both',
                         transform=data_crs)

        mask_c, _ = cutil.add_cyclic_point(masks[name].values.astype(float),
                                           coord=masks['lon'].values)
        ax.contourf(lon_c, lat, mask_c, levels=[0.5, 1.5], colors='none',
                    hatches=['////'], transform=data_crs)
        ax.contour(lon_c, lat, mask_c, levels=[0.5], colors='k',
                   linewidths=0.8, transform=data_crs)

        if name.startswith('mask_s20'):
            ax.contour(lon_c, lat, eis_ann, levels=[S20_EIS_THR],
                       colors='darkgreen', linewidths=1.2, linestyles='--',
                       transform=data_crs)

        lat_max = MN13_LAT_MAX if name == 'mask_mn13' else S20_LAT_MAX
        for lat_line in (-lat_max, lat_max):
            ax.plot([0, 360], [lat_line, lat_line], color='k', lw=0.8,
                    ls=':', transform=data_crs)

        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=2)
        ax.coastlines(linewidth=0.5, zorder=3)
        ax.set_extent([0, 359.9, -65, 65], crs=data_crs)
        ax.set_title(f'{lowers[j]}) {label} '
                     f'({100 * area_fraction(masks[name]):.1f}% of globe)',
                     fontsize=13)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = gl.right_labels = False

    cbar = fig.colorbar(cf, ax=axes, orientation='horizontal',
                        fraction=0.04, pad=0.05, aspect=40)
    cbar.set_label('Annual mean ω$_{700}$ (hPa/day)', fontsize=12)
    return fig, axes


def main():
    target = xr.open_dataset(GRID_FILE)[['lat', 'lon']]

    era5_sing = _era5_rename(xr.open_dataset(SINGLE_FILE))
    era5_pres = _era5_rename(xr.open_dataset(PRES_FILE))

    print('Building conservative regridder ...', flush=True)
    regridder = build_regridder(era5_sing, target)

    print('Computing EIS and w_700 ...', flush=True)
    ds = load_eis_w700(regridder, era5_sing, era5_pres)
    ocean_frac = ocean_fraction(regridder, era5_sing)

    masks = build_masks(ds, ocean_frac)
    for name in MASK_LABELS:
        print(f'{name}: {int(masks[name].sum())} grid boxes, '
              f'{100 * area_fraction(masks[name]):.1f}% of globe')

    out = xr.merge([ds, masks])
    for name in list(MASK_LABELS) + ['mn13_active']:
        out[name] = out[name].astype('int8')
    out.to_netcdf(OUT_FILE)
    print(f'Saved {OUT_FILE}')

    fig, _ = plot_masks(masks)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    print(f'Saved {FIG_PATH}')
    plt.show()


if __name__ == '__main__':
    main()
