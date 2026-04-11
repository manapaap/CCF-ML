# -*- coding: utf-8 -*-
"""
Assessing Climatological Extent of Low Cloud Cover

This will help ideitify regions within which we will train machine learning
models. 
"""


import xarray as xr
import matplotlib.pyplot as plt
from os import chdir
import pandas as pd
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as patches


chdir('C:/Users/aakas/Documents/CCF-ML/')
import scripts.utils as utils


def plot_field_patches(data, title='', cbar_lab='',
                      levels=4, to='', cent_lon=180):
    """
    Contour plot of a scalar field by providing the data directly. Also
    has the addition of drawing domains over the data to highlight key regions
    of intrest
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
    gl = ax.gridlines(draw_labels=True, zorder=5, alpha=0.75)
    gl.right_labels = False
    gl.top_labels = False

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical',
                        pad=0.05, shrink=0.65, format='%02d')
    cbar.set_label(cbar_lab)
    
    rect1 = patches.Rectangle((210, 15), 35, 25, linewidth=2,
                             edgecolor='cyan', facecolor='lightcoral',
                             label='NEP', alpha=0.9, fill=False,
                             linestyle='dashed', 
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect1)
    
    rect2 = patches.Rectangle((250, -30), 40, 25, linewidth=2,
                             edgecolor='lime', facecolor='lightcoral',
                             label='SEP', alpha=0.9, fill=False,
                             linestyle='dashed', 
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect2)
    
    rect3 = patches.Rectangle((305, 10), 35, 25, linewidth=2,
                             edgecolor='indigo', facecolor='lightcoral',
                             label='NEA', alpha=0.9, fill=False,
                             linestyle='dashed', 
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect3)
    
    rect4 = patches.Rectangle((340, -30), 35, 25, linewidth=2,
                             edgecolor='dodgerblue', facecolor='lightcoral',
                             label='SEA', alpha=0.9, fill=False,
                             linestyle='dashed', 
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect4)
    
    rect5 = patches.Rectangle((75, -45), 35, 25, linewidth=2,
                             edgecolor='darkgrey', facecolor='lightcoral',
                             label='SEI', alpha=0.9, fill=False,
                             linestyle='dashed', 
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect5)
    
    plt.legend(loc='lower right', ncols=5)

    if to:
        fig.savefig(f'figures\saves\{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)      
    plt.show()


def main():
    ceres_syn = xr.load_dataset('raw_data/ceres_syn.nc')
    ceres_syn['time'] = ceres_syn['time'] - pd.Timedelta(days=14) 
    ceres_syn['cldarea_low_adj'] = utils.low_cloud_adj(ceres_syn)
    # Take climatology
    clim = ceres_syn.mean(dim='time')
    utils.plot_scalar_field(clim['cldarea_low_adj'], cent_lon=0,
                            title='Climatological Low Cloud Cover')
    utils.plot_scalar_field(clim['cldarea_low_adj'] >= 50, cent_lon=0,
                            title='Low Cloud Cover > 50%')
    # Vibes based boxes
    plot_field_patches(clim['cldarea_low_adj'], cent_lon=0,
                       title='Climatological Low Cloud Cover', cbar_lab='%')
    # This information is now in utils.py
    # as the function get_stratocumulus_regions()
    # there is also a helper function to select data- region_sel()


if __name__ == "__main__":
    main()

