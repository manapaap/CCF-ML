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
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        fig.savefig(f'figures/saves/{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)      
    plt.show()


def plot_field_patches_4panel(data, eis, w_700, fig_label='',
                               levels=8, to='', cent_lon=180):
    proj = ccrs.PlateCarree(central_longitude=cent_lon)

    datasets = [data, eis, w_700]
    panel_labels = ['(a)', '(b)', '(c)']
    cbar_labels = ['Low Cloud Cover', 'EIS', 'ω₇₀₀ ']
    units = ['%', 'K', 'Pa s⁻¹']

    fig, axes = plt.subplots(2, 2, figsize=(12, 4), dpi=300,
                             subplot_kw={'projection': proj},
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.15})
    axes = axes.flatten()

    regions = [
        (210, 15,  35, 25, 'cyan',       'NEP'),
        (250, -30, 40, 25, 'lime',       'SEP'),
        (305, 10,  35, 25, 'indigo',     'NEA'),
        (340, -30, 35, 25, 'dodgerblue', 'SEA'),
        (75,  -45, 35, 25, 'darkgrey',   'SEI'),
    ]

    for i, (ax, ds, plabel, clab, unit) in enumerate(
            zip(axes, datasets, panel_labels, cbar_labels, units)):

        era5 = ds.fillna(0).copy()
        lon = era5.lon.values
        lat = era5.lat.values
        lon2d, lat2d = np.meshgrid(lon, lat)

        vmin, vmax = np.nanpercentile(era5.values, [0.5, 99.5])
        if vmin >= 0:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
            cmap = 'Reds'
        elif vmax <= 0:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
            cmap = 'Blues_r'
        else:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = 'RdBu_r'

        ax.set_global()
        ax.set_ylim((-60, 60))
        pcm = ax.pcolormesh(lon2d, lat2d, era5,
                            transform=ccrs.PlateCarree(),
                            shading='nearest', cmap=cmap, norm=norm)

        lon1d = lon2d.reshape(-1)
        lat1d = lat2d.reshape(-1)
        era1d = np.asarray(era5).reshape(-1)
        contour_levels = np.linspace(vmin, vmax, levels)
        contour = ax.tricontour(lon1d, lat1d, era1d,
                                levels=contour_levels,
                                colors='black', linewidths=0.6,
                                transform=ccrs.PlateCarree())
        ax.clabel(contour, inline=True, fontsize=7)

        ax.coastlines(linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, zorder=5, alpha=0.6,
                          linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False
        gl.left_labels = (i % 2 == 0)
        gl.bottom_labels = (i >= 2)

        # Colorbar matched to axes height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05,
                                  axes_class=plt.Axes)
        cbar = fig.colorbar(pcm, cax=cax, format='%.2g')
        cbar.set_label(unit, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(f'{plabel} {clab}')

        legend_handles = []
        for (lon0, lat0, dlon, dlat, color, label) in regions:
            rect = patches.Rectangle(
                (lon0, lat0), dlon, dlat,
                linewidth=1.8, edgecolor=color, facecolor='none',
                linestyle='dashed', transform=ccrs.PlateCarree(), zorder=10)
            ax.add_patch(rect)
            legend_handles.append(
                mlines.Line2D([], [], color=color, linewidth=1.8,
                              linestyle='dashed', label=label))

    ax4_pos = axes[3].get_position()
    axes[3].set_visible(False)

    # Create a plain axes in the same spot
    legend_ax = fig.add_axes(ax4_pos)
    legend_ax.set_axis_off()
    
    # Draw proxy artists directly into that axes
    for (lon0, lat0, dlon, dlat, color, label) in regions:
        legend_ax.plot([], [], color=color, linewidth=1.8,
                       linestyle='dashed', label=label)
    
    legend_ax.legend(loc='center', ncols=1, fontsize=11,
                     framealpha=0.8, title='Regions', title_fontsize=11)
    axes[3].set_visible(False)

    if fig_label:
        fig.text(0.5, -0.06, fig_label,
                 ha='center', va='top', fontsize=9, wrap=True)

    if to:
        fig.savefig(f'figures/saves/{to}.png', dpi=300,
                    bbox_inches='tight', pad_inches=0.1)
    plt.show()


def main():
    era5_1deg = xr.open_dataset('clean_data/ccf_clouds_raw.nc')
    
    # Take climatology
    clim = era5_1deg.mean(dim='time')
    utils.plot_scalar_field(clim['cldarea_low_adj'], cent_lon=0,
                            title='Climatological Low Cloud Cover')
    utils.plot_scalar_field(clim['cldarea_low_adj'] >= 50, cent_lon=0,
                            title='Low Cloud Cover > 50%')
    # Vibes based boxes
    plot_field_patches(clim['cldarea_low_adj'], cent_lon=0,
                       title='Climatological Low Cloud Cover', cbar_lab='%')
    plot_field_patches_4panel(clim['cldarea_low_adj'], 
                              clim['eis'], clim['w_700'], cent_lon=0)
    # This information is now in utils.py
    # as the function get_stratocumulus_regions()
    # there is also a helper function to select data- region_sel()


if __name__ == "__main__":
    main()

