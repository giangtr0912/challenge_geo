from __future__ import division

# Standard library imports
import logging
import os

# Third party imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def plot_soil_profiles(df, out_dir, ignore_cols =[],  attribute=""):
    """Creates errror bars plot(s) for soil profiles."""

    params=['max', 'mean', 'min', 'std']
    if ignore_cols:
        df = df.loc[:, ~df.columns.isin(ignore_cols)]
        wave_lengh = [int(item.split('_')[1]) for item in df.columns.tolist() if 'nir_' in item]
        if attribute != "":
            for field_name in set(df[attribute].tolist()):
                plt.figure(figsize=(20, 10))
                plt.clf
                out_fig = os.path.join(out_dir, 'inspect_data/{}_soil_profiles_plot.png'.format(field_name))

                y_max, y_mean, y_min, yerr = df.loc[df[attribute] == field_name].describe().loc[params].to_numpy()
                # plt.plot(wave_lengh, y_mean, label="{}_mean".format(field_name))
                # plt.plot(wave_lengh, y_min, label="{}_min".format(field_name))
                # plt.plot(wave_lengh, y_max, label="{}_max".format(field_name))
                plt.errorbar(wave_lengh, y_mean, yerr, label=field_name, fmt="ob", capsize=1, ecolor="k")

                plt.xlabel("Wavelength [nm]", fontsize=16)
                plt.ylabel("Soil reflectance", fontsize=16)
                plt.title("Soil reflectance spectroscopy (errror bars plot), location: {}".format(field_name), fontsize=18)
                plt.legend(fontsize=18)

                plt.savefig(out_fig, dpi=350)
                plt.close('all')


def plot_som_hist(df, out_dir,  attribute="", threshold_value = np.PINF):
    """Creates histogram plot(s)."""
    
    '''
    ax = df_clean['som'][df_clean['som'] < 6].plot.hist(bins=100, alpha=0.5)
    plt.savefig('soil_profiles_plot_1.png', dpi=500)
    plt.close('all')
    '''


    if attribute != "":
        for field_name in set(df[attribute].tolist()):
            plt.figure(figsize=(20, 10))
            plt.clf
            out_fig = os.path.join(out_dir, 'inspect_data/{}_som_histogram_plot.png'.format(field_name))
            df['som'][(df['som'] < threshold_value) & (df[attribute] == field_name)].plot.hist(bins=100, alpha=0.5)

            plt.xlabel("Soil organic matter values [%]", fontsize=16)
            plt.ylabel("Number of samples", fontsize=16)
            plt.title("Histogram of soil organic matter values, location: {}".format(field_name), fontsize=18)
            plt.legend(fontsize=18)

            plt.savefig(out_fig, dpi=350)
            plt.close('all')
            
            
def plot_map(geojson_fn, out_dir):

    import geopandas as gpd
    import geoplot as gplt
    import geoplot.crs as gcrs
    import mapclassify as mc

    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    gdf = gpd.read_file(geojson_fn)
    scheme = mc.Quantiles(gdf['som'], k=5)
    
    # plt.figure(figsize=(20, 10))
    plt.clf
    out_fig = os.path.join(out_dir, 'inspect_data/map_plot.png')
    
    ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())
    # gplt.pointplot(gdf, ax=ax)
    gplt.pointplot(
    gdf, projection=gcrs.AlbersEqualArea(),
    scale='som', limits=(-1, 20),
    hue='som', cmap='viridis', scheme=scheme,
    legend=True, legend_var='hue',
    ax=ax)

    plt.savefig(out_fig, dpi=350)
    plt.close('all')


def plot_stenon_polaris(df, 
                        field_name, 
                        y_name = 'som_stenon', 
                        features_names = ['som_polaris_mean_0_5', 'som_polaris_mode_0_5', 'som_polaris_p5_0_5', 'som_polaris_p50_0_5', 'som_polaris_p95_0_5', 
                                          'som_polaris_mean_5_15', 'som_polaris_mode_5_15', 'som_polaris_p5_5_15', 'som_polaris_p50_5_15', 'som_polaris_p95_5_15', 
                                          'som_polaris_mean_15_30', 'som_polaris_mode_15_30', 'som_polaris_p5_15_30', 'som_polaris_p50_15_30', 'som_polaris_p95_15_30'], 
                        x_lim_max=None,
                        y_lim_max=None
                        ):

    g = sns.FacetGrid(pd.DataFrame(features_names), col=0, col_wrap=5, sharex=False)
    for ax, x_var in zip(g.axes, features_names):
        sns.scatterplot(data=df, x=x_var, y=y_name, ax=ax)
    g.tight_layout()

    g.tight_layout()
    if x_lim_max is not None:
        g.set(xlim=(-0.15, x_lim_max))
    if y_lim_max is not None:
        g.set(ylim=(-0.15, y_lim_max))

    #move overall title up
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('SOM (Stenon vs Polaris_median, {})'.format(field_name), fontsize=20)

    plt.savefig('./Compare_with_POLARIS_SOM_{}.png'.format(field_name))