from __future__ import division

# Standard library imports
import logging
import os

# Third party imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def plot_soil_profiles(df, ignore_cols=[],  attribute="", params=['max', 'mean', 'min', 'std']):
    """Creates plots for soil profiles."""

    if ignore_cols:
        df = df.loc[:, ~df.columns.isin(ignore_cols)]
        wave_lengh = [int(item.split('_')[1]) for item in df.columns.tolist() if 'nir_' in item]

        plt.figure(figsize=(20, 10))
        plt.clf
        if attribute != "":
            for field_name in set(df[attribute].tolist()):
                y_max, y_mean, y_min, yerr = df.loc[df[attribute] == field_name].describe().loc[params].to_numpy()
                # plt.plot(wave_lengh, y_mean, label="{}_mean".format(field_name))
                # plt.plot(wave_lengh, y_min, label="{}_min".format(field_name))
                # plt.plot(wave_lengh, y_max, label="{}_max".format(field_name))
                plt.errorbar(wave_lengh, y_mean, yerr, label=field_name)
                plt.xlabel("Wavelength [nm]", fontsize=18)
                plt.ylabel("Soil reflectance", fontsize=18)

            plt.legend()

            plt.savefig('./soil_profiles_plot.png', dpi=500)
            plt.close('all')
