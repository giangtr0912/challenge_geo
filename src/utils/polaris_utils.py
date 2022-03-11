# Standard library imports
import os
import glob
import math
import logging

# Third party imports
import wget
import numpy as np


# Default POLARIS setting values
DEFAULT_POLARIS_URL = 'http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{0}/{1}/{2}/lat{3}{4}_lon{5}{6}.tif'

DEFAULT_POLARIS_LAYERS = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']
DEFAULT_POLARIS_STATISTICS = ['mean', 'mode', 'p5', 'p50', 'p95']
DEFAULT_POLARIS_PARAMETERS = ['alpha', 'bd', 'clay', 'hb', 'ksat', 'lambda', 'n', 'om', 'ph', 'sand', 'silt', 'theta_r', 'theta_s']

DEFAULT_POLARIS_WORKING_DIRECTORY = './data/polaris'

layers = DEFAULT_POLARIS_LAYERS
statistics = DEFAULT_POLARIS_STATISTICS
parameters = DEFAULT_POLARIS_PARAMETERS
template_url = DEFAULT_POLARIS_URL

output_base_path = DEFAULT_POLARIS_WORKING_DIRECTORY


def download_polaris_data(area_extent, output_base_path=output_base_path, layers=layers, statistics=statistics, parameters=parameters):

    # import pdb;pdb.set_trace()
    (minLon, maxLon, minLat, maxLat) = area_extent
    domain_extent = {'lon': [math.floor(minLon), int(math.ceil(maxLon))], 'lat':[math.floor(minLat), int(math.ceil(maxLat))]}

    def generate_url_path(output_base_path, domain_extent):
        ''' TODO': Write discription'''

        url_path_lst = []
        lat_range = range(domain_extent['lat'][0],domain_extent['lat'][1])
        lon_range = range(domain_extent['lon'][0],domain_extent['lon'][1])
        for layer in layers:
            for stat in statistics:
                for param in parameters:
                    for lat in lat_range:
                        for lon in lon_range:
                            url = template_url.format(param,stat,layer,str(lat),str(lat+1),str(lon),str(lon+1))
                            temp_path = os.path.join(output_base_path, '{}/{}/{}/'.format(param,stat,layer))
                            if not os.path.exists(temp_path):
                                os.makedirs(temp_path)
                            url_path_lst += [[url, temp_path]]

        return url_path_lst

    
    # Generate URL path for the data that we intersted in
    url_path_lst = generate_url_path(output_base_path, domain_extent)

    for url, path in url_path_lst:
        print('Beginning file download with wget module {n}'.format(n=url))
        wget.download(url, out=path)

    for url, path in url_path_lst:
      if os.path.exists(path):
        print("File(s) in {} downloaded successfully from {}".format(path, url))
      else:
        print("Failed when download file {} from {}".format(path, url))

    return url_path_lst

def compute_logData(x):
   return np.log(x)

def plot_stenon_polaris(df, 
                        field_name, 
                        y_name = 'som_stenon', 
                        features_names = ['som_polaris_mean_0_5', 'som_polaris_mode_0_5', 'som_polaris_p5_0_5', 'som_polaris_p50_0_5', 'som_polaris_p95_0_5', 
                                          'som_polaris_mean_5_15', 'som_polaris_mode_5_15', 'som_polaris_p5_5_15', 'som_polaris_p50_5_15', 'som_polaris_p95_5_15', 
                                          'som_polaris_mean_15_30', 'som_polaris_mode_15_30', 'som_polaris_p5_15_30', 'som_polaris_p50_15_30', 'som_polaris_p95_15_30']
, 
                        x_lim_max=None,
                        y_lim_max=None
                        ):

  g = sns.FacetGrid(pd.DataFrame(features_names), col=0, col_wrap=5, sharex=False)
  for ax, x_var in zip(g.axes, features_names):
      sns.scatterplot(data=df, x=x_var, y=y_name, ax=ax, marker="+", color='blue')
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


def plot_stenon_polaris_0_30(df, 
                        field_name, 
                        y_name = 'som_stenon', 
                        features_names = ['som_polaris_mean_0_30', 'som_polaris_mode_0_30', 'som_polaris_p50_0_30', 'som_polaris_p95_0_30'], 
                        t_oulier=None,
                        x_lim_max=None,
                        y_lim_max=None
                        ):

  g = sns.FacetGrid(pd.DataFrame(features_names), col=0, col_wrap=2, sharex=False, size = 6)
  for ax, x_var in zip(g.axes, features_names):
    if t_oulier is not None:
      df = df[df[y_name] < t_oulier]

    sns.regplot(data=df, x=x_var, y=y_name, ax=ax, 
                scatter_kws={"color": "blue"}, line_kws={"color": "red"},
                scatter = True, ci = 0, fit_reg = True, marker="+")

    r, p = sp.stats.pearsonr(df[x_var], df[y_name])
    ax.text(.02, 1.05, 'r={:.2f}, p={:.2g}'.format(r, p), weight='bold', fontsize=15, transform=ax.transAxes)

  g.tight_layout()
  if x_lim_max is not None:
    g.set(xlim=(-0.15, x_lim_max))
  if y_lim_max is not None:
    g.set(ylim=(-0.15, y_lim_max))

  #move overall title up
  g.fig.subplots_adjust(top=0.9)
  g.fig.suptitle('SOM (Stenon vs Polaris_median, {})'.format(field_name), fontsize=20)

  plt.savefig('./Compare_with_POLARIS_SOM_{}_0_30_median_t_oulier_{}.png'.format(field_name, t_oulier))