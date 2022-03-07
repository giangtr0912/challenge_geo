# Standard library imports
import os
import glob
import math
import logging
import multiprocessing

# Third party imports
import wget


# Default POLARIS setting values
DEFAULT_POLARIS_URL = 'http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{0}/{1}/{2}/lat{3}{4}_lon{5}{6}.tif'

DEFAULT_POLARIS_LAYERS = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']
DEFAULT_POLARIS_STATISTICS = ['mean', 'mode', 'p5', 'p50', 'p95']
DEFAULT_POLARIS_PARAMETERS = ['alpha', 'bd', 'clay', 'hb', 'ksat', 'lambda', 'n', 'om', 'ph', 'sand', 'silt', 'theta_r', 'theta_s']

DEFAULT_POLARIS_WORKING_DIRECTORY = '../data/polaris'

layers = DEFAULT_POLARIS_LAYERS
statistics = DEFAULT_POLARIS_STATISTICS
parameters = DEFAULT_POLARIS_PARAMETERS
template_url = DEFAULT_POLARIS_URL

output_base_path = DEFAULT_POLARIS_WORKING_DIRECTORY
os.makedirs(output_base_path, exist_ok=True)

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


    def run_process(url, output_path):
        wget.download(url, out=output_path)


    # Generate URL path for the data that we intersted in
    url_path_lst = generate_url_path(output_base_path, domain_extent)

    cpus = multiprocessing.cpu_count()
    max_pool_size = 6
    pool = multiprocessing.Pool(cpus if cpus < max_pool_size else max_pool_size)

    for url, path in url_path_lst:
        print('Beginning file download with wget module {n}'.format(n=url))
        pool.apply_async(run_process, args=(url, path, ))

    pool.close()
    pool.join()

    print("Finish downloading data from POLARIS soild database.")
