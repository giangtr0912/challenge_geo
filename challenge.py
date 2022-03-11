import os
import sys
import logging
import warnings
import argparse                                                                  

import pandas as pd

from src.utils import io_utils, plot_utils, polaris_utils

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Default CHALLENGE setting values
DEFAULT_CSV_FILENAME = 'data.csv'
DEFAULT_GOJ_FILENAME = 'fields_boundary.geojson'
OUTLIER_REMOVAL_ALGORITHM_NAME = 'IQ'
DEFAULT_OM_LAYERS = ['0_5', '5_15', '15_30']
DEFAULT_OM_LAYER_STATISTICS = ['mean', 'mode', 'p5', 'p50', 'p95']
DEFAULT_POLARIS_PARAMETERS = ['om']

DEFAULT_INPUT_DIRECTORY = './data/input'
DEFAULT_OUTPUT_DIRECTORY = './data/output'
DEFAULT_OUTPUT_MODEL_DIRECTORY = './model'


class CleanData(object):
    def __init__(self, args):
        print ("Start cleanning data!")
        self.in_dir = args.in_dir
        self.out_dir = args.out_dir
        if not os.path.exists(self.out_dir):os.makedirs(self.out_dir)

        self.csv_fn = args.input_csv_fn
        self.goj_fn = args.input_vector_fn
        self.in_csv_fp = os.path.join(self.in_dir, self.csv_fn)
        self.in_goj_fp = os.path.join(self.in_dir, self.goj_fn)
        self.csv_fn_no_ext = os.path.splitext(self.csv_fn)[0]

    def data_preparation(self):
        logger.info("Step: Data preparation")
        print ("Load data from input csv file: {}".format(self.in_csv_fp))
        df = io_utils.load_data_from_csv_file(self.in_csv_fp)

        logger.info("Primary data cleaning")
        df_prm =  io_utils.primary_clean_data(df)

        logger.info("Create Lat Long columns from LatLng column")
        df_prm_ll = io_utils.create_lat_lng_cols(df_prm)

        logger.info("Convert data to vector geojson format for further inspection")
        out_goj_for_inspection_fn = os.path.join(self.out_dir, '{}_for_inspection.geojson'.format(self.csv_fn_no_ext))
        out_goj_for_inspection_obj = io_utils.pandas_to_geojson(df_prm_ll, out_goj_for_inspection_fn)
        df_prm_ll.to_csv(os.path.join(self.out_dir, '{}_for_inspection.csv'.format(self.csv_fn_no_ext)))

        df_prm_ll_correct_label = io_utils.correct_data_label(out_goj_for_inspection_obj, self.in_goj_fp)
        out_goj_label_corrected_for_inspection_fn = os.path.join(self.out_dir, \
            '{}_label_corrected_for_inspection.geojson'.format(self.csv_fn_no_ext))
        
        io_utils.pandas_to_geojson(df_prm_ll_correct_label, out_goj_label_corrected_for_inspection_fn)
        df_prm_ll_correct_label.to_csv(os.path.join(self.out_dir, \
            '{}_label_corrected_for_inspection.csv'.format(self.csv_fn_no_ext)))

        return df_prm_ll_correct_label


class PolarisData(CleanData):
    def __init__(self, args):
        super().__init__(args)
        print ("Star working with POLARIS data")

        self.pp = args.polaris_parameters
        self.pl = args.polaris_parameter_layers
        self.ps = args.polaris_parameter_layer_statistics
        print ("POLARIS data specifications: parameters {}, layers {},\
             statistics {}".format(self.pp, self.pl, self.ps))

        self.in_csv_corrected_fp = os.path.join(self.out_dir, \
            '{}_label_corrected_for_inspection.csv'.format(self.csv_fn_no_ext))
        self.in_goj_corrected_fp = os.path.join(self.out_dir, \
            '{}_label_corrected_for_inspection.geojson'.format(self.csv_fn_no_ext))

        if os.path.isfile(self.in_csv_corrected_fp) and os.path.isfile(self.in_goj_corrected_fp):
            print ('Input data: {}, {} are available! Begin with next processing steps!'\
                .format(self.in_csv_corrected_fp, self.in_goj_corrected_fp))
        else:
            super().data_preparation()


    def analyze_polaris_data(self, interest_cols=['measurement_ID', 'location', 'som']):
        logger.info("Download data from POLARIS soil database")

        df = io_utils.load_data_from_csv_file(self.in_csv_corrected_fp)
        aoi_extent = df['lng'].min(), df['lng'].max(), df['lat'].min(), df['lat'].max()

        url_path_lst = polaris_utils.download_polaris_data(aoi_extent, layers=self.pl, \
                                                           statistics=self.ps, parameters=self.pp)
        
        logger.info("Extract data from POLARIS soil database")
        in_rst_list =['{}{}'.format(item[1], os.path.basename(item[0])) for item in url_path_lst]
        out_geojson_with_polaris = os.path.join(self.out_dir, 'data_with_SOM_from_polaris.geojson')
        df_data_with_polaris_data = io_utils.ZonalStats(self.in_goj_corrected_fp, in_rst_list, \
            out_vec=out_geojson_with_polaris, interest_cols=interest_cols)

        df_data_with_polaris_data['som_stenon'] = \
            df_data_with_polaris_data[['som']].apply(polaris_utils.compute_logData, axis=1)

        # features_names = df_data_with_polaris_data.columns[3:-1].tolist()
        # polaris_utils.plot_stenon_polaris(df_data_with_polaris_data, \
        #     features_names=features_names, field_name = 'field_A&B', x_lim_max=1.5, y_lim_max=1.5)

        # # Combine data from multiple-layers to generate 0_30 layer data
        # new_features_names = []
        # ts_hilo_corr_attributes = df_data_with_polaris_data.columns[3:-1].tolist()
        # for i, name in enumerate(self.ps):
        #     df_data_with_polaris_data["som_polaris_%s_0_30"%name] = [np.median(row) for row in df_data_with_polaris_data[features_names[i::5]].itertuples(index=False)]
        #     new_features_names.append("som_polaris_%s_0_30"%name)

        # select_features_names = [item for item in new_features_names if item != 'som_polaris_p5_0_30']
        # polaris_utils.plot_stenon_polaris_0_30(df_data_with_polaris_data, \
        #     features_names=select_features_names, field_name='field_A&B', t_oulier=2, x_lim_max=2, y_lim_max=2)

        # logger.info("Group points sample by location")
        # selected_cols = ['lat', 'lng']
        # out_geojson_with_group_by_location_fp = os.path.join(self.out_dir, 'data_clean_with_group_by_location.geojson')
        # df_with_group_by_location = io_utils.group_points_sample(self.in_goj_corrected_fp, \
        #     out_vec=out_geojson_with_group_by_location_fp, interest_cols=selected_cols)
        # df_with_group_by_location.to_csv(os.path.join(self.out_dir, 'data_clean_with_group_by_location.csv'))


class SoilOrganicMatterPrediction(CleanData):
    def __init__(self, args):
        super().__init__(args)
        print ("Soil Organic Matter Prediction")
        self.ora = args.outlier_removal_algorithm

        self.out_model_dir = args.out_model_dir
        if not os.path.exists(self.out_model_dir):os.makedirs(self.out_model_dir)

        self.in_csv_corrected_fp = os.path.join(self.out_dir, \
            '{}_label_corrected_for_inspection.csv'.format(self.csv_fn_no_ext))

        if os.path.isfile(self.in_csv_corrected_fp):
            print ('Input data: {} are available! Begin with next processing steps!'.format(self.in_csv_corrected_fp))
            self.df = io_utils.load_data_from_csv_file(self.in_csv_corrected_fp)
        else:
            self.df = super().data_preparation()

        if self.ora is not None:
            self.df_A = io_utils.outlier_removal(self.df, 'field_A')
            self.df_B = io_utils.outlier_removal(self.df, 'field_B')

            self.df = pd.concat([self.df_A, self.df_B])
            self.df.to_csv('./data/data_clean.csv')

    def data_preparation(self):
        ''' Data preparation for train the model '''
        # Full data (Field_A and Field_B)
        self.train_targets_df = self.df['som']
        self.train_features_df = self.df.filter(regex='nir_')
        self.train_targets_arr = self.train_targets_df.to_numpy()
        self.train_features_arr = self.train_features_df.to_numpy()

        # Field_A data only
        self.train_field_A_features_df = self.df.loc[self.df['location']=='field_A'].filter(regex='nir_')
        self.train_field_A_targets_df = self.df.loc[self.df['location']=='field_A']['som']
        self.train_field_A_features_arr = self.train_field_A_features_df.to_numpy()
        self.train_field_A_targets_arr = self.train_field_A_targets_df.to_numpy()

        # Field_B data only
        self.train_field_B_features_df = self.df.loc[self.df['location']=='field_B'].filter(regex='nir_')
        self.train_field_B_targets_df = self.df.loc[self.df['location']=='field_B']['som']
        self.train_field_B_features_arr = self.train_field_B_features_df.to_numpy()
        self.train_field_B_targets_arr = self.train_field_B_targets_df.to_numpy()

    def train(self):
        # List of Models for performance evaluation
        models = ['Ridge', 'PLS']
        out_model_fn = 'ALL_train'

        # Train the model
        for index, model_selection in enumerate(models):
            io_utils.som_models(self.train_features_arr, self.train_targets_arr, self.out_model_dir, model_selection, out_model_fn)


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)                                 
    parent_parser.add_argument('-i', '--in-dir', default=DEFAULT_INPUT_DIRECTORY)
    parent_parser.add_argument('-o', '--out-dir', default=DEFAULT_OUTPUT_DIRECTORY) 
    parent_parser.add_argument('-ifn', '--input-csv-fn', default=DEFAULT_CSV_FILENAME)
    parent_parser.add_argument('-ivfn', '--input-vector-fn', default=DEFAULT_GOJ_FILENAME)             

    parser = argparse.ArgumentParser(add_help=False) 
    subparsers = parser.add_subparsers()                                             

    # subcommand clean data                                                                  
    parser_clean = subparsers.add_parser('clean', parents = [parent_parser])                          
    parser_clean.add_argument('-D', '--daemon', action='store_true')                     

    parser_clean.add_argument('-L', '--log', default='/tmp/test.log')                    

    # subcommand inteact with polaris data                                                                
    parser_polaris = subparsers.add_parser('polaris', parents = [parent_parser])
    parser_polaris.add_argument('-p', '--polaris-parameters', action='store_true', default=DEFAULT_POLARIS_PARAMETERS)                
    parser_polaris.add_argument('-pl', '--polaris-parameter-layers', action='store_true', default=DEFAULT_OM_LAYERS)                     
    parser_polaris.add_argument('-pls', '--polaris-parameter-layer-statistics', action='store_true', default=DEFAULT_OM_LAYER_STATISTICS)                     

    # subcommand predict Organic Soil Matter                                                           
    parser_predict = subparsers.add_parser('predict', parents = [parent_parser])
    parser_predict.add_argument('-ora', '--outlier-removal-algorithm', default=OUTLIER_REMOVAL_ALGORITHM_NAME)                             
    parser_predict.add_argument('-omd', '--out-model-dir', default=DEFAULT_OUTPUT_MODEL_DIRECTORY)             
                       
    args = parser.parse_args()

    clean_data = CleanData(args)
    if sys.argv[1] == 'clean':
        clean_data.data_preparation()
    if sys.argv[1] == 'polaris':
        polaris_data = PolarisData(args)
        polaris_data.analyze_polaris_data()
    if sys.argv[1] == 'predict':
        som_predict = SoilOrganicMatterPrediction(args)
        som_predict.data_preparation()
        som_predict.train()