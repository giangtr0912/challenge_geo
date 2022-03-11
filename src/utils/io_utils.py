# Standard library imports
import logging
import os
import csv

# Third party imports
import json
import geojson
import numpy as np
import pandas as pd

import geopandas as gpd
from geojson import Feature, FeatureCollection, Point

from rasterstats import point_query

# Local application imports
from src.utils.core_utils import _inputfile_exists


logger = logging.getLogger(__name__)
pd.set_option("display.max_rows", None, "display.max_columns", None)


@_inputfile_exists
def load_data_from_csv_file(filepath) :
    """Loads a set of edges from a .mat or .geojson-file."""
    
    _, file_extension = os.path.splitext(filepath)
    if file_extension == ".csv":
        params = check_csv_valid(filepath)
        if params:
            delimiter, has_header, _ = params
            df = pd.read_csv(filepath, delimiter = delimiter)
            return df
    else:
        raise ValueError("Input file must be .csv file.")


@_inputfile_exists
def check_csv_valid(filepath, arbitrary_number=2048):
    """Loads a set of edges from a .mat or .geojson-file."""

    # An arbitrary number is an entirely arbitrary number. It just needs to be big enough to read in at least two or three CSV rows.
    params = []
    with open(filepath, 'r') as csvfile:
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(csvfile.read(arbitrary_number))

            # Perform various checks on the dialect (e.g., lineseparator, delimiter) to make sure it's sane
            delimiter, lineseparator = dialect.delimiter, dialect.lineterminator

            # Reset the read position back to the start of the file before reading any entries.
            csvfile.seek(0)

            # Perform header check to make sure it's sane
            has_header = sniffer.has_header(csvfile.read(arbitrary_number))
            logger.info("Input CSV file is valid. Ready to open and explorer!")

            params = [delimiter, has_header, lineseparator]
        except csv.Error:
            # File appears not to be in CSV format; move along
            logger.error ("Invalid CSV file")
    
    return params


def primary_clean_data(df) :
    """Loads a set of edges from a .mat or .geojson-file."""

    # Check the data type of all columns in the DataFrame
    # print (df.dtypes)

    wave_length_cols_dtype = [df[col].dtype for col in df.columns if (("nir_" in col) and (df[col].dtype != "float64"))]
    if wave_length_cols_dtype:
            logger.error ("There are {} having un-proper data type".format(len(wave_length_cols_dtype)))

    # Count number of Rows having NaN or Empty element(s)
    nrows_having_NaN = df.shape[0] - df.dropna().shape[0]
    print ("Total number of Rows contain NaN or Empty element(s): {}".format(nrows_having_NaN))

    nrows_index_having_NaN = df.index[df.isnull().any(axis=1)]
    print ("Following Rows: {} having NaN or Empty element(s)".format(nrows_index_having_NaN.tolist()))
    if nrows_index_having_NaN.tolist():
        df = df.dropna()

    # Count number of Rows having  all element are NaN or Empty
    nrows_all_NaN = df.shape[0] - df.isnull().all(axis=1).shape[0] 
    print ("Total number of Rows having  all elements are NaN or Empty: {}".format(nrows_all_NaN))
    
    if nrows_all_NaN> 0:
        print (df.loc[nrows_all_NaN])
        df = df.loc[df.index.drop(nrows_all_NaN)]
        
    return df


def create_lat_lng_cols(df):
    temp = df["lat_lng"].str.slice(start=1,stop=-1).str.split(",", n = 1, expand = True)
    df["lat"], df['lng'] = temp[0], temp[1]
    df[["lat", "lng"]] = df[["lat", "lng"]].apply(pd.to_numeric)

    return df

def outlier_removal(df, location_name, col='location', attribute='som'):
  df_field = df[df[col] == location_name]

  Q1 = df_field[attribute].quantile(0.25) # Same as np.percentile but maps (0,1) and not (0,100)
  Q3 = df_field[attribute].quantile(0.75)
  IQR = Q3 - Q1

  # Return a boolean array of the rows with (any) non-outlier column values
  condition = ~((df_field[attribute] < (Q1 - 1.5 * IQR)) | (df_field[attribute] > (Q3 + 1.5 * IQR)))
  filtered_df_field = df_field.loc[condition]


def correct_data_label(out_goj_for_inspection, fields_boundary_fp, col_name='location'):
    poly_gdf = gpd.read_file(fields_boundary_fp)
    point_gdf = gpd.GeoDataFrame.from_features(out_goj_for_inspection['features'], crs=4326)

    # Make sure they're using the same projection reference before perform geometry Joins

    # If Geopandas sjoin query is not working, please check out:
    # https://stackoverflow.com/questions/67021748/importerror-spatial-indexes-require-either-rtree-or-pygeos-in-geopanda-but
    # for the solution
    # pip uninstall rtree
    # sudo apt install libspatialindex-dev
    # pip install rtree

    if point_gdf.crs == poly_gdf.crs:
      join_inner_gdf = point_gdf.sjoin(poly_gdf, how="inner")
    else:
      logger.info('The projection reference not the same!')

    join_inner_gdf.rename({'{}_right'.format(col_name): '{}_correct'.format(col_name), \
        '{}_left'.format(col_name): col_name}, axis=1, inplace=True)
    df = pd.DataFrame(join_inner_gdf.drop(columns='geometry'))

    return df

    
def pandas_to_geojson(df, out_geojson=None, latitude_longitude="lat_lng", encoding="utf-8"):
    """Creates points for a Pandas DataFrame and exports data as a GeoJSON.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame.
        out_geojson (str): The file path to the exported GeoJSON. Default to None.
        latitude (str, optional): The name of the column containing latitude coordinates. Defaults to "latitude".
        longitude (str, optional): The name of the column containing longitude coordinates. Defaults to "longitude".
        encoding (str, optional): The encoding of characters. Defaults to "utf-8".

    """

    features = df.apply(
        lambda row: Feature(
            geometry=Point((float(row[latitude_longitude][1:-1].split(',')[1]), float(row[latitude_longitude][1:-1].split(',')[0]))),
            properties=dict(row),
        ),
        axis=1,
    ).tolist()

    geojson_obj = FeatureCollection(features=features)

    if out_geojson is not None:
        with open(out_geojson, "w", encoding=encoding) as f:
            f.write(json.dumps(geojson_obj))
            
    return geojson_obj
    

def ZonalStats(in_vec, in_rst_list, out_vec=None, interest_cols=[], attribute_name='som_polaris'):
    """TODO"""
    # in_vec - shapefile path
    # in_rst_list - raster path
    # the result is df as DataFrame

    shape_gdf = gpd.read_file(in_vec)
    for in_rst in in_rst_list:
        file_pattern = '_'.join(in_rst.split('/')[-3:-1])
        column_name = "{}_{}".format(attribute_name, file_pattern)
        zonalSt = point_query(in_vec, in_rst, band=1, nodata=-9999, interpolate='nearest')
        df = pd.DataFrame (zonalSt, index=shape_gdf.index, columns = [column_name])
        shape_gdf = pd.concat([shape_gdf, df], axis=1)
        interest_cols.append(column_name)

    interest_cols.append('geometry')
    # re-order the columns
    gdf = gpd.GeoDataFrame(shape_gdf, geometry=shape_gdf.geometry)
    gdf = gdf[interest_cols]

    if out_vec is not None:
        # Alternatively, you can write GeoJSON to file:
        gdf.to_file(out_vec, driver="GeoJSON")  
    
    df = gdf.drop(['geometry'], axis=1, errors='ignore')

    return df


def group_points_sample(in_vec, out_vec, interest_cols=['lat', 'lng'], eps_value=0.00007, min_samples=2):
    from sklearn.cluster import DBSCAN

    # https://medium.com/@agarwalvibhor84/lets-cluster-data-points-using-dbscan-278c5459bee5
    df = gpd.read_file(in_vec)

    X = df[interest_cols].to_numpy()
    dbscan = DBSCAN(eps = eps_value, min_samples=min_samples)
    model = dbscan.fit(X)

    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    df_label = pd.DataFrame(model.labels_, index=df.index, columns = ["group"])
    shape_gdf = pd.concat([df, df_label], axis=1)
    gdf = gpd.GeoDataFrame(shape_gdf, geometry=shape_gdf.geometry)

    if out_vec is not None:
        # Alternatively, you can write GeoJSON to file:
        gdf.to_file(out_vec, driver="GeoJSON")  

    df = gdf.drop(['geometry'], axis=1, errors='ignore')
    
    return df


def outlier_removal(df, location_name, col='location', attribute='som'):
  df_field = df[df[col] == location_name]

  Q1 = df_field[attribute].quantile(0.25) # Same as np.percentile but maps (0,1) and not (0,100)
  Q3 = df_field[attribute].quantile(0.75)
  IQR = Q3 - Q1

  # Return a boolean array of the rows with (any) non-outlier column values
  condition = ~((df_field[attribute] < (Q1 - 1.5 * IQR)) | (df_field[attribute] > (Q3 + 1.5 * IQR)))
  filtered_df_field = df_field.loc[condition]

#   plt.figure(figsize=(15, 8))
#   plt.clf

#   plt.hist(filtered_df_field[attribute], bins=100, histtype='bar', rwidth=0.8, color='blue')
#   plt.xlim(filtered_df_field[attribute].min(), filtered_df_field[attribute].max())
#   plt.xlabel("{} [%]".format(attribute), fontsize=16)
#   plt.title ("{} (soil organic matter content data, n_obs = {} samples)".format(location_name, filtered_df_field[attribute].shape[0]), fontsize=18)
#   plt.legend()

  return filtered_df_field


def som_models(features, targets, out_dir, model_selection, out_model_fn):

    # Third party imports for regression Model training and validation
    import pickle
    from sklearn import metrics
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    from sklearn import linear_model  # the Linear regression model
    from sklearn.ensemble import RandomForestRegressor  # the Random forest regression model
    import xgboost as xgb  # the XGBoost regression model
    from sklearn.linear_model import Ridge  # the Ridge regression model
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.cross_decomposition import PLSRegression # the PLSRegression regression model

    # Split the dataset into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=0)

    model_build = False
    if model_selection == 'Ridge':
        # define cross-validation method to evaluate model
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1) # 
        model = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
        model.fit(X_train, y_train)
        model_build = True
    elif model_selection == 'PLS':
        model = PLSRegression(n_components=150)
        model.fit(X_train, y_train)
        model_build = True

    if model_build:
        pickle.dump(model, open('./{}/{}_{}.pkl'.format(out_dir, model_selection, out_model_fn), 'wb'))