# Standard library imports
import logging
import os
import csv

# Third party imports
import pandas as pd
import numpy as np
import geopandas as gpd

# Local application imports
from utils.core_utils import _inputfile_exists


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


def inspect_data(df) :
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
    
    # Count number of Rows having  all element are NaN or Empty
    nrows_all_NaN = df.shape[0] - df.isnull().all(axis=1).shape[0] 
    print ("Total number of Rows having  all elements are NaN or Empty: {}".format(nrows_all_NaN))
    
    if nrows_all_NaN> 0:
        print (df.loc[nrows_all_NaN])
        df = df.loc[df.index.drop(nrows_all_NaN)]

    #TODO:  Further check: new data frame with split value columns
    df_new = df["lat_lng"].str.slice(start=1,stop=-1).str.split(",", n = 1, expand = True)
    df["lat"] = df_new[0]
    df['lng'] = df_new[1]
    df[["lat", "lng"]] = df[["lat", "lng"]].apply(pd.to_numeric)

    return df , nrows_index_having_NaN


def clean_data(df, idx_to_del=[]) :
    """Loads a set of edges from a .mat or .geojson-file."""

    # Drop Rows with missing values or NaN
    # import pdb; pdb.set_trace()
    if idx_to_del.tolist():
        # df_clean = df.loc[df.index.drop(idx_to_del)]
        df = df.dropna()
        
    # Drop ZERO LAT LON record(s)
    df = df.loc[(df['lat'] != 0) & (df['lng'] != 0)]

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

    import json
    from geojson import Feature, FeatureCollection, Point

    features = df.apply(
        lambda row: Feature(
            geometry=Point((float(row[latitude_longitude][1:-1].split(',')[1]), float(row[latitude_longitude][1:-1].split(',')[0]))),
            properties=dict(row),
        ),
        axis=1,
    ).tolist()

    geojson = FeatureCollection(features=features)

    if out_geojson is not None:
        with open(out_geojson, "w", encoding=encoding) as f:
            f.write(json.dumps(geojson))

    return geojson


def get_bounding_box(geometry):
    """TODO"""

    import geojson

    coords = np.array(list(geojson.utils.coords(geometry)))

    #delete [0, 0] from coords
    coords = np.delete(coords, [0, 0], axis=0)
    params = coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max()

    return params


def ZonalStats(in_vec, in_rst_list, out_vec=None, interest_cols=[], attribute_name='som_polaris'):
    """TODO"""
    # in_vec - shapefile path
    # in_rst_list - raster path
    # the result is df as DataFrame

    from rasterstats import point_query

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


def group_points_sample(in_vec, out_vec, interest_cols=['lat', 'lng']):
    from sklearn.cluster import DBSCAN

    # https://medium.com/@agarwalvibhor84/lets-cluster-data-points-using-dbscan-278c5459bee5
    df = gpd.read_file(in_vec)

    X = df[interest_cols].to_numpy()
    dbscan = DBSCAN(eps = 0.00007, min_samples = 2)
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
