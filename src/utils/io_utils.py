# Standard library imports
import logging
import os
import csv

# Third party imports
import pandas as pd
import pdb

# Local application imports
from utils.core_utils import _inputfile_exists


logger = logging.getLogger(__name__)
pd.set_option("display.max_rows", None, "display.max_columns", None)


@_inputfile_exists
def load_data_from_csv_file(filepath) :
    """TODO"""
    
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
    """TODO"""

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
    """TODO"""

    # Check the data type of all columns in the DataFrame
    print (df.dtypes)

    # Count number of Rows having NaN or Empty element(s)
    nrows_having_NaN = df.shape[0] - df.dropna().shape[0]
    print ("Total number of Rows having NaN or Empty element(s): {}".format(nrows_having_NaN))

    nrows_index_having_NaN = df.index[df.isnull().any(axis=1)]
    print (df.loc[nrows_index_having_NaN])
    
    # Count number of Rows having  all element are NaN or Empty
    nrows_all_NaN = df.shape[0] - df.isnull().all(axis=1).shape[0] 
    print ("Total number of Rows having  all elements are NaN or Empty: {}".format(nrows_all_NaN))
    
    if nrows_all_NaN> 0:
        print (df.loc[nrows_all_NaN])
        logger.info("Rows having  all elements are NaN or Empty will be dropped from dataframe!")
        df = df.loc[df.index.drop(nrows_all_NaN)]

    return df , nrows_index_having_NaN


def clean_data(df, idx_to_del=[]) :
    """TODO"""

    # Drop Rows with missing values or NaN
    if idx_to_del.tolist():
        df_clean = df.loc[df.index.drop(idx_to_del)]
        
    return df