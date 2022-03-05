# Local application imports
from utils import io_utils, plot_utils


in_csv = "../data/challenge_geoDS.csv"

# Load data from input csv file
df = io_utils.load_data_from_csv_file(in_csv)

# Data inspection
df_inspection, idx_to_del =  io_utils.inspect_data(df)

# Data cleaning
data_clean = io_utils.clean_data(df_inspection, idx_to_del)

# Data inspection (after cleaning) via plots
ignore_cols = ['measurement_ID', 'lat_lng', 'som']
attribute =  "location"
params=['max', 'mean', 'min', 'std']
plot_utils.plot_soil_profiles(data_clean, ignore_cols=ignore_cols, attribute=attribute, params=params)