import matplotlib.pyplot as plt
import numpy as np
import tfs 
import pandas as pd

nominal_df = tfs.read('nominal_twiss_parameters.tfs')
magnet_errors_df = tfs.read('magneterrors_twiss_parameters.tfs')

nominal_df = nominal_df.apply(pd.to_numeric, errors='coerce')
magnet_errors_df = magnet_errors_df.apply(pd.to_numeric, errors='coerce')

# Exclude rows containing strings only
nominal_df = nominal_df.dropna(how='all')
magnet_errors_df = magnet_errors_df.dropna(how='all')


# Compute the difference
difference_table = nominal_df - magnet_errors_df

for column in difference_table.columns:
    difference_table[column].fillna(nominal_df[column], inplace=True)
# Save the difference table
tfs.write("path_to_save_difference_table.tfs", difference_table)



