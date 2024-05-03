import matplotlib.pyplot as plt
import numpy as np
import cpymad.madx
import tfs


# Read data from TFS files into pandas DataFrames
nominal_df = tfs.read('nominal_twiss_parameters.tfs')
magnet_errors_df = tfs.read('magneterrors_twiss_parameters.tfs')

# Convert infinite values to NaN
#nominal_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#magnet_errors_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Extract columns you want to plot
s_nominal = nominal_df['s']
betx_nominal = nominal_df['betx']
betx_errored = magnet_errors_df['betx']
beta_difference=(betx_nominal-betx_errored)/betx_nominal

#phase advance
mux_nominal = nominal_df['mux']
mux_errored = magnet_errors_df['mux']
difference=mux_nominal-mux_errored

#dispersion




# Plot beta
plt.plot(s_nominal, betx_nominal, label='Nominal')
plt.plot(s_nominal, betx_errored, label='Errored')
plt.xlabel('S')
plt.ylabel('BETX')
plt.title('BETX vs S')
plt.legend()
plt.grid(True)
plt.show()
"""
#plot phase advance
plt.plot(s_nominal, mux_nominal, label='Nominal')
plt.plot(s_nominal, mux_errored, label='Errored')
plt.xlabel('S')
plt.ylabel('phase advance_x')
plt.title('phase advance_x vs S')
plt.legend()
plt.grid(True)
plt.show()
"""
"""
plt.plot(s_nominal, beta_difference, label='Errored')
plt.xlabel('S')
plt.ylabel('difference_phase advance_x')
plt.title('difference between phase advance_x vs S')
plt.legend()
plt.grid(True)
plt.show()
"""