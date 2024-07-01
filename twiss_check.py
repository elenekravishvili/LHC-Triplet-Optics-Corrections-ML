import numpy as np
import random
import tfs
import pandas as pd
import matplotlib.pyplot as plt
"""
B1_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
B2_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")

QX = 62.28
QY = 60.31

def get_input_for_beam(twiss_df, meas_mdl, beam):
    ip_bpms_b1 = ["BPMSW.1L1.B1", "BPMSW.1R1.B1", "BPMSW.1L2.B1", "BPMSW.1R2.B1", "BPMSW.1L5.B1", "BPMSW.1R5.B1", "BPMSW.1L8.B1", "BPMSW.1R8.B1"]
    ip_bpms_b2 = ["BPMSW.1L1.B2", "BPMSW.1R1.B2", "BPMSW.1L2.B2", "BPMSW.1R2.B2", "BPMSW.1L5.B2", "BPMSW.1R5.B2", "BPMSW.1L8.B2", "BPMSW.1R8.B2"]

    tw_perturbed_elements = uppercase_twiss(tw_df=twiss_df.copy()) 
    
    

    #tw_perturbed_elements = tfs.read_tfs(twiss_pert_elements_path).set_index("NAME")
    #tw_perturbed_elements = twiss_df.set_index("name") 
    # Uppercase and taking the relimport matplotlib.pyplot as plt
    #tw_perturbed_elements.columns = [col.upper() for col in tw_perturbed_elements.columns]

    tw_perturbed = tw_perturbed_elements[tw_perturbed_elements.index.isin(meas_mdl.index)]

    ip_bpms = ip_bpms_b1 if beam == 1 else ip_bpms_b2
    # phase advance deviations
    phase_adv_x = get_phase_adv(tw_perturbed['MUX'], QX)
    phase_adv_y = get_phase_adv(tw_perturbed['MUY'], QY)
    mdl_ph_adv_x = get_phase_adv(meas_mdl['MUX'], QX)
    mdl_ph_adv_y = get_phase_adv(meas_mdl['MUY'], QY)
    delta_phase_adv_x = phase_adv_x - mdl_ph_adv_x
    delta_phase_adv_y = phase_adv_y - mdl_ph_adv_y



    # beta deviations at bpms around IPs
    delta_beta_star_x = (np.array(tw_perturbed.loc[ip_bpms, "BETX"] - meas_mdl.loc[ip_bpms, "BETX"]))/meas_mdl.loc[ip_bpms, "BETX"]
    delta_beta_star_y = np.array(tw_perturbed.loc[ip_bpms, "BETY"] - meas_mdl.loc[ip_bpms, "BETY"])/meas_mdl.loc[ip_bpms, "BETY"]

    #normalized dispersion deviation
    n_disp = 0 # tw_perturbed['NDX']
    
    # Adding betas at bpms for phase advance measurement in order to compute the noise

    beta_bpms_x = np.array(tw_perturbed["BETX"])
    beta_bpms_y = np.array(tw_perturbed["BETY"])
    
    #print("Beta Beat", meas_mdl, tw_perturbed)
    #plot_example_betabeat(meas_mdl, tw_perturbed, beam)
    
    return np.array(delta_beta_star_x), np.array(delta_beta_star_y), \
        np.array(delta_phase_adv_x), np.array(delta_phase_adv_y), np.array(n_disp),\
        np.array(beta_bpms_x), np.array(beta_bpms_y)

def uppercase_twiss(tw_df):
    tw_df.columns = [col.upper() for col in tw_df.columns]
    tw_df.apply(lambda x: x.astype(str).str.upper())
    tw_df.columns = [col.upper() for col in tw_df.columns]
    tw_df = tw_df.set_index("NAME")
    tw_df.index = tw_df.index.str.upper() 
    tw_df.index = tw_df.index.str[:-2]
    tw_df['KEYWORD'] = tw_df['KEYWORD'].str.upper()
    return tw_df

def get_phase_adv(total_phase, tune):
    total_phase = np.array(total_phase)
    phase_diff = np.diff(total_phase)
    last_to_first = total_phase[0] - (total_phase[-1] - tune)
    phase_adv = np.append(phase_diff, last_to_first)
    return phase_adv


#tw_before_match=tfs.read("check_twiss_before_match.tfs")
tw_before_off = tfs.read("check_twiss_beforeoff.tfs")
tw_off = tfs.read("check_final_twiss.tfs")

delta_beta_star_x, delta_beta_star_y, \
        delta_phase_adv_x, delta_phase_adv_y, n_disp,\
        beta_bpms_x, beta_bpms_y = get_input_for_beam(tw_off, B1_MONITORS_MDL_TFS, 1)

print(delta_beta_star_x)
"""
"""
all_samples = np.load('./data_include_offset/100%triplet_ip1_20%ip5_6575.npy', allow_pickle=True)

delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
    delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
        delta_muy_b2,\
        beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
        triplet_errors, dpp_1, dpp_2 = all_samples.T

input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
        np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
        np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)   

#output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)
output_data = np.vstack(triplet_errors)
# betas = beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2
beta_x_b1, beta_y_b1, beta_x_b2, beta_y_b2, input_data, output_data, off_1, off_2 = np.vstack(beta_bpm_x_b1), np.vstack(beta_bpm_y_b1), np.vstack(beta_bpm_x_b2), np.vstack(beta_bpm_y_b2),\
     input_data, output_data, dpp_1, dpp_2
"""     

B1_ELEMENTS_MDL_TFS = tfs.read_tfs("b1_nominal_elements.dat").set_index("NAME")
B2_ELEMENTS_MDL_TFS = tfs.read_tfs("b2_nominal_elements.dat").set_index("NAME")

K1L_b1 =np.array(B1_ELEMENTS_MDL_TFS["K1L"])
K1L_b2 =np.array(B2_ELEMENTS_MDL_TFS["K1L"])
S = np.array(B1_ELEMENTS_MDL_TFS["S"])
betx = np.array(B1_ELEMENTS_MDL_TFS["BETX"])

bety = np.array(B1_ELEMENTS_MDL_TFS["BETY"])



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# First plot: Beta functions
ax1.plot(S, betx, label=r'$\beta_x$', linestyle='-')
ax1.plot(S, bety, label=r'$\beta_y$', linestyle='-')
#ax1.set_title('Beta functions and quadrupole strength in a FODO lattice', fontsize=16)
ax1.set_ylabel(r'$\beta$ [m]', fontsize=20)
ax1.legend(fontsize=12)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=12)
#ax1.set_xlim(9800, 10600)
ax1.set_xlim(775, 1025)




ax1.set_ylim(0, 200)

# Second plot: Vertical lines for K1L
ax2.vlines(x=S, ymin=0, ymax=K1L_b1, color='red', linestyle='-', linewidth=5)
ax2.axhline(0, color='black', linestyle='--')  # Baseline
ax2.set_xlabel('Longitudinal position [m]', fontsize=14)
ax2.set_ylabel(r'$K_{1L} \ [m^{-2}]$', fontsize=18)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=12)
#ax2.set_xlim(9800, 10600)
ax2.set_xlim(775, 1025)

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()


import matplotlib.pyplot as plt

# Define a function to set the template for the plots
def set_plot_template():
    plt.rcParams.update({
        'figure.figsize': (10, 6),        # Size of the figure
        'axes.titlesize': 25,             # Title font size
        'axes.labelsize': 25,             # X and Y label font size
        'xtick.labelsize': 18,            # X tick label font size
        'ytick.labelsize': 18,            # Y tick label font size
        'axes.titlepad': 25,              # Distance of the title from the plot
        'axes.labelpad': 20,              # Distance of the labels from the plot
        'legend.fontsize': 18,            # Legend font size
        'legend.frameon': True,           # Legend frame
        'legend.loc': 'best',             # Legend location
        'lines.linewidth': 2,             # Line width
        'grid.linestyle': '--',           # Grid line style
        'grid.alpha': 0.7,                # Grid line transparency
        'savefig.dpi': 300,               # Dots per inch for saved figures
        'savefig.bbox': 'tight',          # Bounding box for saved figures
        'savefig.format': 'png',          # Default save format
        'figure.autolayout': True         # Automatic layout adjustment
    })

# Apply the template
set_plot_template()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# First plot: Beta functions
ax1.plot(S, betx, label=r'$\beta_x$', linestyle='-')
ax1.plot(S, bety, label=r'$\beta_y$', linestyle='-')
ax1.set_ylabel(r'$\beta$ [m]', fontsize=25)
ax1.legend(fontsize=18)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_xlim(775, 1025)
ax1.set_ylim(0, 200)

# Second plot: Vertical lines for K1L
ax2.vlines(x=S, ymin=0, ymax=K1L_b1, color='red', linestyle='-', linewidth=5)
ax2.axhline(0, color='black', linestyle='--')  # Baseline
ax2.set_xlabel('Longitudinal position [m]', fontsize=25)
ax2.set_ylabel(r'$K_{1L} \ [m^{-2}]$', fontsize=25)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_xlim(775, 1025)

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()



# Define a function to set the template for the plots
def set_plot_template():
    plt.rcParams.update({
        'figure.figsize': (10, 6),        # Size of the figure
        'axes.titlesize': 25,             # Title font size
        'axes.labelsize': 25,             # X and Y label font size
        'xtick.labelsize': 18,            # X tick label font size
        'ytick.labelsize': 18,            # Y tick label font size
        'axes.titlepad': 25,              # Distance of the title from the plot
        'axes.labelpad': 20,              # Distance of the labels from the plot
        'legend.fontsize': 18,            # Legend font size
        'legend.frameon': True,           # Legend frame
        'legend.loc': 'best',             # Legend location
        'lines.linewidth': 2,             # Line width
        'grid.linestyle': '--',           # Grid line style
        'grid.alpha': 0.7,                # Grid line transparency
        'savefig.dpi': 300,               # Dots per inch for saved figures
        'savefig.bbox': 'tight',          # Bounding box for saved figures
        'savefig.format': 'png',          # Default save format
        'figure.autolayout': True         # Automatic layout adjustment
    })

# Apply the template
set_plot_template()



fig = plt.figure(figsize=(10, 8))

# First plot: Beta functions
ax1 = fig.add_axes([0.16, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
ax1.plot(S, betx, label=r'$\beta_x$', linestyle='-')
ax1.plot(S, bety, label=r'$\beta_y$', linestyle='-')
ax1.set_ylabel(r'$\beta$ [m]', fontsize=25)
ax1.legend(fontsize=18)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_xlim(775, 1025)
ax1.set_ylim(0, 200)

# Second plot: Vertical lines for K1L
ax2 = fig.add_axes([0.16, 0.19, 0.8, 0.25], sharex=ax1)  # [left, bottom, width, height]
ax2.vlines(x=S, ymin=0, ymax=K1L_b1, color='red', linestyle='-', linewidth=5)
ax2.axhline(0, color='black', linestyle='--')  # Baseline
ax2.set_xlabel('Longitudinal position [m]', fontsize=25)
ax2.set_ylabel(r'$K_{1L} \ [m^{-2}]$', fontsize=25)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_xlim(775, 1025)


plt.tight_layout()


plt.show()

