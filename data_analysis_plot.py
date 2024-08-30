import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
from pathlib import Path
from joblib import load
from cpymad import madx
from data_utils import add_phase_noise
from data_utils import add_beta_star_noise
from madx_job_sbs import madx_ml_op
import seaborn as sns



def merge_data(data_path, noise):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    for file_name in file_names:
        print(file_name)
        aux_input, aux_output = load_data(file_name, noise)
        input_data.append(aux_input)
        output_data.append(aux_output)

    return np.concatenate(input_data), np.concatenate(output_data)


def load_data(set_name, noise):
    
    #all_samples = np.load('./data_phase_adv_triplet/{}.npy'.format(set_name), allow_pickle=True)
    #all_samples = np.load('./data_include_offset/{}.npy'.format(set_name), allow_pickle=True)
    #all_samples = np.load('./test_set_witout_misalignment/{}.npy'.format(set_name), allow_pickle=True)
    all_samples = np.load('./data_right_offset/{}.npy'.format(set_name), allow_pickle=True)


    #The configuration of samples are different in two cass, data having only magnet errors and data having for magnet errors and offsets
    
    #Following one down is for the data where only magnet errors are
    
    #delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
     # delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
       #     delta_muy_b2,b1_disp, b2_disp,\
        #    beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
         #   triplet_errors = all_samples.T
    
    #This is for the data where magnets and offsets are
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors, dpp_1, dpp_2 = all_samples.T
        
        
        
    delta_mux_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b1, beta_bpm_x_b1)]
    delta_muy_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b1, beta_bpm_y_b1)]
    delta_mux_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b2, beta_bpm_x_b2)]
    delta_muy_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b2, beta_bpm_y_b2)]


    delta_beta_star_x_b1_with_noise = add_beta_star_noise(delta_beta_star_x_b1)
    delta_beta_star_y_b1_with_noise = add_beta_star_noise(delta_beta_star_y_b1)
    delta_beta_star_x_b2_with_noise = add_beta_star_noise(delta_beta_star_x_b2)
    delta_beta_star_y_b2_with_noise = add_beta_star_noise(delta_beta_star_y_b2)
    
    #Input and output can be changed depending on the desired model configuration
    
    input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
           np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
           np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
            np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    #input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
     #      np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    #input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1_with_noise), np.vstack(delta_beta_star_y_b1_with_noise), \
     #   np.vstack(delta_beta_star_x_b2_with_noise), np.vstack(delta_beta_star_y_b2_with_noise)), axis=1)  
    
    output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)
    #output_data =  np.vstack(triplet_errors)

    return input_data, output_data

def plot_example_errors(input_data, output_data, estimator):
    test_idx = sorted(np.load("./data_analysis/test_idx.npy"))[0:1]
    pred_triplet, true_triplet, pred_arc,\
    true_arc, pred_mqt, true_mqt = obtain_errors(input_data[test_idx], 
                                                    output_data[test_idx], 
                                                    estimator)
    
    errors = (("Triplet Errors: ", pred_triplet, true_triplet), 
            ("Arc Errors: ", pred_arc, true_arc), 
            ("MQT Knob: ", pred_mqt, true_mqt))

    for idx, (name, pred_error, true_error) in enumerate(errors):
        x = [idx for idx, error in enumerate(true_error)]
        plt.bar(x, true_error, label="True")
        plt.bar(x, pred_error, label="Pred")
        plt.bar(x, pred_error-true_error, label="Res")

        plt.title(f"{name}")
        plt.xlabel(r"MQ [#]")
        plt.ylabel(r"Absolute Error: $\Delta k$")
        plt.legend()
        plt.savefig(f"./figures/error_bars_{name[:-2]}.pdf")
        plt.show()
        plt.clf()

def obtain_errors(input_data, output_data, estimator, NORMALIZE=False):
    # Function that gives the predicted and real values of errors for a numpy array inputs
    pred_data = estimator.predict(input_data)
    if NORMALIZE==True:
        pred_data = normalize_errors(pred_data)
        output_data = normalize_errors(output_data)

    pred_triplet = np.hstack(pred_data[:,:32])
    true_triplet = np.hstack(output_data[:,:32])

    pred_arc = np.hstack(pred_data[:,32:1248])
    true_arc = np.hstack(output_data[:,32:1248])

    pred_mqt = np.hstack(pred_data[:,1248:])
    true_mqt = np.hstack(output_data[:,1248:])

    return pred_triplet, true_triplet, pred_arc, true_arc, pred_mqt, true_mqt

def normalize_errors(data):
    # Function with error values as input and outputs the results normalized by the nominal value
    with open("./data_analysis/mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "").upper().split(':')[0] for name in lines][:-4]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_elements_30.dat").set_index("NAME")
    B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_elements_30.dat").set_index("NAME")
    ELEMENTS_MDL_TFS = tfs.frame.concat([B1_ELEMENTS_MDL_TFS, B2_ELEMENTS_MDL_TFS])
    TRIPLET_NOM_K1 = ELEMENTS_MDL_TFS.loc[names, "K1L"][:64]
    TRIPLET_NOM_K1 = TRIPLET_NOM_K1[TRIPLET_NOM_K1.index.duplicated(keep="first")]

    ARC_NOM_K1 = ELEMENTS_MDL_TFS.loc[names, "K1L"][64:]

    nom_k1 = tfs.frame.concat([TRIPLET_NOM_K1, ARC_NOM_K1])
    nom_k1 = np.append(nom_k1, [1,1,1,1]) # Not the cleanest function ever
    
    for i, sample in enumerate(data):
        data[i] = (sample)/nom_k1
    return data

def normalize(data):
    TRIPLET_NAMES = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                    "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                    "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                    "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_elements.dat").set_index("NAME")
    TRIPLET_NOM_K1 = B1_ELEMENTS_MDL_TFS.loc[TRIPLET_NAMES, "K1L"]

    nom_k1 = tfs.frame.concat([TRIPLET_NOM_K1])
    #nom_k1 = np.append(nom_k1, [1,1,1,1]) # Not the cleanest function ever
    nom_k1=np.array(nom_k1)
    nom_k1=np.hstack(nom_k1)


    normalized_data = np.zeros_like(data)
    for i in range(len(data)):
        normalized_data[i] = data[i] / nom_k1

    return normalized_data



def obtain_triplet_errors(input_data, output_data_all, estimator):
    # Function that gives the predicted and real values of errors for a numpy array inputs
    pred_data_all = estimator.predict(input_data)
    pred_data = pred_data_all[:, :32]
    output_data = output_data_all[:, :32]
    pred_data = normalize(pred_data)
    output_data = normalize(output_data)

    pred_triplet = np.hstack(pred_data)
    true_triplet = np.hstack(output_data)

    return pred_triplet, true_triplet

def obtain_offset(input_data, output_data, estimator):
    all_pred_triplet = []
    all_pred_dpp1 = []
    all_pred_dpp2 = []
    # Function that gives the predicted and real values of errors for a numpy array inputs
    predicted_output = estimator.predict(input_data)

    predicted_trip = predicted_output[:, :32] #first 32
    true_trip = output_data[:, :32] 
    predicted_trip = normalize(predicted_trip)
    true_trip= normalize(true_trip)
    pred_triplet = np.hstack(predicted_trip)
    true_triplet = np.hstack(true_trip)
    #all_pred_triplet.append(predicted_triplet)

    predicted_dpp_1 = predicted_output[:, -2] #second last  
    true_dpp_1 = output_data[:, -2]
    pred_dpp1 = np.hstack(predicted_dpp_1)
    true_dpp1 = np.hstack(true_dpp_1)

    predicted_dpp_2 =  predicted_output[:, -1] #last
    true_dpp_2 = output_data[:, -1]
    pred_dpp2 = np.hstack(predicted_dpp_2)
    true_dpp2 = np.hstack(true_dpp_2)

    #all_pred_dpp2.append(predicted_dpp_2)

    return pred_dpp1, true_dpp1, pred_dpp2, true_dpp2, pred_triplet, true_triplet


def triplet_error_dist_plot(input_data, output_data, estimator, noise, alpha):
    set_plot_template()
    pred_triplet, true_triplet= obtain_triplet_errors(input_data, output_data, estimator)
    residue=true_triplet-pred_triplet
    residue_dp = pd.DataFrame(residue)
    
    # Ensure all column names are strings
    residue_dp.columns = residue_dp.columns.astype(str)
    
    #tfs.write("base_model_corrected_trip.tfs", residue_dp)
    mean = np.mean(residue)
    std = np.std(residue)
    rms = np.sqrt(np.mean(np.square(residue)))
    rms_true = np.sqrt(np.mean(np.square(true_triplet)))
    print(rms_true)
    
    
    
    #print(mean, std)
    bin_edges = np.linspace(-0.003, 0.003, 50)
    plt.hist(true_triplet, bins=bin_edges,  alpha=0.5, label='True')
    plt.hist(pred_triplet, bins=bin_edges,  alpha=0.5, label='Predicted')
    plt.hist(residue, bins=bin_edges, alpha=0.5, label='Residuals')
    plt.xlabel('Triplet errors')
    plt.ylabel('Counts')
    plt.ylim(0, 750)
    #plt.title(f'noise={noise}, alpha={alpha}, mean={mean}, std={std}')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    #plt.savefig(f"./triplet_distr/noise_{noise}_alpha_{alpha}.png")
    plt.show()
    #plt.close()
    
    
def off_dist_plot(input_data, output_data, estimator):
    pred_dpp1, true_dpp1, pred_dpp2, true_dpp2, pred_triplet, true_triplet = obtain_offset(input_data, output_data, estimator)
    residue_triplet = true_triplet - pred_triplet
    residue_dpp1 = true_dpp1 - pred_dpp1
    residue_dpp2 = true_dpp2 - pred_dpp2
    std_dpp1 = np.std(residue_dpp1)
    rms_dpp1 = np.sqrt(np.mean(np.square(residue_dpp1)))
    
    std_dpp2 = np.std(true_dpp2)
    rms_dpp2 = np.sqrt(np.mean(np.square(true_dpp2)))
    
    std_res = np.std(residue_triplet)
    rms_res = np.sqrt(np.mean(np.square(residue_triplet)))
    
    std_true = np.std(true_triplet)
    rms_true = np.sqrt(np.mean(np.square(true_triplet)))
    
    print("dpp1", std_dpp1, rms_dpp1)
    
    print("dpp2_true", std_dpp2, rms_dpp2)
    print("trip_true", std_true, rms_true)

    set_plot_template()
    # Plot for triplet errors
    fig, ax = plt.subplots(figsize=(10, 6))
    #bin_edges = np.linspace(-0.0002, 0.0002, 50)
    
    bin_edges_1 = np.linspace(-0.003, 0.003, 50)

    ax.hist(true_triplet, bins=bin_edges_1, alpha=0.5)
    #ax.hist(pred_triplet, bins=bin_edges_1, alpha=0.5, label='Predicted')
    #ax.hist(residue_triplet, bins=bin_edges_1, alpha=0.5, label='Residuals')
    ax.set_xlabel(r'$\frac{\Delta K_{1}}{K_{1}}$')
    #ax.set_ylabel('Counts')
    ax.set_ylim(0, 5000)
    ax.legend()
    ax.grid(True)
    plt.show()

    set_plot_template()
    # Plot for dpp1 and dpp2 errors combined
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 12))
    bin_edges = np.linspace(-0.3, 0.3, 85)

    # Plot for dpp1 errors
    axes[0].hist(true_dpp1*1000, bins=bin_edges, alpha=0.5)
    #axes[0].hist(pred_dpp1*1000, bins=bin_edges, alpha=0.5, label='Predicted')
    #axes[0].hist(residue_dpp1*1000, bins=bin_edges, alpha=0.5, label='Residuals')
    axes[0].set_xlabel(r'$\frac{\Delta p}{p}$  $[10^{{-3}}]$')
    axes[0].set_title('Beam 1')
    #axes[0].set_ylabel('Counts')
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].grid(True)
    #axes[0].text(0.5, 0.95, r'$\sigma$ = {:.2f} $\times 10^{{-5}}$ \n RMS = {:.2f} $\times 10^{{-5}}$'.format(std_dpp1*1e5, rms_dpp1*1e5), 
      #       transform=axes[0].transAxes, fontsize=25, 
       #      verticalalignment='top', horizontalalignment='left')

    # Plot for dpp2 errors
    axes[1].hist(true_dpp2*1000, bins=bin_edges, alpha=0.5, label='True')
    axes[1].hist(pred_dpp2*1000, bins=bin_edges, alpha=0.5, label='Predicted')
    axes[1].hist(residue_dpp2*1000, bins=bin_edges, alpha=0.5, label='Residuals')
    axes[1].set_xlabel(r'$\frac{\Delta p}{p}$  $[10^{{-3}}]$')
    axes[1].set_title('Beam 2')
    axes[1].set_ylim(0, 600)

    axes[1].set_ylabel('Counts')
    axes[1].legend()
    axes[1].grid(True)
   # axes[1].text(0.95, 0.95, f'Std: {std_dpp2*1e5:.2f}e-5\nRMS: {rms_dpp2*1e5:.2f}e-5', 
    #         transform=axes[1].transAxes, fontsize=25, 
     #        verticalalignment='top', horizontalalignment='right')
    # Adjust layout
    plt.tight_layout()

    # Show the combined plot
    plt.show()
    
    
def get_mean_value_of_triplet_errors(input_data, output_data, estimator, noise, alpha):
    pred_triplet, true_triplet= obtain_triplet_errors(input_data, output_data, estimator)
    #pred_triplet_norm = normalize(pred_triplet)
    #true_triplet_norm = normalize(true_triplet)
    residue=true_triplet-pred_triplet
    residue_dp = pd.DataFrame(residue)
    
    # Ensure all column names are strings
    residue_dp.columns = residue_dp.columns.astype(str)
    
    #tfs.write("base_model_corrected_trip.tfs", residue_dp)
    mean = np.mean(residue)
    rms = np.sqrt(np.mean(np.square(residue)))
    std = np.std(residue)
    
    mean_modulus = np.mean(np.abs(residue))

    return true_triplet, pred_triplet, residue, rms

def save_np_errors_tfs(np_errors):
    TRIPLET_NAMES = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                        "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                        "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                        "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]

    #This is the tfs format that can be read, this model of file is then copied and filled
    #error_tfs_model_b1 = tfs.read_tfs("./data_analysis/errors_b1.tfs")
    #error_tfs_model_b2 = tfs.read_tfs("./data_analysis/errors_b2.tfs")

    #Function that takes np errors and outputs .tfs file with all error values
    #with open("./data_analysis/mq_names_best_know.txt", "r") as f:
     #   lines = f.readlines()
      #  names = [name.replace("\n", "") for name in lines]

    #names = names[:32] # If we only predict triplets
    names = TRIPLET_NAMES

    # Recons_df is a dataframe with the correct names and errors but not format
    recons_df = pd.DataFrame(columns=["NAME","K1L"])
    recons_df.K1L = np_errors #[:-32] This is if we try to predict misalignment
    #ds_errors = np_errors[-32:]
    #recons_df.DS = [0 if i>=32 else ds_errors[i] for i, name in enumerate(names)]
    recons_df.NAME = names

    #for beam, error_tfs_model in enumerate([error_tfs_model_b1, error_tfs_model_b2]):
     #   for i in range(len(error_tfs_model)):
      #      # check if the name is in recons_df
     #       if error_tfs_model.loc[i, 'NAME'] in list(recons_df['NAME']):
       #         error_tfs_model.loc[i, 'K1L'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][1]
        #        #error_tfs_model.loc[i, 'DS'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][2]
            
    #tfs.writer.write_tfs(tfs_file_path=f"./data_analysis/b1_{filename}", data_frame=error_tfs_model_b1)
    tfs.write("./data_analysis/errors_1.tfs", recons_df)
    return recons_df

#x = tfs.read("/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/twiss_reconstruction/data_analysis/b1_pred_best_know_err.tfs")
#tfs.write("example.tfs", x)

def set_plot_template():
    plt.rcParams.update({
        'figure.figsize': (10, 6),        # Size of the figure
        'axes.titlesize': 35,             # Title font size
        'axes.labelsize': 30,             # X and Y label font size
        'xtick.labelsize': 25,            # X tick label font size
        'ytick.labelsize': 25,            # Y tick label font size
        'axes.titlepad': 25,              # Distance of the title from the plot
        'axes.labelpad': 20,              # Distance of the labels from the plot
        'legend.fontsize': 25,            # Legend font size
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

#Example of beta beating in the segment simulation with the quadrupole field strengths
def sbs_beta_beat(input_data, output_data, estimator, beam, index):
    predicted_data = estimator.predict([input_data[index]])
    corrected_errors = output_data[index] - predicted_data
    err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 = corrected_errors[0][-8:]
    pred_k1l = predicted_data[0][-8:]
    real_k1l = output_data[index][-8:]
    res_k1l = real_k1l - pred_k1l
    print(pred_k1l)
    mdx = madx_ml_op()
    # Beam 1 calculations
    mdx.job_init_sbs_b1(beam, 1) #here 1 is in the place of IP, but I was changing it manually, so this argument is not used
    mdx.sbs_twiss_b1()
    twiss_df_b1 = mdx.table.twiss.dframe()
    mdx.set_triplet_errors_b1(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
    mdx.sbs_twiss_b1()
    twiss_err_df_b1 = mdx.table.twiss.dframe()
     
    #filter data from drifts and also spike at IP (this spike location was named as solenoid)   
    prefixes_to_filter = ['ip', 'drift', 'mbas']

    # Apply the filtering function to each DataFrame
    twiss_df_b1_filtered = filter_indices(twiss_df_b1, prefixes_to_filter)
    #tfs.write("twiss_df_b1_filtered.tfs", twiss_df_b1_filtered)

    twiss_err_df_b1_filtered = filter_indices(twiss_err_df_b1, prefixes_to_filter)
    #tfs.write("twiss_err_df_b1_filtered.tfs", twiss_err_df_b1_filtered)
    s_b1 = twiss_df_b1_filtered["s"]
    betx_b1_nom = twiss_df_b1_filtered["betx"]
    bety_b1_nom = twiss_df_b1_filtered["bety"]
    
    betx_b1_err = twiss_err_df_b1_filtered["betx"]
    bety_b1_err = twiss_err_df_b1_filtered["bety"]
    
    # Beam 2 calculations
    mdx.job_init_sbs_b2(beam, 1) 
    mdx.sbs_twiss_b2()
    twiss_df_b2 = mdx.table.twiss.dframe()
    mdx.set_triplet_errors_b2(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
    mdx.sbs_twiss_b2()
    twiss_err_df_b2 = mdx.table.twiss.dframe()
    twiss_df_b2_filtered = filter_indices(twiss_df_b2, prefixes_to_filter)
    #tfs.write("twiss_df_b2_filtered.tfs", twiss_df_b2_filtered)

    twiss_err_df_b2_filtered = filter_indices(twiss_err_df_b2, prefixes_to_filter)
    #tfs.write("twiss_err_df_b2_filtered.tfs", twiss_err_df_b2_filtered)
    mqx_nom_values = twiss_df_b2_filtered.filter(like='mqx', axis=0)
    s_k1l = mqx_nom_values.s
    
    
    s_b2 = twiss_df_b2_filtered["s"]

    betx_b2_nom = twiss_df_b2_filtered["betx"]
    bety_b2_nom = twiss_df_b2_filtered["bety"]
    
    betx_b2_err = twiss_err_df_b2_filtered["betx"]
    bety_b2_err = twiss_err_df_b2_filtered["bety"]
    

    fig, axs = plt.subplots(3, 2, figsize=(18, 24), gridspec_kw={'height_ratios': [6, 6, 4]}, sharex='col')

    
    # Top-left subplot for beta beating in x for beam 1
    axs[0, 0].plot(s_b1, 100*(betx_b1_err - betx_b1_nom) / betx_b1_nom, color='#1f77b4')
    axs[0, 0].set_ylabel(r'$\frac{\Delta \beta_x}{\beta_x}$ [%]', fontsize = 29)
    axs[0, 0].set_ylim(-5, 5)
    axs[0, 0].set_title('Beam 1')
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(0, 1000)
    axs[0, 0].legend(fontsize=16)

    
        # Top-right subplot for beta beating in x for beam 2
    axs[0, 1].plot(s_b2, 100*(betx_b2_err - betx_b2_nom) / betx_b2_nom, color='#ff7f0e')
    #axs[0, 1].set_xlabel('Longitudinal Position [m]')
    axs[0, 1].set_ylabel#(r'$\frac{\Delta \beta_x}{\beta_x}$ [%]')
    axs[0, 1].set_title('Beam 2')
    axs[0, 1].set_ylim(-5, 5)
    axs[0, 1].grid(True)
    axs[0, 1].set_xlim(0, 1000)
    axs[0, 1].legend(fontsize=16)
    axs[0, 1].yaxis.set_visible(False)
    
    # Bottom-left subplot for beta beating in y for beam 1
    axs[1, 0].plot(s_b1, 100*(bety_b1_err - bety_b1_nom) / bety_b1_nom,  color='#1f77b4')
    axs[1, 0].set_ylabel(r'$\frac{\Delta \beta_y}{\beta_y}$ [%]', fontsize = 29)
    axs[1, 0].set_ylim(-5, 5)
    #axs[1, 0].set_xlabel('Longitudinal Position [m]')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlim(0, 1000)
    axs[1, 0].legend(fontsize=16)

    # Add vertical lines plot below bottom-left subplot
    add_vertical_lines(axs[2, 0], s_k1l, real_k1l, pred_k1l, res_k1l)
    axs[2, 0].set_xlabel('Longitudinal Position [m]', fontsize=20)
    axs[2, 0].set_ylabel (r'$\Delta K_{1L}  [m^{-2}]$', fontsize=20)
    axs[2, 0].set_xlim(0, 1000)
    #axs[2, 0].tick_params(labelleft=False)
    axs[2, 0].yaxis.set_visible(True)

    # Bottom-right subplot for beta beating in y for beam 2
    axs[1, 1].plot(s_b2, 100*(bety_b2_err - bety_b2_nom) / bety_b2_nom, color='#ff7f0e')
    axs[1, 1].set_xlabel#('Longitudinal Position [m]')
    axs[1, 1].set_ylim(-5, 5)
    axs[1, 1].grid(True)
    axs[1, 1].legend(fontsize=16)
    axs[1, 1].yaxis.set_visible(False)

    # Add vertical lines plot below bottom-right subplot
    add_vertical_lines(axs[2, 1], s_k1l, real_k1l, pred_k1l, res_k1l)
    axs[2, 1].set_xlabel('Longitudinal Position [m]', fontsize=20)
    axs[2, 1].set_xlim(0, 1000)
    #axs[2, 1].tick_params(labelleft=False)
    axs[2, 1].yaxis.set_visible(False)

    plt.tight_layout()
    plt.show()
    
 
        
def filter_indices(df, prefixes):
    mask = df.index.to_series().apply(lambda x: not any(x.startswith(prefix) for prefix in prefixes))
    return df[mask]

def add_vertical_lines(ax, s_k1l, real_k1l, pred_k1l, res_k1l):
        ax.vlines(x=s_k1l, ymin=0, ymax=real_k1l, color='red', linestyle='-', linewidth=5, label='Real')
        ax.vlines(x=s_k1l, ymin=0, ymax=pred_k1l, color='blue', linestyle='-', linewidth=5, label='Predicted')
        ax.vlines(x=s_k1l, ymin=0, ymax=res_k1l, color='green', linestyle='-', linewidth=5, label='Residual')
        ax.axhline(0, color='black', linestyle='--')  # Baseline
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.78, 1), ncol=3, fontsize=10)  
  
#Example of delta phi in the segment with quadrupole field strengths
def sbs_phase_adv(input_data, output_data, estimator, beam, index):
    predicted_data = estimator.predict([input_data[index]])
    corrected_errors = output_data[index] - predicted_data
    err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 = corrected_errors[0][-8:]
    pred_k1l = predicted_data[0][-8:]
    real_k1l = output_data[index][-8:]
    res_k1l = real_k1l - pred_k1l
 
    mdx = madx_ml_op()
    # Beam 1 calculations
    mdx.job_init_sbs_b1(beam, 1) #same here about the second variable 
    mdx.sbs_twiss_b1()
    twiss_df_b1 = mdx.table.twiss.dframe()
    mdx.set_triplet_errors_b1(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
    mdx.sbs_twiss_b1()
    twiss_err_df_b1 = mdx.table.twiss.dframe()
    
    
    prefixes_to_filter = ['ip', 'drift', 'mbas', 'mbcs']

    # Apply the filtering function to each DataFrame
    twiss_df_b1_filtered = filter_indices(twiss_df_b1, prefixes_to_filter)
    #tfs.write("twiss_df_b1_filtered.tfs", twiss_df_b1_filtered)

    twiss_err_df_b1_filtered = filter_indices(twiss_err_df_b1, prefixes_to_filter)
    #tfs.write("twiss_err_df_b1_filtered.tfs", twiss_err_df_b1_filtered)

    s_b1 = twiss_df_b1_filtered["s"]
    mux_b1_nom = twiss_df_b1_filtered["mux"]
    muy_b1_nom = twiss_df_b1_filtered["muy"]
    
    mux_b1_err = twiss_err_df_b1_filtered["mux"]
    muy_b1_err = twiss_err_df_b1_filtered["muy"]
    
    # Beam 2 calculations
    mdx.job_init_sbs_b2(beam, 1)
    mdx.sbs_twiss_b2()
    twiss_df_b2 = mdx.table.twiss.dframe()
    mdx.set_triplet_errors_b2(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
    mdx.sbs_twiss_b2()
    twiss_err_df_b2 = mdx.table.twiss.dframe()
    twiss_df_b2_filtered = filter_indices(twiss_df_b2, prefixes_to_filter)
    #tfs.write("twiss_df_b2_filtered.tfs", twiss_df_b2_filtered)

    twiss_err_df_b2_filtered = filter_indices(twiss_err_df_b2, prefixes_to_filter)
    #tfs.write("twiss_err_df_b2_filtered.tfs", twiss_err_df_b2_filtered)
    mqx_nom_values = twiss_df_b2_filtered.filter(like='mqx', axis=0)
    s_k1l = mqx_nom_values.s
    
    s_b2 = twiss_df_b2_filtered["s"]

    mux_b2_nom = twiss_df_b2_filtered["mux"]
    muy_b2_nom = twiss_df_b2_filtered["muy"]
    betx = twiss_df_b2_filtered.betx
    
    mux_b2_err = twiss_err_df_b2_filtered["mux"]
    muy_b2_err = twiss_err_df_b2_filtered["muy"]
    betx_err = twiss_err_df_b2_filtered.betx
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 24), gridspec_kw={'height_ratios': [6, 6, 4]}, sharex='col')

    # Top-left subplot
    axs[0, 0].plot(s_b1, mux_b1_err - mux_b1_nom, color='#1f77b4', lw=2)
    axs[0, 0].set_ylabel(r'$\Delta \phi_x$', fontsize=29)
    axs[0, 0].grid(True)
    axs[0, 0].set_title('Beam 1')
    axs[0, 0].set_ylim(-0.005, 0.005)
    axs[0, 0].set_xlim(0, 1000)
    axs[0, 0].legend(fontsize=18)

    # Top-right subplot
    axs[0, 1].plot(s_b2, mux_b2_err - mux_b2_nom, color='#ff7f0e', lw=2)
    axs[0, 1].set_ylabel(r'$\Delta \phi_x$', fontsize=29)
    axs[0, 1].grid(True)
    axs[0, 1].set_title('Beam 2')
    axs[0, 1].set_ylim(-0.005, 0.005)
    axs[0, 1].set_xlim(0, 1000)
    axs[0, 1].legend(fontsize=18)
    axs[0, 1].yaxis.set_visible(False)

    # Bottom-left subplot
    axs[1, 0].plot(s_b1, muy_b1_err - muy_b1_nom, color='#1f77b4', lw=2)
    axs[1, 0].set_ylabel(r'$\Delta \phi_y$', fontsize=29)
    axs[1, 0].grid(True)
    axs[1, 0].set_ylim(-0.005, 0.005)
    axs[1, 0].set_xlim(0, 1000)
    axs[1, 0].legend(fontsize=18)

    # Add vertical lines plot below bottom-left subplot
    add_vertical_lines(axs[2, 0], s_k1l, real_k1l, pred_k1l, res_k1l)
    axs[2, 0].set_xlabel('Longitudinal Position [m]', fontsize=20)
    axs[2, 0].set_ylabel (r'$\Delta K_{1L}  [m^{-2}]$', fontsize=20)
    axs[2, 0].set_xlim(0, 1000)
    #axs[2, 0].tick_params(labelleft=False)
    axs[2, 0].yaxis.set_visible(True)

    # Bottom-right subplot
    axs[1, 1].plot(s_b2, muy_b2_err - muy_b2_nom, color='#ff7f0e', lw=2)
    axs[1, 1].set_ylabel(r'$\Delta \phi_y$', fontsize=29)
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(-0.005, 0.005)
    axs[1, 1].set_xlim(0, 1000)
    axs[1, 1].legend(fontsize=18)
    axs[1, 1].yaxis.set_visible(False)

    # Add vertical lines plot below bottom-right subplot
    add_vertical_lines(axs[2, 1], s_k1l, real_k1l, pred_k1l, res_k1l)
    axs[2, 1].set_xlabel('Longitudinal Position [m]', fontsize=20)
    axs[2, 1].set_xlim(0, 1000)
    #axs[2, 1].tick_params(labelleft=False)
    axs[2, 1].yaxis.set_visible(False)

    plt.tight_layout()
    plt.show()



#Ploting of tripelt fiels distributions for the 3 values of noises in the training set with the fixed alpha parameter
def compare_triplet_pred_for_diff_noises(input_data, output_data):
    alphas = [0.001]
    noises = [0.0001, 0.001, 0.01]
    estimators = {}
    i = 0
    estimators_path = './estimators/'
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    bin_edges = np.linspace(-0.003, 0.003, 50)

    for alpha in alphas:
            for noise in noises:    
                filename = f"test_for_new_data_with_betastar_triplet_phases_028_alpha_{alpha}_ridge_{noise}.pkl"
                file_path = estimators_path + filename
                estimator = load(file_path)
                estimators[f"estimator_noise_{str(noise).replace('.', '')}_alpha{str(alpha).replace('.', '')}"] = estimator
                #triplet_error_dist_plot(input_data, output_data, estimator, noise, alpha)
                true_triplet, pred_triplet, residue, rms = get_mean_value_of_triplet_errors(input_data, output_data, estimator, noise, alpha)
                #mean_values.loc[alpha, noise] = mean
                #sbs_phase_adv(input_data, output_data, estimator, 1, 2)
                #sbs_beta_beat(input_data, output_data, estimator, 1, 2)
                axes[i].hist(true_triplet, bins=bin_edges, alpha=0.5, label='True')
                axes[i].hist(pred_triplet, bins=bin_edges, alpha=0.5, label='Predicted')
                axes[i].hist(residue, bins=bin_edges, alpha=0.5, label='Residuals')
                #axes[0].set_xlabel('Energy offset for the beam 1')
                axes[i].set_title(f'Noise: {noise}')
                axes[i].text(0.00015, 7000, r'$RMS = {:.2f} \times 10^{{-4}}$'.format(rms * 1e4), color='green', ha='left', fontsize=18)
                axes[i].set_ylim(0, 14000)
                axes[i].legend()
                 # if i > 0:
                #    axes[i].yaxis.set_visible(False) 
                axes[i].grid(True)

                i = i+1
                      
    axes[0].set_ylabel('Counts')
    plt.tight_layout()
    plt.show()
    
#Plot for tripled residuals RMS for the different pairs of noise and alpha in the training.
def rms_triplet_residuals_heatmap(input_data, output_data):    
    alphas = [0.001] 
    noises = [0.001, 0.01, 2, 5]
    estimators = {}
    estimators_path = './estimators/'
    rms_values = pd.DataFrame(index=alphas, columns=noises)

    for alpha in alphas:
            for noise in noises:
                #estimator names are corresponding noise and alpha with which the models were trained    
                filename = f"test_for_new_data_with_betastar_triplet_phases_028_alpha_{alpha}_ridge_{noise}.pkl"
                file_path = estimators_path + filename
                estimator = load(file_path)
                estimators[f"estimator_noise_{str(noise).replace('.', '')}_alpha{str(alpha).replace('.', '')}"] = estimator
                #triplet_error_dist_plot(input_data, output_data, estimator, noise, alpha)
                true_triplet, pred_triplet, residue, rms = get_mean_value_of_triplet_errors(input_data, output_data, estimator, noise, alpha)
                rms_values.loc[alpha, noise] = rms
                #sbs_phase_adv(input_data, output_data, estimator, 1, 2)
                #sbs_beta_beat(input_data, output_data, estimator, 1, 2)
           
    # Convert DataFrame values to float
    mean_values = rms_values.astype(float)

    # Fill NaN values with a placeholder (e.g., 0)
    mean_values.fillna(0, inplace=True)

    # Ensure all data is numeric
    mean_values = mean_values.apply(pd.to_numeric)
    mean_values *= 10000
           
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_values, annot=True, fmt=".4f", cmap='viridis', cbar=True)
    plt.xlabel('Noise')
    plt.ylabel(r'$\alpha$')
    #colorbar.set_label('Values multiplied by $10^4$')
    plt.title(r'RMS $[10^{{-4}}]$')
    plt.gca().invert_yaxis() 
    plt.show()        
    
        
def compare_beta_beat():
    rms_beta_beat_x_b1_noise_0001 = tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.0001.tfs")
    rms_beta_beat_y_b1_noise_0001 = tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.0001.tfs")

    
    rms_beta_beat_x_b1_noise_001 = tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.001.tfs")
    rms_beta_beat_y_b1_noise_001 = tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.001.tfs")
    
    rms_beta_beat_x_b1_noise_01 = tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.01.tfs")
    rms_beta_beat_y_b1_noise_01 = tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.01.tfs")
    
    rms_beta_beat_x_b1 = tfs.read("beta_beat/rms_beta_beat_y_b1_improved_model_estimator_noise_0.0001.tfs")
    rms_beta_beat_y_b1 = tfs.read("beta_beat/rms_beta_beat_y_b1_improved_model_estimator_noise_0.0001.tfs")
    # Reading the datasets
    datasets = {
        "x_0.0001": tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.0001.tfs"),
        "y_0.0001": tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.0001.tfs"),
        "x_0.001": tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.001.tfs"),
        "y_0.001": tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.001.tfs"),
        "x_0.01": tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.01.tfs"),
        "y_0.01": tfs.read("beta_beat/rms_beta_beat_y_b1_corrected_improved_model_estimator_noise_0.01.tfs"),
    }

    # Calculate mean values
    mean_values = {key: dataset.mean().mean() for key, dataset in datasets.items()}  # Get the overall mean

    # Create DataFrames for the heatmaps
    noise_levels = ["0.0001", "0.001", "0.01"]

    mean_x = pd.DataFrame(index=["0.001"], columns=noise_levels)
    mean_y = pd.DataFrame(index=["0.001"], columns=noise_levels)

    for noise in noise_levels:
        mean_x.loc["0.001", noise] = mean_values[f"x_{noise}"]
        mean_y.loc["0.001", noise] = mean_values[f"y_{noise}"]

    # Plot heatmaps
    plt.figure(figsize=(12, 6))

    # Heatmap for x
    plt.subplot(1, 2, 1)
    sns.heatmap(mean_x.astype(float), annot=True, cmap="viridis")
    plt.title("Mean RMS Values for x")
    plt.xlabel("Noise Level")
    plt.ylabel("Alpha")

    # Heatmap for y
    plt.subplot(1, 2, 2)
    sns.heatmap(mean_y.astype(float), annot=True, cmap="viridis")
    plt.title("Mean RMS Values for y")
    plt.xlabel("Noise Level")
    plt.ylabel("Alpha")

    plt.tight_layout()
    plt.show()


#Seeing beta beating coming from different scenarios, when only triplet erros were assignes in the data generation, when only arcs, when only systemati c in the triplet,
#when only misalignemt in the triplet
#Also seing beta beating coming from the enegy offset solely
def error_contributions(): 
    tw_nominal = tfs.read("twiss_nominal.tfs")
    tw_triplet_errors = tfs.read("b1_twiss_triplet_errors.tfs") #Twiss with only triplet errors (both systematic and misalig)
    tw_all_errors = tfs.read("b1_twiss_all_errors.tfs") #Twiss with arc and triplet errros
    tw_arc_errors = tfs.read("b1_twiss_arc_errors.tfs") #twiss with only arcs
    tw_syst_misalign = tfs.read("triplet_systematic_misalign_errors.tfs") #basically only triplets, with systematic and misalign
    tw_syst = tfs.read("triplet_systematic_errors.tfs") #twiss with only systematic in the triplet
    tw_mis = tfs.read("triplet_misalignment_errors.tfs") #twiss with only misalign in the triplet
    twiss_off = tfs.read("twiss_final.tfs") #when only offset was assigned
    twiss_nom = tfs.read("b1_nominal_elements.dat")


    beta_nom_x_b1 = tw_nominal.betx
    s = tw_nominal.s
    beta_triplet_x_b1 = tw_triplet_errors.betx
    beta_all_x_b1 = tw_all_errors.betx
    beta_arc_x_b1 = tw_arc_errors.betx
    beta_syst_mis = tw_syst_misalign.betx
    beta_syst = tw_syst.betx
    beta_mis = tw_mis.betx
    beta_off = twiss_off.BETX
    beta_nom = twiss_nom.BETX
    S = twiss_nom.S
    

    b2_tw_nominal = tfs.read("b2_twiss_nom.tfs")
    b2_tw_triplet_errors = tfs.read("b2_twiss_triplet_errors.tfs")
    b2_tw_all_errors = tfs.read("b2_twiss_all_errors.tfs")
    b2_tw_arc_errors = tfs.read("b2_twiss_arc_errors.tfs")

    beta_nom_x_b2 = b2_tw_nominal.betx
    s_b2 = b2_tw_nominal.s
    beta_triplet_x_b2 = b2_tw_triplet_errors.betx
    beta_all_x_b2 = b2_tw_all_errors.betx
    beta_arc_x_b2 = b2_tw_arc_errors.betx
    
    
    plt.figure(figsize=(10, 6))
    

    #plt.plot(S, 100 * (beta_off-beta_nom)/beta_nom, lw=2)
    plt.plot(s, 100 * (beta_syst-beta_nom_x_b1)/beta_nom_x_b1, label='Systematic', lw=2)
    plt.plot(s, 100*(beta_mis-beta_nom_x_b1)/beta_nom_x_b1, label='Misalignment', lw=2)

    plt.xlabel('Longitudinal Position (s)', fontsize=36)
    plt.ylabel(r'$\frac{\Delta \beta_x}{\beta_x}\, [\%]$', fontsize=36)
    #plt.title(r'ML model includes beta star in the features, noise $10^{-4}$ ')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for Beam 1
    ax1.set_title("Beam 1")
    ax1.set_ylabel(r"$\frac{\Delta \beta_x}{\beta_x} \, (\%)$")
    ax1.plot(s, 100 * (beta_triplet_x_b1 - beta_nom_x_b1) / beta_nom_x_b1, label="Triplet errors")
    ax1.plot(s, 100 * (beta_arc_x_b1 - beta_nom_x_b1) / beta_nom_x_b1, label="Arc errors")
    ax1.legend(loc='upper right')  # Position the legend in the upper right corner

    # Plot for Beam 2
    ax2.set_title("Beam 2")
    ax2.set_xlabel("S")
    ax2.set_ylabel(r"$\frac{\Delta \beta_x}{\beta_x} \, (\%)$")
    ax2.plot(s_b2, 100 * (beta_triplet_x_b2 - beta_nom_x_b2) / beta_nom_x_b2, label="Triplet errors")
    ax2.plot(s_b2, 100 * (beta_arc_x_b2 - beta_nom_x_b2) / beta_nom_x_b2, label="Arc errors")
    ax2.legend(loc='upper right')  # Position the legend in the upper right corner

    plt.tight_layout()
    plt.show()

#Just testing beam performance at IP8 in diffrenet scenarios
def ip8_twiss():
    tw_nominal_b1 = tfs.read("IP5_twiss_nominal_b1.tfs")
    tw_corrected_b1 =tfs.read("IP5_twiss_corrected_b1.tfs")
    tw_error_b1 =tfs.read("IP5_twiss_full_error_b1.tfs")

    tw_nominal_b2 = tfs.read("IP5_twiss_nominal_b2.tfs")
    tw_corrected_b2 =tfs.read("IP5_twiss_corrected_b2.tfs")
    tw_error_b2 =tfs.read("IP5_twiss_full_error_b2.tfs")
    
    s_b1=tw_nominal_b1.s
    betx_b1 = tw_nominal_b1.betx
    betx_corr_b1 = tw_corrected_b1.betx
    betx_err_b1 = tw_error_b1.betx

    mux_b1 = tw_nominal_b1.mux
    mux_corr_b1 = tw_corrected_b1.mux
    mux_err_b1 = tw_error_b1.mux

    
    s_b2=tw_nominal_b2.s
    betx_b2 = tw_nominal_b2.betx
    betx_corr_b2 = tw_corrected_b2.betx
    betx_err_b2 = tw_error_b2.betx
    mux_b2 = tw_nominal_b2.mux
    mux_corr_b2 = tw_corrected_b2.mux
    mux_err_b2 = tw_error_b2.mux
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for Beam 1
    ax1.set_title("Beam 1")
    ax1.set_ylabel(r"$\frac{\Delta \beta_x}{\beta_x} \, (\%)$")
    ax1.plot(s_b1, 100 * (betx_corr_b1 - betx_b1) / betx_b1, label="corrected ")
    ax1.plot(s_b1, 100 * (betx_err_b1 - betx_b1) / betx_b1, label="error ")

    ax1.legend(loc='upper right')  # Position the legend in the upper right corner

    # Plot for Beam 2
    ax2.set_title("Beam 2")
    ax2.set_xlabel("S")
    ax2.set_ylabel(r"$\frac{\Delta \beta_x}{\beta_x} \, (\%)$")
    ax2.plot(s_b2, 100 * (betx_corr_b2 - betx_b2) / betx_b2, label="corrceted ")
    ax2.plot(s_b2, 100 * (betx_err_b2 - betx_b2) / betx_b2, label=" error")

    ax2.legend(loc='upper right')  # Position the legend in the upper right corner

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    

    plt.plot(s_b1, betx_b1, label='beta nom', lw=2)
    plt.plot(s_b1, betx_err_b1, label='beta err', lw=2)
    plt.xlabel('Longitudinal Position (s)')
    plt.ylabel(r'$\beta$-function')
    #plt.title(r'ML model includes beta star in the features, noise $10^{-4}$ ')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    

    plt.plot(s_b1, mux_corr_b1-mux_b1, label='corrected',  lw=2)
    plt.plot(s_b1, mux_err_b1-mux_b1, label='error',  lw=2)

    plt.xlabel('Longitudinal Position (s)')
    plt.ylabel(r'$\Delta \mu$')
    #plt.title(r'ML model includes beta star in the features, noise $10^{-4}$ ')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for Beam 1
    ax1.set_title("Beam 1")
    ax1.set_ylabel(r'$\Delta \mu$')
    ax1.plot(s_b1, mux_corr_b1-mux_b1, label='corrected',  lw=2)
    ax1.plot(s_b1, mux_err_b1-mux_b1, label='error',  lw=2)

    ax1.legend(loc='upper right')  # Position the legend in the upper right corner

    # Plot for Beam 2
    ax2.set_title("Beam 2")
    ax2.set_xlabel("S")
    ax2.set_ylabel(r'$\Delta \mu$')
    ax2.plot(s_b2, mux_corr_b2-mux_b2, label='corrected',  lw=2)
    ax2.plot(s_b2, mux_err_b2-mux_b2, label='error',  lw=2)

    ax2.legend(loc='upper right')  # Position the legend in the upper right corner

    plt.tight_layout()
    plt.show()
    


#Main
data_path = "data_phase_adv_triplet"   #In the data only magnet errors are assigned
data_path_with_off = "data_include_offset" #Here magnet errors and offsets are assigned
test_data_with_off = "test_data_with_offset" #test data from the one having offset, They are sorted and lats 20% is the test set, I copied several files from the data folder
test_data_with_right_off = "test_data_right_offset" #Same as previous, Right, because in the precious I did not have normal distributions of the values and I changed
test_data_path = "test_data" #This is test data of the set with only magnet errors, copied few files from last 20%
data_path_without_missalign = "test_set_witout_misalignment" #Misaligmnets were cousing the problem for local corrections, Here is 100 simulation for the tes data excluding misalignment in the triplets
noise =  1e-3 #noise value on the test set, on th phase advances

input_data, output_data = merge_data(test_data_with_right_off, noise) #It takes input and output data separatelly and sets noise on it

# This input and output are very oftes argument for the plotting functions
right_ffset_est = load('./estimators/with_off_test_for_new_data_with_betastar_alpha_0.01_ridge_0.001.pkl')
estimator_001_noise_alpha_01 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.01_ridge_0.001.pkl')
estimator_001_noise_alpha_001 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.001_ridge_0.01.pkl')
estimator_0001_noise = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_1e-05_ridge_0.0.pkl')
off_estimator = load('./estimators/right_off_test_for_new_data_with_betastar_triplet_phases_028_alpha_0.0001_ridge_0.0.pkl')
off_dist_plot(input_data, output_data, off_estimator)
#triplet_error_dist_plot(input_data, output_data,estimator_001_noise_alpha_01, 0 , 0 )
#sbs_beta_beat(input_data, output_data, estimator_001_noise_alpha_001, 1, 35)
#sbs_phase_adv(input_data, output_data, estimator_001_noise_alpha_001, 1, 35)
#estimator_01_noise_alpha_001 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.001_ridge_0.01.pkl')#
#off_dist_plot(input_data, output_data, right_ffset_est)
#compare_triplet_pred_for_diff_noises(input_data, output_data)

#triplet_error_dist_plot(input_data, output_data,estimator_01_noise_alpha_001, 1, 1 )
alphas = [0.0001, 0.001, 0.01, 0.1]
noises = [0.001]
estimators = {}
#estimators_path = './estimators/'
#rms_values = pd.DataFrame(index=alphas, columns=noises)
#right_ffset_est_with_zero_noise = load('./estimators/right_off_test_for_new_data_with_betastar_triplet_phases_028_alpha_0.0001_ridge_0.0.pkl')
#estimator_01_noise_alpha_001 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.001_ridge_0.01.pkl')
#rms_triplet_residuals_heatmap(input_data, output_data)
#off_dist_plot(input_data, output_data, right_ffset_est_with_zero_noise)
#sbs_beta_beat(input_data, output_data, estimator_001_noise_alpha_01, 1, 2)
#compare_beta_beat()
#estimator_betastar_only_feature = load('./estimators/test_for_new_data_without_betastar_only_028_alpha_0.001_ridge_0.pkl')
#estimator_betastar_only_feature_offset_pred = load('./estimators/with_off_test_for_new_data_without_betastar_only_028_alpha_0.001_ridge_0.pkl')
#off_dist_plot(input_data, output_data, estimator_betastar_only_feature_offset_pred )
#sbs_phase_adv(input_data, output_data, estimator_01_noise_alpha_001, 1, 35)


OPTICS_30CM_2024 = '/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx'

