import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pn
from pathlib import Path
from joblib import load



def merge_data(data_path, noise):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
    B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")
    beta_x_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["BETX"])
    beta_y_b1_nom =np.array(B1_ELEMENTS_MDL_TFS.BETY)
    beta_x_b2_nom =np.array(B2_ELEMENTS_MDL_TFS["BETX"])
    beta_y_b2_nom =np.array(B2_ELEMENTS_MDL_TFS.BETY)
    beta_beat_x_b1, beta_beat_y_b1, beta_beat_x_b2, beta_beat_y_b2 = [], [], [], []
    rms_x_b1, rms_y_b1, rms_x_b2, rms_y_b2  = [], [], [], []
    for file_name in file_names:
        beta_x_b1, beta_y_b1, beta_x_b2, beta_y_b2 = load_data(file_name, noise)
    
        for i in range(50):
            bb_x1 = (beta_x_b1[i]-beta_x_b1_nom)/beta_x_b1_nom
            rms_x1 = np.sqrt(np.mean(bb_x1**2, axis=0))
            rms_x_b1.append(rms_x1)

            bb_y1 = (beta_y_b1[i]-beta_y_b1_nom)/beta_y_b1_nom
            rms_y1 = np.sqrt(np.mean(bb_y1**2, axis=0))
            rms_y_b1.append(rms_y1)
            
            bb_x2 = (beta_x_b2[i]-beta_x_b2_nom)/beta_x_b2_nom
            rms_x2 = np.sqrt(np.mean(bb_x2**2, axis=0))
            rms_x_b2.append(rms_x2)

            bb_y2 = (beta_y_b2[i]-beta_y_b2_nom)/beta_y_b2_nom
            rms_y2 = np.sqrt(np.mean(bb_y2**2, axis=0))
            rms_y_b2.append(rms_y2) 
    #x=np.hstack(beta_beat_x_b1)
    rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
    rms_beta_beat_y_b1 = np.hstack(rms_y_b1)
    rms_beta_beat_x_b2 = np.hstack(rms_x_b2)
    rms_beta_beat_y_b2 = np.hstack(rms_y_b2)
    
    
    plt.hist(rms_beta_beat_x_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, x')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show()

    plt.hist(rms_beta_beat_y_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, y')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show() 

    plt.hist(rms_beta_beat_x_b2, bins=15, color='green', alpha=0.5, label='rms of beta beating beam2, x')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show()

    plt.hist(rms_beta_beat_y_b2, bins=15, color='green', alpha=0.5, label='rms of beta beating beam2, y')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show()



def load_data(set_name, noise):
    
    all_samples = np.load('./data_include_offset/{}.npy'.format(set_name), allow_pickle=True)

    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors, dpp_1, dpp_2 = all_samples.T
    
    input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
            np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
            np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
            np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    #input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
     #       np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    
    output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)

    betas = beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2

    return np.vstack(beta_bpm_x_b1), np.vstack(beta_bpm_y_b1), np.vstack(beta_bpm_x_b2), np.vstack(beta_bpm_y_b2)



data_path = "data_include_offset/"
noise = 1E-4
Nth_array=68

merge_data(data_path, noise)
