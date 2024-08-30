
#%%

import numpy as np
import pandas as pd
import tfs
from pathlib import Path
import joblib

""" ------------------------------------------------------- 0 --------------------------------------------------
Script with auxiliary functions for loading data specially, adding noise, choosing which variables to load...

----------------------------------------------------------- 0 --------------------------------------------------  """

def main():        
    data_path = "data_2/"
    noise = 1E-4
    
    #output_example_result_tfs()
    input_data, output_data = merge_data(data_path, noise)
    print(output_data.shape)

def load_data(set_name, noise):
    
    all_samples = np.load('./test_set_witout_misalignment/{}.npy'.format(set_name), allow_pickle=True)

    #all_samples = np.load('./data_phase_adv_triplet/{}.npy'.format(set_name), allow_pickle=True)
    #all_samples = np.load('./data_include_offset/{}.npy'.format(set_name), allow_pickle=True) #path to the data when offset is included in the targets
    #all_samples = np.load('./data_right_offset/{}.npy'.format(set_name), allow_pickle=True)

    
    #The configuration of samples are different in two cass, data having only magnet errors and data having for magnet errors and offsets
    
    #Following one down is for the data where only magnet errors are
    
    #delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
     # delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
      #      delta_muy_b2,\
       #      beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
        #    triplet_errors, dpp_1, dpp_2 = all_samples.T
    
     #This is for the data where magnets and offsets are
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
       delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,disp_1, disp_2,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors = all_samples.T


    #, arc_errors_b1, arc_errors_b2, \
    #        mqt_errors_b1, mqt_errors_b2, mqt_knob_errors_b1, mqt_knob_errors_b2, misalign_errors
    
    #B1_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
    #B2_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")
    
    #tw_perturbed_b1 = pd.DataFrame(columns=["BETX", "BETY"])
    #tw_perturbed_b1.BETX=beta_bpm_x_b1[0]
    #tw_perturbed_b1.BETY=beta_bpm_y_b1[0]

    #tw_perturbed_b2 = pd.DataFrame(columns=["BETX", "BETY"])    
    #tw_perturbed_b2.BETX=beta_bpm_x_b2[0]
    #tw_perturbed_b2.BETY=beta_bpm_y_b2[0]

    #print(len(B1_MONITORS_MDL_TFS))
    #print(len(tw_perturbed_b1))
    
    #plot_example_betabeat(B1_MONITORS_MDL_TFS, tw_perturbed_b1, 1)
    #plot_example_betabeat(B2_MONITORS_MDL_TFS, tw_perturbed_b2, 2)
    
    # select features for input
    # Optionally: add noise to simulated optics functions
    #n_disp_b1 = [add_dispersion_noise(n_disp, noise) for n_disp in n_disp_b1]  
    #n_disp_b2 = [add_dispersion_noise(n_disp, noise) for n_disp in n_disp_b2]

    
    delta_mux_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b1, beta_bpm_x_b1)]
    delta_muy_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b1, beta_bpm_y_b1)]
    delta_mux_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b2, beta_bpm_x_b2)]
    delta_muy_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b2, beta_bpm_y_b2)]


    delta_beta_star_x_b1_with_noise = add_beta_star_noise(delta_beta_star_x_b1)
    delta_beta_star_y_b1_with_noise = add_beta_star_noise(delta_beta_star_y_b1)
    delta_beta_star_x_b2_with_noise = add_beta_star_noise(delta_beta_star_x_b2)
    delta_beta_star_y_b2_with_noise = add_beta_star_noise(delta_beta_star_y_b2)

    
    #Input and output can be changed depending on the desired model configuration
    input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1_with_noise), np.vstack(delta_beta_star_y_b1_with_noise), \
        np.vstack(delta_beta_star_x_b2_with_noise), np.vstack(delta_beta_star_y_b2_with_noise),\
        np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    
    #input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
     #   np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)  
    
    #input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1_with_noise), np.vstack(delta_beta_star_y_b1_with_noise), \
     #   np.vstack(delta_beta_star_x_b2_with_noise), np.vstack(delta_beta_star_y_b2_with_noise)), axis=1)  
    
    #output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)
    output_data = np.vstack(triplet_errors)
    #
    return input_data, output_data


    #input_data = np.concatenate((np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
    #    np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
    #    np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
    #    np.vstack(delta_mux_b2), np.vstack(delta_muy_b2), \
    #    np.vstack(n_disp_b1), np.vstack(n_disp_b2)
    #    ), axis=1)

    # select targets for output

    
    #output_data = np.concatenate((
    # np.vstack(triplet_errors), np.vstack(arc_errors_b1),\
    #                            np.vstack(arc_errors_b2), np.vstack(mqt_knob_errors_b1), \
    #                            np.vstack(mqt_knob_errors_b2), np.vstack(misalign_errors)), axis=1)                         

def merge_data(data_path, noise):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = sorted(list(Path(data_path).glob('**/*.npy')))
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    for file_name in file_names:
        print(f"Processing file: {file_name}") 
        aux_input, aux_output = load_data(file_name, noise)
        input_data.append(aux_input)
        output_data.append(aux_output)

    return np.concatenate(input_data), np.concatenate(output_data)


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

def save_np_errors_tfs(np_errors, filename):
    #This is the tfs format that can be read, this model of file is then copied and filled
    error_tfs_model_b1 = tfs.read_tfs("./data_analysis/errors_b1.tfs")
    error_tfs_model_b2 = tfs.read_tfs("./data_analysis/errors_b2.tfs")

    #Function that takes np errors and outputs .tfs file with all error values
    with open("./data_analysis/mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "") for name in lines]

    # Recons_df is a dataframe with the correct names and errors but not format
    recons_df = pd.DataFrame(columns=["NAME","K1L"])
    recons_df.K1L = np_errors
    recons_df.NAME = names
    
    for beam, error_tfs_model in enumerate([error_tfs_model_b1, error_tfs_model_b2]):
        for i in range(len(error_tfs_model)):
            # check if the name is in recons_df
            if error_tfs_model.loc[i, 'NAME'] in list(recons_df['NAME']):
                error_tfs_model.loc[i, 'K1L'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][1]
            
    tfs.writer.write_tfs(tfs_file_path=f"./data_analysis/b1_{filename}", data_frame=error_tfs_model_b1)
    tfs.writer.write_tfs(tfs_file_path=f"./data_analysis/b2_{filename}", data_frame=error_tfs_model_b2)

def add_phase_noise(phase_errors, betas, expected_noise):
    #Add noise to generated phase advance deviations as estimated from measurements
    my_phase_errors = np.array(phase_errors)
    noises = np.random.standard_normal(phase_errors.shape)
    betas_fact = (expected_noise * (171**0.5) / (betas**0.5))
    noise_with_beta_fact = np.multiply(noises, betas_fact)

    phase_errors_with_noise = my_phase_errors + noise_with_beta_fact
    return phase_errors_with_noise


def add_dispersion_noise(disp_errors, noise):
    # Add noise to generated dispersion deviations as estimated from measurements in 2018
    my_disp_errors = np.array(disp_errors)
    noises = noise * np.random.noncentral_chisquare(4, 0.0035, disp_errors.shape)
    disp_errors_with_noise = my_disp_errors + noises
    
    return disp_errors_with_noise

def add_beta_star_noise_for_training(beta_star, noise):
    my_beta_star = np.array(beta_star)
    noises = np.random.standard_normal(beta_star.shape)
    # Truncate noise within 5 sigma
    max_val = 5 * noise
    min_val = -5 * noise
    clipped_noises = np.clip(noises, min_val, max_val)
    beta_with_noise = my_beta_star + noise * clipped_noises
    return beta_with_noise

#for prediction add 3% noise of the beta star
#def add_beta_star_noise(delta_beta_star_x_b1, noise_percentage=0.03):
 #   noise_factor = 1 + noise_percentage
  #  noisy_value = delta_beta_star_x_b1 * noise_factor
   # return noisy_value


#Adds noise on beta beating around IPs, (star is not quite relevant name). Noise values is 3% of the actual beating with the normal dist truncated in 3 sigma
def add_beta_star_noise(delta_beta_star_x_b1, noise_percentage=0.03, sigma=3):
    noise = noise_percentage * delta_beta_star_x_b1
    raw_noise = np.random.randn(*delta_beta_star_x_b1.shape)
    
    # Truncate noise within 3 sigma
    truncated_noise = np.clip(raw_noise, -sigma, sigma)
    
    noisy_value = delta_beta_star_x_b1 + noise * truncated_noise
    return noisy_value



def output_example_result_tfs():
    input_data, output_data = load_data("test", 1e-3)
    estimator = joblib.load(f'./estimators/estimator_ridge_0.001.pkl') 

    true_error = output_data[:1]
    #pred_error = estimator.predict(input_data[:1])

    save_np_errors_tfs(true_error[0], "true_example.tfs")
    #save_np_errors_tfs(pred_error[0], "pred_example.tfs")


if __name__ == "__main__":
    main()
# %%
