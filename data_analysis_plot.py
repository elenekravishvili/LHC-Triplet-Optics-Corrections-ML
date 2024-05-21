import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
from pathlib import Path
from joblib import load
from cpymad import madx



def merge_data(data_path, noise):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    for file_name in file_names:
        aux_input, aux_output = load_data(file_name, noise)
        input_data.append(aux_input)
        output_data.append(aux_output)

    return np.concatenate(input_data), np.concatenate(output_data)


def load_data(set_name, noise):
    
    all_samples = np.load('./data_include_offset/{}.npy'.format(set_name), allow_pickle=True)

    #delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
     #   delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
      #      delta_muy_b2,b1_disp, b2_disp,\
       #     beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
        #    triplet_errors = all_samples.T
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
     #      np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    
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


def triplet_error_dist_plot(input_data, output_data, estimator):
    pred_triplet, true_triplet= obtain_triplet_errors(input_data, output_data, estimator)
    residue=true_triplet-pred_triplet
    plt.hist(pred_triplet, bins=15, color='blue', alpha=0.5, label='predicted')
    plt.hist(true_triplet, bins=15, color='red', alpha=0.5, label='true')
    plt.hist(residue, bins=15, color='green', alpha=0.5, label='residuals')
    plt.xlabel('Normalized triplet errors')
    plt.ylabel('Counts')
    plt.title('Histogram of Normalized Data')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show()





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


#mdx = madx_ml_op(stdout=False)
OPTICS_30CM_2024 = '/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx'
data_path = "data_phase_adv_triplet"
data_path_with_off = "data_include_offset"
noise = 1E-4
Nth_array=68
loaded_estimator_with_off = load('./estimators/test_2_triplet_phases_028_ridge_0.0001.pkl')
loaded_estimator = load('./estimators/triplet_phases_only_028_ridge_0.001.pkl')

input_data, output_data = merge_data(data_path_with_off, noise)
#triplet_error_dist_plot(input_data, output_data, loaded_estimator)


predicted_data = loaded_estimator_with_off.predict(input_data)


pred_triples = predicted_data[:, :32]
real_triplet = output_data [:,:32]
real_energy_off = output_data [:, -2:]
energy_off_pred = np.hstack(predicted_data[:, -2:])
energy_off_real = np.hstack(real_energy_off)
print(pred_triples.shape)

res = energy_off_real-energy_off_pred

plt.hist(energy_off_pred, bins=20, color='blue', alpha=0.5, label='predicted', histtype=u'step')
plt.hist(energy_off_real, bins=20, color='red', alpha=0.5, label='true', histtype=u'step')
plt.hist(res, bins=20, color='green', alpha=0.5, label='residuals', histtype=u'step')
plt.xlabel('Energy offser')
plt.ylabel('Counts')
plt.title('Histogram of Energy offset')
plt.legend()  # Add legend to display labels
plt.grid(True)
plt.show()



"""
tw_nominal=tfs.read("twiss_nominal.tfs")
#tw_nominal=tw_nominal.set_index('NAME')
tw_off=tfs.read("twiss_corrected.tfs")
#tw_off=tw_off.set_index('NAME')

beta_nom=tw_nominal.BETX.to_numpy()
beta_off=tw_off.BETX.to_numpy()
beta_beating=(beta_off-beta_nom)/beta_nom
s=tw_nominal.S.to_numpy()

plt.title("Beta function for nominal and energy offset case")
plt.xlabel("S")
plt.ylabel("Beta funtion")
plt.plot(s, beta_nom, label="Nominal Beta")
plt.plot(s, beta_off, label="Off Beta")
plt.legend()
plt.show()



plt.title("Beta beating: energy offset")
plt.xlabel("S")
plt.ylabel("Beta beating")
plt.plot(s, beta_beating, label="Beta beating")
plt.legend()
plt.show()
"""