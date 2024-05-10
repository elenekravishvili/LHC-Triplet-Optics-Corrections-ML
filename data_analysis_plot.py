import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pn
from pathlib import Path
from joblib import load



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
    
    all_samples = np.load('./data_2/{}.npy'.format(set_name), allow_pickle=True)

    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2, n_disp_b1, n_disp_b2,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors = all_samples.T
    
    input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
            np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
            np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
            np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    
    output_data = np.vstack(triplet_errors)
                
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



def obtain_triplet_errors(input_data, output_data, estimator):
    # Function that gives the predicted and real values of errors for a numpy array inputs
    pred_data = estimator.predict(input_data)
   
    pred_data = normalize(pred_data)
    output_data = normalize(output_data)

    pred_triplet = np.hstack(pred_data)
    true_triplet = np.hstack(output_data)

    return pred_triplet, true_triplet


def triplet_error_dist_plot(input_data, output_data, estimator):
    pred_triplet, true_triplet= obtain_triplet_errors(input_data, output_data, estimator)
    residue=true_triplet-pred_triplet
    plt.hist(pred_triplet, bins=20, color='blue', alpha=0.5, label='predicted')
    plt.hist(true_triplet, bins=20, color='red', alpha=0.5, label='true')
    plt.hist(residue, bins=20, color='green', alpha=0.5, label='residuals')
    plt.xlabel('Normalized Data')
    plt.ylabel('Counts')
    plt.title('Histogram of Normalized Data')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show()



data_path = "data_2/"
noise = 1E-4
Nth_array=68
#output_example_result_tfs()
input_data, output_data = merge_data(data_path, noise)


# Load the saved model
loaded_estimator = load('./estimators/with_betastartriplet_phases_only_028_ridge_0.0001.pkl')


triplet_error_dist_plot(input_data, output_data, loaded_estimator)

# Convert the list of arrays to a single 2D array
#normalized_true_data = np.array(normalized_predicted_data)




#normalized_data=[]
#for i in range(len(predictions)):
#    normalized_data.append(predictions[i] / nom_k1)

#dif=predicted_output-real_output
#diff=dif.flatten()

#x_coordinates = np.arange(len(real_output))



"""
# Plotting the histogram
plt.figure(figsize=(12, 6))

# Plot the real values
#plt.bar(x_coordinates, real_output, color='blue', alpha=0.5, label='Real')

# Plot the predicted values
#plt.bar(x_coordinates, predicted_output, color='red', alpha=0.5, label='Predicted')

# Plot the differences
plt.bar(x_coordinates, diff, color='green', alpha=0.5, label='Difference')

plt.title('Histogram of Real Values, Predicted Values, and Differences')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

"""

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