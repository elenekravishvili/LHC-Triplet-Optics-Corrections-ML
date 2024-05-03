import numpy as np
import pandas as pd
import tfs
from pathlib import Path
import joblib
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


from plots import plot_noise_vs_metrics
from plots import plot_learning_curve

from data_utils import load_data
from data_utils import merge_data
from data_utils import save_np_errors_tfs

import random
import matplotlib.pyplot as plt

import scipy.stats



def add_phase_noise(phase_errors, betas, expected_noise):
    #Add noise to generated phase advance deviations as estimated from measurements
    my_phase_errors = np.array(phase_errors)
    noises = np.random.standard_normal(phase_errors.shape)
    betas_fact = (expected_noise * (171**0.5) / (betas**0.5))
    noise_with_beta_fact = np.multiply(noises, betas_fact)
    phase_errors_with_noise = my_phase_errors + noise_with_beta_fact
    return phase_errors_with_noise


def load_data(set_name, noise):
    
    all_samples = np.load('./data/{}.npy'.format(set_name), allow_pickle=True)
    
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2, n_disp_b1, n_disp_b2,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors = all_samples.T
    
 

    delta_mux_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b1, beta_bpm_x_b1)]
    delta_muy_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b1, beta_bpm_y_b1)]
    delta_mux_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b2, beta_bpm_x_b2)]
    delta_muy_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b2, beta_bpm_y_b2)]

    input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    

    
    output_data = np.vstack(triplet_errors)
 

    return input_data, output_data




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


def train_model(input_data, output_data, algorithm, noise):  

    #Function that loads the data and trains the chosen model with a given noise level
    indices = np.arange(len(input_data))
    # Generating new test split or loading old one
    if GEN_TEST_SPLIT==True:
        train_inputs, test_inputs, train_outputs, test_outputs, indices_train, indices_test = train_test_split(
            input_data, output_data, indices, test_size=0.2, random_state=None)
        np.save("./data_analysis/test_idx.npy", indices_test)
        np.save("./data_analysis/train_idx.npy", indices_train)
    else:
        indices_test = np.load("./data_analysis/test_idx.npy")
        indices_train = np.load("./data_analysis/train_idx.npy")
        train_inputs, test_inputs, train_outputs, test_outputs = input_data[indices_train], input_data[indices_test],\
                                                                output_data[indices_train], output_data[indices_test]
    
    # create and fit a regression model
    if algorithm == "ridge":
        ridge = linear_model.Ridge(tol=1e-50, alpha=5e-4) #normalize=false
        estimator = BaggingRegressor(base_estimator=ridge, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=1, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "linear":
        linear = linear_model.LinearRegression()
        estimator = BaggingRegressor(base_estimator=linear, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)    
        
    elif algorithm == "tree":
        tree = DecisionTreeRegressor(criterion="squared_error", max_depth=10) #5
        estimator = tree
        estimator.fit(train_inputs, train_outputs)

    # Optionally: save fitted model or load already trained model        
    joblib.dump(estimator, f'./estimators/triplet_phases_only_028_{algorithm}_{noise}.pkl')           

    # Check scores: explained variance and MAE

    y_true_train, y_pred_train = train_outputs, estimator.predict(train_inputs) 
    y_true_test, y_pred_test = test_outputs, estimator.predict(test_inputs) 


    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    mae_test = mean_absolute_error(y_true_test, y_pred_test)

    r2_train = r2_score(y_true_train, y_pred_train)
    r2_test = r2_score(y_true_train, y_pred_train)

    mae_train_triplet = mean_absolute_error(y_true_train[:,:32], y_pred_train[:,:32])
    mae_test_triplet = mean_absolute_error(y_true_test[:,:32], y_pred_test[:,:32])

    return  mae_train, mae_test,  mae_test_triplet, r2_train, r2_test

GEN_TEST_SPLIT = True # If a new test split is needed
data_path = "data/"
set_name = "100%triplet_ip1_100%ip5_56205"
TRAIN = True
MERGE = True
algorithm = "ridge"

metrics, n_samples = [], []
noises = [1e-2] # np.logspace(-5, -2, num=10)



for noise in noises:

    if MERGE == True:
        input_data, output_data = merge_data(data_path, noise)

        print(input_data.shape)
    else:
        input_data, output_data = load_data(set_name, noise)


    if TRAIN==True:
        n_splits=15
        input_data = np.array_split(input_data, n_splits, axis=0)
        output_data = np.array_split(output_data, n_splits, axis=0)
        print(input_data[0].shape)
        print(input_data[1].shape)
        print(input_data[2].shape)

        for i in range(n_splits):
            results = train_model(np.vstack(input_data[:i+1]), np.vstack(output_data[:i+1]),
                                    algorithm=algorithm, noise=noise)
            n_samples.append(len(np.vstack(input_data[:i+1])))
            metrics.append(results)

        #plot_learning_curve(n_samples, metrics, algorithm)


metrics = np.array(metrics, dtype=object)
#MAE
print(n_samples, metrics[:,1])                                                                                                                      
plt.title("Mean Average Error")
plt.xlabel("N Samples")
plt.ylabel("MAE")
plt.plot(n_samples, metrics[:,0], label="Train", marker='o')
plt.plot(n_samples, metrics[:,1], label="Test", marker='o')
plt.legend()
plt.show()
#plt.savefig(f"./figures/test1_mae_{algorithm}.pdf")
plt.clf()
plt.title("Correlation Coefficient")
plt.xlabel("N Samples")
plt.ylabel(r"$R^2$")
plt.plot(n_samples, metrics[:,3], label="Train", marker='o')
plt.plot(n_samples, metrics[:,4], label="Test", marker='o')
plt.legend()
plt.show()



"""   
for i in range(n_splits):
    results = train_model(np.vstack(input_data[:i+1]), np.vstack(output_data[:i+1]),
                            algorithm=algorithm, noise=noise)
    n_samples.append(len(np.vstack(input_data[:i+1])))
    metrics.append(results)

plot_learning_curve(n_samples, metrics, algorithm)



print("begin")
print(x)
print("middle")
print(result)
print("end")





"""




"""
OPTICS_30CM_2024 = '/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx'

index=1


QX = 62.28
QY = 60.31

np.random.seed(seed=None)
seed = random.randint(0, 999999999)
mdx = madx_ml_op()
mdx.job_nominal2024()
tw_nominal=mdx.table.twiss.dframe()
#tfs.write('test_tw_nominal.tfs', tw_nominal)
# BEAM 1

mdx.job_magneterrors_b1(OPTICS_30CM_2024, str(index), seed)
b1_tw_before_match = mdx.table.twiss.dframe() # Twiss before match
#tfs.write('test_b1_tw_before_match.tfs', b1_tw_before_match)

mdx.match_tunes_b1()
b1_tw_after_match = mdx.table.twiss.dframe()# Twiss after match
#tfs.write('test_b1_tw_after_match.tfs', b1_tw_after_match)
common_errors = mdx.table.cetab.dframe() # Errors for both beams, triplet errors
#tfs.write('test_common_errors.tfs', common_errors)
b1_errors = mdx.table.etabb1.dframe() # Table error for MQ- magnets
#tfs.write('test_b1_errors.tfs', b1_errors)

#tfs.writer.write_tfs(tfs_file_path=f"b1_correct_example.tfs", data_frame=b1_errors)
#tfs.writer.write_tfs(tfs_file_path=f"b1_common_errors.tfs", data_frame=common_errors)
#tfs.writer.write_tfs(tfs_file_path=f"b1_example_twiss.tfs", data_frame=b1_tw_after_match)

# BEAM 2
mdx.job_magneterrors_b2(OPTICS_30CM_2024, str(index), seed)

b2_tw_before_match = mdx.table.twiss.dframe() # Twiss before match
#tfs.write('test_b2_tw_before_match.tfs', b2_tw_before_match)
mdx.match_tunes_b2()

b2_tw_after_match = mdx.table.twiss.dframe()# Twiss after match
tfs.write('test_b2_tw_after_match_check.tfs', b2_tw_after_match)
#twiss_data_b2 = mdx.table.twiss.dframe() # Relevant to training Twiss data
b2_errors= mdx.table.etabb2.dframe() # Table error for MQX magnets
#tfs.write('test_b2_errors.tfs', b2_errors)





B1_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
B2_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")











"""







"""


def get_phase_adv(total_phase, tune):
    total_phase = np.array(total_phase)
    phase_diff = np.diff(total_phase)
    last_to_first = total_phase[0] - (total_phase[-1] - tune)
    phase_adv = np.append(phase_diff, last_to_first)
    return phase_adv

def uppercase_twiss(tw_df):
    tw_df.columns = [col.upper() for col in tw_df.columns]
    tw_df.apply(lambda x: x.astype(str).str.upper())
    tw_df.columns = [col.upper() for col in tw_df.columns]
    tw_df = tw_df.set_index("NAME")
    tw_df.index = tw_df.index.str.upper() 
    tw_df.index = tw_df.index.str[:-2]
    #tw_df.index=tw_df.index.str[:-1] + '2'
    tw_df['KEYWORD'] = tw_df['KEYWORD'].str.upper()
    #tw_df.reset_index(inplace=True)

    return tw_df

def get_input_for_beam(twiss_df, meas_mdl, beam):

    ip_bpms_b1 = ["BPMSW.1L1.B1", "BPMSW.1R1.B1", "BPMSW.1L2.B1", "BPMSW.1R2.B1", "BPMSW.1L5.B1", "BPMSW.1R5.B1", "BPMSW.1L8.B1", "BPMSW.1R8.B1"]
    ip_bpms_b2 = ["BPMSW.1L1.B2", "BPMSW.1R1.B2", "BPMSW.1L2.B2", "BPMSW.1R2.B2", "BPMSW.1L5.B2", "BPMSW.1R5.B2", "BPMSW.1L8.B2", "BPMSW.1R8.B2"]

    #tw_perturbed_elements = tfs.read_tfs(twiss_pert_elements_path).set_index("NAME")
    tw_perturbed_elements = uppercase_twiss(twiss_df) 
    #x=tw_perturbed_elements.index
    #tfs.write('test_indexes_perturbed.tfs',x)
    #print(tw_perturbed_elements.index)
    #x=tw_perturbed_elements.index
    #tfs.write('test_beam_2_index.tfs', x)
    ip_bpms = ip_bpms_b1 if beam == 1 else ip_bpms_b2
   
    

    tw_perturbed = tw_perturbed_elements[tw_perturbed_elements.index.isin(meas_mdl.index)]

 
    phase_adv_x = get_phase_adv(tw_perturbed['MUX'], QX)
    phase_adv_y = get_phase_adv(tw_perturbed['MUY'], QY)
    mdl_ph_adv_x = get_phase_adv(meas_mdl['MUX'], QX)
    mdl_ph_adv_y = get_phase_adv(meas_mdl['MUY'], QY)
    delta_phase_adv_x = phase_adv_x - mdl_ph_adv_x
    delta_phase_adv_y = phase_adv_y - mdl_ph_adv_y

    delta_beta_star_x = np.array(tw_perturbed.loc[ip_bpms, "BETX"] - meas_mdl.loc[ip_bpms, "BETX"])
    delta_beta_star_y = np.array(tw_perturbed.loc[ip_bpms, "BETY"] - meas_mdl.loc[ip_bpms, "BETY"])

    n_disp = tw_perturbed['NDX']

    beta_bpms_x = np.array(tw_perturbed["BETX"])
    beta_bpms_y = np.array(tw_perturbed["BETY"])

    return  np.array(delta_beta_star_x), np.array(delta_beta_star_y), \
        np.array(delta_phase_adv_x), np.array(delta_phase_adv_y), np.array(n_disp),\
        np.array(beta_bpms_x), np.array(beta_bpms_y)

x, y, z, a, b, c, d =get_input_for_beam(b2_tw_after_match, B2_MONITORS_MDL_TFS, 2)

print(z)


"""
