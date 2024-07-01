from madx_job_sbs import madx_ml_op
import matplotlib.pyplot as plt
import tfs
import numpy as np
from pathlib import Path
from joblib import load
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data_utils import add_phase_noise
from data_utils import load_data
#mdx = madx_ml_op(stdout=False)

# Create an instance of your class
#madx_instance = madx_ml_op()

# Run the job_sbs method
#madx_instance.job_sbs(1, 5)


#twiss_table = madx_instance.table.twiss


#twiss = tfs.read("twiss_IP1.dat")
#twiss_modif = tfs.read("twiss_IP1_cor.dat")
#mux = twiss.MUX
#mux_mod = twiss_modif.MUX
#print(twiss.BETX-twiss_modif.BETX)
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




def rms_mu_dist(data_path, noise, estimator, mdx, beam, IP, alpha):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]


    mu_x, mu_y = [],  []
    rms_mu_x, rms_mu_y  = [], []
    rms_mu_x_all, rms_mu_y_all = [], []
    for file_name in file_names:
        input_data, output_data = load_data(file_name, noise)
    
        for i in range(1):
            try:
                predicted_data = estimator.predict([input_data[i]])
                corrected_errors = output_data[i] -predicted_data
                err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1  = corrected_errors[0][-8:]
                mdx.job_sbs(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 )
                twiss = tfs.read("twiss_IP1.dat")
                twiss_modif = tfs.read("twiss_IP1_cor.dat")
                mux = twiss.MUX
                mux_mod = twiss_modif.MUX
                muy = twiss.MUY
                muy_mod = twiss_modif.MUY

                mu_x = mux_mod-mux
                rms_mu_x = np.sqrt(np.mean(mu_x**2, axis=0))
                rms_mu_x_all.append(rms_mu_x)
                
                print(mu_x) 
                
                mu_y = muy_mod-muy
                rms_mu_y = np.sqrt(np.mean(mu_y**2, axis=0))
                rms_mu_y_all.append(rms_mu_y)
                
            except Exception as e:
                # Code to handle the exception
                print(f"An error occurred with {i}: {e}")

    rms_mux_df = pd.DataFrame({'rms_mux': rms_mu_x_all})
    rms_muy_df = pd.DataFrame({'rms_muy': rms_mu_y_all})
    print("Hello")
    # Save using tfs.write
    tfs.write("./rms_mu_changing_parameters/rms_mux_noise_{}_alpha{}.tfs".format(noise, alpha), rms_mux_df)
    tfs.write("./rms_mu_changing_parameters/rms_muy_noise_{}_alpha{}.tfs".format(noise, alpha), rms_muy_df)

    




    
data_path = "data_phase_adv_triplet"
noise = 2E-4
alpha = 4E-4
Nth_array=68
mdx = madx_ml_op()
#loaded_estimator_001 = load('./estimators/triplet_phases_only_028_ridge_0.001.pkl')
#estimator_noise_0002_alpha001 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_ridge_0.0002_alpha(1e-3).pkl')
estimator_noise_0002_alpha0002 = load('./estimators_for_parameters/test_for_new_data_with_betastar_triplet_phases_028_ridge_0.0002_alpha(2e-4).pkl')
#estimator_noise_0002_alpha0004 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_ridge_0.0002_alpha(4e-4).pkl')
#estimator_noise_0002_alpha0006 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_ridge_0.0002_alpha(6e-4).pkl')
#estimator_noise_0002_alpha0008 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_ridge_0.0002_alpha(8e-4).pkl')

estimator_noise_0002_alpha0004 = load('./estimators_for_parameters/test_for_new_data_with_betastar_triplet_phases_028_ridge_0.0002_alpha(4e-4).pkl')
rms_mu_dist(data_path, noise, estimator_noise_0002_alpha0004, mdx, 1, 1, alpha)






