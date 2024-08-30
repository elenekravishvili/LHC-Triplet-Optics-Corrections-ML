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
import pickle
import time
from data_utils import merge_data
import random

#At the IPs, at locations of solenois there is a spike, which can be removed from the data. This function filters that. 
def filter_indices(df, prefixes):
    mask = df.index.to_series().apply(lambda x: not any(x.startswith(prefix) for prefix in prefixes))
    return df[mask]

#Simulates the segment with the corrected errors, then calculates the delta phase for x and y for beam one and beam two
#noise variable is not used, IP and beam are not used, I was manually changing them in the madx script for different Ips and beams
def get_delta_mu(input_data, output_data, noise, estimator, beam, IP, index):

    overall_rms_mu_x_all = []
    overall_rms_mu_y_all = []
    try:
        # Predict data and correct errors
        predicted_data = estimator.predict([input_data[index]])
        corrected_errors = output_data[index] - predicted_data
        err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 = corrected_errors[0][-8:] #[0][16:24]#[-8:] #[0][8:16]

        mdx = madx_ml_op()
        
        # Beam 1 calculations
        mdx.job_init_sbs_b1(beam, IP)
        mdx.sbs_twiss_b1()
        twiss_df_b1 = mdx.table.twiss.dframe()
        mdx.set_triplet_errors_b1(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
        mdx.sbs_twiss_b1()
        twiss_err_df_b1 = mdx.table.twiss.dframe()

        # Beam 2 calculations
        mdx.job_init_sbs_b2(beam, IP)
        mdx.sbs_twiss_b2()

        twiss_df_b2 = mdx.table.twiss.dframe()
        mdx.set_triplet_errors_b2(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
        mdx.sbs_twiss_b2()
        twiss_err_df_b2 = mdx.table.twiss.dframe()

        #removes spike at IP, keeps only PBMs
        prefixes_to_filter = ['ip', 'drift', 'mbcs', 'mbas']

        # Apply the filtering function to each DataFrame
        twiss_df_b1_filtered = filter_indices(twiss_df_b1, prefixes_to_filter)
        #tfs.write("IP5_twiss_nominal_b1.tfs", twiss_df_b1_filtered)

        twiss_err_df_b1_filtered = filter_indices(twiss_err_df_b1, prefixes_to_filter)
        #tfs.write("IP5_twiss_corrected_b1.tfs", twiss_err_df_b1_filtered)

        twiss_df_b2_filtered = filter_indices(twiss_df_b2, prefixes_to_filter)
        #tfs.write("IP5_twiss_nominal_b2.tfs", twiss_df_b2_filtered)

        twiss_err_df_b2_filtered = filter_indices(twiss_err_df_b2, prefixes_to_filter)
        #tfs.write("IP5_twiss_corrected_b2.tfs", twiss_err_df_b2_filtered)

   
        delta_mu_x_b1 = twiss_err_df_b1_filtered['mux'] - twiss_df_b1_filtered['mux']
        delta_mu_y_b1 = twiss_err_df_b1_filtered['muy'] - twiss_df_b1_filtered['muy']
        
        delta_mu_x_b2 = twiss_err_df_b2_filtered['mux'] - twiss_df_b2_filtered['mux']
        delta_mu_y_b2 = twiss_err_df_b2_filtered['muy'] - twiss_df_b2_filtered['muy']
        
        combined_delta_mu_x = np.concatenate((delta_mu_x_b1, delta_mu_x_b2))
        combined_delta_mu_y = np.concatenate((delta_mu_y_b1, delta_mu_y_b2))

        #sample = combined_delta_mu_x, combined_delta_mu_y
        sample = delta_mu_x_b1, delta_mu_x_b2, delta_mu_y_b1, delta_mu_y_b2
        
    except Exception as e:
        print(f"An error occurred with: {e}")

    return sample


def main():
    start_time = time.time()  # Start time
    data_path = "test_set_witout_misalignment"
    #mdx = madx_ml_op()
    #indexes = random.sample(range(700), 2)
    num_sim = 10
    valid_samples = []
    GENERATE_DATA = True
    noise_on_test = 0.001 #this noise is no the phase advances only, on beta-beating it is set to 3% of the value
    estimators = {}
    input_data, output_data = merge_data(data_path, noise_on_test)



    
    
 
    alphas = [ 0.1, 0.01, 0.001] #[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001] #[0.0001, 0.0002, 0.0005, 0.0008, 0.001]  
    noises = [ 0.1, 0.01, 0.001] #[0.1, 0.05, 0.01, 0.001, 0.0008, 0.0005, 0.0002, 0.0001, 0.00001] #[0.00031622776601683794, 0.001] 
    estimators_path = './estimators/'
    
    for alpha in alphas:
        for noise in noises:    
            filename = f"test_for_new_data_with_betastar_triplet_phases_028_alpha_{alpha}_ridge_{noise}.pkl"
            file_path = estimators_path + filename
            estimator = load(file_path)
            estimators[f"estimator_noise_{str(noise).replace('.', '')}_alpha{str(alpha).replace('.', '')}"] = estimator
            
            all_samples = [get_delta_mu(input_data, output_data, noise, estimator, 1, 1, i) for i in range(15, 20)]  # 1, 1 are on the place of beam and IP, they are not used
         
            #print(all_samples)
            #print(all_samples.shape)

            #print("Number of generated samples: {}".format(len(valid_samples)))
            np.save(f'./IP1_sbs_filtered/mux_muy_b1_b2_noise_{noise}_alpha_{alpha}.compare2.npy', np.array(all_samples, dtype=object))
    
    end_time = time.time()  # End time
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 25, 30, 35]

    # Plot the data
    plt.figure()
    plt.plot(x, y, marker='o')

    # Customize the plot
    plt.title("Computation Result")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()







