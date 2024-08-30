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


#This script is for simulation os the segment but it is crushing, so sbs_generate.py is used rather than this


def rms_mu_dist_old(data_path, noise, estimator, mdx, beam, IP, alpha):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]


    mu_x, mu_y = [],  []
    rms_mu_x, rms_mu_y  = [], []
    rms_mu_x_all, rms_mu_y_all = [], []
    for file_name in file_names:
        input_data, output_data = load_data(file_name, noise)
    
        for i in range(15):
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

def calc_rms_mu_old(data_path, mdx, beam, IP):   
    # Define the lists for alpha and noise values
    alphas = [0.0008]
    noises = [0.00031622776601683794, 0.0005623413251903491, 0.0001]

    # Define the base path for the estimators
    estimators_path = './estimators/'

    # Dictionary to store the estimators
    estimators = {}
    input_data, output_data, beta  = [], [], []

    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    # Loop through each alpha and noise value to construct the filenames and load estimators
    for alpha in alphas:
        for noise in noises:
            # Construct the filename using the current alpha and noise values
            filename = f"test_for_new_data_with_betastar_triplet_phases_028_alpha({alpha})_ridge_{noise}.pkl"
            file_path = estimators_path + filename
            
            try:
                estimator = load(file_path)
                
                # Assign the loaded estimator to the dynamic variable name
                var_name = f"estimator_noise_{str(noise).replace('.', '')}_alpha{str(alpha).replace('.', '')}"
                estimators[var_name] = estimator

                mu_x, mu_y = [], []
                rms_mu_x_all, rms_mu_y_all = [], []

                for file_name in file_names:
                    input_data, output_data = load_data(file_name, noise)
                    
                    for i in range(2):
                        try:
                            predicted_data = estimator.predict([input_data[i]])
                            corrected_errors = output_data[i] - predicted_data
                            err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 = corrected_errors[0][-8:]

                            mdx.job_init_sbs(beam, 1)
                            mdx.sbs_twiss()
                            twiss_df = mdx.table.twiss.dframe()
                            mdx.set_triplet_errors(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
                            mdx.sbs_twiss()
                            twiss_err_df = mdx.table.twiss.dframe()
                            #twiss = tfs.read("twiss_IP1.dat")
                            #twiss_modif = tfs.read("twiss_IP1_cor.dat")
                            #mux = twiss.MUX
                            #mux_mod = twiss_modif.MUX
                            #muy = twiss.MUY
                            #muy_mod = twiss_modif.MUY

                            delta_mu_x = twiss_err_df['mux'] - twiss_df['mux']
                            rms_mu_x = np.sqrt(np.mean(delta_mu_x**2, axis=0))
                            rms_mu_x_all.append(rms_mu_x)
                            
                            delta_mu_y = twiss_err_df['muy'] - twiss_df['muy']
                            rms_mu_y = np.sqrt(np.mean(delta_mu_y**2, axis=0))
                            rms_mu_y_all.append(rms_mu_y)

                        except Exception as e:
                            print(f"An error occurred with {file_name} at index {i}: {e}")

                rms_mux_df = pd.DataFrame({'rms_mux': rms_mu_x_all})
                rms_muy_df = pd.DataFrame({'rms_muy': rms_mu_y_all})
                
                noise_str = f"{noise:.18f}".rstrip('0').rstrip('.')
                tfs.write(f"./rms_mu_changing_parameters/rms_mux_noise_{noise_str}_alpha{alpha}.tfs", rms_mux_df)
                tfs.write(f"./rms_mu_changing_parameters/rms_muy_noise_{noise_str}_alpha{alpha}.tfs", rms_muy_df)
                print(f"Written files for noise {noise} and alpha {alpha}")

            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred while loading {filename}: {e}")

    for key, value in estimators.items():
        print(f"{key}: {value}")
"""
def calc_rms_mu(data_path, mdx, beam, IP, alphas, noises, estimators_path):   
    estimators = {}
    input_data, output_data, beta = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    for alpha in alphas:
        for noise in noises:
            filename = f"test_for_new_data_with_betastar_triplet_phases_028_alpha({alpha})_ridge_{noise}.pkl"
            file_path = estimators_path + filename
            
            try:
                estimator = load(file_path)
                estimators[f"estimator_noise_{str(noise).replace('.', '')}_alpha{str(alpha).replace('.', '')}"] = estimator

                rms_mu_x_all, rms_mu_y_all = rms_mu_dist(data_path, noise, estimator, mdx, beam, IP)

                yield noise, alpha, rms_mu_x_all, rms_mu_y_all

            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred while loading {filename}: {e}")

    for key, value in estimators.items():
        print(f"{key}: {value}")

def main():
    start_time = time.time()  # Start time
    data_path = "data_phase_adv_triplet"
    mdx = madx_ml_op()
    
    alphas = [0.001] #[0.0001, 0.0002, 0.0005, 0.0008, 0.001]
    noises = [0.00031622776601683794, 0.0005623413251903491, 0.001] # [0.0001, 0.00017782794100389227, 0.00031622776601683794, 0.0005623413251903491, 0.001]
    estimators_path = './estimators/'
    
    for noise, alpha, rms_mu_x_all, rms_mu_y_all in calc_rms_mu(data_path, mdx, 1, 1, alphas, noises, estimators_path):
        rms_mu_df = pd.DataFrame({'rms_mux': rms_mu_x_all, 'rms_muy': rms_mu_y_all})  
        #rms_muy_df = pd.DataFrame({'rms_muy': rms_mu_y_all})
        noise_str = f"{noise:.18f}".rstrip('0').rstrip('.')
        
        #tfs.write(f"./rms_mu_changing_parameters/rms_mux_noise_{noise_str}_alpha{alpha}.tfs", rms_mux_df)
        #tfs.write(f"./rms_mu_changing_parameters/rms_muy_noise_{noise_str}_alpha{alpha}.tfs", rms_muy_df)
        #print(f"Written files for noise {noise} and alpha {alpha}")
        #np.save(f"./rms_mu_changing_parameters/rms_mu_noise_{noise_str}_alpha{alpha}.npy", np.array(rms_mu_df, dtype=object))
        np.save(f"./rms_mu_changing_parameters/rms_mu_noise_{noise_str}_alpha{alpha}.npy", rms_mu_df.to_dict('list'))
        #np.save(f"./rms_mu_changing_parameters/rms_muy_noise_{noise_str}_alpha{alpha}.npy", np.array(rms_mu_df, dtype=object))
        print(f"Written files for noise {noise} and alpha {alpha}")
        
    end_time = time.time()  # End time
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    main()

"""        


def rms_mu_dist(input_data, output_data, noise, estimator, beam, IP):
    #input_data, output_data = [], []
    #pathlist = Path(data_path).glob('**/*.npy')
    #file_names = [str(path).split('/')[-1][:-4] for path in pathlist]
   
    #input_data, output_data = merge_data(data_path, noise)

    overall_rms_mu_x_all = []
    overall_rms_mu_y_all = []

    for i in range(100, 110):
        try:
            predicted_data = estimator.predict([input_data[i]])
            corrected_errors = output_data[i] - predicted_data
            err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 = corrected_errors[0][-8:]
            mdx = madx_ml_op()
            # Beam 1 calculations
            mdx.job_init_sbs_b1(beam, 1)
            mdx.sbs_twiss_b1()
            twiss_df_b1 = mdx.table.twiss.dframe()
            mdx.set_triplet_errors_b1(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
            mdx.sbs_twiss_b1()
            twiss_err_df_b1 = mdx.table.twiss.dframe()
            
            # Beam 2 calculations
            mdx.job_init_sbs_b2(beam, 1)
            mdx.sbs_twiss_b2()
            twiss_df_b2 = mdx.table.twiss.dframe()
            mdx.set_triplet_errors_b2(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1)
            mdx.sbs_twiss_b2()
            twiss_err_df_b2 = mdx.table.twiss.dframe()

            # Calculate delta values and combine for both beams
            delta_mu_x_b1 = twiss_err_df_b1['mux'] - twiss_df_b1['mux']
            delta_mu_y_b1 = twiss_err_df_b1['muy'] - twiss_df_b1['muy']
            
            delta_mu_x_b2 = twiss_err_df_b2['mux'] - twiss_df_b2['mux']
            delta_mu_y_b2 = twiss_err_df_b2['muy'] - twiss_df_b2['muy']
            
            combined_delta_mu_x = np.concatenate((delta_mu_x_b1, delta_mu_x_b2))
            combined_delta_mu_y = np.concatenate((delta_mu_y_b1, delta_mu_y_b2))

            # Calculate RMS for combined deltas
            overall_rms_mu_x = np.sqrt(np.mean(combined_delta_mu_x**2, axis=0))
            overall_rms_mu_y = np.sqrt(np.mean(combined_delta_mu_y**2, axis=0))

            overall_rms_mu_x_all.append(overall_rms_mu_x)
            overall_rms_mu_y_all.append(overall_rms_mu_y)
            
        except Exception as e:
            print(f"An error occurred with {i}: {e}")

    return overall_rms_mu_x_all, overall_rms_mu_y_all


def calc_rms_mu(data_path, beam, IP, alphas, noises, estimators_path):   

    #pathlist = Path(data_path).glob('**/*.npy')
    #file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    estimators = {}
    input_data, output_data = merge_data(data_path, 0.001)

    results = []

    for alpha in alphas:
        for noise in noises:
            filename = f"test_for_new_data_with_betastar_triplet_phases_028_alpha_{alpha}_ridge_{noise}.pkl"
            file_path = estimators_path + filename
            
            try:
                estimator = load(file_path)
                estimators[f"estimator_noise_{str(noise).replace('.', '')}_alpha{str(alpha).replace('.', '')}"] = estimator

                rms_mu_x_all, rms_mu_y_all = rms_mu_dist(input_data, output_data, noise, estimator, beam, IP)

                results.append((noise, alpha, rms_mu_x_all, rms_mu_y_all))

            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred while loading {filename}: {e}")

    return results

def main():
    start_time = time.time()  # Start time
    data_path = "test_data"
    #mdx = madx_ml_op()
    #indexes = random.sample(range(700), 2)
    
    alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001] #[0.0001, 0.0002, 0.0005, 0.0008, 0.001] 
    noises = [0.001, 0.0008, 0.0005, 0.0002, 0.0001, 0.00001]  #[0.00031622776601683794, 0.001] 
    estimators_path = './estimators/'
    
    results = calc_rms_mu(data_path, 1, 1, alphas, noises, estimators_path)
    
    # Combine all results into a single DataFrame with MultiIndex columns
    combined_data = {}

    for noise, alpha, rms_mu_x_all, rms_mu_y_all in results:
        for i, (rms_mu_x, rms_mu_y) in enumerate(zip(rms_mu_x_all, rms_mu_y_all)):
            if i not in combined_data:
                combined_data[i] = {}
            combined_data[i][(f"rms_mux_noise_{noise}_alpha_{alpha}", f"rms_muy_noise_{noise}_alpha_{alpha}")] = (rms_mu_x, rms_mu_y)

    # Create a list of tuples for columns
    columns = []
    for key in combined_data[0].keys():
        columns.append(key[0])
        columns.append(key[1])
    
    rows = [row for row in combined_data.keys()]
    data = []

    for row in rows:
        data_row = []
        for key in combined_data[row].keys():
            data_row.extend(combined_data[row][key])
        data.append(data_row)

    # Debugging print statements
    print(f"Columns: {columns}")
    print(f"Rows: {rows}")
    print(f"Data: {data}")

    # Check the shape of data
    data_length = len(data[0])
    columns_length = len(columns)
    #random_suffix = random.randint(1000, 9999)
    if data_length != columns_length:
        print(f"Data length: {data_length}, Columns length: {columns_length}")
        raise ValueError("Mismatch between data length and columns length")

    combined_df = pd.DataFrame(data, index=rows, columns=columns)
    
    combined_df.to_csv(f'./rms_mu_when_noisy_data/rms_mu_combined_100-110.csv', index=False)
    print(f"Written combined RMS values to rms_mu_combined.csv")
    
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







