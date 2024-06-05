from madx_job_sbs import madx_ml_op
import matplotlib.pyplot as plt
import tfs
import numpy as np
from pathlib import Path
from joblib import load
from sklearn.ensemble import RandomForestRegressor
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



def load_data_mod(set_name, noise):
    
    all_samples = np.load('./data_phase_adv_triplet/{}.npy'.format(set_name), allow_pickle=True)

    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,b1_disp, b2_disp,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors = all_samples.T
    
    input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
           np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    #input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
     #   np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
      #  np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
       # np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)  
    
    output_data =  np.vstack(triplet_errors)
    # betas = beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2

    return  input_data, output_data

def rms_mu_dist(data_path, noise, estimator, mdx, beam, IP):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]


    mu_x, mu_y = [],  []
    rms_mu_x, rms_mu_y  = [], []
    rms_mu_x_all, rms_mu_y_all = [], []
    for file_name in file_names:
        input_data, output_data = load_data_mod(file_name, noise)
    
        for i in range(10):
            try:
                predicted_data = estimator.predict([input_data[i]])
                corrected_errors = output_data[i]-predicted_data
                err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1  = corrected_errors[0][-8:]
                mdx.job_sbs(beam, 1, err1_1, err2_1, err3_1, err4_1, err5_1, err6_1, err7_1, err8_1 )
                twiss = tfs.read("twiss_IP1.dat")
                twiss_modif = tfs.read("twiss_IP1_cor.dat")
                mux = twiss.MUX
                mux_mod = twiss_modif.MUX
                muy = twiss.MUY
                muy_mod = twiss_modif.MUY

                mu_x = (mux_mod-mux)/mux
                rms_mu_x = np.sqrt(np.mean(mu_x**2, axis=0))
                rms_mu_x_all.append(rms_mu_x)

                mu_y = (muy_mod-muy)/muy
                rms_mu_y = np.sqrt(np.mean(mu_y**2, axis=0))
                rms_mu_y_all.append(rms_mu_y)
                
            except Exception as e:
                # Code to handle the exception
                print(f"An error occurred with {i}: {e}")

    rms_mux = np.hstack(rms_mu_x_all)
    tfs.write("rms_mux_noise_0.0001.tfs", rms_mux)
    rms_muy = np.hstack(rms_mu_y_all)
    tfs.write("rms_muy_noise_0.0001.tfs", rms_muy)

    bin_edges = np.linspace(0, 1, 16)
    
    plt.hist(rms_mux, bins=bin_edges, color='green', alpha=0.5, label='rms of mux')
    plt.hist(rms_mu_y, bins=bin_edges, color='blue', alpha=0.5, label='rms of muy')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    #plt.savefig("beta_beat_recons_b1x.pdf")
    plt.show()



    
data_path = "data_phase_adv_triplet"
noise = 1E-4
Nth_array=68
mdx = madx_ml_op()
loaded_estimator_001 = load('./estimators/triplet_phases_only_028_ridge_0.001.pkl')
loaded_estimator_0001 = load('./estimators/triplet_phases_028_ridge_0.0001.pkl')
loaded_estimator_01 = load('./estimators/triplet_phases_only_028_ridge_0.01.pkl')

rms_mu_dist(data_path, noise, loaded_estimator_0001, mdx, 1, 1)
