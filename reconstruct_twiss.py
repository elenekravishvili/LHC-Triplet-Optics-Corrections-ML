#from cpymad import madx
import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
import os
from pathlib import Path
from joblib import load
from cpymad import madx
#from sklearn.ensemble import RandomForestRegressor
from data_utils import add_beta_star_noise
from data_utils import add_phase_noise

def save_np_errors_tfs(np_errors):
    TRIPLET_NAMES = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                        "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                        "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                        "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]


    names = TRIPLET_NAMES
    recons_df = pd.DataFrame(columns=["NAME","K1L"])
    recons_df.K1L = np_errors 
    recons_df.NAME = names
    #print(recons_df)

    tfs.write("./predicted_errors/test_predicted_triplet_errors.tfs", recons_df)
    print("tfs write has happend")

def save_errors_rel_formal():
    names = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                        "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                        "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                        "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]

    example = tfs.read("example.tfs").set_index("NAME")
    errors = tfs.read("./predicted_errors/test_predicted_triplet_errors.tfs",).set_index("NAME")

    example.loc[names, "K1L"] = errors.loc[names, "K1L"]

    x = example
    x.reset_index(inplace=True)

    tfs.write("example_redefine.tfs", x)
    print("tfs write has happend")

def recons_twiss( beam, mdx):
    mdx.options(echo=False)
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2023.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx")
    mdx.call(file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq")
    mdx.options(echo=True)

    mdx.input("exec, define_nominal_beams(energy=6800);")
    mdx.call(file="/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx")
    mdx.input("exec, cycle_sequences();")

    mdx.use(sequence=f"LHCB{beam}")

    mdx.options(echo=False)

    mdx.input(f"exec, match_tunes(62.28, 60.31, {beam});")
    #mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0

    #Assigning errors
    mdx.input(f"""
                !readtable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam{beam}/error_tables_6.5TeV/MBx-0001.errors", table=errtab;
                !seterr, table=errtab;
                READMYTABLE, file="/afs/cern.ch/user/e/ekravish/work/linear/example_redefine.tfs", table=errtab;
                SETERR, TABLE=errtab;""")

    mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0
    mdx.input(f"exec, match_tunes(62.28, 60.31, {beam});")
    mdx.input(f"""etable, table="final_error";""")
    
    mdx.twiss(sequence=f"LHCB{beam}", file="")    

    #print("Eroror", mdx.table.final_error.dframe())
    #tfs.writer.write_tfs(tfs_file_path=f"final_errors.tfs", data_frame=mdx.table.final_error.dframe())

    # Generate twiss with columns needed for training data
    mdx.input(f"""ndx := table(twiss,dx)/sqrt(table(twiss,betx));
               select, flag=twiss, clear;
              select, flag=twiss, pattern="^BPM.*B{beam}$", column=name, s, betx, bety, ndx,
                                             mux, muy;
           twiss, chrom, sequence=LHCB{beam}, deltap=0.0, file="";""")

    return mdx.table.twiss.dframe()



def load_data_mod(set_name, noise):
    
    #all_samples = np.load('./data_phase_adv_triplet/{}.npy'.format(set_name), allow_pickle=True)
    all_samples = np.load('./data_right_offset/{}.npy'.format(set_name), allow_pickle=True)
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
    
    
    delta_mux_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b1, beta_bpm_x_b1)]
    delta_muy_b1 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b1, beta_bpm_y_b1)]
    delta_mux_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_mux_b2, beta_bpm_x_b2)]
    delta_muy_b2 = [add_phase_noise(delta_mu, beta_bpm, noise) for delta_mu, beta_bpm in zip(delta_muy_b2, beta_bpm_y_b2)]

    delta_beta_star_x_b1_with_noise = add_beta_star_noise(delta_beta_star_x_b1, noise)
    delta_beta_star_y_b1_with_noise = add_beta_star_noise(delta_beta_star_y_b1, noise)
    delta_beta_star_x_b2_with_noise = add_beta_star_noise(delta_beta_star_x_b2, noise)
    delta_beta_star_y_b2_with_noise = add_beta_star_noise(delta_beta_star_y_b2, noise)
    
    
    #input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
     #      np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1_with_noise), np.vstack(delta_beta_star_y_b1_with_noise), \
        np.vstack(delta_beta_star_x_b2_with_noise), np.vstack(delta_beta_star_y_b2_with_noise), \
        np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)  
    
    output_data =  np.vstack(triplet_errors)
    betas = beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2

    return np.vstack(beta_bpm_x_b1), np.vstack(beta_bpm_y_b1), np.vstack(beta_bpm_x_b2), np.vstack(beta_bpm_y_b2), input_data, output_data



def beta_beat_dist(data_path, noise, estimator, mdx):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
    B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")
    B1_ELEMENTS_MDL_TFS_rec = tfs.read_tfs("reconst_nom_twiss.tfs")
    tw_nom_rec = make_twiss_good(B1_ELEMENTS_MDL_TFS_rec)
    beta_x_b1_nom_rec =np.array(tw_nom_rec.BETX)
    beta_y_b1_nom_rec =np.array(tw_nom_rec.BETY)
    beta_x_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["BETX"])
    beta_y_b1_nom =np.array(B1_ELEMENTS_MDL_TFS.BETY)
    beta_x_b2_nom =np.array(B2_ELEMENTS_MDL_TFS["BETX"])
    beta_y_b2_nom =np.array(B2_ELEMENTS_MDL_TFS.BETY)
    beta_beat_x_b1, beta_beat_y_b1, beta_beat_x_b2, beta_beat_y_b2 = [], [], [], []
    rms_x_b1, rms_y_b1, rms_x_b2, rms_y_b2  = [], [], [], []
    rms_x_b1_pred, rms_y_b1_pred, rms_x_b2_pred, rms_y_b2_pred  = [], [], [], []
    for file_name in file_names:
        print(file_name)
        beta_x_b1, beta_y_b1, beta_x_b2, beta_y_b2, input_data, output_data = load_data_mod(file_name, noise)
    
        for i in range(150):
            try:
                predicted_data = estimator.predict([input_data[i]])
                predicted_triplet = predicted_data[:, :32]
                corrected_errors = output_data[i]-predicted_triplet
                save_np_errors_tfs(corrected_errors[0])
                #save_np_errors_tfs(output_data[i])
                save_errors_rel_formal()
                
                tw_recons_first = recons_twiss( 1, mdx)
                tw_recons = make_twiss_good(tw_recons_first)
            
                betax = np.array(tw_recons.BETX)
                betay = np.array(tw_recons.BETY)

                beta_beating_x = (betax-beta_x_b1_nom_rec)/beta_x_b1_nom_rec
                rms_x1_pred = np.sqrt(np.mean(beta_beating_x**2, axis=0))
                rms_x_b1_pred.append(rms_x1_pred)

                beta_beating_y = (betay-beta_y_b1_nom_rec)/beta_y_b1_nom_rec
                rms_y1_pred = np.sqrt(np.mean(beta_beating_y**2, axis=0))
                rms_y_b1_pred.append(rms_y1_pred)
                

                #bb_x1 = (beta_x_b1[i]-beta_x_b1_nom)/beta_x_b1_nom
                #rms_x1 = np.sqrt(np.mean(bb_x1**2, axis=0))
                #rms_x_b1.append(rms_x1)

                #bb_y1 = (beta_y_b1[i]-beta_y_b1_nom)/beta_y_b1_nom
                #rms_y1 = np.sqrt(np.mean(bb_y1**2, axis=0))
                #rms_y_b1.append(rms_y1)
                
                #bb_x2 = (beta_x_b2[i]-beta_x_b2_nom)/beta_x_b2_nom
                #rms_x2 = np.sqrt(np.mean(bb_x2**2, axis=0))
                #rms_x_b2.append(rms_x2)

                #bb_y2 = (beta_y_b2[i]-beta_y_b2_nom)/beta_y_b2_nom
                #rms_y2 = np.sqrt(np.mean(bb_y2**2, axis=0))
                #rms_y_b2.append(rms_y2) 
            except Exception as e:
                # Code to handle the exception
                print(f"An error occurred with {i}: {e}")

    #x=np.hstack(beta_beat_x_b1)
    #rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
    #rms_beta_beat_x_b1_df = pd.DataFrame({'rms_betabeat': rms_beta_beat_x_b1})
    #tfs.write("./beta_beat/rms_beta_beat_improved_model_estimator_noise_0.0001.tfs", rms_beta_beat_x_b1_df)
    #rms_beta_beat_y_b1 = np.hstack(rms_y_b1)
    #rms_beta_beat_y_b1_df = pd.DataFrame({'rms_betabeat': rms_beta_beat_y_b1})
    #tfs.write("./beta_beat/rms_beta_beat_y_b1_improved_model_estimator_noise_0.0001.tfs", rms_beta_beat_y_b1_df)
    #rms_beta_beat_x_b2 = np.hstack(rms_x_b2)
    #rms_beta_beat_y_b2 = np.hstack(rms_y_b2)
    rms_beta_beat_x_b1_pred = np.hstack(rms_x_b1_pred)
    rms_beta_beat_x_b1_pred_df = pd.DataFrame({'rms_betabeat': rms_beta_beat_x_b1_pred})
    tfs.write("./beta_beat/rms_beta_beat_x_b1_offsetdata_regularestimator_corrects_triplet.tfs", rms_beta_beat_x_b1_pred_df)
    rms_beta_beat_y_b1_pred = np.hstack(rms_y_b1_pred)
    rms_beta_beat_y_b1_pred_df = pd.DataFrame({'rms_betabeat': rms_beta_beat_y_b1_pred})
    tfs.write("./beta_beat/rms_beta_beat_y_b1_offsetdata_regularestimator_corrects_triplet.tfs", rms_beta_beat_y_b1_pred_df)

    bin_edges = np.linspace(0, 1, 16)
    """
    plt.hist(rms_beta_beat_x_b1, bins=bin_edges, color='green', alpha=0.5, label='rms of beta beating beam1, x')
    plt.hist(rms_beta_beat_x_b1_pred, bins=bin_edges, color='blue', alpha=0.5, label='rms of beta beating beam1 corrected x')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    #plt.savefig("beta_beat_recons_b1x.pdf")
    plt.show()


    plt.hist(rms_beta_beat_y_b1, bins=bin_edges, color='green', alpha=0.5, label='rms of beta beating beam1, y')
    plt.hist(rms_beta_beat_y_b1_pred, bins=bin_edges, color='blue', alpha=0.5, label='rms of beta beating beam1 corrected, y')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    #plt.savefig("beta_beat_recons_b1y.pdf")
    plt.show() 
    """
def uppercase_twiss(tw_df):
    tw_df.columns = [col.upper() for col in tw_df.columns]
    tw_df.apply(lambda x: x.astype(str).str.upper())
    tw_df.columns = [col.upper() for col in tw_df.columns]
    tw_df = tw_df.set_index("NAME")
    tw_df.index = tw_df.index.str.upper() 
    tw_df.index = tw_df.index.str[:-2]
    tw_df['KEYWORD'] = tw_df['KEYWORD'].str.upper()
    return tw_df

def make_twiss_good(tw):
    B1_ELEMENTS_MDL_TFS_rec = tfs.read_tfs("reconst_nom_twiss.tfs")
    twiss_nom_upp = uppercase_twiss(tw)
    common_indices = twiss_nom_upp.index.intersection(example.index)
    tw_upp_good = twiss_nom_upp.loc[common_indices]
    return tw_upp_good


example = tfs.read("b1_nominal_monitors.dat").set_index("NAME")
data_path = "test_data"
data_path_with_off = "test_data_right_offset"
noise = 0
Nth_array=68
mdx = madx.Madx()
loaded_estimator_noise_0001 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.001_ridge_0.0001.pkl')
loaded_estimator_noise_001 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.0001_ridge_0.001.pkl')
loaded_estimator_noise_01 = load('./estimators/test_for_new_data_with_betastar_triplet_phases_028_alpha_0.001_ridge_0.01.pkl')
new_estimator_base_model = load('./estimators/test_for_new_data_without_betastar_triplet_phases_028_alpha_0.001_ridge_0.01.pkl')
offset_estimator = load('./estimators/off_test_for_new_data_with_betastar_triplet_phases_028_alpha_0.0001_ridge_0.001.pkl')


beta_beat_dist(data_path_with_off, noise, loaded_estimator_noise_001, mdx)



"""

input_data, output_data, beta  = [], [], []
pathlist = Path(data_path).glob('**/*.npy')
file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")
B1_ELEMENTS_MDL_TFS_rec = tfs.read_tfs("reconst_nom_twiss.tfs")
tw_nom_rec = make_twiss_good(B1_ELEMENTS_MDL_TFS_rec)
beta_x_b1_nom_rec =np.array(tw_nom_rec.BETX)
beta_y_b1_nom_rec =np.array(tw_nom_rec.BETY)
beta_x_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["BETX"])
beta_y_b1_nom =np.array(B1_ELEMENTS_MDL_TFS.BETY)
beta_x_b2_nom =np.array(B2_ELEMENTS_MDL_TFS["BETX"])
beta_y_b2_nom =np.array(B2_ELEMENTS_MDL_TFS.BETY)
beta_beat_x_b1, beta_beat_y_b1, beta_beat_x_b2, beta_beat_y_b2 = [], [], [], []
rms_x_b1, rms_y_b1, rms_x_b2, rms_y_b2  = [], [], [], []
rms_x_b1_pred, rms_y_b1_pred, rms_x_b2_pred, rms_y_b2_pred  = [], [], [], []
for file_name in file_names:
    beta_x_b1, beta_y_b1, beta_x_b2, beta_y_b2, input_data, output_data = load_data_mod(file_name, noise)

    for i in range(3):
        try:
            #predicted_data = estimator.predict([input_data[i]])
            #corrected_errors = output_data[i]-predicted_data
            #save_np_errors_tfs(corrected_errors[0])
            real_errors = output_data[i]
            save_np_errors_tfs(real_errors[0])
            save_errors_rel_formal()
            tw_recons_first = recons_twiss( 1, mdx)
            tw_recons = make_twiss_good(tw_recons_first)
        
            betax = np.array(tw_recons.BETX)
            betay = np.array(tw_recons.BETY)

            beta_beating_x = (betax-beta_x_b1_nom_rec)/beta_x_b1_nom_rec
            rms_x1_pred = np.sqrt(np.mean(beta_beating_x**2, axis=0))
            rms_x_b1_pred.append(rms_x1_pred)

            beta_beating_y = (betay-beta_y_b1_nom_rec)/beta_y_b1_nom_rec
            rms_y1_pred = np.sqrt(np.mean(beta_beating_y**2, axis=0))
            rms_y_b1_pred.append(rms_y1_pred)
            

            bb_x1 = (beta_x_b1[i]-beta_x_b1_nom)/beta_x_b1_nom
            rms_x1 = np.sqrt(np.mean(bb_x1**2, axis=0))
            rms_x_b1.append(rms_x1)

            bb_y1 = (beta_y_b1[i]-beta_y_b1_nom)/beta_y_b1_nom
            rms_y1 = np.sqrt(np.mean(bb_y1**2, axis=0))
            rms_y_b1.append(rms_y1)
            
            #bb_x2 = (beta_x_b2[i]-beta_x_b2_nom)/beta_x_b2_nom
            #rms_x2 = np.sqrt(np.mean(bb_x2**2, axis=0))
            #rms_x_b2.append(rms_x2)

            #bb_y2 = (beta_y_b2[i]-beta_y_b2_nom)/beta_y_b2_nom
            #rms_y2 = np.sqrt(np.mean(bb_y2**2, axis=0))
            #rms_y_b2.append(rms_y2) 
        except Exception as e:
            # Code to handle the exception
            print(f"An error occurred with {i}: {e}")

#x=np.hstack(beta_beat_x_b1)
rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
tfs.write("rms_beta_beat_x_b1.tfs", rms_beta_beat_x_b1)
rms_beta_beat_y_b1 = np.hstack(rms_y_b1)
tfs.write("rms_beta_beat_y_b1.tfs", rms_beta_beat_y_b1)
#rms_beta_beat_x_b2 = np.hstack(rms_x_b2)
#rms_beta_beat_y_b2 = np.hstack(rms_y_b2)
rms_beta_beat_x_b1_pred = np.hstack(rms_x_b1_pred)
tfs.write("rms_beta_beat_x_b1_corrected.tfs", rms_beta_beat_x_b1_pred)
rms_beta_beat_y_b1_pred = np.hstack(rms_y_b1_pred)
tfs.write("rms_beta_beat_y_b1_corrected.tfs", rms_beta_beat_y_b1_pred)


plt.hist(rms_beta_beat_x_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, x')
plt.hist(rms_beta_beat_x_b1_pred, bins=15, color='blue', alpha=0.5, label='rms of beta beating beam1 corrected x')
plt.xlabel('rms')
plt.ylabel('Counts')
plt.title('rms')
plt.legend()  # Add legend to display labels
plt.grid(True)
plt.savefig("beta_beat_recons_b1x.pdf")
plt.show()


plt.hist(rms_beta_beat_y_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, y')
plt.hist(rms_beta_beat_y_b1_pred, bins=15, color='blue', alpha=0.5, label='rms of beta beating beam1 corrected, y')
plt.xlabel('rms')
plt.ylabel('Counts')
plt.title('rms')
plt.legend()  # Add legend to display labels
plt.grid(True)
plt.savefig("beta_beat_recons_b1y.pdf")
plt.show() 
"""





"""
B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
B1_ELEMENTS_MDL_TFS_rec = tfs.read_tfs("reconst_nom_twiss.tfs")
tw_nom_rec = make_twiss_good(B1_ELEMENTS_MDL_TFS_rec)
beta_x_b1_nom_rec =np.array(tw_nom_rec.BETX)
beta_y_b1_nom_rec =np.array(tw_nom_rec.BETY)
beta_x_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["BETX"])
beta_y_b1_nom =np.array(B1_ELEMENTS_MDL_TFS.BETY)

all_samples = np.load('./data_phase_adv_triplet/100%triplet_ip1_20%ip5_2202.npy', allow_pickle=True)

delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
    delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
        delta_muy_b2,b1_disp, b2_disp,\
        beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
        triplet_errors = all_samples.T

input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    


beta_x_b1, beta_y_b1  = np.vstack(beta_bpm_x_b1), np.vstack(beta_bpm_y_b1)
output_data =  np.vstack(triplet_errors)
rms_x_b1, rms_y_b1, rms_x_b2, rms_y_b2  = [], [], [], []
rms_x_b1_pred, rms_y_b1_pred, rms_x_b2_pred, rms_y_b2_pred  = [], [], [], []

for i in range(1):
    try:
        save_np_errors_tfs(output_data[i])
        save_errors_rel_formal()
        tw_recons_first = recons_twiss( 1, mdx)
        print(tw_recons_first)
        tw_recons = make_twiss_good(tw_recons_first)
        print(tw_recons)
        betax = np.array(tw_recons.BETX)
        betay = np.array(tw_recons.BETY)

        print(betax, betay)

        beta_beating_x = (betax-beta_x_b1_nom_rec)/beta_x_b1_nom_rec
        rms_x1_pred = np.sqrt(np.mean(beta_beating_x**2, axis=0))
        rms_x_b1_pred.append(rms_x1_pred)

        bb_x1 = (beta_x_b1[i]-beta_x_b1_nom)/beta_x_b1_nom
        rms_x1 = np.sqrt(np.mean(bb_x1**2, axis=0))
        rms_x_b1.append(rms_x1)
    except Exception as e:
        # Code to handle the exception
        print(f"An error occurred with {i}: {e}")

rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
#tfs.write("rms_beta_beat_x_b1_only_real_triplet_errors.tfs", rms_beta_beat_x_b1)

rms_beta_beat_x_b1_pred = np.hstack(rms_x_b1_pred)
#tfs.write("rms_beta_beat_x_b1_real_arc_and_tripplet_errors.tfs", rms_beta_beat_x_b1_pred)    

rms_beta_beat_x_b1 = tfs.read("rms_beta_beat_x_b1_only_real_triplet_errors.tfs")
rms_beta_beat_x_b1_pred = tfs.read("rms_beta_beat_x_b1_real_arc_and_tripplet_errors.tfs")
x = np.array(rms_beta_beat_x_b1)
y = np.array(rms_beta_beat_x_b1_pred)
n_bin = 20

plt.hist(rms_beta_beat_x_b1, bins=7, color='blue', alpha=0.5, label='rms of beta beating beam1, x, when  real arc and triplet errors are assigned')
plt.hist(rms_beta_beat_x_b1_pred, bins=20, color='green', alpha=0.5, label='rms of beta beating beam1, x, when only real triplet errors are assigned')
plt.xlabel('rms')
plt.ylabel('Counts')
plt.title('rms')
plt.legend()  # Add legend to display labels
plt.grid(True)
plt.savefig("compare_beta_beat_with_and_without_arcs_1.png")
#plt.show()
"""
"""
def plot_betabeat_reconstruction(tw_true, tw_recons, beam):
    if beam==1:
        tw_nominal = tfs.read_tfs("b1_nominal_monitors.dat").set_index("NAME")
    elif beam==2:
        tw_nominal = tfs.read_tfs("b2_nominal_monitors.dat").set_index("NAME")

    tw_recons = tw_recons.set_index("name") 
    tw_recons.index = [(idx.upper()).split(':')[0] for idx in tw_recons.index]
    tw_recons.columns = [col.upper() for col in tw_recons.columns]

    tw_recons = tw_recons[tw_recons.index.isin(tw_nominal.index)]
    
    bbeat_x_recons = 100*(np.array(tw_recons.BETX - tw_nominal.BETX))/tw_nominal.BETX
    bbeat_y_recons = 100*(np.array(tw_recons.BETY - tw_nominal.BETY))/tw_nominal.BETY

    fig, axs = plt.subplots(2)
    axs[0].plot(tw_true.S, tw_true.BETX, label="True", alpha=0.7)
    axs[0].plot(tw_recons.S, bbeat_x_recons, label="Rec", alpha=0.7)
    axs[0].plot(tw_recons.S, bbeat_x_recons-tw_true.BETX, label="Res", alpha=0.7)
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_ylabel(r"$\Delta \beta _x / \beta _x [\%]$")
    axs[0].set_xticklabels(labels=['IP2', 'IP3', 'IP4', 'IP5', 'IP6', 'IP7', 'IP8', 'IP1'])
    axs[0].set_xticks([i for i in np.linspace(0, int(tw_true.S[-1]), num=8)])
    axs[0].legend()

    axs[1].plot(tw_true.S, tw_true.BETY, label="True", alpha=0.7)
    axs[1].plot(tw_recons.S, bbeat_y_recons, label="Rec", alpha=0.7)
    axs[1].plot(tw_recons.S, bbeat_y_recons-tw_true.BETY, label="Res", alpha=0.7)
    axs[1].set_ylabel(r"$\Delta \beta _y / \beta _y [\%]$")
    axs[1].set_xlabel(r"Longitudinal location $[m]$")
    axs[1].legend()

    fig.suptitle(f"Beam {beam}")
    fig.savefig(f"../generate_data/figures/example_twiss_beam{beam}.pdf")
    fig.show()
"""