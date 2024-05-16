from cpymad import madx
import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
import os
from pathlib import Path
from joblib import load
from cpymad import madx
from sklearn.ensemble import RandomForestRegressor


def save_np_errors_tfs(np_errors):
    TRIPLET_NAMES = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                        "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                        "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                        "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]


    names = TRIPLET_NAMES
    recons_df = pd.DataFrame(columns=["NAME","K1L"])
    recons_df.K1L = np_errors 
    recons_df.NAME = names

    tfs.write("./twiss_reconstruct/predicted_errors/predicted_errors.tfs", recons_df)

def save_errors_rel_formal():
    names = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                        "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                        "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                        "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]

    example = tfs.read("example.tfs").set_index("NAME")
    errors = tfs.read("./twiss_reconstruct/predicted_errors/predicted_errors.tfs",).set_index("NAME")

    example.loc[names, "K1L"] = errors.loc[names, "K1L"]

    x = example
    x.reset_index(inplace=True)

    tfs.write("example_redefine.tfs", x)

def recons_twiss( beam, mdx):
    mdx.options(echo=False)
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2023.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx")
    mdx.call(file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq")
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
                READMYTABLE, file="/home/ekravishvili/Desktop/ML-Optic-Correction-main/src_afs/linear/example_redefine.tfs", table=errtab;
                SETERR, TABLE=errtab;""")

    mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0
    mdx.input(f"match_tunes(62.28, 60.31, {beam});")
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
    
    all_samples = np.load('./data_phase_adv_triplet/{}.npy'.format(set_name), allow_pickle=True)

    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,b1_disp, b2_disp,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors = all_samples.T
    
    #input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
     #       np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
      #      np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
       #     np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
           np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    
    # output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)
    output_data =  np.vstack(triplet_errors)

    return input_data, output_data







"""
input_data, output_data = merge_data(data_path, noise)
loaded_estimator = load('./estimators/triplet_phases_only_028_ridge_0.001.pkl')

beta_beat_x_b1, beta_beat_y_b1 = [], []
rms_x_b1, rms_y_b1, rms_x_b2, rms_y_b2  = [], [], [], []

for i in range(150):
    predicted_data = loaded_estimator.predict(input_data)[i]
    save_np_errors_tfs(predicted_data)
    save_errors_rel_formal()

    tw_recons = recons_twiss( 1, mdx)

    betax = np.array(tw_recons.betx)
    betay = np.array(tw_recons.bety)
    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("reconst_nom_twiss.tfs").set_index("name")
    beta_x_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["betx"])
    beta_y_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["bety"])

    beta_beating_x = (betax-beta_x_b1_nom)/beta_x_b1_nom
    rms_x1 = np.sqrt(np.mean(beta_beating_x**2, axis=0))
    rms_x_b1.append(rms_x1)

    beta_beating_y = (betax-beta_y_b1_nom)/beta_y_b1_nom
    rms_y1 = np.sqrt(np.mean(beta_beating_y**2, axis=0))
    rms_y_b1.append(rms_y1)



rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
rms_beta_beat_y_b1 = np.hstack(rms_y_b1)



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
"""

def load_data_mod(set_name, noise):
    
    all_samples = np.load('./data_phase_adv_triplet/{}.npy'.format(set_name), allow_pickle=True)

    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,b1_disp, b2_disp,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors = all_samples.T
    
    input_data = np.concatenate( (np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
           np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)    
    
    output_data =  np.vstack(triplet_errors)
    betas = beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2

    return np.vstack(beta_bpm_x_b1), np.vstack(beta_bpm_y_b1), np.vstack(beta_bpm_x_b2), np.vstack(beta_bpm_y_b2), input_data



def merge_data_mod(data_path, noise, estimator, mdx):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
    B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")
    B1_ELEMENTS_MDL_TFS_rec = tfs.read_tfs("reconst_nom_twiss.tfs").set_index("name")
    beta_x_b1_nom_rec =np.array(B1_ELEMENTS_MDL_TFS_rec["betx"])
    beta_y_b1_nom_rec =np.array(B1_ELEMENTS_MDL_TFS_rec["bety"])
    beta_x_b1_nom =np.array(B1_ELEMENTS_MDL_TFS["BETX"])
    beta_y_b1_nom =np.array(B1_ELEMENTS_MDL_TFS.BETY)
    beta_x_b2_nom =np.array(B2_ELEMENTS_MDL_TFS["BETX"])
    beta_y_b2_nom =np.array(B2_ELEMENTS_MDL_TFS.BETY)
    beta_beat_x_b1, beta_beat_y_b1, beta_beat_x_b2, beta_beat_y_b2 = [], [], [], []
    rms_x_b1, rms_y_b1, rms_x_b2, rms_y_b2  = [], [], [], []
    rms_x_b1_pred, rms_y_b1_pred, rms_x_b2_pred, rms_y_b2_pred  = [], [], [], []

    for file_name in file_names:
        beta_x_b1, beta_y_b1, beta_x_b2, beta_y_b2, input_data = load_data_mod(file_name, noise)
    
        for i in range(20):
            predicted_data = estimator.predict([input_data[i]])
            save_np_errors_tfs(predicted_data[0])
            save_errors_rel_formal()
            tw_recons = recons_twiss( 1, mdx)

            betax = np.array(tw_recons.betx)
            betay = np.array(tw_recons.bety)

            beta_beating_x = (betax-beta_x_b1_nom_rec)/beta_x_b1_nom_rec
            rms_x1_pred = np.sqrt(np.mean(beta_beating_x**2, axis=0))
            rms_x_b1_pred.append(rms_x1_pred)

            beta_beating_y = (betax-beta_y_b1_nom_rec)/beta_y_b1_nom_rec
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
    #x=np.hstack(beta_beat_x_b1)
    rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
    rms_beta_beat_y_b1 = np.hstack(rms_y_b1)
    #rms_beta_beat_x_b2 = np.hstack(rms_x_b2)
    #rms_beta_beat_y_b2 = np.hstack(rms_y_b2)
    rms_beta_beat_x_b1_pred = np.hstack(rms_x_b1_pred)
    rms_beta_beat_y_b1_pred = np.hstack(rms_y_b1_pred)


    plt.hist(rms_beta_beat_x_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, x')
    plt.hist(rms_beta_beat_x_b1_pred, bins=15, color='blue', alpha=0.5, label='rms of beta beating beam1 predected x')
    plt.hist(rms_beta_beat_x_b1-rms_beta_beat_x_b1_pred, bins=15, color='red', alpha=0.5, label='rms of beta beating beam1 resid, x')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show()

    plt.hist(rms_beta_beat_y_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, y')
    plt.hist(rms_beta_beat_y_b1_pred, bins=15, color='blue', alpha=0.5, label='rms of beta beating beam1 pred, y')
    plt.hist(rms_beta_beat_y_b1-rms_beta_beat_y_b1_pred, bins=15, color='red', alpha=0.5, label='rms of beta beating beam1 resid, y')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.show() 


data_path = "data_phase_adv_triplet"
noise = 1E-4
Nth_array=68
mdx = madx.Madx()
loaded_estimator = load('./estimators/triplet_phases_only_028_ridge_0.001.pkl')

merge_data_mod(data_path, noise, loaded_estimator, mdx)


