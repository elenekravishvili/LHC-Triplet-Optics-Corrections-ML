import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
import os
from pathlib import Path
from joblib import load
from cpymad import madx
#from sklearn.ensemble import RandomForestRegressor


def merge_data(data_path, noise, estimator):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    for file_name in file_names:
        beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2,\
        input_data, output_data, dpp1, dpp_2 = load_data_mod(file_name, noise)
        for i in range(1):
            predicted_data = estimator.predict([input_data[i]])
            corrected_errors = output_data[i]-predicted_data
            print(corrected_errors.shape)

    # return np.concatenate(input_data), np.concatenate(output_data)

def load_data_mod(set_name, noise):
    
    all_samples = np.load('./data_include_offset/{}.npy'.format(set_name), allow_pickle=True)

    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2,\
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors, dpp_1, dpp_2 = all_samples.T
    
    input_data = np.concatenate( (np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
            np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
            np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
            np.vstack(delta_mux_b2), np.vstack(delta_muy_b2)), axis=1)   
    
    #output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)
    output_data = np.vstack(triplet_errors)
    # betas = beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2

    return np.vstack(beta_bpm_x_b1), np.vstack(beta_bpm_y_b1), np.vstack(beta_bpm_x_b2), np.vstack(beta_bpm_y_b2),\
     input_data, output_data, dpp_1, dpp_2



def recons_twiss_with_off( beam, mdx, dpp_1, dpp_2):
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
                READMYTABLE, file="/afs/cern.ch/user/e/ekravish/public/src_afs/linear/example_redefine.tfs", table=errtab;
                SETERR, TABLE=errtab;""")

    mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0
    mdx.input(f"match_tunes(62.28, 60.31, {beam});")
    mdx.input(f"""etable, table="final_error";""")
    
    mdx.twiss(sequence=f"LHCB{beam}", file="")    

    mdx.input(f"match_tunes(62.28, 60.31, {beam});")
    mdx.input(f"""etable, table="final_error";""")
    
    mdx.twiss(sequence=f"LHCB{beam}", file="")   

    
    mdx.input(f"""dpp_offset={dpp_1};
            exec, do_twiss_elements(LHCB1, "twiss_off.tfs", dpp_offset);
            correct,mode=svd; 
            exec, do_twiss_elements(LHCB1, "twiss_corrected.tfs", dpp_offset);  
            exec, find_complete_tunes(62.28, 60.31, 1);
            match, deltap=dpp_offset;
            vary, name=dQx.b1_op;
            vary, name=dQy.b1_op;
            constraint, range=#E, mux=62.28, muy=60.31;
            lmdif;
            endmatch;
            exec, do_twiss_elements(LHCB1, "twiss_corrected.tfs", dpp_offset);
              
               """)


    #print("Eroror", mdx.table.final_error.dframe())
    #tfs.writer.write_tfs(tfs_file_path=f"final_errors.tfs", data_frame=mdx.table.final_error.dframe())

    # Generate twiss with columns needed for training data


    return mdx.table.twiss.dframe()


def job_energyoffset_b1(self, seed):
    self.input('''
        EOPTION, SEED = %(SEED)s;
        GCUTR = 3; ! Cut for truncated gaussians (sigmas)
        err = RANF()*TGAUSS(GCUTR);
        dpp_offset=0.0001*err;
                

        ! twiss, chrom, sequence=LHCB1, deltap=dpp_offset, file="twiss_off.tfs";
        exec, do_twiss_elements(LHCB1, "twiss_off.tfs", dpp_offset);
        ! etable, table="cetab";    
                        
        correct,mode=svd;  
        
        ! twiss, chrom, sequence=LHCB1, deltap=dpp_offset, file="twiss_corrected.tfs";
        exec, do_twiss_elements(LHCB1, "twiss_corrected.tfs", dpp_offset);            
                    
        match_tunes_dpp(nqx, nqy, beam_number): macro = {
            exec, find_complete_tunes(nqx, nqy, beam_number);
            exec, match_tunes_op_dpp(total_qx, total_qy, beam_number);
        };

        match_tunes_op_dpp(nqx, nqy, beam_number): macro = {
            match, deltap=dpp_offset;
            vary, name=dQx.bbeam_number_op;
            vary, name=dQy.bbeam_number_op;
            constraint, range=#E, mux=nqx, muy=nqy;
            lmdif;
            endmatch;
        };
        
        exec, match_tunes_dpp(62.28, 60.31, 1);
        ! twiss, chrom, sequence=LHCB1, deltap=dpp_offset, file="twiss_final.tfs";
        exec, do_twiss_elements(LHCB1, "twiss_final.tfs", dpp_offset);
        

        ! twiss, chrom;                                    
    '''% {"SEED": seed})

def save_np_errors_tfs(np_errors):
    TRIPLET_NAMES = ["MQXA.3L2", "MQXB.B2L2", "MQXB.A2L2", "MQXA.1L2",  "MQXA.1R2", "MQXB.A2R2",  "MQXB.B2R2" , "MQXA.3R2", \
                        "MQXA.3L5" , "MQXB.B2L5", "MQXB.A2L5", "MQXA.1L5",  "MQXA.1R5", "MQXB.A2R5",  "MQXB.B2R5" , "MQXA.3R5",\
                        "MQXA.3L8" , "MQXB.B2L8", "MQXB.A2L8", "MQXA.1L8",  "MQXA.1R8", "MQXB.A2R8",  "MQXB.B2R8" , "MQXA.3R8",\
                        "MQXA.3L1" , "MQXB.B2L1", "MQXB.A2L1", "MQXA.1L1",  "MQXA.1R1", "MQXB.A2R1",  "MQXB.B2R1" , "MQXA.3R1"]


    names = TRIPLET_NAMES
    recons_df = pd.DataFrame(columns=["NAME","K1L"])
    recons_df.K1L = np_errors 
    recons_df.NAME = names
    print(recons_df)

    tfs.write("./predicted_errors/test_predicted_triplet_errors.tfs", recons_df)

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


def beta_beat_dist(data_path, noise, mdx):
    #Takes folder path for all different data files and merges them
    input_data, output_data, beta  = [], [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    dpp_1 = tfs.read("./predicted_errors/pred_dpp_1.tfs")
    dpp_2 = tfs.read("./predicted_errors/pred_dpp_2.tfs")
    pred_triplet_err = tfs.read("./predicted_errors/pred_triplet_incl_off.tfs")
    dpp_1_array = np.array(dpp_1)
    dpp_2_array = np.array(dpp_2)
    pred_triplet_err_array = np.array(pred_triplet_err)

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
    number = 0
    for file_name in file_names:
        beta_x_b1, beta_y_b1, beta_x_b2, beta_y_b2, input_data, output_data, off_1, off_2 = load_data_mod(file_name, noise)
    
        for i in range(1):
            try:
                predicted_data = pred_triplet_err_array[i]
                corrected_errors = output_data[i]-predicted_data
                save_np_errors_tfs(corrected_errors[0])
                save_errors_rel_formal()


                deltap_1 =off_1[i] - dpp_1_array[i]
                deltap_2 = off_2[i] - dpp_2_array[i]

                tw_recons_first = recons_twiss_with_off(1, mdx, deltap_1, deltap_2)
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
                number +=1
                #bb_x2 = (beta_x_b2[i]-beta_x_b2_nom)/beta_x_b2_nom
                #rms_x2 = np.sqrt(np.mean(bb_x2**2, axis=0))
                #rms_x_b2.append(rms_x2)

                #bb_y2 = (beta_y_b2[i]-beta_y_b2_nom)/beta_y_b2_nom
                #rms_y2 = np.sqrt(np.mean(bb_y2**2, axis=0))
                #rms_y_b2.append(rms_y2) 
            except Exception as e:
                # Code to handle the exception
                print(f"An error occurred with {i}: {e}")
    print(number)
    #x=np.hstack(beta_beat_x_b1)
    """
    rms_beta_beat_x_b1 = np.hstack(rms_x_b1)
    tfs.write("rms_beta_beat_x_b1.tfs", rms_beta_beat_x_b1)
    rms_beta_beat_y_b1 = np.hstack(rms_y_b1)
    tfs.write("rms_beta_beat_y_b1.tfs", rms_beta_beat_y_b1)
    #rms_beta_beat_x_b2 = np.hstack(rms_x_b2)
    #rms_beta_beat_y_b2 = np.hstack(rms_y_b2)
    rms_beta_beat_x_b1_pred = np.hstack(rms_x_b1_pred)
    tfs.write("test_incl_off_rms_beta_beat_x_b1_corrected.tfs", rms_beta_beat_x_b1_pred)
    rms_beta_beat_y_b1_pred = np.hstack(rms_y_b1_pred)
    tfs.write("rms_beta_beat_y_b1_corrected.tfs", rms_beta_beat_y_b1_pred)

    
    plt.hist(rms_beta_beat_x_b1, bins=15, color='green', alpha=0.5, label='rms of beta beating beam1, x')
    plt.hist(rms_beta_beat_x_b1_pred, bins=15, color='blue', alpha=0.5, label='rms of beta beating beam1 corrected x')
    plt.xlabel('rms')
    plt.ylabel('Counts')
    plt.title('rms')
    plt.legend()  # Add legend to display labels
    plt.grid(True)
    plt.savefig("test_incl_off_beta_beat_recons_b1x.pdf")
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
data_path = "data_include_offset"
noise = 1E-4
Nth_array=68
mdx = madx.Madx()
#loaded_estimator = load('./estimators/test_2_triplet_phases_028_ridge_0.0001.pkl')
#merge_data(data_path, noise, loaded_estimator)
dpp_1 = tfs.read("./predicted_errors/pred_dpp_1.tfs")
dpp_2 = tfs.read("./predicted_errors/pred_dpp_2.tfs")
pred_triplet_err = tfs.read("./predicted_errors/pred_triplet_incl_off.tfs")
dpp_1_array = np.array(dpp_1)
dpp_2_array = np.array(dpp_2)
pred_triplet_err_array = np.array(pred_triplet_err)

beta_beat_dist(data_path, noise, mdx)

#x = recons_twiss_with_off(1, mdx, 0.0005, 0.00034)
#print(x)