import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
import os
from pathlib import Path
from joblib import load
from cpymad import madx
from sklearn.ensemble import RandomForestRegressor


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
    
    output_data = np.concatenate( (np.vstack(triplet_errors), np.vstack(dpp_1), np.vstack(dpp_2)), axis=1)
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



example = tfs.read("b1_nominal_monitors.dat").set_index("NAME")
data_path = "data_include_offset"
noise = 1E-4
Nth_array=68
mdx = madx.Madx()
loaded_estimator = load('./estimators/test_2_triplet_phases_028_ridge_0.0001.pkl')
merge_data(data_path, noise, loaded_estimator)
