import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd
import os
from pathlib import Path
from joblib import load
from cpymad import madx


dpp_1= 1
dpp_2 = 2



def recons_twiss( beam, mdx, dpp_1, dpp_2):
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
            exec, match, deltap=dpp_offset;
            exec, vary, name=dQx.bbeam_number_op;
            exec, vary, name=dQy.bbeam_number_op;
            exec, constraint, range=#E, mux=nqx, muy=nqy;
            exec, lmdif;
            exec, endmatch;
            exec, do_twiss_elements(LHCB1, "twiss_final.tfs", dpp_offset);
               """)


    #print("Eroror", mdx.table.final_error.dframe())
    #tfs.writer.write_tfs(tfs_file_path=f"final_errors.tfs", data_frame=mdx.table.final_error.dframe())

    # Generate twiss with columns needed for training data
    mdx.input(f"""ndx := table(twiss,dx)/sqrt(table(twiss,betx));
               select, flag=twiss, clear;
              select, flag=twiss, pattern="^BPM.*B{beam}$", column=name, s, betx, bety, ndx,
                                             mux, muy;
           twiss, chrom, sequence=LHCB{beam}, deltap=0.0, file="";""")

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