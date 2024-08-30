import matplotlib.pyplot as plt
import numpy as np
import cpymad.madx


#This is separate script for energy offset, for testing. It is not usedn in the, offset job is included in the madx_jobs.py
class Nominal:
    def __init__(self):
        self.madx = cpymad.madx.Madx()

        self.madx.input("""
            ! MAD-X script for nominal case here
            call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
            call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
            call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx";
            call, file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq";
            exec, define_nominal_beams(energy=6800);
            call, file='/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx';
            exec, cycle_sequences();
            use, period = LHCB1;
            exec, do_twiss_elements(LHCB1, "twiss_nominal.tfs" 0.0);
        """)

class Energy_offset:
    def __init__(self):
        self.madx = cpymad.madx.Madx()

        self.madx.input("""
            ! MAD-X script for energy dispersion case here
            call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
            call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
            call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx";
            call, file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq";
            exec, define_nominal_beams(energy=6800);
            call, file='/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx';
            exec, cycle_sequences();
            use, period = LHCB1;
            dpp_offset=0.0002;

                   

            exec, do_twiss_elements(LHCB1, "twiss_off.tfs", dpp_offset);
            ! etable, table="cetab";    

                                    
            correct,mode=svd;  
            
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
            exec, do_twiss_elements(LHCB1, "twiss_final.tfs", dpp_offset);
            
            ! twiss, chrom;                                    
        """)

nominal_instance = Nominal()
energy_offset_instance = Energy_offset()


"""
Qx_nom=nominal_instance.madx.table.summ.q1
Qy_nom=nominal_instance.madx.table.summ.q2 
deltap_nom=nominal_instance.madx.table.summ.deltap

Qx_offset=energy_offset_instance.madx.table.summ.q1
Qy_offset=energy_offset_instance.madx.table.summ.q2
deltap_offset=energy_offset_instance.madx.table.summ.deltap
print("nominal tunes:", Qx_nom, Qy_nom)
print("tunes with energy offset:", Qx_offset, Qy_offset)
print("nominal offset (0) :", deltap_nom, "offset ofsset:", deltap_offset)

"""




"""
beta_x = np.array(nominal_instance.madx.table.twiss.betx)
beta_x_disp = np.array(energy_disp_instance.madx.table.twiss.betx)

s = np.array(nominal_instance.madx.table.twiss.s)
beta_beat=(beta_x_disp-beta_x)/beta_x
error_table=energy_disp_instance.madx.table.cetab.dframe()
summ_table=energy_disp_instance.madx.table.summ.dframe().deltap

print(summ_table)

plt.plot(s, beta_x, label='Beta_nominal')
plt.plot(s, beta_x_disp, label='Beta_Disp')
plt.xlabel('S')
plt.ylabel('Beta_x')
plt.title('Beta_x vs S')
plt.legend()
plt.grid(True)
plt.show()
"""