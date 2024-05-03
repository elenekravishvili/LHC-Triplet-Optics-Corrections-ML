import matplotlib.pyplot as plt
import numpy as np
import cpymad.madx

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
            exec, do_twiss_monitors(LHCB1, 0.0);
        """)

class EnergyDispersion:
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

            do_twiss_elements(use_sequence, output_file, dpp): macro = {
            twiss, chrom, deltap=0.0037;
            };            

            exec, do_twiss_elements;
            etable, table="cetab";                                         
        """)

nominal_instance = Nominal()
energy_disp_instance = EnergyDispersion()


beta_x = np.array(nominal_instance.madx.table.twiss.betx)
beta_x_disp = np.array(energy_disp_instance.madx.table.twiss.betx)

s = np.array(nominal_instance.madx.table.twiss.s)
beta_beat=(beta_x_disp-beta_x)/beta_x
error_table=energy_disp_instance.madx.table.cetab.dframe()
summ_table=energy_disp_instance.madx.table.summ.dframe().deltap

print(summ_table)
"""
#plt.plot(s, beta_beat, label='Beta_beat')
plt.plot(s, beta_x, label='Beta_nominal')
plt.plot(s, beta_x_disp, label='Beta_Disp')
plt.xlabel('S')
plt.ylabel('Beta_x')
plt.title('Beta_x vs S')
plt.legend()
plt.grid(True)
plt.show()
"""