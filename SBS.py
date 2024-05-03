import matplotlib.pyplot as plt
import numpy as np
import cpymad.madx
import tfs

madx=cpymad.madx.Madx()

class nominal:
        def __init__(self):
                self.madx = cpymad.madx.Madx()

                self.madx.input("""
                        
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
class magneterrors:
        def __init__(self):
                self.madx = cpymad.madx.Madx()

                self.madx.input("""
                        
                call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
                call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
                call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx";
                        
      

                call, file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq";

                exec, define_nominal_beams(energy=6800);
                call, file='/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx';
                exec, cycle_sequences();  

                use, period = LHCB1;   
                Rr = 0.050;
                
                SetEfcomp_QEL: macro = {
                Efcomp, radius = Rr, order= 1,
                        dknr:={0, 0.0035};
                        };
                        
                select, flag=error, clear;
                select, flag=error, pattern = "^MQX[AB]\..*[R][1]";
                exec, SetEfcomp_QEL;          
                        

                exec, do_twiss_monitors(LHCB1, "", 0.0);                                   

                """)


run_sim = False
if run_sim:
    nominal_instance = nominal()
    magnet_errors_instance = magneterrors()

    tw_nominal = nominal_instance.madx.table.twiss.dframe()
    tw_errored = magnet_errors_instance.madx.table.twiss.dframe()

    tfs.write('twiss_nominal.tfs', tw_nominal)
    tfs.write('twiss_error.tfs', tw_errored)
else:
    tw_nominal = tfs.read('twiss_nominal.tfs')
    tw_errored = tfs.read('twiss_error.tfs')

tw_nominal = tw_nominal.set_index('name')
tw_errored = tw_errored.set_index('name')

print(tw_nominal)

#s=np.array(nominal_instance.madx.table.twiss.s)

""""
plt.plot(s, phase_adv_x, label='Phase_adv')
plt.plot(s, phase_adv_x_errored, label='Phase_adv error')
plt.xlabel('S')
plt.ylabel('Phase_adv')
plt.title('phase_adv vs S')
plt.legend()
plt.grid(True)
plt.show()
"""


Ref_element = 'bpm.12l1.b1:1'

'''
Phase_SBS_nominal = []
Ref_element_val = phase_adv_x[Ref_element]
for i in range(Ref_element, Nth_Element):
    delta = phase_adv_x[i] - Ref_element_val
    Phase_SBS_nominal.append(delta)

Phase_SBS_error = []
Ref_element_err = phase_adv_x_errored[Ref_element]
for i in range(Ref_element, Nth_Element):
    delta = phase_adv_x_errored[i] - Ref_element_err
    Phase_SBS_error.append(delta)

Phase_SBS_error_array = np.array(Phase_SBS_error)
Phase_SBS_nominal_array = np.array(Phase_SBS_nominal)

SBS = Phase_SBS_error_array - Phase_SBS_nominal_array
s = np.arange(Ref_element, Nth_Element)
'''

nominal_sbs = tw_nominal['mux'] - tw_nominal.loc[Ref_element, 'mux']
error_sbs = tw_errored['mux'] - tw_errored.loc[Ref_element, 'mux']

plt.plot(tw_nominal['s'], error_sbs-nominal_sbs, label='SBS')
plt.xlabel('S')
plt.ylabel('SBS')
plt.title('phase_adv_SBS vs S')
plt.legend()
plt.grid(True)
plt.show()
        