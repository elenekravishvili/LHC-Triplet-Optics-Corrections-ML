#%%
import cpymad.madx
#import pymadng

""" ------------------------------------------------------- 0 --------------------------------------------------

----------------------------------------------------------- 0 --------------------------------------------------  """

class madx_ml_op(cpymad.madx.Madx):
    '''Normal cpymad wrapper with extra methods for this exact project'''
    def job_sbs(self, beam, IP, err1, err2, err3, err4, err5, err6, err7, err8 ):
        self.input('''
        option, -echo;

        ! call, file = "/afs/cern.ch/work/f/fcarlier/public/BBsrc/madx/lib/segments.macros.madx";
        
        
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2023.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx";
        
        call, file = "/home/ekravish/Desktop/src_afs/src_afs/linear/segments.macros.madx"
        
        
        option, echo;

        !@require lhc_runIII_2022
        !@require segments

        option, -echo;

        call, file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq";
        exec, define_nominal_beams();
        call, file='/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx';
        exec, set_default_crossing_scheme();

        ! Cycle the sequence in the start point to
        ! avoid negative length sequence.
        seqedit, sequence=LHCB%(beam)s;
        flatten;
        cycle, start=BPM.12L%(IP)s.B1;
        endedit;

        use, period = LHCB%(beam)s;

        option, echo;

        
        SAVEBETA, LABEL=sbs_init_ip1, PLACE=BPM.12L%(IP)s.B1, SEQUENCE=LHCB1;
        TWISS;
                
        exec, extract_segment_sequence(
            LHCB%(beam)s,
            front_LHCB%(beam)s, back_LHCB%(beam)s,
            BPM.12L%(IP)s.B%(beam)s, BPM.12R%(IP)s.B%(beam)s
        );
        exec, beam_LHCB%(beam)s(front_LHCB%(beam)s);

        exec, twiss_segment(front_LHCB%(beam)s, "twiss_IP1.dat", sbs_init_ip1);
        ! call, file="corrections_IP1.madx"
        ! READMYTABLE, file="/afs/cern.ch/user/e/ekravish/public/src_afs/linear/example.tfs", table=errtab;
        ! SETERR, TABLE=errtab;
        
        MQXA.3L%(IP)s->K1 = MQXA.3L%(IP)s->K1 + %(err1)s/MQXA.3L%(IP)s->L;
        MQXB.B2L%(IP)s->K1 = MQXB.B2L%(IP)s->K1 + %(err2)s/MQXB.B2L%(IP)s->L;
        MQXB.A2L%(IP)s->K1 = MQXB.A2L%(IP)s->K1 + %(err3)s/MQXB.A2L%(IP)s->L;
        MQXA.1L%(IP)s->K1 = MQXA.1L%(IP)s->K1 + %(err4)s/MQXA.1L%(IP)s->L;
        MQXA.1R%(IP)s->K1 = MQXA.1R%(IP)s->K1 + %(err5)s/MQXA.1R%(IP)s->L;
        MQXB.A2R%(IP)s->K1 = MQXB.A2R%(IP)s->K1 + %(err6)s/MQXB.A2r%(IP)s->L;
        MQXB.B2R%(IP)s->K1 = MQXB.B2R%(IP)s->K1 + %(err7)s/MQXB.B2R%(IP)s->L;
        MQXA.3R%(IP)s->K1 = MQXA.3R%(IP)s->K1 + %(err8)s/MQXA.3R%(IP)s->L;
        
                
        exec, twiss_segment(front_LHCB%(beam)s, "twiss_IP1_cor.dat", sbs_init_ip1);
        twiss;
        '''% {"beam": beam, "IP": IP, "err1": err1, "err2": err2, "err3": err3, "err4": err4, "err5": err5, "err6": err6, "err7": err7, "err8": err8,})        
# %%
