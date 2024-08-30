#%%
import cpymad.madx


""" ------------------------------------------------------- 0 --------------------------------------------------

----------------------------------------------------------- 0 --------------------------------------------------  """

class madx_ml_op(cpymad.madx.Madx):
    '''Normal cpymad wrapper with extra methods for this exact project'''
    def job_init_sbs_b1(self, beam, IP):
        self.input('''
        option, -echo;
        
        ! call, file = "/afs/cern.ch/work/f/fcarlier/public/BBsrc/madx/lib/segments.macros.madx";
        
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2023.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx";
        
        ! call, file = "/home/ekravish/Desktop/src_afs/src_afs/linear/segments.macros.madx";
        call, file  = "/afs/cern.ch/work/e/ekravish/public/linear/segments.macros.madx";
                

        !@require lhc_runIII_2022
        !@require segments


        call, file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq";
        exec, define_nominal_beams();
        call, file='/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx';
        ! exec, set_default_crossing_scheme();

        ! Cycle the sequence in the start point to
        ! avoid negative length sequence.
        seqedit, sequence=LHCB1;
        flatten;
        cycle, start=BPM.12L1.B1;
        endedit;

        use, period = LHCB1;

        
        SAVEBETA, LABEL=sbs_init_ip1, PLACE=BPM.12L1.B1, SEQUENCE=LHCB1;
        TWISS;
                
        exec, extract_segment_sequence(
            LHCB1,
            front_LHCB1, back_LHCB1,
            BPM.12L1.B1, BPM.12R1.B1
        );
        
        exec, beam_LHCB1(front_LHCB1);
        ''')

    def sbs_twiss_b1(self):
        self.input('''
        exec, twiss_segment(front_LHCB1, sbs_init_ip1);
        ''')
        
    def set_triplet_errors_b1(self, beam, IP, err1, err2, err3, err4, err5, err6, err7, err8 ):
        self.input('''
        
        ! call, file="corrections_IP1.madx"
        ! READMYTABLE, file="/afs/cern.ch/user/e/ekravish/public/src_afs/linear/example.tfs", table=errtab;
        ! SETERR, TABLE=errtab;
        
        MQXA.3L1->K1 = MQXA.3L1->K1 + (%(err1)s/MQXA.3L1->L);
        MQXB.B2L1->K1 = MQXB.B2L1->K1 + (%(err2)s/MQXB.B2L1->L);
        MQXB.A2L1->K1 = MQXB.A2L1->K1 + (%(err3)s/MQXB.A2L1->L);
        MQXA.1L1->K1 = MQXA.1L1->K1 + (%(err4)s/MQXA.1L1->L);
        MQXA.1R1->K1 = MQXA.1R1->K1 + (%(err5)s/MQXA.1R1->L);
        MQXB.A2R1->K1 = MQXB.A2R1->K1 + (%(err6)s/MQXB.A2R1->L);
        MQXB.B2R1->K1 = MQXB.B2R1->K1 + (%(err7)s/MQXB.B2R1->L);
        MQXA.3R1->K1 = MQXA.3R1->K1 + (%(err8)s/MQXA.3R1->L);
        '''% {"beam": beam, "IP": IP, "err1": err1, "err2": err2, "err3": err3, "err4": err4, "err5": err5, "err6": err6, "err7": err7, "err8": err8})
    
    def job_init_sbs_b2(self, beam, IP):
        self.input('''
        option, -echo;
        
        ! call, file = "/afs/cern.ch/work/f/fcarlier/public/BBsrc/madx/lib/segments.macros.madx";
        
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2023.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2024.macros.madx";
        
        ! call, file = "/home/ekravish/Desktop/src_afs/src_afs/linear/segments.macros.madx";
        call, file  = "/afs/cern.ch/work/e/ekravish/public/linear/segments.macros.madx";
                

        !@require lhc_runIII_2022
        !@require segments

        option, -echo;

        call, file = "/afs/cern.ch/eng/acc-models/lhc/2024/lhc.seq";
        exec, define_nominal_beams();
        call, file='/afs/cern.ch/eng/acc-models/lhc/2024/operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx';
        ! exec, set_default_crossing_scheme();

        ! Cycle the sequence in the start point to
        ! avoid negative length sequence.
        seqedit, sequence=LHCB2;
        flatten;
        cycle, start=BPM.12L1.B2;
        endedit;

        use, period = LHCB2;

        
        SAVEBETA, LABEL=sbs_init_ip2, PLACE=BPM.12L1.B2, SEQUENCE=LHCB2;
        TWISS;
                
        exec, extract_segment_sequence(
            LHCB2,
            front_LHCB2, back_LHCB2,
            BPM.12L1.B2, BPM.12R1.B2
        );
        
        exec, beam_LHCB2(front_LHCB2);
        ''')

    def sbs_twiss_b2(self):
        self.input('''
        exec, twiss_segment(front_LHCB2, sbs_init_ip2);
        ''')
        
    def set_triplet_errors_b2(self, beam, IP, err1, err2, err3, err4, err5, err6, err7, err8 ):
        self.input('''
        
        ! call, file="corrections_IP1.madx"
        ! READMYTABLE, file="/afs/cern.ch/user/e/ekravish/public/src_afs/linear/example.tfs", table=errtab;
        ! SETERR, TABLE=errtab;
        
        MQXA.3L1->K1 = MQXA.3L1->K1 + (%(err1)s/MQXA.3L1->L);
        MQXB.B2L1->K1 = MQXB.B2L1->K1 + (%(err2)s/MQXB.B2L1->L);
        MQXB.A2L1->K1 = MQXB.A2L1->K1 + (%(err3)s/MQXB.A2L1->L);
        MQXA.1L1->K1 = MQXA.1L1->K1 + (%(err4)s/MQXA.1L1->L);
        MQXA.1R1->K1 = MQXA.1R1->K1 + (%(err5)s/MQXA.1R1->L);
        MQXB.A2R1->K1 = MQXB.A2R1->K1 + (%(err6)s/MQXB.A2R1->L);
        MQXB.B2R1->K1 = MQXB.B2R1->K1 + (%(err7)s/MQXB.B2R1->L);
        MQXA.3R1->K1 = MQXA.3R1->K1 + (%(err8)s/MQXA.3R1->L);
        '''% {"beam": beam, "IP": IP, "err1": err1, "err2": err2, "err3": err3, "err4": err4, "err5": err5, "err6": err6, "err7": err7, "err8": err8})    
    
            
# %%
