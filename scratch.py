from nems_lbhb.baphy_experiment import BAPHYExperiment

mfile = '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP.m'

options = {'rasterfs': 10, 'resp': True}

manager = BAPHYExperiment(parmfile=[mfile])
rec = manager.get_recording(**options)