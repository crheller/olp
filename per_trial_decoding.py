"""
Over the course of the trial, plot the projections onto:
    - decoding axis (computed per bin? If so, show similarity over time with matrix...)
    - FG encoding axis (PC1 of fg in isolation)
    - BG encoding axis (PC1 of bg in isolation)
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import numpy as np
import matplotlib.pyplot as plt
import helpers as helpers
from sklearn.decomposition import PCA

mfile = '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP.m'

options = {'rasterfs': 10, 'resp': True}
manager = BAPHYExperiment(parmfile=[mfile])
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

# REFERENCE time
nbins = rec['resp'].extract_epoch('REFERENCE').shape[-1]
time = np.linspace(0, nbins / rec['resp'].fs, nbins)
prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rec['resp'].fs
poststim = time[-1] - rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rec['resp'].fs

# get epochs
fgs, bgs = helpers.get_fg_bg_epochs(rec)
fg_bg_epochs = helpers.get_fg_bg_transitions(rec) 
bg_fg_epochs = helpers.get_bg_fg_transitions(rec)

# get projection axes
fg_pcs = helpers.get_fg_pcs(rec)
bg_pcs = helpers.get_bg_pcs(rec)
decoder = helpers.get_decoder(rec)

# project / plot single trial data for fg to bg + fg transitions
proj_axis = bg_pcs[0, :]
f, ax = plt.subplots(len(fg_bg_epochs), 1, figsize=(8, 12), sharey=True, sharex=True)

r_fgbg = rec['resp'].extract_epochs(fg_bg_epochs)
for trans, a in zip(fg_bg_epochs, ax):
    resp = r_fgbg[trans]
    resp = resp.transpose([0, 2, 1]).dot(proj_axis.T)
    mresp = resp.mean(axis=0)
    sem = resp.std(axis=0) #/ np.sqrt(resp.shape[0])

    # plot
    a.plot(time, mresp, lw=2, color='tab:blue')
    a.fill_between(time, mresp-sem, mresp+sem, alpha=0.5, lw=0, color='tab:blue')
    a.set_title(trans)
    a.axvline(prestim + 0.5, linestyle='--', color='red')
    a.axvline(prestim, linestyle='--', color='k')
    a.axvline(poststim, linestyle='--', color='k')

f.tight_layout()

plt.show()