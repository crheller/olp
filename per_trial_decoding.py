"""
Over the course of the trial, plot the projections onto:
    - decoding axis (computed per bin? If so, show similarity over time with matrix...)
    - FG encoding axis (PC1 of fg in isolation)?
    - BG encoding axis (PC1 of bg in isolation)?
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import numpy as np
import matplotlib.pyplot as plt
import helpers as helpers
from sklearn.decomposition import PCA

mfile = '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP.m'

options = {'rasterfs': 20, 'resp': True}
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

# project / plot single trial data for fg to bg + fg transitions
proj_axis = fg_pcs[0, :]
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


# if collapse, single axis for each time point

fg_epoch = '01Fight_Squeak-0-1'
bg_epoch = '03Insect_Buzz-0-1'
fg_bg_epoch = 'STIM_03Insect_Buzz-0.5-1_01Fight_Squeak-0-1'
decoder = helpers.get_decoder(rec, fg_epoch, bg_epoch, collapse=False)

f = plt.figure(figsize=(11, 3))

tser = plt.subplot2grid((1, 5), (0, 0), colspan=3)
sim = plt.subplot2grid((1, 5), (0, 3), colspan=2)

fg_resp = rec['resp'].extract_epoch('STIM_null_'+fg_epoch)
bg_resp = rec['resp'].extract_epoch('STIM_'+bg_epoch+'_null')
fg_bg_resp = rec['resp'].extract_epoch(fg_bg_epoch)

# project single trials onto each decoding axis
# responses in isolation
fg_resp = np.concatenate([fg_resp[:, :, i] @ decoder[[i], :].T for i in range(decoder.shape[0])], axis=-1)
bg_resp = np.concatenate([bg_resp[:, :, i] @ decoder[[i], :].T for i in range(decoder.shape[0])], axis=-1)
# response to combined
fg_bg_resp = np.concatenate([fg_bg_resp[:, :, i] @ decoder[[i], :].T for i in range(decoder.shape[0])], axis=-1)

tser.plot(time, fg_resp.mean(axis=0), '--', label='fg (isolation)', color='tab:blue', lw=2)
tser.plot(time, bg_resp.mean(axis=0), '--', label='bg (isolation)', color='tab:orange', lw=2)
m = fg_bg_resp.mean(axis=0)
sem = fg_bg_resp.std(axis=0)
tser.plot(time, m, lw=2, color='grey')
tser.fill_between(time, m-sem, m+sem, lw=0, alpha=0.5, color='grey', label='Combination')
tser.axvline(prestim+0.5, color='k', lw=2, label='transition', zorder=0)

tser.legend(frameon=False, bbox_to_anchor=(1,1), loc="upper left")
tser.set_ylabel('Decoding axis projection')
tser.set_xlabel('Trial time')
tser.set_title(fg_bg_epoch)
tser.set_xlim((time[0], time[-1]))

sim.imshow(np.abs(decoder @ decoder.T), aspect='auto', cmap='Reds', vmin=0, vmax=1, extent=[time[0], time[-1], time[0], time[-1]])
sim.set_title('Decoding axes similarity')

f.tight_layout()

plt.show()