from nems_lbhb.baphy_experiment import BAPHYExperiment
import numpy as np
import matplotlib.pyplot as plt

mfile = '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP.m'

options = {'rasterfs': 10, 'resp': True}

manager = BAPHYExperiment(parmfile=[mfile])
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

fgs = np.unique([e.split('STIM_null_')[1] for e in rec.epochs.name.unique() if e.startswith('STIM_null_')]).tolist()
bgs = np.unique([e.split('_null')[0].split('STIM_')[1] for e in rec.epochs.name.unique() if e.endswith('_null')]).tolist()

prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
poststim = rec['resp'].extract_epoch('PostStimSilence').shape[-1]

# plot mean response to each background / foreground in isolation
fg_mean = {}
bg_mean = {}
for fg in fgs:
    resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].sum(axis=-1).mean()
    fg_mean[fg] = resp
for bg in bgs:
    resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].sum(axis=-1).mean()
    bg_mean[bg] = resp

f, ax = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

xpos = 0
for k, v in fg_mean.items():
    if k.endswith('0-1'):
        ax[0].scatter(xpos, v, s=50, color='k')
    elif k.endswith('0.5-1'):
        ax[0].scatter(xpos, v, s=50, color='grey')
        xpos += 1

ax[0].set_title('Foreground')
ax[0].set_ylabel('Mean pop. response')
ax[0].set_xticks(np.arange(xpos))
ax[0].set_xticklabels(list(fg_mean.keys())[::2], rotation=90)

xpos = 0
for k, v in bg_mean.items():
    if k.endswith('0-1'):
        ax[1].scatter(xpos, v, s=50, color='k')
    elif k.endswith('0.5-1'):
        ax[1].scatter(xpos, v, s=50, color='grey')
        xpos += 1
ax[1].set_title('Background')
ax[1].set_ylabel('Mean pop. response')
ax[1].set_xticks(range(xpos))
ax[1].set_xticklabels(list(bg_mean.keys())[::2], rotation=90)

f.tight_layout()

plt.show()

