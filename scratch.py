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
nCells = rec['resp'].shape[0]

fgs, bgs = helpers.get_fg_bg_epochs(rec)

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


# plot all (isolated 1s) responses in PC space (of fg PCs / bg PCs)
s = 20
f, ax = plt.subplots(2, 2, figsize=(10, 8))

fgs1 = [f for f in fgs if f.endswith('0-1')]
bgs1 = [b for b in bgs if b.endswith('0-1')]

fg_resp = np.zeros((len(fgs1), nCells))
for i, fg in enumerate(fgs1):
    resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].sum(axis=-1).mean(axis=0)
    prespont = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
    fg_resp[i, :] = resp - prespont

fg_pca = PCA()
fg_pca.fit(fg_resp)
# plot scree plot
ax[0, 1].bar(range(fg_pca.components_.shape[0]), fg_pca.explained_variance_ratio_, edgecolor='k', width=0.5, color='grey')
ax[0, 1].set_ylabel('Variance explained ratio')
ax[0, 1].set_xlabel('PC')

# project single trials into fg_space (pc1 / pc2) and plot
for bg in bgs1:
    resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].sum(axis=-1)
    prespont = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
    proj = (resp - prespont).dot(fg_pca.components_[0:2, :].T)

    ax[0, 0].scatter(proj[:, 0], proj[:, 1], s=s, label=bg, color='lightgrey', edgecolor='white')

for fg in fgs1:
    resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].sum(axis=-1)
    prespont = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
    proj = (resp - prespont).dot(fg_pca.components_[0:2, :].T)

    ax[0, 0].scatter(proj[:, 0], proj[:, 1], s=s, edgecolor='white', label=fg)

ax[0, 0].axhline(0, linestyle='--', color='k')
ax[0, 0].axvline(0, linestyle='--', color='k')
ax[0, 0].set_xlabel(r"$PC_1$")
ax[0, 0].set_ylabel(r"$PC_2$")
ax[0, 0].legend(fontsize=8, frameon=False, bbox_to_anchor=(1,1), loc="upper left")
ax[0, 0].set_title('Foreground PCs')

bg_resp = np.zeros((len(bgs1), nCells))
for i, bg in enumerate(bgs1):
    resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].sum(axis=-1).mean(axis=0)
    prespont = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
    bg_resp[i, :] = resp - prespont

bg_pca = PCA()
bg_pca.fit(bg_resp)
# plot scree plot
ax[1, 1].bar(range(bg_pca.components_.shape[0]), bg_pca.explained_variance_ratio_, edgecolor='k', width=0.5, color='grey')
ax[1, 1].set_ylabel('Variance explained ratio')
ax[1, 1].set_xlabel('PC')

# project single trials into bg_space (pc1 / pc2) and plot
for fg in fgs1:
    resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].sum(axis=-1)
    prespont = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
    proj = (resp - prespont).dot(bg_pca.components_[0:2, :].T)

    ax[1, 0].scatter(proj[:, 0], proj[:, 1], s=s, edgecolor='white', label=fg, color='lightgrey')

for bg in bgs1:
    resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].sum(axis=-1)
    prespont = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
    proj = (resp - prespont).dot(bg_pca.components_[0:2, :].T)

    ax[1, 0].scatter(proj[:, 0], proj[:, 1], s=s, label=bg, edgecolor='white')

ax[1, 0].axhline(0, linestyle='--', color='k')
ax[1, 0].axvline(0, linestyle='--', color='k')
ax[1, 0].set_xlabel(r"$PC_1$")
ax[1, 0].set_ylabel(r"$PC_2$")
ax[1, 0].legend(fontsize=8, frameon=False, bbox_to_anchor=(1,1), loc="upper left")
ax[1, 0].set_title('Background PCs')

f.tight_layout()

plt.show()

ex_fg = '01Fight_Squeak'  #'07Kit_High-0-1'
ex_bg = '07Thunder'


