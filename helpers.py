
import numpy as np
from sklearn.decomposition import PCA

def get_fg_bg_epochs(rec):
    """
    Get fg / bg epoch names
    """

    fgs = np.unique([e.split('STIM_null_')[1] for e in rec.epochs.name.unique() if e.startswith('STIM_null_')]).tolist()
    bgs = np.unique([e.split('_null')[0].split('STIM_')[1] for e in rec.epochs.name.unique() if e.endswith('_null')]).tolist()

    return fgs, bgs


def get_fg_bg_transitions(rec):
    """
    get epochs that transitioned from only fg to fg + bg
    """
    trans_epochs = [e for e in rec.epochs.name.unique() if ('null' not in e) & ('STIM_' in e)]
    fgs, bgs = get_fg_bg_epochs(rec)
    lfgs = [f for f in fgs if '-0-1' in f]
    sbgs = [b for b in bgs if '-0.5-1' in b]
    fg_bg = [e for e in trans_epochs if any(ele in e for ele in lfgs) & any(ele in e for ele in sbgs)]
    return fg_bg

def get_bg_fg_transitions(rec):
    """
    get epochs that transitioned from only bg to fg + bg
    """
    trans_epochs = [e for e in rec.epochs.name.unique() if ('null' not in e) & ('STIM_' in e)]
    fgs, bgs = get_fg_bg_epochs(rec)
    sfgs = [f for f in fgs if '-0.5-1' in f]
    lbgs = [b for b in bgs if '-0-1' in b]
    bg_fg = [e for e in trans_epochs if any(ele in e for ele in sfgs) & any(ele in e for ele in lbgs)]
    return bg_fg

def get_fg_bg_full_overlap(rec):
    """
    get epochs that overlapped for the full period
    """
    trans_epochs = [e for e in rec.epochs.name.unique() if ('null' not in e) & ('STIM_' in e)]
    fgs, bgs = get_fg_bg_epochs(rec)
    lfgs = [f for f in fgs if '-0-1' in f]
    lbgs = [b for b in bgs if '-0-1' in b]
    bg_fg = [e for e in trans_epochs if any(ele in e for ele in lfgs) & any(ele in e for ele in lbgs)]
    return bg_fg


def get_fg_pcs(rec, collapse=False):
    """
    Get top pc axes of FG evoked responses
    """
    prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
    poststim = rec['resp'].extract_epoch('PostStimSilence').shape[-1]
    nCells = rec['resp'].shape[0]

    fgs, bgs = get_fg_bg_epochs(rec)
    fgs1 = [f for f in fgs if f.endswith('0-1')]
    if collapse:
        fg_resp = np.zeros((len(fgs1), nCells))
    else: 
        fg_resp = []
    for i, fg in enumerate(fgs1):
        if collapse:
            resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].sum(axis=-1).mean(axis=0)
            prespont = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
            fg_resp[i, :] = resp - prespont
        else:
            resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].mean(axis=0)
            prespont = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, 0:prestim].mean(axis=(0, -1))
            fg_resp.append((resp.T - prespont))

    if type(fg_resp) == list:
        fg_resp = np.concatenate(fg_resp, axis=0)
    fg_pca = PCA()
    fg_pca.fit(fg_resp)
    return fg_pca.components_


def get_bg_pcs(rec, collapse=False):
    """
    Get top pc axes of BG evoked responses
    """
    prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
    poststim = rec['resp'].extract_epoch('PostStimSilence').shape[-1]
    nCells = rec['resp'].shape[0]
    
    fgs, bgs = get_fg_bg_epochs(rec)
    bgs1 = [b for b in bgs if b.endswith('0-1')]
    if collapse:
        bg_resp = np.zeros((len(bgs1), nCells))
    else:
        bg_resp = []
    for i, bg in enumerate(bgs1):
        if collapse:
            resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].sum(axis=-1).mean(axis=0)
            prespont = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
            bg_resp[i, :] = resp - prespont
        else:
            resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].mean(axis=0)
            prespont = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, 0:prestim].mean(axis=(0, -1))
            bg_resp.append((resp.T - prespont))

    if type(bg_resp) == list:
        bg_resp = np.concatenate(bg_resp, axis=0)
    bg_pca = PCA()
    bg_pca.fit(bg_resp)
    return bg_pca.components_


def get_decoder(rec, fg_epoch, bg_epoch, collapse=False):
    """
    Get decoding axis for each time bin for the given epoch pair
    """

    prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
    poststim = rec['resp'].extract_epoch('PostStimSilence').shape[-1]
    tbins = rec['resp'].extract_epoch('REFERENCE').shape[-1]
    nCells = rec['resp'].shape[0]

    fg_resp = rec['resp'].extract_epoch('STIM_null_'+fg_epoch) 
    bg_resp = rec['resp'].extract_epoch('STIM_'+bg_epoch+'_null') 

    decoding_axes = np.zeros((tbins, nCells))
    if collapse:
        # single decoding axis, repped out for all tbins
        fr = fg_resp[:, :, prestim:-poststim].transpose([0, 2, 1]).reshape(-1, nCells)
        br = bg_resp[:, :, prestim:-poststim].transpose([0, 2, 1]).reshape(-1, nCells)
        axis = fr.mean(axis=0) - br.mean(axis=0)
        axis /= np.linalg.norm(axis)
        decoding_axes = np.tile(axis, [tbins, 1])
    else:
        # different decoding axis for each time point
        for t in range(fg_resp.shape[-1]):
            fr = fg_resp[:, :, t]
            br = bg_resp[:, :, t]
            axis = fr.mean(axis=0) - br.mean(axis=0)
            axis /= np.linalg.norm(axis)
            decoding_axes[t, :] = axis

    return decoding_axes