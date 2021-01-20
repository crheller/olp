
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


def get_fg_pcs(rec):
    """
    Get top pc axes of FG evoked responses
    """
    prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
    poststim = rec['resp'].extract_epoch('PostStimSilence').shape[-1]
    nCells = rec['resp'].shape[0]

    fgs, bgs = get_fg_bg_epochs(rec)
    fgs1 = [f for f in fgs if f.endswith('0-1')]
    fg_resp = np.zeros((len(fgs1), nCells))
    for i, fg in enumerate(fgs1):
        resp = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, prestim:-poststim].sum(axis=-1).mean(axis=0)
        prespont = rec['resp'].extract_epoch('STIM_null_'+fg)[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
        fg_resp[i, :] = resp - prespont

    fg_pca = PCA()
    fg_pca.fit(fg_resp)
    return fg_pca.components_


def get_bg_pcs(rec):
    """
    Get top pc axes of BG evoked responses
    """
    prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
    poststim = rec['resp'].extract_epoch('PostStimSilence').shape[-1]
    nCells = rec['resp'].shape[0]
    
    fgs, bgs = get_fg_bg_epochs(rec)
    bgs1 = [b for b in bgs if b.endswith('0-1')]
    bg_resp = np.zeros((len(bgs1), nCells))
    for i, bg in enumerate(bgs1):
        resp = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, prestim:-poststim].sum(axis=-1).mean(axis=0)
        prespont = rec['resp'].extract_epoch('STIM_' + bg + '_null')[:, :, 0:prestim].sum(axis=-1).mean(axis=0)
        bg_resp[i, :] = resp - prespont

    bg_pca = PCA()
    bg_pca.fit(bg_resp)
    return bg_pca.components_