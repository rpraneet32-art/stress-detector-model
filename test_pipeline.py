import sys
sys.path.insert(0, '.')

from config import seed_everything, ALL_SUBJECTS, WINDOW_SAMPLES, GLOBAL_SEED, subject_pkl_path
import os

seed_everything(GLOBAL_SEED)
print('Config OK | Subjects:', ALL_SUBJECTS)
print('Window samples:', WINDOW_SAMPLES)

missing = [s for s in ALL_SUBJECTS if not os.path.exists(subject_pkl_path(s))]
print('Missing subjects:', missing if missing else 'None - all 15 found!')

from data.loader import load_subject, channel_stats
sid = 2
print(f'\nLoading S{sid}...')
sigs, labs = load_subject(sid)
print(f'  signals shape: {sigs.shape}')
print(f'  labels  shape: {labs.shape}')
print(f'  label values : {sorted(set(labs.tolist()))}')
stats = channel_stats(sigs, labs, sid)
cb = stats['class_balance']
print(f'  class balance: non_stress={cb["n_non_stress"]} stress={cb["n_stress"]} discard={cb["n_discard"]} stress_frac={cb["stress_frac"]}')
ecg = stats['ECG']
print(f'  ECG stats: min={ecg["min"]:.4f} max={ecg["max"]:.4f} mean={ecg["mean"]:.4f} std={ecg["std"]:.4f}')
print('Loader OK!')

from data.preprocessing import preprocess
import numpy as np
filt = preprocess(sigs)
print(f'\nPreprocessing OK | filtered shape: {filt.shape}')
print(f'  ECG  after z-score: mean={filt[:,0].mean():.4f} std={filt[:,0].std():.4f}')
print(f'  PPG  after z-score: mean={filt[:,1].mean():.4f} std={filt[:,1].std():.4f}')
print(f'  Temp after z-score: mean={filt[:,2].mean():.4f} std={filt[:,2].std():.4f}')

from data.windowing import window_subject, N_FEATURES, FEATURE_NAMES
tens, feats, wlabs, wsids = window_subject(filt, labs, sid)
print(f'\nWindowing OK:')
print(f'  tensors  : {tens.shape}')
print(f'  features : {feats.shape}  (expected {N_FEATURES} features)')
print(f'  labels   : {wlabs.shape}  stress={int((wlabs==1).sum())} non_stress={int((wlabs==0).sum())}')
print(f'\nFeature names ({N_FEATURES}): {FEATURE_NAMES}')
print('\nALL LOCAL TESTS PASSED (no torch needed)')
