import mne
import numpy as np
import pandas as pd
import os
from glob import glob

def get_full_file_map():
    file_map = {}
    # Список всех папок, где могут быть PSG
    folders = ['1-5', '6-10', '11-15', '16-17', 'Непонятно']
    
    for i in range(1, 20):
        sub_id = f"01-02-{i:04d}"
        entry = {'psg': None, 'base': None, 'spindles': None}
        
        # Ищем PSG и Base в любой из папок
        for folder in folders:
            psg = os.path.join(folder, f"{sub_id} PSG.edf")
            base = os.path.join(folder, f"{sub_id} Base.edf")
            if os.path.exists(psg): entry['psg'] = psg
            if os.path.exists(base): entry['base'] = base
            
        # Спиндлы всегда в annotationsss2
        spindles = os.path.join("annotationsss2", f"{sub_id} Spindles_E1.edf")
        if os.path.exists(spindles): entry['spindles'] = spindles
        
        if entry['psg'] and entry['base'] and entry['spindles']:
            file_map[sub_id] = entry
            
    return file_map

def extract_features(sub_id, paths, epoch_duration=3.0):
    print(f"Processing {sub_id}...")
    try:
        raw = mne.io.read_raw_edf(paths['psg'], preload=True, verbose=False)
        # Ищем канал C3 (может называться по-разному)
        target_ch = [ch for ch in raw.ch_names if 'C3' in ch][0]
        raw.pick_channels([target_ch])
        sfreq = raw.info['sfreq']
        
        base = mne.io.read_raw_edf(paths['base'], verbose=False)
        n2_mask = [a for a in base.annotations if 'stage 2' in a['description'].lower()]
        
        spindles = mne.io.read_raw_edf(paths['spindles'], verbose=False)
        sp_annots = spindles.annotations
        
        epochs, labels = [], []
        for annot in n2_mask:
            start, dur = annot['onset'], annot['duration']
            for t in np.arange(start, start + dur - epoch_duration, epoch_duration):
                s_idx = int(t * sfreq)
                e_idx = s_idx + int(epoch_duration * sfreq)
                data = raw.get_data(start=s_idx, stop=e_idx)[0]
                
                has_sp = 0
                for sp_on, sp_dur in zip(sp_annots.onset, sp_annots.duration):
                    if max(t, sp_on) < min(t + epoch_duration, sp_on + sp_dur):
                        has_sp = 1; break
                
                epochs.append(data)
                labels.append(has_sp)
        return np.array(epochs), np.array(labels), sfreq
    except Exception as e:
        print(f"Skip {sub_id}: {e}")
        return None, None, None

def compute_feats(X, sfreq):
    res = []
    for sig in X:
        rms = np.sqrt(np.mean(sig**2))
        psd = np.abs(np.fft.rfft(sig))**2
        freqs = np.fft.rfftfreq(len(sig), 1/sfreq)
        s_p = np.sum(psd[(freqs >= 11) & (freqs <= 16)])
        a_p = np.sum(psd[(freqs >= 8) & (freqs < 11)])
        res.append([rms, s_p, s_p / (a_p + 1e-9)])
    return np.array(res)

file_map = get_full_file_map()
print(f"Found {len(file_map)} complete subjects.")

# Разделение
train_subs = [f"01-02-{i:04d}" for i in range(1, 16)]
test_subs = [f"01-02-{i:04d}" for i in range(16, 20)]

def collect_data(sub_list):
    all_x, all_y = [], []
    for sid in sub_list:
        if sid in file_map:
            X, y, sfreq = extract_features(sid, file_map[sid])
            if X is not None:
                all_x.append(compute_feats(X, sfreq))
                all_y.append(y)
    return np.vstack(all_x), np.concatenate(all_y)

print("\n--- Collecting Train Data (1-15) ---")
X_train, y_train = collect_data(train_subs)
print("\n--- Collecting Test Data (16-19) ---")
X_test, y_test = collect_data(test_subs)

np.savez("full_dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(f"\nDone! Train size: {len(y_train)}, Test size: {len(y_test)}")
