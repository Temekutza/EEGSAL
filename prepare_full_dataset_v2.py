import mne
import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis

def get_full_file_map():
    file_map = {}
    folders = ['1-5', '6-10', '11-15', '16-17', 'Непонятно']
    for i in range(1, 20):
        sub_id = f"01-02-{i:04d}"
        entry = {'psg': None, 'base': None, 'spindles': None}
        for folder in folders:
            psg = os.path.join(folder, f"{sub_id} PSG.edf")
            base = os.path.join(folder, f"{sub_id} Base.edf")
            if os.path.exists(psg): entry['psg'] = psg
            if os.path.exists(base): entry['base'] = base
        spindles = os.path.join("annotationsss2", f"{sub_id} Spindles_E1.edf")
        if os.path.exists(spindles): entry['spindles'] = spindles
        if entry['psg'] and entry['base'] and entry['spindles']:
            file_map[sub_id] = entry
    return file_map

def extract_features_v2(sub_id, paths, epoch_duration=3.0):
    print(f"Processing {sub_id} with Advanced Features...")
    try:
        raw = mne.io.read_raw_edf(paths['psg'], preload=True, verbose=False)
        target_ch = [ch for ch in raw.ch_names if 'C3' in ch][0]
        raw.pick_channels([target_ch])
        sfreq = raw.info['sfreq']
        
        # Получаем данные всей N2 стадии для нормализации амплитуды
        base = mne.io.read_raw_edf(paths['base'], verbose=False)
        n2_annots = [a for a in base.annotations if 'stage 2' in a['description'].lower()]
        
        # Разметка спиндлов
        spindles = mne.io.read_raw_edf(paths['spindles'], verbose=False)
        sp_annots = spindles.annotations
        
        # Считаем средний RMS всей записи для этого пациента (нормализация)
        full_data = raw.get_data()[0]
        global_rms = np.sqrt(np.mean(full_data**2))
        
        epochs, labels = [], []
        for annot in n2_annots:
            start, dur = annot['onset'], annot['duration']
            for t in np.arange(start, start + dur - epoch_duration, epoch_duration):
                s_idx = int(t * sfreq)
                e_idx = s_idx + int(epoch_duration * sfreq)
                data = raw.get_data(start=s_idx, stop=e_idx)[0]
                
                # 1. Relative RMS (Амплитуда относительно среднего)
                rms = np.sqrt(np.mean(data**2)) / (global_rms + 1e-9)
                
                # 2. Kurtosis (Эксцесс - поиск острых спайков)
                kurt = kurtosis(data)
                
                # 3. Spectral features
                psd = np.abs(np.fft.rfft(data))**2
                freqs = np.fft.rfftfreq(len(data), 1/sfreq)
                s_p = np.sum(psd[(freqs >= 11) & (freqs <= 16)])
                a_p = np.sum(psd[(freqs >= 8) & (freqs < 11)])
                ratio = s_p / (a_p + 1e-9)
                
                # Проверка на спиндл
                has_sp = 0
                for sp_on, sp_dur in zip(sp_annots.onset, sp_annots.duration):
                    if max(t, sp_on) < min(t + epoch_duration, sp_on + sp_dur):
                        has_sp = 1; break
                
                epochs.append([rms, kurt, s_p, ratio])
                labels.append(has_sp)
        return np.array(epochs), np.array(labels)
    except Exception as e:
        print(f"Skip {sub_id}: {e}"); return None, None

file_map = get_full_file_map()
train_subs = [f"01-02-{i:04d}" for i in range(1, 16)]
test_subs = [f"01-02-{i:04d}" for i in range(16, 20)]

def collect(sub_list):
    x, y = [], []
    for sid in sub_list:
        feat, lbl = extract_features_v2(sid, file_map[sid])
        if feat is not None: x.append(feat); y.append(lbl)
    return np.vstack(x), np.concatenate(y)

X_tr, y_tr = collect(train_subs)
X_te, y_te = collect(test_subs)
np.savez("full_dataset_v2.npz", X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
print("Dataset V2 (Advanced) created!")
