import mne
import numpy as np
import pandas as pd
import os
from glob import glob

def get_file_map():
    file_map = {}
    # Ищем все PSG файлы
    for folder in ['1-5', '6-10', '11-15', '16-17']:
        psg_files = glob(os.path.join(folder, "* PSG.edf"))
        for psg in psg_files:
            sub_id = os.path.basename(psg).split()[0] # 01-02-0001
            base = psg.replace("PSG.edf", "Base.edf")
            
            # Разметка в annotationsss2
            spindles = os.path.join("annotationsss2", f"{sub_id} Spindles_E1.edf")
            kcomplex = os.path.join("annotationsss2", f"{sub_id} KComplexes_E1.edf")
            
            if os.path.exists(psg) and os.path.exists(base) and os.path.exists(spindles):
                file_map[sub_id] = {
                    'psg': psg,
                    'base': base,
                    'spindles': spindles,
                    'kcomplex': kcomplex if os.path.exists(kcomplex) else None
                }
    return file_map

def extract_features_from_subject(sub_id, paths, epoch_duration=3.0):
    print(f"Loading {sub_id}...")
    try:
        raw = mne.io.read_raw_edf(paths['psg'], preload=True, verbose=False)
        raw.pick_channels(['EEG C3-CLE'])
        sfreq = raw.info['sfreq']
        
        # Стадии сна
        base = mne.io.read_raw_edf(paths['base'], verbose=False)
        n2_mask = [a for a in base.annotations if 'stage 2' in a['description'].lower()]
        
        # Спиндлы
        spindles = mne.io.read_raw_edf(paths['spindles'], verbose=False)
        sp_annots = spindles.annotations
        
        epochs = []
        labels = []
        
        for annot in n2_mask:
            start, dur = annot['onset'], annot['duration']
            for t in np.arange(start, start + dur - epoch_duration, epoch_duration):
                start_samp = int(t * sfreq)
                end_samp = start_samp + int(epoch_duration * sfreq)
                
                data = raw.get_data(start=start_samp, stop=end_samp)[0]
                
                # Проверка на спиндл
                has_sp = 0
                for sp_on, sp_dur in zip(sp_annots.onset, sp_annots.duration):
                    if max(t, sp_on) < min(t + epoch_duration, sp_on + sp_dur):
                        has_sp = 1
                        break
                
                epochs.append(data)
                labels.append(has_sp)
        
        return np.array(epochs), np.array(labels), sfreq
    except Exception as e:
        print(f"Error processing {sub_id}: {e}")
        return None, None, None

def compute_advanced_features(X, sfreq):
    feats = []
    for sig in X:
        # RMS
        rms = np.sqrt(np.mean(sig**2))
        # Спектр
        psd = np.abs(np.fft.rfft(sig))**2
        freqs = np.fft.rfftfreq(len(sig), 1/sfreq)
        
        # Sigma band (11-16 Hz)
        sigma_idx = (freqs >= 11) & (freqs <= 16)
        sigma_p = np.sum(psd[sigma_idx])
        
        # Alpha band (8-12 Hz) - для исключения ложных срабатываний
        alpha_idx = (freqs >= 8) & (freqs < 11)
        alpha_p = np.sum(psd[alpha_idx])
        
        # Ratio
        ratio = sigma_p / (alpha_p + 1e-9)
        
        feats.append([rms, sigma_p, ratio])
    return np.array(feats)

# Собираем данные по первым 3 пациентам для теста
file_map = get_file_map()
all_features = []
all_labels = []

for sub_id in list(file_map.keys())[:3]:
    X_raw, y, sfreq = extract_features_from_subject(sub_id, file_map[sub_id])
    if X_raw is not None:
        feats = compute_advanced_features(X_raw, sfreq)
        all_features.append(feats)
        all_labels.append(y)

X_final = np.vstack(all_features)
y_final = np.concatenate(all_labels)

df = pd.DataFrame(X_final, columns=['rms', 'sigma_power', 'sigma_alpha_ratio'])
df['target'] = y_final
df.to_csv("multi_subject_data.csv", index=False)
print(f"Total epochs: {len(df)}. Spindles: {sum(y_final)}")
