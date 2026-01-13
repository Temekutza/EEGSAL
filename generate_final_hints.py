import torch
import mne
import numpy as np
import os
from scipy.stats import kurtosis
from train_v2 import SpindleNetV2

# Загрузка модели и параметров
params = np.load("norm_params.npz")
mean, std = params['mean'], params['std']

model = SpindleNetV2()
model.load_state_dict(torch.load('spindlenet_v2.pt'))
model.eval()

def process_subject(sub_id, psg_path, base_path):
    print(f"Generating hints for {sub_id}...")
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    target_ch = [ch for ch in raw.ch_names if 'C3' in ch][0]
    raw.pick_channels([target_ch])
    sfreq = raw.info['sfreq']
    
    # Считаем глобальный RMS для нормализации
    global_rms = np.sqrt(np.mean(raw.get_data()[0]**2))
    
    base = mne.io.read_raw_edf(base_path, verbose=False)
    n2_mask = [a for a in base.annotations if 'stage 2' in a['description'].lower()]
    
    hints = []
    epoch_dur = 3.0
    
    for annot in n2_mask:
        start, dur = annot['onset'], annot['duration']
        for t in np.arange(start, start + dur - epoch_dur, epoch_dur):
            s_idx = int(t * sfreq)
            e_idx = s_idx + int(epoch_dur * sfreq)
            data = raw.get_data(start=s_idx, stop=e_idx)[0]
            
            # Извлечение признаков V2
            rms = np.sqrt(np.mean(data**2)) / (global_rms + 1e-9)
            kurt = kurtosis(data)
            psd = np.abs(np.fft.rfft(data))**2
            freqs = np.fft.rfftfreq(len(data), 1/sfreq)
            s_p = np.sum(psd[(freqs >= 11) & (freqs <= 16)])
            a_p = np.sum(psd[(freqs >= 8) & (freqs < 11)])
            
            feat = (np.array([[rms, kurt, s_p, s_p/(a_p+1e-9)]]) - mean) / (std + 1e-9)
            feat = torch.FloatTensor(feat.astype(np.float32))
            
            with torch.no_grad():
                probs = model(feat)[0].numpy()
            
            p_yes = probs[1]
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
            
            label = ""
            if p_yes > 0.8: label = "AI_Spindle_CONFIRMED"
            elif entropy > 0.85: label = "AI_ANOMALY_CHECK_EPI"
            elif p_yes > 0.5: label = "AI_Spindle_PROBABLE"
            
            if label:
                hints.append([f"{t:.3f}", label, f"{epoch_dur:.1f}"])

    output_file = f"AI_HINTS_{sub_id}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in hints:
            f.write("\t".join(line) + "\n")
    print(f"Saved: {output_file}")

# Файлы пациентов
tasks = [
    ("0016", "16-17/01-02-0016 PSG.edf", "16-17/01-02-0016 Base.edf"),
    ("0017", "16-17/01-02-0017 PSG.edf", "16-17/01-02-0017 Base.edf"),
    ("0018", "Непонятно/01-02-0018 PSG.edf", "Непонятно/01-02-0018 Base.edf"),
    ("0019", "Непонятно/01-02-0019 PSG.edf", "Непонятно/01-02-0019 Base.edf")
]

for sid, psg, base in tasks:
    if os.path.exists(psg) and os.path.exists(base):
        process_subject(sid, psg, base)
    else:
        print(f"Files for {sid} not found. Check paths.")
