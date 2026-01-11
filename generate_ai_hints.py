import torch
import mne
import numpy as np
import os
import sys
import pandas as pd
from train_max_active import SpindleNet
from skorch import NeuralNetClassifier

def get_trained_learner():
    data = np.load("full_dataset.npz")
    X = data['X_train'].astype(np.float32)
    y = data['y_train'].astype(np.int64)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std
    
    net = NeuralNetClassifier(SpindleNet, max_epochs=5, lr=0.05, batch_size=128, device='cpu', verbose=0)
    net.fit(X[:10000], y[:10000]) 
    return net, X_mean, X_std

def generate_hints_for_edfbrowser(psg_path, base_path, output_txt):
    net, mean, std = get_trained_learner()
    
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    target_ch = [ch for ch in raw.ch_names if 'C3' in ch][0]
    raw.pick_channels([target_ch])
    sfreq = raw.info['sfreq']
    
    base = mne.io.read_raw_edf(base_path, verbose=False)
    n2_mask = [a for a in base.annotations if 'stage 2' in a['description'].lower()]
    
    hints = []
    epoch_dur = 3.0
    
    print(f"Analyzing {os.path.basename(psg_path)}...")
    for annot in n2_mask:
        start, dur = annot['onset'], annot['duration']
        for t in np.arange(start, start + dur - epoch_dur, epoch_dur):
            s_idx = int(t * sfreq)
            e_idx = s_idx + int(epoch_dur * sfreq)
            sig = raw.get_data(start=s_idx, stop=e_idx)[0]
            
            rms = np.sqrt(np.mean(sig**2))
            psd = np.abs(np.fft.rfft(sig))**2
            f = np.fft.rfftfreq(len(sig), 1/sfreq)
            s_p = np.sum(psd[(f >= 11) & (f <= 16)])
            a_p = np.sum(psd[(f >= 8) & (f < 11)])
            feat = (np.array([[rms, s_p, s_p/(a_p+1e-9)]]) - mean) / std
            
            probs = net.predict_proba(feat.astype(np.float32))[0]
            p_yes = probs[1]
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
            
            # Формируем метки
            label = ""
            if p_yes > 0.8: label = "AI_Spindle_CONFIRMED"
            elif entropy > 0.85: label = "AI_ANOMALY_CHECK_EPI"
            elif p_yes > 0.5: label = "AI_Spindle_PROBABLE"
            
            if label:
                # ПОРЯДОК СТОЛБЦОВ: 1:Onset, 2:Description, 3:Duration
                # РАЗДЕЛИТЕЛЬ: Табуляция (\t)
                hints.append([f"{t:.3f}", label, f"{epoch_dur:.1f}"])

    # Сохраняем без заголовка, разделитель - табуляция
    with open(output_txt, 'w', encoding='utf-8') as f:
        for line in hints:
            f.write("\t".join(line) + "\n")
            
    print(f"Done! File created: {output_txt}")

# Запуск для 0018
psg_0018 = r"C:\Users\Vlad\Desktop\mass_1\ss2\Непонятно\01-02-0018 PSG.edf"
base_0018 = r"C:\Users\Vlad\Desktop\mass_1\ss2\Непонятно\01-02-0018 Base.edf"
generate_hints_for_edfbrowser(psg_0018, base_0018, "AI_HINTS_0018.txt")
