import mne
import numpy as np
import pandas as pd
import os

def load_data(subject_id, psg_path, base_path, spindle_path):
    # Load raw PSG
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    # Target C3 channel
    raw.pick_channels(['EEG C3-CLE'])
    
    # Load hypnogram from Base.edf
    raw_base = mne.io.read_raw_edf(base_path, preload=False, verbose=False)
    hypno_annots = raw_base.annotations
    
    # Load spindles from Spindles_E1.edf
    raw_spindles = mne.io.read_raw_edf(spindle_path, preload=False, verbose=False)
    spindle_annots = raw_spindles.annotations
    
    return raw, hypno_annots, spindle_annots

def extract_n2_epochs(raw, hypno_annots, spindle_annots, epoch_duration=3.0):
    sfreq = raw.info['sfreq']
    n_samples_epoch = int(epoch_duration * sfreq)
    
    # Filter N2 segments
    n2_onsets = [a['onset'] for a in hypno_annots if 'stage 2' in a['description'].lower()]
    n2_durations = [a['duration'] for a in hypno_annots if 'stage 2' in a['description'].lower()]
    
    X = []
    y = []
    
    # For each N2 segment, slice it into epochs
    for onset, duration in zip(n2_onsets, n2_durations):
        start_sample = int(onset * sfreq)
        end_sample = int((onset + duration) * sfreq)
        
        for s in range(start_sample, end_sample - n_samples_epoch, n_samples_epoch):
            epoch_data = raw.get_data(start=s, stop=s + n_samples_epoch)[0]
            
            # Check if this epoch contains a spindle
            epoch_start_time = s / sfreq
            epoch_end_time = (s + n_samples_epoch) / sfreq
            
            # A spindle is in this epoch if it overlaps
            has_spindle = 0
            for sp_onset, sp_dur in zip(spindle_annots.onset, spindle_annots.duration):
                sp_end = sp_onset + sp_dur
                # Overlap check
                if max(epoch_start_time, sp_onset) < min(epoch_end_time, sp_end):
                    has_spindle = 1
                    break
            
            X.append(epoch_data)
            y.append(has_spindle)
            
    return np.array(X), np.array(y)

def compute_features(X, sfreq):
    features = []
    for signal in X:
        # 1. RMS amplitude
        rms = np.sqrt(np.mean(signal**2))
        
        # 2. Spectral power in sigma band (11-16 Hz)
        # Using simple FFT for speed in this demo
        freqs = np.fft.rfftfreq(len(signal), 1/sfreq)
        psd = np.abs(np.fft.rfft(signal))**2
        
        sigma_mask = (freqs >= 11) & (freqs <= 16)
        sigma_power = np.sum(psd[sigma_mask])
        total_power = np.sum(psd)
        rel_sigma_power = sigma_power / total_power if total_power > 0 else 0
        
        # 3. Peak frequency in sigma band
        if np.any(sigma_mask):
            peak_sigma_freq = freqs[sigma_mask][np.argmax(psd[sigma_mask])]
        else:
            peak_sigma_freq = 0
            
        features.append([rms, rel_sigma_power, peak_sigma_freq])
        
    return np.array(features)

# Main processing for subject 0001
sub = "0001"
psg = os.path.join("1-5", f"01-02-{sub} PSG.edf")
base = os.path.join("1-5", f"01-02-{sub} Base.edf")
spindles = os.path.join("annotationsss2", f"01-02-{sub} Spindles_E1.edf")

print(f"Processing Subject {sub}...")
raw, hypno, sp_annots = load_data(sub, psg, base, spindles)
X_raw, y = extract_n2_epochs(raw, hypno, sp_annots)
features = compute_features(X_raw, raw.info['sfreq'])

df = pd.DataFrame(features, columns=['rms', 'rel_sigma_power', 'peak_sigma_freq'])
df['target'] = y

print(f"Extracted {len(df)} epochs.")
print(f"Spindle distribution: {df['target'].value_counts(normalize=True).to_dict()}")

# Save for next step
df.to_csv("processed_features_0001.csv", index=False)
print("Saved to processed_features_0001.csv")
