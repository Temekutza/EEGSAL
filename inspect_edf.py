import mne
import os

def inspect_file(filepath):
    print(f"\n=== Inspecting: {filepath} ===")
    try:
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
        print(f"Channels: {raw.ch_names}")
        print(f"Duration: {raw.times[-1]}s")
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        annotations = raw.annotations
        if len(annotations) > 0:
            print(f"Number of annotations: {len(annotations)}")
            unique_annots = set(annotations.description)
            print(f"Unique annotations: {unique_annots}")
            # Show first 5 annotations
            for i in range(min(5, len(annotations))):
                print(f"  - {annotations.onset[i]}s: {annotations.description[i]} (duration: {annotations.duration[i]}s)")
        else:
            print("No annotations found in this file.")
    except Exception as e:
        print(f"Error reading file: {e}")

# Check files for Subject 0001
psg_file = os.path.join("1-5", "01-02-0001 PSG.edf")
base_file = os.path.join("1-5", "01-02-0001 Base.edf")
spindles_file = os.path.join("annotationsss2", "01-02-0001 Spindles_E1.edf")

inspect_file(psg_file)
inspect_file(base_file)
inspect_file(spindles_file)
