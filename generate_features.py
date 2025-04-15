import os
import numpy as np
from speech_features import extract_features

AUDIO_DIR = "audio"
MFCC_DIR = "mfccs"

os.makedirs(MFCC_DIR, exist_ok=True)

def generate_mfccs():
    for phrase_dir in os.listdir(AUDIO_DIR):
        phrase_path = os.path.join(AUDIO_DIR, phrase_dir)
        if not os.path.isdir(phrase_path):
            continue

        mfcc_phrase_path = os.path.join(MFCC_DIR, phrase_dir)
        os.makedirs(mfcc_phrase_path, exist_ok=True)

        for file in os.listdir(phrase_path):
            if not file.endswith(".wav"):
                continue

            wav_path = os.path.join(phrase_path, file)
            features = extract_features(file_path=wav_path)

            base_name = os.path.splitext(file)[0]
            npy_path = os.path.join(mfcc_phrase_path, base_name + ".npy")
            np.save(npy_path, features)
            print(f"[OK] {wav_path} → {npy_path}")

if __name__ == "__main__":
    generate_mfccs()
