import numpy as np
import os
from speech_features import extract_features, compare_features

MFCC_DIR = "mfccs"
AUDIO_PATH = "mic_record_test/test.wav"  # caminho do wav a comparar
PHRASES = os.listdir(MFCC_DIR)


def recognize_from_wav(wav_path):
    input_features = extract_features(file_path=wav_path)

    best_phrase = None
    best_score = float('inf')

    for phrase in PHRASES:
        mfcc_path = os.path.join(MFCC_DIR, phrase)
        for file in os.listdir(mfcc_path):
            if not file.endswith('.npy'):
                continue
            known_features = np.load(os.path.join(mfcc_path, file))
            score = compare_features(input_features, known_features)
            if score < best_score:
                best_score = score
                best_phrase = phrase

    print(f"Melhor correspondência: {best_phrase} (Distância: {best_score})")


if __name__ == "__main__":
    recognize_from_wav(AUDIO_PATH)
