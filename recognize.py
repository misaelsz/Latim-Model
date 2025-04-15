import numpy as np
import sounddevice as sd
from speech_features import extract_features, compare_features
import os

MFCC_DIR = "mfccs"
PHRASES = os.listdir(MFCC_DIR)

SAMPLING_RATE = 16000
AUDIO_DURATION = 2  # segundos


def record_audio():
    print("Gravando... Fale agora!")
    audio = sd.rec(int(AUDIO_DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Gravação finalizada.")
    return audio.flatten()


def recognize():
    signal = record_audio()
    input_features = extract_features(signal=signal)

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
    recognize()
