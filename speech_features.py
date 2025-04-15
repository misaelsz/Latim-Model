import numpy as np
import librosa
from scipy.spatial.distance import cdist

# Extrai os MFCCs do sinal ou de um caminho para arquivo wav
def extract_features(file_path=None, signal=None, sr=16000):
    if file_path:
        signal, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return mfcc.T  # transposto pra facilitar a comparação


# Faz a comparação dos MFCCs (quanto menor a distância, mais parecido)
def compare_features(features1, features2):
    # Garante que os dois sejam 2D
    features1 = np.atleast_2d(features1)
    features2 = np.atleast_2d(features2)

    distances = cdist(features1, features2, metric='euclidean')
    min_distance = np.min(distances)
    return min_distance
