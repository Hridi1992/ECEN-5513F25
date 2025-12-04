import numpy as np
from scipy.signal import find_peaks
from utils import normalize
from load_data import load_data

def compute_features(ecg, fs=360):
    ecg = normalize(ecg)

    # R-peaks
    peaks, _ = find_peaks(ecg, distance=int(0.2 * fs))

    heart_rate = len(peaks) * (60 / (len(ecg) / fs))
    rms = np.sqrt(np.mean(ecg**2))
    energy = np.sum(ecg**2)

    return [heart_rate, rms, energy]

if __name__ == "__main__":
    signals, labels = load_data()
    feats = np.array([compute_features(s) for s in signals])
    print("Features shape:", feats.shape)
