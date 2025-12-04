import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def butter_bandpass(lowcut=0.5, highcut=40, fs=360, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass(signal, low=0.5, high=40, fs=360):
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, signal)

def plot_signal(signal, title="ECG Signal"):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
