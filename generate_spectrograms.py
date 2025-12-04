import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from load_data import load_data

OUTPUT_DIR = "spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_spectrogram(signal, label, index, fs=360):
    # Compute STFT
    f, t, Zxx = stft(signal, fs=fs, nperseg=256)

    # Magnitude of STFT
    S = np.abs(Zxx)

    # Plot spectrogram
    plt.figure(figsize=(3,3))
    plt.pcolormesh(t, f, S, shading='gouraud', cmap='jet')
    plt.axis('off')

    filename = f"{OUTPUT_DIR}/{label}_{index}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

signals, labels = load_data()

for i, (sig, label) in enumerate(zip(signals, labels)):
    create_spectrogram(sig, label, i)

print("Spectrograms saved!")
