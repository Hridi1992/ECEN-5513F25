import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from load_data import load_data

OUTPUT_DIR = "scalograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_scalogram(signal, label, index):
    scales = np.arange(1, 128)
    coef, freqs = pywt.cwt(signal, scales, 'morl')

    plt.figure(figsize=(3,3))
    plt.imshow(np.abs(coef), aspect='auto', cmap='jet')
    plt.axis('off')

    filename = f"{OUTPUT_DIR}/{label}_{index}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

signals, labels = load_data()

for i, (sig, label) in enumerate(zip(signals, labels)):
    create_scalogram(sig, label, i)

print("Scalograms saved!")
