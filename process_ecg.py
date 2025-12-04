import numpy as np
from load_data import load_data
from utils import apply_bandpass

signals, labels = load_data()

filtered_signals = np.array([apply_bandpass(s) for s in signals])

print("Filtered signals shape:", filtered_signals.shape)
np.save("filtered_signals.npy", filtered_signals)
print("Saved filtered_signals.npy")

