from load_data import load_data
from utils import plot_signal, apply_bandpass

signals, labels = load_data()

# pick first signal
signal = signals[0]
label = labels[0]

plot_signal(signal, title=f"Raw ECG - {label}")

filtered = apply_bandpass(signal)
plot_signal(filtered, title=f"Filtered ECG - {label}")
