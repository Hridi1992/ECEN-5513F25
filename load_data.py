import scipy.io as sio
import numpy as np

def load_data(path="../ECGData/ECGData.mat"):
    mat = sio.loadmat(path)

    E = mat["ECGData"]

    signals = E["Data"][0, 0]             # shape (162, 65536)
    labels_raw = E["Labels"][0, 0]        # shape (162, 1)

    labels = [lbl[0] for lbl in labels_raw[:, 0]]

    return signals, labels


if __name__ == "__main__":
    signals, labels = load_data()
    print("Signals shape:", signals.shape)
    print("First 10 labels:", labels[:10])
