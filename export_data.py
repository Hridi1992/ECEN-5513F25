import numpy as np
import pandas as pd
from extract_features import compute_features
from load_data import load_data

signals, labels = load_data()
features = np.array([compute_features(s) for s in signals])

df = pd.DataFrame(features, columns=["HeartRate", "RMS", "Energy"])
df["Label"] = labels

df.to_csv("ecg_features.csv", index=False)
print("Saved ecg_features.csv")

np.save("signals.npy", signals)
np.save("labels.npy", labels)
np.save("features.npy", features)
print("Saved signals.npy, labels.npy, features.npy")