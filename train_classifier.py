import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from extract_features import compute_features
from load_data import load_data

signals, labels = load_data()

# Convert class labels → numeric
classes = sorted(set(labels))
label_map = {c: i for i, c in enumerate(classes)}
y = np.array([label_map[l] for l in labels])

# Feature extraction
X = np.array([compute_features(s) for s in signals])

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
