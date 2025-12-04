import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from extract_features import compute_features
from load_data import load_data

# ---------------------
# Load the data
# ---------------------
signals, labels = load_data()

# Convert class names → numeric values
classes = sorted(set(labels))
label_map = {c: i for i, c in enumerate(classes)}
y = np.array([label_map[l] for l in labels])

# Extract features for each ECG signal
X = np.array([compute_features(s) for s in signals])

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# ---------------------
# Train/test split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------
# Train lightweight classifier
# ---------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------
# Evaluate
# ---------------------
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nTrain accuracy:", train_acc)
print("Test accuracy:", test_acc)

# Confusion matrix
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
