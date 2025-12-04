import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "spectrograms"
IMG_SIZE = (64, 64)   # smaller → faster training

# ---------------------------------------------------------------------
# Load all images from folder structure:
# spectrograms/
#      ├── classA/
#      ├── classB/
#      └── classC/
# ---------------------------------------------------------------------

def load_images():
    X = []
    y = []
    class_names = sorted(os.listdir(DATA_DIR))

    for label, class_name in enumerate(class_names):
        folder = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)

                X.append(np.array(img).flatten())  
                y.append(label)

            except:
                pass  # skip unreadable images

    return np.array(X), np.array(y), class_names


print("Loading images...")
X, y, class_names = load_images()

print("Dataset shape:", X.shape)
print("Classes:", class_names)

# ---------------------------------------------------------------------
# Train-test split
# ---------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# Train RandomForest classifier
# ---------------------------------------------------------------------

print("Training model...")
model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Save model
import joblib
joblib.dump((model, class_names), "rf_ecg_image_model.pkl")

print("Model saved as rf_ecg_image_model.pkl")
