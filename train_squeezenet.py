# train_squeezenet.py
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os

# === Config ===
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "spectrograms"   # folder where generate_spectrograms.py wrote images
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 1e-4
INPUT_SIZE = 224
TEST_SIZE = 0.30
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# === Dataset and split ===
dataset = datasets.ImageFolder(DATA_DIR)
num_samples = len(dataset)
if num_samples == 0:
    raise RuntimeError(f"No images found in {DATA_DIR}. Run generate_spectrograms.py first.")

# create stratified split
labels = np.array([s[1] for s in dataset.samples])
train_idx, val_idx = train_test_split(np.arange(num_samples), test_size=TEST_SIZE,
                                     stratify=labels, random_state=RANDOM_STATE)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

# attach transforms
train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)
print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

# === Model ===
model = models.squeezenet1_1(pretrained=True)
# Replace final conv layer to match num_classes
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
model.num_classes = num_classes
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training loop ===
best_val_acc = 0.0
best_path = MODEL_DIR / "squeezenet_ecg_best.pth"

for epoch in range(1, NUM_EPOCHS+1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), outputs.size(1))  # (N, C)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == targets).sum().item()
        total += inputs.size(0)

    train_loss = running_loss / total
    train_acc = running_corrects / total

    # Validation
    model.eval()
    val_total = 0
    val_correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), outputs.size(1))
            preds = outputs.argmax(dim=1)
            val_correct += (preds == targets).sum().item()
            val_total += inputs.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    val_acc = val_correct / val_total
    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  time={elapsed:.1f}s")

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, best_path)
        print(f"Saved best model -> {best_path}")

print("Training finished. Best val acc:", best_val_acc)

# final evaluation (print classification report/confusion)
from sklearn.metrics import classification_report, confusion_matrix
print("\nFinal evaluation on validation set:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
