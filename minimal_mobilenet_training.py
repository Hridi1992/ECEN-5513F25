import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ---- SETTINGS ----
DATA_DIR = "data"       # put your ECG images here (train/ and val/ subfolders)
IMG_SIZE = 224
BATCH = 16
EPOCHS = 5

# ---- DATA ----
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    DATA_DIR + "/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR + "/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

# ---- MODEL ----
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save("mobilenet_ecg.h5")
print("Training complete.")
