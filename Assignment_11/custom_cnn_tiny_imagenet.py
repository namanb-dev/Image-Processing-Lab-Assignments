import os
import zipfile
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

np.random.seed(42)
tf.random.set_seed(42)

# 1. Download and extract Tiny ImageNet if not already present
DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_PATH = "tiny-imagenet-200.zip"
EXTRACT_DIR = "tiny-imagenet-200"

if not os.path.exists(EXTRACT_DIR):
    if not os.path.exists(ZIP_PATH):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

# 2. Select 3 classes with 200 images each (Total = 600 images of size 64x64x3)
train_dir = os.path.join(EXTRACT_DIR, "train")
selected_classes = sorted(os.listdir(train_dir))[:3]
print("Selected Classes:", selected_classes)

X, y = [], []
for label_idx, class_name in enumerate(selected_classes):
    img_folder = os.path.join(train_dir, class_name, "images")
    img_files = sorted(os.listdir(img_folder))[:200]
    for fname in img_files:
        img_path = os.path.join(img_folder, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(label_idx)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int32)
print(f"Dataset Loaded: X shape = {X.shape}, y shape = {y.shape}")

# 3. 70% Train / 15% Validation / 15% Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"Split Sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 4. Build Custom CNN using Task 1 components (Conv, Padding='same', ReLU, MaxPool, GAP, BatchNorm, Dropout, Dense)
def build_custom_cnn(num_classes=3):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

custom_cnn = build_custom_cnn(num_classes=3)
custom_cnn.summary()

# 5. Train with Early Stopping
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = custom_cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on Test Set
test_loss, test_acc = custom_cnn.evaluate(X_test, y_test, verbose=0)
print(f"\nCustom CNN Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

# Save trained model
custom_cnn.save("custom_cnn_model.keras")

# 6. Plot Training vs Validation Loss & Accuracy Curves (Underfitting/Overfitting Analysis)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(history.history["loss"], label="Training Loss", marker="o", markersize=3)
axes[0].plot(history.history["val_loss"], label="Validation Loss", marker="s", markersize=3)
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(history.history["accuracy"], label="Training Accuracy", marker="o", markersize=3)
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s", markersize=3)
axes[1].set_title("Training vs Validation Accuracy")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("custom_cnn_loss_curves.png", dpi=300)
plt.show()
