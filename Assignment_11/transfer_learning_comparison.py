import os
import zipfile
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16, MobileNet, ResNet50
from tensorflow.keras.applications import vgg16, mobilenet, resnet50

np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the exact same 600-image Tiny ImageNet subset (3 classes x 200 images)
DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_PATH = "tiny-imagenet-200.zip"
EXTRACT_DIR = "tiny-imagenet-200"

if not os.path.exists(EXTRACT_DIR):
    if not os.path.exists(ZIP_PATH):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

train_dir = os.path.join(EXTRACT_DIR, "train")
selected_classes = sorted(os.listdir(train_dir))[:3]

X, y = [], []
for label_idx, class_name in enumerate(selected_classes):
    img_folder = os.path.join(train_dir, class_name, "images")
    for fname in sorted(os.listdir(img_folder))[:200]:
        img = cv2.imread(os.path.join(img_folder, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(label_idx)

X = np.array(X, dtype=np.float32)  # Keep in [0, 255] for model-specific preprocessing
y = np.array(y, dtype=np.int32)

# Same 70% / 15% / 15% split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 2. Custom CNN Baseline
def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1.0 / 255.0),
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
        layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# 3. Transfer Learning Model Builder (VGG16, MobileNet, ResNet50)
def build_transfer_model(base_fn, preprocess_fn):
    inputs = layers.Input(shape=(64, 64, 3))
    x = layers.Lambda(preprocess_fn)(inputs)
    base_model = base_fn(weights="imagenet", include_top=False, input_tensor=x)
    base_model.trainable = False  # Freeze pretrained backbone

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

models_dict = {
    "Custom CNN": build_custom_cnn(),
    "VGG16": build_transfer_model(VGG16, vgg16.preprocess_input),
    "MobileNet": build_transfer_model(MobileNet, mobilenet.preprocess_input),
    "ResNet50": build_transfer_model(ResNet50, resnet50.preprocess_input),
}

results = {}

for name, model in models_dict.items():
    print(f"\n--- Training {name} ---")
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results[name] = (acc, prec, rec, f1)
    print(f"{name:12s} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# 4. Print Summary Comparison Table
print("\n" + "=" * 68)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("=" * 68)
for name, (acc, prec, rec, f1) in results.items():
    print(f"{name:<15} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
print("=" * 68)

# 5. Plot and Save Grouped Bar Chart Comparison
model_names = list(results.keys())
metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
scores_matrix = np.array([results[m] for m in model_names])

x_pos = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))
for idx, metric_name in enumerate(metrics_labels):
    ax.bar(x_pos + idx * width, scores_matrix[:, idx], width, label=metric_name)

ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Test Performance Comparison: Custom CNN vs Transfer Learning Models")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("transfer_learning_comparison.png", dpi=300)
plt.show()
