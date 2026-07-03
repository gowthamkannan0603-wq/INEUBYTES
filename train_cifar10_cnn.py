"""
Task 1 - Part A: Traditional CNN (AlexNet-style) on CIFAR-10
Author: Gowtham K | Reg No: INBT020009 | Course: AIINB10626

Goal: Build a clean AlexNet-inspired CNN adapted for 32x32 CIFAR-10 images
      and hit at least 70% test accuracy as the baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 32
NUM_CLASSES = 10
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 0.001

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── 1. Load & preprocess CIFAR-10 ─────────────────────────────────────────────
print("\n[1] Loading CIFAR-10 dataset...")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()

# Normalise pixel values to [0, 1]
x_train = x_train_raw.astype("float32") / 255.0
x_test  = x_test_raw.astype("float32")  / 255.0

# Flatten label arrays
y_train = y_train_raw.flatten()
y_test  = y_test_raw.flatten()

# Fixed 80/20 train-validation split (same split reused in Part B)
val_cutoff = int(len(x_train) * 0.8)
x_val, y_val   = x_train[val_cutoff:], y_train[val_cutoff:]
x_train, y_train = x_train[:val_cutoff], y_train[:val_cutoff]

print(f"   Train: {x_train.shape}  |  Val: {x_val.shape}  |  Test: {x_test.shape}")

# ── 2. Build AlexNet-style CNN (adapted for 32×32) ───────────────────────────
print("\n[2] Building AlexNet-style model...")

def build_alexnet_small(input_shape=(32, 32, 3), num_classes=10):
    """
    Classic AlexNet redesigned for 32×32 inputs.
    Original AlexNet expects 227×227 and uses 96/256/384 filters —
    we scale the filter sizes down and use 52×52-friendly kernel choices
    adapted further for CIFAR's small spatial dimensions.
    """
    inputs = keras.Input(shape=input_shape, name="cifar_input")

    # Block 1 — Large receptive field opener (like AlexNet conv1)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same",
                      activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)

    # Block 2 — Feature enrichment (like AlexNet conv2)
    x = layers.Conv2D(128, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)

    # Block 3/4/5 — Deeper representations (like AlexNet conv3-5)
    x = layers.Conv2D(256, kernel_size=3, padding="same",
                      activation="relu", name="conv3")(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same",
                      activation="relu", name="conv4")(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same",
                      activation="relu", name="conv5")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)

    # Classifier head (like AlexNet FC6/7/8)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc6")(x)
    x = layers.Dense(256, activation="relu", name="fc7")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="fc8_out")(x)

    model = keras.Model(inputs, outputs, name="AlexNet_CIFAR10")
    return model

model_a = build_alexnet_small()
model_a.summary()

total_params = model_a.count_params()
print(f"\n   Total parameters: {total_params:,}")

# ── 3. Compile ────────────────────────────────────────────────────────────────
model_a.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ── 4. Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint("partA_best_model.keras",
                                    save_best_only=True, verbose=0)
]

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\n[3] Training AlexNet-style CNN (no augmentation in Part A)...")
start_time = time.time()

history = model_a.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

train_duration = time.time() - start_time
print(f"\n   Training completed in {train_duration/60:.1f} minutes")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n[4] Evaluating on test set...")
test_loss, test_acc = model_a.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss    : {test_loss:.4f}")
print(f"   Test Accuracy: {test_acc*100:.2f}%")

# Detailed classification report
y_pred_probs = model_a.predict(x_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ── 7. Save metrics for Part B comparison ────────────────────────────────────
os.makedirs("results", exist_ok=True)
np.save("results/partA_history.npy", history.history)
np.save("results/partA_predictions.npy", y_pred)

partA_stats = {
    "test_accuracy"   : float(test_acc),
    "test_loss"       : float(test_loss),
    "total_params"    : int(total_params),
    "training_time_s" : float(train_duration),
    "epochs_run"      : int(len(history.history["loss"]))
}
np.save("results/partA_stats.npy", partA_stats)
print("\n   Stats saved to results/partA_stats.npy")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Part A – AlexNet-style CNN on CIFAR-10", fontsize=14, fontweight="bold")

# Training curves — accuracy
ax = axes[0]
ax.plot(history.history["accuracy"],     label="Train Acc",  color="#2196F3")
ax.plot(history.history["val_accuracy"], label="Val Acc",    color="#FF9800")
ax.axhline(0.70, linestyle="--", color="red", alpha=0.6, label="70% threshold")
ax.set_title("Accuracy over Epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(alpha=0.3)

# Training curves — loss
ax = axes[1]
ax.plot(history.history["loss"],     label="Train Loss", color="#2196F3")
ax.plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
ax.set_title("Loss over Epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# Confusion matrix
ax = axes[2]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=False, annot_kws={"size": 7})
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)

plt.tight_layout()
plt.savefig("results/partA_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n   Plot saved to results/partA_results.png")

# ── 9. Summary printout ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PART A SUMMARY")
print("="*55)
print(f"  Model        : AlexNet-style CNN (adapted for 32×32)")
print(f"  Parameters   : {total_params:,}")
print(f"  Epochs run   : {len(history.history['loss'])}")
print(f"  Training time: {train_duration/60:.1f} min")
print(f"  Train Acc    : {history.history['accuracy'][-1]*100:.2f}%")
print(f"  Val Acc      : {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"  Test Acc     : {test_acc*100:.2f}%")
print(f"  Target met   : {'✓ YES' if test_acc >= 0.70 else '✗ Not yet'} (≥70%)")
print("="*55)
