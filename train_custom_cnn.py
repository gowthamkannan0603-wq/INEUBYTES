"""
Task 1 - Part B: Customized CNN — Beating the AlexNet Baseline
Author: Gowtham K | Reg No: INBT020009 | Course: AIINB10626

Goal: Design a custom CNN that beats Part A by ≥3 percentage points
      using architectural improvements and training tricks.

Changes from Part A (and why each one helps):
  1. Batch Normalisation after every conv block → stabilises gradients,
     allows higher LR, speeds convergence significantly.
  2. Dropout (0.4 before Dense) → regularises the classifier head,
     prevents memorising training samples.
  3. Data Augmentation (flip + slight shift + rotation) → artificially
     expands the training set; forces the model to learn rotation/flip-invariant
     features rather than pixel positions.
  4. Residual-style skip connection in the middle block → helps gradients
     flow back through deeper layers without vanishing.
  5. Slightly tuned LR + cosine decay → smoother convergence vs fixed LR.
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

# ── Reproducibility (SAME seed as Part A) ────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 32
NUM_CLASSES = 10
BATCH_SIZE  = 64
EPOCHS      = 40          # Part B allowed to run longer, but same budget noted
LR          = 0.001

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── 1. Load CIFAR-10 (SAME fixed split as Part A) ────────────────────────────
print("\n[1] Loading CIFAR-10 (same split as Part A)...")
(x_raw, y_raw), (x_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()

x_all  = x_raw.astype("float32") / 255.0
x_test = x_test_raw.astype("float32") / 255.0
y_all  = y_raw.flatten()
y_test = y_test_raw.flatten()

val_cutoff = int(len(x_all) * 0.8)
x_val,   y_val   = x_all[val_cutoff:],  y_all[val_cutoff:]
x_train, y_train = x_all[:val_cutoff],  y_all[:val_cutoff]

print(f"   Train: {x_train.shape}  |  Val: {x_val.shape}  |  Test: {x_test.shape}")

# ── 2. Data Augmentation pipeline ────────────────────────────────────────────
# Part A had NO augmentation — this is one of our key improvements
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomRotation(factor=0.1),
], name="augmentation")

# Build augmented tf.data pipelines
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=10000, seed=SEED)
    .batch(BATCH_SIZE)
    .map(lambda img, lbl: (augment(img, training=True), lbl),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ── 3. Custom CNN with residual block ─────────────────────────────────────────
print("\n[2] Building Custom CNN with BatchNorm + Dropout + Skip connection...")

def residual_block(x, filters, name_prefix):
    """Mini residual unit: two 3×3 convs with BN + ReLU, plus a 1×1 projection shortcut."""
    shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                              name=f"{name_prefix}_proj")(x)
    shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    return x


def build_custom_cnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape, name="input_img")

    # ── Entry block ───────────────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="entry_conv")(inputs)
    x = layers.BatchNormalization(name="entry_bn")(x)
    x = layers.Activation("relu", name="entry_relu")(x)
    x = layers.MaxPooling2D(2, name="entry_pool")(x)      # 16×16

    # ── Stage 1: residual block ───────────────────────────────────────────
    x = residual_block(x, filters=128, name_prefix="stage1")
    x = layers.MaxPooling2D(2, name="stage1_pool")(x)     # 8×8

    # ── Stage 2: residual block ───────────────────────────────────────────
    x = residual_block(x, filters=256, name_prefix="stage2")
    x = layers.MaxPooling2D(2, name="stage2_pool")(x)     # 4×4

    # ── Stage 3: plain conv block (deeper) ───────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="stage3_conv")(x)
    x = layers.BatchNormalization(name="stage3_bn")(x)
    x = layers.Activation("relu", name="stage3_relu")(x)

    # ── Classifier head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)   # replaces Flatten + huge Dense
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="Custom_CNN_v1")
    return model

model_b = build_custom_cnn()
model_b.summary()

total_params_b = model_b.count_params()
print(f"\n   Total parameters: {total_params_b:,}")

# ── 4. Compile with cosine-decay LR ──────────────────────────────────────────
steps_per_epoch = len(x_train) // BATCH_SIZE
cosine_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR,
    decay_steps=EPOCHS * steps_per_epoch,
    alpha=1e-5
)

model_b.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ── 5. Callbacks ──────────────────────────────────────────────────────────────
callbacks_b = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint("partB_best_model.keras",
                                    save_best_only=True, verbose=0)
]

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("\n[3] Training Custom CNN (with augmentation + BN + Dropout + skip)...")
start_b = time.time()

history_b = model_b.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_b,
    verbose=1
)

train_duration_b = time.time() - start_b
print(f"\n   Training done in {train_duration_b/60:.1f} minutes")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
print("\n[4] Evaluating on test set...")
test_loss_b, test_acc_b = model_b.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss    : {test_loss_b:.4f}")
print(f"   Test Accuracy: {test_acc_b*100:.2f}%")

y_pred_b = np.argmax(model_b.predict(x_test, verbose=0), axis=1)

print("\n   Classification Report:")
print(classification_report(y_test, y_pred_b, target_names=CLASS_NAMES))

# ── 8. Load Part A stats for comparison ──────────────────────────────────────
os.makedirs("results", exist_ok=True)
try:
    partA_stats = np.load("../partA/results/partA_stats.npy", allow_pickle=True).item()
    acc_a = partA_stats["test_accuracy"]
    params_a = partA_stats["total_params"]
    time_a   = partA_stats["training_time_s"]
except FileNotFoundError:
    print("   (Part A stats not found — run partA first for full comparison)")
    acc_a, params_a, time_a = None, None, None

# ── 9. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Part B – Custom CNN vs AlexNet Baseline (CIFAR-10)", fontsize=14, fontweight="bold")

# Accuracy curves
ax = axes[0, 0]
ax.plot(history_b.history["accuracy"],     label="Train Acc",  color="#4CAF50")
ax.plot(history_b.history["val_accuracy"], label="Val Acc",    color="#FF5722")
ax.axhline(0.70, linestyle="--", color="gray", alpha=0.6, label="70% baseline target")
ax.set_title("Custom CNN – Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(alpha=0.3)

# Loss curves
ax = axes[0, 1]
ax.plot(history_b.history["loss"],     label="Train Loss", color="#4CAF50")
ax.plot(history_b.history["val_loss"], label="Val Loss",   color="#FF5722")
ax.set_title("Custom CNN – Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# Confusion matrix
ax = axes[1, 0]
cm_b = confusion_matrix(y_test, y_pred_b)
sns.heatmap(cm_b, annot=True, fmt="d", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=False, annot_kws={"size": 7})
ax.set_title("Confusion Matrix – Custom CNN")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)

# Side-by-side comparison bar chart
ax = axes[1, 1]
if acc_a is not None:
    models   = ["Part A\n(AlexNet-style)", "Part B\n(Custom CNN)"]
    acc_vals = [acc_a * 100, test_acc_b * 100]
    bars = ax.bar(models, acc_vals, color=["#2196F3", "#4CAF50"], width=0.4)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(70, linestyle="--", color="red", alpha=0.5, label="70% threshold")
    ax.set_title("Test Accuracy Comparison")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
else:
    ax.text(0.5, 0.5, "Run Part A first\nfor comparison chart",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig("results/partB_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Plot saved → results/partB_results.png")

# ── 10. Final comparison summary ─────────────────────────────────────────────
improvement = (test_acc_b - acc_a) * 100 if acc_a is not None else None

print("\n" + "="*60)
print("  TASK 1 FINAL COMPARISON")
print("="*60)
print(f"  {'Metric':<25} {'Part A':>12} {'Part B':>12}")
print(f"  {'-'*49}")
if acc_a:
    print(f"  {'Test Accuracy':<25} {acc_a*100:>11.2f}% {test_acc_b*100:>11.2f}%")
    print(f"  {'Parameters':<25} {params_a:>12,} {total_params_b:>12,}")
    print(f"  {'Training Time (s)':<25} {time_a:>12.1f} {train_duration_b:>12.1f}")
    print(f"  {'Improvement':<25} {'—':>12} {improvement:>+11.2f}%")
    print(f"\n  Target (≥3pp gain): {'✓ ACHIEVED' if improvement >= 3 else '✗ Not met'}")
else:
    print(f"  Part B Test Accuracy: {test_acc_b*100:.2f}%")
print("="*60)

print("""
WHY THE CUSTOM CNN PERFORMS BETTER
────────────────────────────────────────────────────────
1. Batch Normalisation  → normalises activations after each block,
                          reduces internal covariate shift, lets the
                          model train faster and more stably.

2. Residual skip links  → gradient can bypass non-linear layers;
                          prevents vanishing gradients in deeper nets.

3. Data Augmentation    → horizontal flips, small shifts and rotations
                          act as free extra training data and teach
                          spatial invariance the baseline never learned.

4. Global Avg Pooling   → replaces Flatten+Dense; massively reduces
                          parameters in the head, cuts overfitting.

5. Dropout (0.4 + 0.3)  → randomly disables neurons during training;
                          ensemble effect at inference time.

6. Cosine LR decay      → smoothly anneals LR rather than sudden drops,
                          helps the optimiser settle into sharper minima.
────────────────────────────────────────────────────────
""")
