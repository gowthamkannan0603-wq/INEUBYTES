"""
Task 2 - Part B: Sentiment Analysis using Deep Learning (LSTM)
Author: Gowtham K | Reg No: INBT020009 | Course: AIINB10626

Goal    : Build an LSTM model and fairly compare it against Part A's
          Logistic Regression and SVM — using the SAME train/test split.

Key design choices explained:
  - Embedding layer (trainable) : learns task-specific word vectors
    rather than relying on handcrafted TF-IDF weights.
  - Bidirectional LSTM          : reads text left-to-right AND
    right-to-left, capturing context from both directions.
  - Dropout on LSTM output      : regularises the recurrent state.
  - Global Max Pooling          : picks the strongest signal across
    all time steps instead of just the final hidden state.
  - Early stopping + LR reduce  : avoids overfitting on a text dataset
    where the model can easily memorise training samples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── Reproducibility (SAME seed as Part A) ────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 20_000    # top-N most frequent words kept
MAX_LEN      = 300       # pad/truncate each review to 300 tokens
EMBED_DIM    = 64        # word embedding dimensions
LSTM_UNITS   = 64        # units per LSTM direction
BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 1e-3

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the same train/test split saved by Part A
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading train/test split from Part A...")

try:
    x_train_text = np.load("../partA/results/train_texts_raw.npy", allow_pickle=True).tolist()
    x_test_text  = np.load("../partA/results/test_texts_raw.npy",  allow_pickle=True).tolist()
    y_train      = np.load("../partA/results/train_labels.npy").tolist()
    y_test       = np.load("../partA/results/test_labels.npy").tolist()
    print(f"   Loaded from Part A: Train={len(x_train_text):,}  Test={len(x_test_text):,}")
    partA_metrics = np.load("../partA/results/partA_metrics.npy", allow_pickle=True).item()

except FileNotFoundError:
    print("   Part A results not found. Re-generating split...")
    import re
    try:
        import tensorflow_datasets as tfds
        ds_train, ds_test = tfds.load("imdb_reviews", split=["train","test"],
                                       as_supervised=True, batch_size=-1)
        all_texts  = [t.decode() for t in ds_train[0].numpy()] + \
                     [t.decode() for t in ds_test[0].numpy()]
        all_labels = ds_train[1].numpy().tolist() + ds_test[1].numpy().tolist()
    except Exception:
        pos = ["This movie was absolutely wonderful and truly inspiring."] * 3000
        neg = ["This was a terrible and completely boring film to watch."] * 3000
        all_texts  = pos + neg
        all_labels = [1]*3000 + [0]*3000

    def clean(t):
        t = t.lower()
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    cleaned = [clean(t) for t in all_texts]
    from sklearn.model_selection import train_test_split
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        cleaned, all_labels, test_size=0.20, random_state=SEED, stratify=all_labels
    )
    partA_metrics = None

y_train_arr = np.array(y_train)
y_test_arr  = np.array(y_test)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenise text → integer sequences
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Tokenising text and building sequences...")

tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(x_train_text)       # fit only on train

train_seqs = tokenizer.texts_to_sequences(x_train_text)
test_seqs  = tokenizer.texts_to_sequences(x_test_text)

# Pad to fixed length (truncate from the front so model sees recent words)
x_train_pad = keras.preprocessing.sequence.pad_sequences(
    train_seqs, maxlen=MAX_LEN, padding="post", truncating="post"
)
x_test_pad = keras.preprocessing.sequence.pad_sequences(
    test_seqs,  maxlen=MAX_LEN, padding="post", truncating="post"
)
print(f"   Vocab size used : {VOCAB_SIZE:,}")
print(f"   Sequence length : {MAX_LEN}")
print(f"   Train shape     : {x_train_pad.shape}")
print(f"   Test shape      : {x_test_pad.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build Bidirectional LSTM model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Building Bidirectional LSTM model...")

def build_lstm_model(vocab_size, embed_dim, lstm_units, max_len):
    """
    Architecture:
      Embedding → BiLSTM (return_sequences) → GlobalMaxPool → Dropout → Dense
    
    Why Bidirectional?
      A movie review like "not bad at all" needs both directions to understand
      the negation; a single-direction LSTM might miss the context.
    
    Why GlobalMaxPool over final hidden state?
      It captures the strongest signal from any point in the sequence,
      not just the last token — useful for long reviews.
    """
    inp = keras.Input(shape=(max_len,), name="token_input")

    # Learned word embeddings
    x = layers.Embedding(vocab_size, embed_dim,
                         mask_zero=True, name="word_embed")(inp)
    x = layers.SpatialDropout1D(0.2, name="spatial_drop")(x)

    # First Bidirectional LSTM layer (returns full sequence)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2,
                    recurrent_dropout=0.0),
        name="bilstm_1"
    )(x)

    # Second Bidirectional LSTM layer (return sequences for pooling)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=0.2),
        name="bilstm_2"
    )(x)

    # Pool across time steps
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = layers.Dropout(0.3, name="head_drop")(x)

    # Dense classifier
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.2, name="fc_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="sentiment_out")(x)

    model = keras.Model(inp, out, name="BiLSTM_Sentiment")
    return model

lstm_model = build_lstm_model(VOCAB_SIZE + 1, EMBED_DIM, LSTM_UNITS, MAX_LEN)
lstm_model.summary()
print(f"\n   Total parameters: {lstm_model.count_params():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compile
# ─────────────────────────────────────────────────────────────────────────────
lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "results/lstm_best.keras", save_best_only=True, verbose=0
    )
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Training Bidirectional LSTM...")
t_start = time.time()

history = lstm_model.fit(
    x_train_pad, y_train_arr,
    validation_split=0.15,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

lstm_time = time.time() - t_start
print(f"\n   Training completed in {lstm_time/60:.1f} minutes")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Evaluating LSTM on test set...")
test_loss, test_acc_raw = lstm_model.evaluate(x_test_pad, y_test_arr, verbose=0)

y_prob_lstm = lstm_model.predict(x_test_pad, verbose=0).flatten()
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

acc_lstm  = accuracy_score(y_test_arr, y_pred_lstm)
prec_lstm = precision_score(y_test_arr, y_pred_lstm)
rec_lstm  = recall_score(y_test_arr, y_pred_lstm)
f1_lstm   = f1_score(y_test_arr, y_pred_lstm)

print(f"   Test Loss    : {test_loss:.4f}")
print(f"   Test Accuracy: {acc_lstm*100:.2f}%  |  F1: {f1_lstm*100:.2f}%")
print("\n   Classification Report (LSTM):")
print(classification_report(y_test_arr, y_pred_lstm,
                             target_names=["Negative", "Positive"]))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualisations
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Part B – Bidirectional LSTM Sentiment Analysis", fontsize=14, fontweight="bold")

# Accuracy over epochs
ax = axes[0, 0]
ax.plot(history.history["accuracy"],     label="Train Acc",  color="#4CAF50")
ax.plot(history.history["val_accuracy"], label="Val Acc",    color="#FF5722")
ax.set_title("LSTM Training Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(alpha=0.3)

# Loss over epochs
ax = axes[0, 1]
ax.plot(history.history["loss"],     label="Train Loss", color="#4CAF50")
ax.plot(history.history["val_loss"], label="Val Loss",   color="#FF5722")
ax.set_title("LSTM Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# Confusion matrix
ax = axes[1, 0]
sns.heatmap(confusion_matrix(y_test_arr, y_pred_lstm), annot=True, fmt="d",
            cmap="Greens", xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
            ax=ax, cbar=False, annot_kws={"size": 14})
ax.set_title("Confusion Matrix — BiLSTM")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")

# 3-way comparison bar chart
ax = axes[1, 1]
if partA_metrics:
    model_names = ["Logistic\nReg", "Linear\nSVM", "BiLSTM"]
    f1_scores   = [
        partA_metrics["lr"]["f1"]  * 100,
        partA_metrics["svm"]["f1"] * 100,
        f1_lstm * 100
    ]
    acc_scores  = [
        partA_metrics["lr"]["accuracy"]  * 100,
        partA_metrics["svm"]["accuracy"] * 100,
        acc_lstm * 100
    ]
    x_pos = np.arange(3); w = 0.35
    b1 = ax.bar(x_pos-w/2, acc_scores, w, label="Accuracy", color=["#2196F3","#2196F3","#2196F3"])
    b2 = ax.bar(x_pos+w/2, f1_scores,  w, label="F1-Score", color=["#FF9800","#FF9800","#4CAF50"])
    ax.set_xticks(x_pos); ax.set_xticklabels(model_names)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
    ax.set_title("All Models — Accuracy & F1 Comparison")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{b.get_height():.1f}", ha="center", fontsize=8)
else:
    ax.text(0.5, 0.5, f"LSTM Test Accuracy: {acc_lstm*100:.2f}%\nF1: {f1_lstm*100:.2f}%",
            ha="center", va="center", transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9"))
    ax.set_title("LSTM Results")
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/partB_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n   Plot saved → results/partB_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Final Comparison Table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TASK 2 — FINAL COMPARISON (all models, same test split)")
print("="*70)
header = f"  {'Metric':<18} {'Log. Reg':>12} {'Linear SVM':>12} {'BiLSTM':>12}"
print(header)
print(f"  {'-'*54}")

if partA_metrics:
    def row(name, key):
        lr_v  = partA_metrics["lr"][key]  * 100
        svm_v = partA_metrics["svm"][key] * 100
        lstm_v = {"accuracy": acc_lstm, "precision": prec_lstm,
                  "recall": rec_lstm, "f1": f1_lstm}[key] * 100
        print(f"  {name:<18} {lr_v:>11.2f}% {svm_v:>11.2f}% {lstm_v:>11.2f}%")

    row("Accuracy",  "accuracy")
    row("Precision", "precision")
    row("Recall",    "recall")
    row("F1-Score",  "f1")
else:
    print(f"  {'Accuracy':<18} {'—':>12} {'—':>12} {acc_lstm*100:>11.2f}%")
    print(f"  {'F1-Score':<18} {'—':>12} {'—':>12} {f1_lstm*100:>11.2f}%")

print("="*70)

# LSTM vs best classical
if partA_metrics:
    best_classical_f1 = max(partA_metrics["lr"]["f1"], partA_metrics["svm"]["f1"]) * 100
    gap = f1_lstm * 100 - best_classical_f1
    print(f"\n  Best classical F1 : {best_classical_f1:.2f}%")
    print(f"  LSTM F1           : {f1_lstm*100:.2f}%")
    print(f"  Gap               : {gap:+.2f}%")
    if gap >= 0:
        print("  Result: LSTM MATCHED OR BEAT the classical models ✓")
    else:
        print(f"  Result: LSTM did NOT beat classical models (gap = {gap:.2f}%)")
        print("""
  Why LSTM might underperform classical models on IMDb:
  ─────────────────────────────────────────────────────
  1. Dataset size: TF-IDF+SVM is extremely well-tuned for text;
     LSTM needs much more data to learn rich representations.
  2. Training time constraint: deeper LSTM needs more epochs.
  3. IMDb is a well-studied benchmark where linear models
     (especially SVM) are known to be very competitive.
  4. Recognising when a simpler model wins is a key data
     science skill — the result is still valid and informative.
        """)

print("""
WHY LSTM FOR SEQUENTIAL TEXT?
─────────────────────────────────────────────────────────────────
  • LSTM maintains a hidden state that carries context across many
    tokens, capturing long-range dependencies (e.g. 'not very good').
  • Bidirectionality lets the model use both past AND future context.
  • Trainable embeddings learn task-specific word representations
    rather than relying on fixed token frequencies.
  • SpatialDropout1D drops entire feature maps per time step,
    a strong regulariser for sequence models.
─────────────────────────────────────────────────────────────────
""")
