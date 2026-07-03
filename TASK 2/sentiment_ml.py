import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()
SEED = 42
np.random.seed(SEED)
MAX_FEATURES  = 30_000
TEST_FRACTION = 0.20
os.makedirs("results", exist_ok=True)
print("\n[1] Loading IMDb dataset...")
try:
    import tensorflow_datasets as tfds
    ds_train, ds_test = tfds.load(
        "imdb_reviews", split=["train", "test"],
        as_supervised=True, batch_size=-1
    )
    x_raw_train = [t.decode() for t in ds_train[0].numpy()]
    y_raw_train = ds_train[1].numpy().tolist()
    x_raw_test  = [t.decode() for t in ds_test[0].numpy()]
    y_raw_test  = ds_test[1].numpy().tolist()
    all_texts  = x_raw_train + x_raw_test
    all_labels = y_raw_train + y_raw_test
    print(f"   Loaded via tensorflow_datasets: {len(all_texts):,} reviews")
except Exception:
    print("   tensorflow_datasets not found — using synthetic fallback dataset.")
    print("   Install with: pip install tensorflow-datasets")
    pos = [
        "This movie was absolutely wonderful and I loved every minute.",
        "A masterpiece of storytelling with brilliant performances.",
        "Highly recommend this film, it was entertaining and moving.",
        "One of the best films I have ever seen in my life.",
        "Fantastic direction, superb acting and a compelling story.",
        "A feel-good film that left me smiling throughout.",
        "Beautiful cinematography and an emotionally rich narrative.",
        "The cast delivered outstanding performances all around.",
        "Gripping from start to finish, a truly great experience.",
        "An inspiring and uplifting movie that I will watch again.",
    ] * 300
    neg = [
        "Terrible film, a complete waste of time and money.",
        "The plot made no sense and the acting was dreadful.",
        "Boring, predictable and painfully slow. Avoid this one.",
        "One of the worst movies I have ever had to sit through.",
        "Poor screenplay, bad direction and zero entertainment value.",
        "The dialogue was cringeworthy and the story was weak.",
        "Disappointing on every level, I regret watching this.",
        "Absolutely dreadful. The characters were flat and uninteresting.",
        "A monotonous and lifeless film with no redeeming qualities.",
        "Painful to watch. The worst movie of the year by far.",
    ] * 300
    all_texts  = pos + neg
    all_labels = [1] * len(pos) + [0] * len(neg)
label_counts = pd.Series(all_labels).value_counts()
print(f"\n   Class distribution:")
print(f"   Positive (1): {label_counts.get(1, 0):,}")
print(f"   Negative (0): {label_counts.get(0, 0):,}")
print("\n[2] Cleaning text data...")
def clean_text(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)
clean_start = time.time()
cleaned_texts = [clean_text(t) for t in all_texts]
print(f"   Cleaned {len(cleaned_texts):,} reviews in {time.time()-clean_start:.1f}s")
x_train_text, x_test_text, y_train, y_test = train_test_split(
    cleaned_texts, all_labels,
    test_size=TEST_FRACTION, random_state=SEED, stratify=all_labels
)
print(f"\n   Train: {len(x_train_text):,}  |  Test: {len(x_test_text):,}")
np.save("results/test_texts_raw.npy",  np.array(x_test_text,  dtype=object))
np.save("results/test_labels.npy",     np.array(y_test))
np.save("results/train_texts_raw.npy", np.array(x_train_text, dtype=object))
np.save("results/train_labels.npy",    np.array(y_train))
print("\n[3] Fitting TF-IDF vectoriser...")
tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.90
)
X_train = tfidf.fit_transform(x_train_text)
X_test  = tfidf.transform(x_test_text)
print(f"   Vocabulary size : {len(tfidf.vocabulary_):,}")
print(f"   Train matrix    : {X_train.shape}")
joblib.dump(tfidf, "results/tfidf_vectorizer.pkl")
print("\n[4] Training Logistic Regression...")
t0 = time.time()
lr_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                               random_state=SEED, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_time = time.time() - t0
y_pred_lr = lr_model.predict(X_test)
acc_lr  = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr  = recall_score(y_test, y_pred_lr)
f1_lr   = f1_score(y_test, y_pred_lr)
print(f"   Done in {lr_time:.1f}s | Accuracy: {acc_lr*100:.2f}% | F1: {f1_lr*100:.2f}%")
print(classification_report(y_test, y_pred_lr, target_names=["Negative", "Positive"]))
joblib.dump(lr_model, "results/logistic_regression_model.pkl")
print("\n[5] Training Linear SVM...")
t0 = time.time()
svm_model = LinearSVC(C=0.5, max_iter=2000, random_state=SEED)
svm_model.fit(X_train, y_train)
svm_time = time.time() - t0
y_pred_svm = svm_model.predict(X_test)
acc_svm  = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
rec_svm  = recall_score(y_test, y_pred_svm)
f1_svm   = f1_score(y_test, y_pred_svm)
print(f"   Done in {svm_time:.1f}s | Accuracy: {acc_svm*100:.2f}% | F1: {f1_svm*100:.2f}%")
print(classification_report(y_test, y_pred_svm, target_names=["Negative", "Positive"]))
joblib.dump(svm_model, "results/svm_model.pkl")
np.save("results/partA_metrics.npy", {
    "lr":  {"accuracy": acc_lr,  "precision": prec_lr,  "recall": rec_lr,  "f1": f1_lr},
    "svm": {"accuracy": acc_svm, "precision": prec_svm, "recall": rec_svm, "f1": f1_svm},
})
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Part A – Sentiment Analysis: TF-IDF + LR vs SVM", fontsize=14, fontweight="bold")
ax = axes[0, 0]
bars = ax.bar(["Negative", "Positive"],
              [label_counts.get(0, 0), label_counts.get(1, 0)],
              color=["#EF5350", "#42A5F5"])
ax.set_title("Class Distribution"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+50,
            f"{int(b.get_height()):,}", ha="center", fontsize=11)
ax = axes[0, 1]
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
x_pos = np.arange(4); w = 0.35
b1 = ax.bar(x_pos-w/2, [v*100 for v in [acc_lr,prec_lr,rec_lr,f1_lr]],
            w, label="Logistic Reg", color="#42A5F5")
b2 = ax.bar(x_pos+w/2, [v*100 for v in [acc_svm,prec_svm,rec_svm,f1_svm]],
            w, label="Linear SVM",  color="#AB47BC")
ax.set_xticks(x_pos); ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 115); ax.set_ylabel("Score (%)"); ax.set_title("LR vs SVM Metrics")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f"{b.get_height():.1f}", ha="center", fontsize=8)
ax = axes[1, 0]
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
            ax=ax, cbar=False, annot_kws={"size": 14})
ax.set_title("Confusion Matrix — Logistic Regression")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax = axes[1, 1]
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Purples",
            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
            ax=ax, cbar=False, annot_kws={"size": 14})
ax.set_title("Confusion Matrix — Linear SVM")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("results/partA_results.png", dpi=150, bbox_inches="tight")
plt.show()
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
feature_names = np.array(tfidf.get_feature_names_out())
coef = lr_model.coef_[0]
top_pos = coef.argsort()[-20:][::-1]
top_neg = coef.argsort()[:20]
axes2[0].barh(feature_names[top_pos][::-1], coef[top_pos][::-1], color="#42A5F5")
axes2[0].set_title("Top 20 Positive-Sentiment Features")
axes2[0].set_xlabel("LR Coefficient")
axes2[0].grid(axis="x", alpha=0.3)
axes2[1].barh(feature_names[top_neg], np.abs(coef[top_neg]), color="#EF5350")
axes2[1].set_title("Top 20 Negative-Sentiment Features")
axes2[1].set_xlabel("|LR Coefficient|")
axes2[1].grid(axis="x", alpha=0.3)
fig2.suptitle("TF-IDF Feature Importance via Logistic Regression Coefficients",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/partA_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n" + "="*60)
print("  PART A — SUMMARY")
print("="*60)
print(f"  {'Metric':<18} {'Logistic Reg':>14} {'Linear SVM':>14}")
print(f"  {'-'*46}")
print(f"  {'Accuracy':<18} {acc_lr*100:>13.2f}% {acc_svm*100:>13.2f}%")
print(f"  {'Precision':<18} {prec_lr*100:>13.2f}% {prec_svm*100:>13.2f}%")
print(f"  {'Recall':<18} {rec_lr*100:>13.2f}% {rec_svm*100:>13.2f}%")
print(f"  {'F1-Score':<18} {f1_lr*100:>13.2f}% {f1_svm*100:>13.2f}%")
print(f"  {'Train time (s)':<18} {lr_time:>14.1f} {svm_time:>14.1f}")
print("="*60)
