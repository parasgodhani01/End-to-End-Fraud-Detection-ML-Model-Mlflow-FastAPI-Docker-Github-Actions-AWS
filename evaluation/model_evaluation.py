import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)

# Style
sns.set_theme(style="darkgrid", palette="muted")
FRAUD_COLOR  = "#e74c3c"
NORMAL_COLOR = "#2ecc71"
ACCENT_COLOR = "#3498db"

#Load best model + test set 

BEST_MODEL_PATH = "models/best_model.pkl"

print("Loading best model …")
with open(BEST_MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
model_name   = bundle["name"]
feature_cols = bundle["feature_cols"]
X_test       = bundle["X_test"]
y_test       = bundle["y_test"]

print(f"  Model   : {model_name}")
print(f"  Test set: {X_test.shape[0]:,} samples  ({y_test.sum():,} fraud)\n")

# Predictions

y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)   # default threshold

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION REPORT

print(classification_report(y_test, y_pred,
                             target_names=["Legitimate", "Fraud"], digits=4))


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRIX

cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=1,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=axes[0], cbar=False)
axes[0].set_title("Raw Counts")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", linewidths=1,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=axes[1], cbar=False)
axes[1].set_title("Row-Normalised (%)")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("plot_eval_01_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved → plot_eval_01_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROC CURVE

fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=ACCENT_COLOR, lw=2,
        label=f"{model_name}  (AUC = {auc_score:.4f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color=ACCENT_COLOR)
ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plot_eval_02_roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  ROC AUC: {auc_score:.4f}")
print("  Saved → plot_eval_02_roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRECISION-RECALL CURVE

precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)
baseline = y_test.mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_arr, precision_arr, color=FRAUD_COLOR, lw=2,
        label=f"{model_name}  (AP = {ap_score:.4f})")
ax.axhline(baseline, color="grey", lw=1, linestyle="--",
           label=f"Baseline (fraud rate = {baseline:.4f})")
ax.fill_between(recall_arr, precision_arr, alpha=0.1, color=FRAUD_COLOR)
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
plt.tight_layout()
plt.savefig("plot_eval_03_pr_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Average Precision: {ap_score:.4f}")
print("  Saved → plot_eval_03_pr_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD ANALYSIS

thresholds_range = np.arange(0.01, 1.0, 0.01)
f1_scores, prec_scores, rec_scores = [], [], []

for t in thresholds_range:
    preds = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds, zero_division=0))
    prec_scores.append(precision_score(y_test, preds, zero_division=0))
    rec_scores.append(recall_score(y_test, preds, zero_division=0))

best_f1_idx = np.argmax(f1_scores)
best_f1_t   = thresholds_range[best_f1_idx]
best_f1_val = f1_scores[best_f1_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_range, f1_scores,   lw=2, color="#9b59b6", label="F1")
ax.plot(thresholds_range, prec_scores, lw=2, color=ACCENT_COLOR, label="Precision")
ax.plot(thresholds_range, rec_scores,  lw=2, color=FRAUD_COLOR,  label="Recall")
ax.axvline(best_f1_t, color="#9b59b6", linestyle="--", alpha=0.7,
           label=f"Best F1 threshold = {best_f1_t:.2f}  (F1={best_f1_val:.4f})")
ax.set_title("Precision / Recall / F1 vs Decision Threshold", fontsize=13, fontweight="bold")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
plt.savefig("plot_eval_04_threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n  Best F1 threshold  : {best_f1_t:.2f}  (F1 = {best_f1_val:.4f})")
print("  Saved → plot_eval_04_threshold_analysis.png")

# Re-evaluate at best F1 threshold
y_pred_best = (y_proba >= best_f1_t).astype(int)
print(f"\n  Classification report at optimal threshold ({best_f1_t:.2f}):")
print(classification_report(y_test, y_pred_best,
                             target_names=["Legitimate", "Fraud"], digits=4))


# 6. FEATURE IMPORTANCE (Top 20)

try:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_features = [feature_cols[i] for i in indices]
    top_scores   = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [FRAUD_COLOR if s > np.percentile(top_scores, 70) else ACCENT_COLOR
              for s in top_scores]
    ax.barh(top_features[::-1], top_scores[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title(f"{model_name} — Top 20 Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plot_eval_05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → plot_eval_05_feature_importance.png")

    print("\n  Top 10 features:")
    for i, (feat, score) in enumerate(zip(top_features[:10], top_scores[:10]), 1):
        print(f"    {i:>2}. {feat:<35} {score:.5f}")
except AttributeError:
    print("  (feature_importances_ not available for this model type)")


# 7. BUSINESS IMPACT SUMMARY

for label, threshold in [("Default (0.50)", 0.50), (f"Optimal F1 ({best_f1_t:.2f})", best_f1_t)]:
    preds = (y_proba >= threshold).astype(int)
    cm_b  = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm_b.ravel()

    total_fraud_amount = y_test.sum()  
    print(f"\n  Threshold: {label}")
    print(f"    True Positives  (fraud caught)    : {tp:>8,}")
    print(f"    False Negatives (fraud missed)    : {fn:>8,}")
    print(f"    False Positives (false alarms)    : {fp:>8,}")
    print(f"    True Negatives  (correct clears)  : {tn:>8,}")
    print(f"    Recall (fraud detection rate)     : {tp/(tp+fn)*100:>7.2f}%")
    print(f"    Precision (accuracy when flagged) : {tp/(tp+fp)*100:>7.2f}%")
    print(f"    False alarm rate                  : {fp/(fp+tn)*100:>7.4f}%")

print("\nEvaluation complete — all plots saved.")
