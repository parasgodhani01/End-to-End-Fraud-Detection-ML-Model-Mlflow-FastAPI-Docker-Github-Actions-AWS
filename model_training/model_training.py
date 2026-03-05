import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score)
import xgboost as xgb

DATA_PATH    = "Dataset/transactions_engineered.parquet"
MODELS_DIR   = "models/"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_JOBS       = -1          
CV_FOLDS     = 5
N_ITER       = 20          


# Load engineered dataset

print("Loading engineered dataset …")
df = pd.read_parquet(DATA_PATH)
print(f"  Shape  : {df.shape}")
print(f"  Fraud %: {df['isFraud'].mean()*100:.4f}%\n")

TARGET = "isFraud"
FEATURE_COLS = [c for c in df.columns if c != TARGET]

X = df[FEATURE_COLS].values
y = df[TARGET].values

fraud_count  = y.sum() # number of fraud cases #type: ignore
normal_count = len(y) - fraud_count
scale_weight = normal_count / fraud_count   

print(f"Fraud cases  : {fraud_count:,}")
print(f"Normal cases : {normal_count:,}")
print(f"Scale weight : {scale_weight:.1f}\n")


# ── Train / Validation / Test Split 
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE) #type: ignore

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE)

print(f"Train : {X_train.shape[0]:>10,}  (fraud: {y_train.sum():,})")
print(f"Val   : {X_val.shape[0]:>10,}  (fraud: {y_val.sum():,})")
print(f"Test  : {X_test.shape[0]:>10,}  (fraud: {y_test.sum():,})\n")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# Random Forest

print("MODEL 1: Random Forest")


rf_param_dist = {
    "n_estimators"     : [200, 300, 500],
    "max_depth"        : [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2"],
    "class_weight"     : ["balanced", "balanced_subsample"],
}

rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

rf_search = RandomizedSearchCV(
    rf_base,
    param_distributions=rf_param_dist,
    n_iter=N_ITER,
    scoring="average_precision",  
    cv=cv,
    n_jobs=N_JOBS,
    verbose=1,
    random_state=RANDOM_STATE,
)

print("Running RandomizedSearchCV for Random Forest …")
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

print(f"\n  Best params : {rf_search.best_params_}")
print(f"  CV PR-AUC  : {rf_search.best_score_:.4f}")

rf_val_proba = rf_best.predict_proba(X_val)[:, 1] # type: ignore
print(f"  Val ROC-AUC : {roc_auc_score(y_val, rf_val_proba):.4f}")
print(f"  Val PR-AUC  : {average_precision_score(y_val, rf_val_proba):.4f}")

# Save model
rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
with open(rf_path, "wb") as f:
    pickle.dump(rf_best, f)
print(f"  Saved → {rf_path}")


# XGBoost

print("MODEL 2: XGBoost")

xgb_param_dist = {
    "n_estimators"     : [300, 500, 800],
    "max_depth"        : [4, 6, 8, 10],
    "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
    "subsample"        : [0.6, 0.8, 1.0],
    "colsample_bytree" : [0.6, 0.8, 1.0],
    "min_child_weight" : [1, 5, 10],
    "gamma"            : [0, 0.1, 0.5],
    "reg_alpha"        : [0, 0.1, 1.0],
    "reg_lambda"       : [1.0, 1.5, 2.0],
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_weight,  
    use_label_encoder=False,
    eval_metric="aucpr",
    tree_method="hist",             
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)

xgb_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=xgb_param_dist,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv,
    n_jobs=N_JOBS,
    verbose=1,
    random_state=RANDOM_STATE,
)

print("Running RandomizedSearchCV for XGBoost …")
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_

print(f"\n  Best params : {xgb_search.best_params_}")
print(f"  CV PR-AUC  : {xgb_search.best_score_:.4f}")

xgb_val_proba = xgb_best.predict_proba(X_val)[:, 1]  # type: ignore
print(f"  Val ROC-AUC : {roc_auc_score(y_val, xgb_val_proba):.4f}")
print(f"  Val PR-AUC  : {average_precision_score(y_val, xgb_val_proba):.4f}")

# Save model
xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
with open(xgb_path, "wb") as f:
    pickle.dump(xgb_best, f)
print(f"  Saved → {xgb_path}")


# Find best thresholds for both models

def find_best_threshold(y_true, y_proba, metric="f1"):
    """
    Sweep thresholds 0.01 → 0.99 and return the one that
    maximises the chosen metric (f1, recall, or precision).
    """
    from sklearn.metrics import f1_score, recall_score, precision_score
    metric_fn = {"f1": f1_score, "recall": recall_score, "precision": precision_score}[metric]

    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (y_proba >= t).astype(int)
        score = metric_fn(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t     = t
    return best_t, best_score

print("THRESHOLD TUNING (on validation set)")

for name, proba in [("Random Forest", rf_val_proba), ("XGBoost", xgb_val_proba)]:
    t_f1,  s_f1  = find_best_threshold(y_val, proba, "f1")
    t_rec, s_rec = find_best_threshold(y_val, proba, "recall")
    print(f"\n  {name}:")
    print(f"    Best F1 threshold      : {t_f1:.2f}  (F1={s_f1:.4f})")
    print(f"    Best Recall threshold  : {t_rec:.2f}  (Recall={s_rec:.4f})")


# Best model selection 

rf_prauc  = average_precision_score(y_val, rf_val_proba)
xgb_prauc = average_precision_score(y_val, xgb_val_proba)

winner_name  = "XGBoost" if xgb_prauc >= rf_prauc else "Random Forest"
winner_model = xgb_best  if xgb_prauc >= rf_prauc else rf_best
winner_path  = os.path.join(MODELS_DIR, "best_model.pkl")

with open(winner_path, "wb") as f:
    pickle.dump({"model": winner_model, "name": winner_name,
                 "feature_cols": FEATURE_COLS,
                 "X_test": X_test, "y_test": y_test}, f)

print(f"\nBest model: {winner_name}  (Val PR-AUC: {max(rf_prauc, xgb_prauc):.4f})")
print(f"   Saved → {winner_path}")
