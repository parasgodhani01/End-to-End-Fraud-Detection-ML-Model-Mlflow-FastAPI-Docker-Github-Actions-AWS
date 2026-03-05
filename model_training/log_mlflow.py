import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

# MLflow Setup
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("fraud-detection")

# Paths 
MODELS_DIR = "../models/"
DATA_PATH  = "Dataset/transactions_engineered.parquet"

print("Loading data to recreate test split ...")
df = pd.read_parquet(DATA_PATH, engine="pyarrow")

TARGET       = "isFraud"
FEATURE_COLS = [col for col in df.columns if col != TARGET]

X = df[FEATURE_COLS].values
y = df[TARGET].values

fraud_count  = y.sum() #type: ignore
normal_count = len(y) - fraud_count
scale_weight = normal_count / fraud_count

# Same split as training
X_temp, X_test, y_temp, y_test = train_test_split( 
    X, y, test_size=0.15, stratify=y, random_state=42) #type: ignore

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

print(f"Val  : {X_val.shape[0]:,}  (fraud: {y_val.sum():,})")
print(f"Test : {X_test.shape[0]:,}  (fraud: {y_test.sum():,})\n")


# LOG XGBOOST

print("=" * 50)
print("Logging XGBoost to MLflow ...")
print("=" * 50)

xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
with open(xgb_path, "rb") as f:
    xgb_model = pickle.load(f)

xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
xgb_roc       = roc_auc_score(y_val, xgb_val_proba)
xgb_pr        = average_precision_score(y_val, xgb_val_proba)

print(f"Val ROC-AUC : {xgb_roc:.4f}")
print(f"Val PR-AUC  : {xgb_pr:.4f}")

with mlflow.start_run(run_name="xgboost-v1"):
    # Log params
    mlflow.log_param("model",        "XGBoost")
    mlflow.log_param("device",       "cuda")
    mlflow.log_param("scale_weight", round(scale_weight, 2))

    # Log model params
    params = xgb_model.get_params()
    for key in ["n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "min_child_weight", "gamma"]:
        mlflow.log_param(key, params.get(key))

    # Log metrics
    mlflow.log_metric("val_roc_auc", xgb_roc) #type: ignore
    mlflow.log_metric("val_pr_auc",  xgb_pr) #type: ignore

    # Log model
    mlflow.xgboost.log_model(xgb_model, "xgboost-model") #type: ignore
    print("XGBoost logged to MLflow successfully.")


# LOG RANDOM FOREST

rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")

if os.path.exists(rf_path):
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_roc       = roc_auc_score(y_val, rf_val_proba)
    rf_pr        = average_precision_score(y_val, rf_val_proba)

    print(f"Val ROC-AUC : {rf_roc:.4f}")
    print(f"Val PR-AUC  : {rf_pr:.4f}")

    with mlflow.start_run(run_name="random-forest-v1"):
        mlflow.log_param("model",    "RandomForest")
        mlflow.log_param("n_jobs",   -1)

        params = rf_model.get_params()
        for key in ["n_estimators", "max_depth", "min_samples_split",
                    "min_samples_leaf", "max_features", "class_weight"]:
            mlflow.log_param(key, params.get(key))

        mlflow.log_metric("val_roc_auc", rf_roc) #type: ignore
        mlflow.log_metric("val_pr_auc",  rf_pr) #type: ignore

        mlflow.sklearn.log_model(rf_model, "random-forest-model") #type: ignore
        print("Random Forest logged to MLflow successfully.")

else:
    print("\nNo random_forest.pkl found — skipping RF logging.")


# COMPARE & SAVE BEST MODEL

print(f"  XGBoost → ROC: {xgb_roc:.4f}  PR-AUC: {xgb_pr:.4f}")

if os.path.exists(rf_path):
    print(f"  RF      → ROC: {rf_roc:.4f}  PR-AUC: {rf_pr:.4f}") #type: ignore
    if xgb_pr >= rf_pr:                                          #type: ignore
        winner_name  = "XGBoost"
        winner_model = xgb_model
        winner_proba = xgb_val_proba
    else:
        winner_name  = "Random Forest"
        winner_model = rf_model                             #type: ignore
        winner_proba = rf_val_proba                         #type: ignore
else: 
    winner_name  = "XGBoost"
    winner_model = xgb_model
    winner_proba = xgb_val_proba

print(f"\n🏆  Winner: {winner_name}")

# -- Save best_model.pkl ------------------------
winner_path = os.path.join(MODELS_DIR, "best_model.pkl")
with open(winner_path, "wb") as f:
    pickle.dump({
        "model"                : winner_model,
        "name"                 : winner_name,
        "feature_cols"         : FEATURE_COLS,
        "X_test"               : X_test,
        "y_test"               : y_test,
        "best_threshold_f1"    : 0.5,
        "best_threshold_recall": 0.3,
    }, f)

print(f"best_model.pkl saved → {winner_path}")

print("\nNow run: venv\\Scripts\\python.exe -m mlflow ui")
print("Then open: http://localhost:5000")