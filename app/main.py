import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

# -- Load model once when server starts
with open("models/best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
feature_cols = bundle["feature_cols"]
threshold    = bundle["best_threshold_f1"]

print(f"Model loaded: {bundle['name']}")
print(f"Threshold   : {threshold}")


# Input schema
class Transaction(BaseModel):
    step           : int
    amount         : float
    oldbalanceOrg  : float
    newbalanceOrig : float
    oldbalanceDest : float
    newbalanceDest : float
    type           : str    


@app.get("/")
def health():
    return {
        "status" : "ok",
        "model"  : bundle["name"],
        "threshold": threshold,
    }


# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    df = pd.DataFrame([transaction.dict()])

    # Feature Engineering 
    df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]

    df["abs_errorBalanceOrig"] = df["errorBalanceOrig"].abs()
    df["abs_errorBalanceDest"] = df["errorBalanceDest"].abs()

    df["isOrigZeroAfter"]  = (df["newbalanceOrig"] == 0).astype(int)
    df["isOrigZeroBefore"] = (df["oldbalanceOrg"]  == 0).astype(int)
    df["isDestZeroBefore"] = (df["oldbalanceDest"] == 0).astype(int)
    df["isDestZeroAfter"]  = (df["newbalanceDest"] == 0).astype(int)

    df["suspiciousWipe"] = (
        (df["isOrigZeroAfter"] == 1) &
        (df["abs_errorBalanceOrig"] > 0)
    ).astype(int)

    df["amountToOldBalanceOrig"] = np.where(
        df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0.0)

    df["amountToOldBalanceDest"] = np.where(
        df["oldbalanceDest"] > 0, df["amount"] / df["oldbalanceDest"], -1.0)

    df["origBalanceChangeRatio"] = np.where(
        df["oldbalanceOrg"] > 0,
        (df["oldbalanceOrg"] - df["newbalanceOrig"]) / df["oldbalanceOrg"], 0.0)

    df["isTransfer"]     = (df["type"] == "TRANSFER").astype(int)
    df["isCashOut"]      = (df["type"] == "CASH_OUT").astype(int)
    df["isHighRiskType"] = ((df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")).astype(int)

    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    df["log_amount"]         = np.log1p(df["amount"])
    df["log_oldbalanceOrg"]  = np.log1p(df["oldbalanceOrg"])
    df["log_newbalanceOrig"] = np.log1p(df["newbalanceOrig"])
    df["log_oldbalanceDest"] = np.log1p(df["oldbalanceDest"])
    df["log_newbalanceDest"] = np.log1p(df["newbalanceDest"])

    bins   = [-1, 1_000, 10_000, 100_000, 1_000_000, np.inf]
    labels = [0, 1, 2, 3, 4]
    df["amountTier"] = pd.cut(df["amount"], bins=bins, labels=labels).astype(int)

    df["origAcctFrequency"]  = 1
    df["destAcctFrequency"]  = 1
    df["isMuleDestCandidate"] = 0

    df["hour_of_day"]  = df["step"] % 24
    df["day_of_month"] = df["step"] // 24
    df["isNightTime"]  = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)

    # Prediction
    X = df[feature_cols].values
    prob = float(model.predict_proba(X)[0][1])

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud"         : prob >= threshold,
        "risk_level"       : "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW",
        "threshold_used"   : threshold,
    }