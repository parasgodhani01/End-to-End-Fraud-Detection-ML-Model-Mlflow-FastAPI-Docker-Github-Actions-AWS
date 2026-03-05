import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_PATH   = "Dataset/fraud_data.csv"

OUTPUT_PATH = "Dataset/transactions_engineered.parquet"

print("Loading dataset---")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")

pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None) 

# -- 1.balances integrity check ------------------------

# Expected vs actual balance for origin account
df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]

# Expected vs actual balance for destination account
df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]

# Absolute mismatch — fraudsters often leave exact-zero balances to hide money
df["abs_errorBalanceOrig"] = df["errorBalanceOrig"].abs()
df["abs_errorBalanceDest"] = df["errorBalanceDest"].abs()

print(df[["errorBalanceOrig", "errorBalanceDest", "abs_errorBalanceOrig", "abs_errorBalanceDest"]].describe())

# -- 2. zero balance falg ------------------------

# Common fraud pattern: orig balance wiped to zero
df["isOrigZeroAfter"]  = (df["newbalanceOrig"] == 0).astype(int)
df["isOrigZeroBefore"] = (df["oldbalanceOrg"]  == 0).astype(int)

# Destination starting from zero (new/temp account)
df["isDestZeroBefore"] = (df["oldbalanceDest"] == 0).astype(int)
df["isDestZeroAfter"]  = (df["newbalanceDest"] == 0).astype(int)

# Both origin balance wipes AND mismatch — very suspicious combo
df["suspiciousWipe"] = (
    (df["isOrigZeroAfter"] == 1) &
    (df["abs_errorBalanceOrig"] > 0)
).astype(int)

# -- 3. Ratio Features ------------------------
# Fraction of origin balance that was transferred
df["amountToOldBalanceOrig"] = np.where(
    df["oldbalanceOrg"] > 0,
    df["amount"] / df["oldbalanceOrg"],
    0.0
)
# Fraction of dest old balance relative to amount received
df["amountToOldBalanceDest"] = np.where(
    df["oldbalanceDest"] > 0,
    df["amount"] / df["oldbalanceDest"],
    -1.0   # -1 signals divide-by-zero case (new account)
)

# Net balance change ratio for origin
df["origBalanceChangeRatio"] = np.where(
    df["oldbalanceOrg"] > 0,
    (df["oldbalanceOrg"] - df["newbalanceOrig"]) / df["oldbalanceOrg"],
    0.0
)

#-- 4.Transaction Type Encoding ------------------------

# Only TRANSFER and CASH_OUT have fraud in this dataset — binary flag is useful
df["isTransfer"]  = (df["type"] == "TRANSFER").astype(int)
df["isCashOut"]   = (df["type"] == "CASH_OUT").astype(int)
df["isHighRiskType"] = ((df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")).astype(int)

# One-hot encode all types (for completeness)
type_dummies = pd.get_dummies(df["type"], prefix="type", drop_first=False)
df = pd.concat([df, type_dummies], axis=1)

# -- 5. Amount Transforms ------------------------

df["log_amount"]         = np.log1p(df["amount"])
df["log_oldbalanceOrg"]  = np.log1p(df["oldbalanceOrg"])
df["log_newbalanceOrig"] = np.log1p(df["newbalanceOrig"])
df["log_oldbalanceDest"] = np.log1p(df["oldbalanceDest"])
df["log_newbalanceDest"] = np.log1p(df["newbalanceDest"])

# Fix — start bins from -1 to include 0 amounts
bins   = [-1, 1_000, 10_000, 100_000, 1_000_000, np.inf]
labels = [0, 1, 2, 3, 4]
df["amountTier"] = pd.cut(df["amount"], bins=bins, labels=labels).astype(int)


# -- 6. Account Velocity ------------------------

# How many times does each account appear in the dataset?
orig_freq = df["nameOrig"].value_counts()
dest_freq = df["nameDest"].value_counts()

df["origAcctFrequency"] = df["nameOrig"].map(orig_freq)
df["destAcctFrequency"] = df["nameDest"].map(dest_freq)

# Accounts appearing 5+ times on destination side = potential mule account
MULE_THRESHOLD = 5
df["isMuleDestCandidate"] = (df["destAcctFrequency"] >= MULE_THRESHOLD).astype(int)

print(df[["origAcctFrequency", "destAcctFrequency", "isMuleDestCandidate"]].describe())


# -- 7. Time Based Features ------------------------

# Each step = 1 hour — dataset spans ~30 days (720 steps)
df["hour_of_day"]  = df["step"] % 24
df["day_of_month"] = df["step"] // 24

# Transactions between midnight and 6am — fraud happens when owners are asleep
df["isNightTime"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)

print(df[["hour_of_day", "day_of_month", "isNightTime"]].describe())


# -- Final Output ------------------------

NEW_FEATURES = [
    "errorBalanceOrig", "errorBalanceDest",
    "abs_errorBalanceOrig", "abs_errorBalanceDest",
    "isOrigZeroAfter", "isOrigZeroBefore",
    "isDestZeroBefore", "isDestZeroAfter", "suspiciousWipe",
    "amountToOldBalanceOrig", "amountToOldBalanceDest",
    "origBalanceChangeRatio",
    "isTransfer", "isCashOut", "isHighRiskType",
    "log_amount", "log_oldbalanceOrg", "log_newbalanceOrig",
    "log_oldbalanceDest", "log_newbalanceDest",
    "amountTier",
    "origAcctFrequency", "destAcctFrequency", "isMuleDestCandidate",
    "hour_of_day", "day_of_month", "isNightTime",
]

FEATURE_COLS = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    *NEW_FEATURES,
    *[c for c in df.columns if c.startswith("type_")],
]

df_model = df[FEATURE_COLS + ["isFraud"]].copy()

print(f"\nNew features added   : {len(NEW_FEATURES)}")
print(f"Final dataset shape  : {df_model.shape}")

df_model.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to → {OUTPUT_PATH} ")