# train_model.py
import os
import sqlite3
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DB_PATH = "customer_churn.db"
TABLE_NAME = "customers"
MODEL_PATH = "models/churn_model.pkl"


def train_model():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ Database not found at {DB_PATH}")

    # ---------------------------
    # Load from DB
    # ---------------------------
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # ---------------------------
    # Features / Target
    # ---------------------------
    if "Churn" not in df.columns:
        raise ValueError("❌ 'Churn' column not found in dataset!")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------
    # Train baseline models
    # ---------------------------
    models = {
        "log_reg": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"accuracy": acc, "auc": auc}

        print(f"{name} → Accuracy={acc:.3f}, AUC={auc:.3f}")

        if auc > best_auc:
            best_model = (model, scaler, X.columns.tolist())
            best_auc = auc

    # ---------------------------
    # Save best model
    # ---------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Best model saved → {MODEL_PATH}")
    return results


if __name__ == "__main__":
    train_model()
