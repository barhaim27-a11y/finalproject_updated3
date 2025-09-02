import os, json, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# === Paths ===
ROOT = Path(__file__).resolve().parents[0]
DATA_PATH = ROOT / "data" / "parkinsons.data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# === Models dict ===
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "SVC": SVC(probability=True),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier()
}

# === Helper: evaluate model ===
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob)
    }

# === Main pipeline function ===
def run_pipeline(X, y):
    # Split Train/Val/Test (70/15/15)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = {}
    val_scores = {}

    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        results[name] = {"val": val_metrics}
        val_scores[name] = val_metrics["roc_auc"]

    # Pick best model by validation ROC-AUC
    best_model_name = max(val_scores, key=val_scores.get)
    best_model = models[best_model_name]

    # Evaluate best model on train and test
    metrics = {
        "train": evaluate_model(best_model, X_train, y_train),
        "test": evaluate_model(best_model, X_test, y_test),
    }

    # Save best model & metrics
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    with open(ASSETS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save all models' results (train/val/test)
    for name, model in models.items():
        results[name]["train"] = evaluate_model(model, X_train, y_train)
        results[name]["test"] = evaluate_model(model, X_test, y_test)

    with open(ASSETS_DIR / "results_all_models.json", "w") as f:
        json.dump(results, f, indent=4)

    return best_model, best_model_name, metrics, results

# === Run standalone ===
if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["status", "name"])
    y = df["status"]
    best_model, best_model_name, metrics, results = run_pipeline(X, y)
    print("✅ Training complete")
    print("⭐ Best model:", best_model_name)
    print("📊 Test ROC-AUC:", metrics["test"]["roc_auc"])
