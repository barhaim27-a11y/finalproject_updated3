import os, json, joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from model_pipeline import evaluate_model, models  # נייבא את הפונקציות והמודלים מה-pipeline

# Paths
ROOT = Path(__file__).resolve().parents[0]
DATA_PATH = ROOT / "data" / "parkinsons.data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
LOG_PATH = ASSETS_DIR / "training_log.csv"

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["status", "name"])
y = df["status"]

# Run pipeline
from model_pipeline import run_pipeline  # ניצור פונקציה ב-model_pipeline שמחזירה best_model, metrics, results
best_model, best_model_name, metrics, results = run_pipeline(X, y)

# Save log
new_entry = {
    "date": datetime.now().isoformat(),
    "dataset": DATA_PATH.name,
    "best_model": best_model_name,
    "roc_auc": metrics["test"]["roc_auc"],
    "accuracy": metrics["test"]["accuracy"],
    "f1": metrics["test"]["f1"]
}
log_df = pd.DataFrame([new_entry])
log_df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

print("✅ Training finished and log updated:", LOG_PATH)
