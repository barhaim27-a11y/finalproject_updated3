import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

# === הגדרת נתיבים עם fallback ===
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("assets/metrics.json")

if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "models" / "best_model.joblib"
if not METRICS_PATH.exists():
    METRICS_PATH = ROOT / "assets" / "metrics.json"

# === כותרת ראשית ===
st.title("🧠 Parkinson's Disease Predictor")

# === טעינת המודל והמטריקות ===
try:
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    st.subheader("✅ Model Loaded Successfully")
    st.json(metrics)

    st.markdown("---")
    st.info("📌 Ready for predictions – UI for manual/CSV input can be added here.")

except FileNotFoundError:
    st.error("❌ Model or metrics not found.\n\nPlease run the training notebook first to generate them.")
