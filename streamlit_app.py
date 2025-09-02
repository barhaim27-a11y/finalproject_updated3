import streamlit as st,pandas as pd,joblib,json; from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MODELS_DIR=ROOT/'models'; ASSETS_DIR=ROOT/'assets'
model=joblib.load(MODELS_DIR/'best_model.joblib'); metrics=json.load(open(ASSETS_DIR/'metrics.json'))
st.title("🧠 Parkinson's Predictor"); st.json(metrics)