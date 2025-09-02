from fastapi import FastAPI; import joblib,json; from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MODELS_DIR=ROOT/'models'; ASSETS_DIR=ROOT/'assets'
model=joblib.load(MODELS_DIR/'best_model.joblib'); metrics=json.load(open(ASSETS_DIR/'metrics.json'))
app=FastAPI(title="Parkinson's API")
@app.get("/") def home(): return {"msg":"API running"}
@app.get("/metrics") def get_metrics(): return metrics