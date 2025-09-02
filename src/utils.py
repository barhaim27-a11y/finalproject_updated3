from pathlib import Path; import joblib,json
ROOT=Path(__file__).resolve().parents[1]; MODELS_DIR=ROOT/'models'; ASSETS_DIR=ROOT/'assets'
BEST_MODEL_PATH=MODELS_DIR/'best_model.joblib'; METRICS_PATH=ASSETS_DIR/'metrics.json'
def save_model(m,path=BEST_MODEL_PATH): joblib.dump(m,path)
def load_model(path=BEST_MODEL_PATH): return joblib.load(path)
def save_metrics(metrics,path=METRICS_PATH): open(path,"w").write(json.dumps(metrics,indent=4))
def load_metrics(path=METRICS_PATH): return json.load(open(path))