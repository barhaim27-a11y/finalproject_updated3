import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, shap, os, io
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# === נתיבים בסיסיים ===
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "assets" / "metrics.json"
RESULTS_PATH = ROOT / "assets" / "results_all_models.json"
DATA_PATH = ROOT / "data" / "parkinsons.data"
LOG_PATH = ROOT / "assets" / "training_log.csv"

# === קונפיגורציית עמוד ===
st.set_page_config(page_title="Parkinson's App", layout="wide")
st.title("🧠 Parkinson's Disease Predictor")

# === טעינת מודל ומטריקות ===
model, metrics, df = None, None, None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

# === Tabs ===
tabs = st.tabs(["📊 Data & EDA", "🤖 Models", "🔮 Prediction", "⚡ Train New Model"])

# -------------------------
# 📊 Tab 1 – Data & EDA
# -------------------------
with tabs[0]:
    st.header("📊 Exploratory Data Analysis (EDA)")

    if df is not None:
        st.subheader("סטטיסטיקות כלליות")
        st.dataframe(df.describe())
        csv_stats = df.describe().to_csv().encode("utf-8")
        st.download_button("⬇️ הורד סטטיסטיקות", csv_stats, "stats.csv", "text/csv")

        # Boxplots
        st.subheader("Boxplots לפי סטטוס")
        num_cols = [c for c in df.columns if c not in ["status","name"]]
        for col in num_cols[:6]:  # להציג חלק לדוגמה
            fig, ax = plt.subplots()
            sns.boxplot(x="status", y=col, data=df, palette="Set2", ax=ax)
            st.pyplot(fig)

        # Histograms + KDE
        st.subheader("התפלגות פיצ'רים")
        for col in num_cols[:6]:  # חלק לדוגמה
            fig, ax = plt.subplots()
            sns.histplot(df, x=col, hue="status", kde=True, element="step", palette="Set1", ax=ax)
            st.pyplot(fig)

        # Feature Importance (אם יש במודל)
        st.subheader("Feature Importance")
        if model is not None:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_[0]
            else:
                importances = None
            if importances is not None:
                fi_df = pd.DataFrame({"feature":num_cols,"importance":importances})
                fi_df = fi_df.sort_values("importance", ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(x="importance", y="feature", data=fi_df.head(10), palette="viridis", ax=ax)
                st.pyplot(fig)

        # Correlation
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        corr = df.drop(columns=["name"]).corr()
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # PCA + t-SNE
        st.subheader("PCA & t-SNE")
        comps = PCA(n_components=2).fit_transform(df[num_cols])
        tsne = TSNE(n_components=2, random_state=42).fit_transform(df[num_cols])
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=comps[:,0], y=comps[:,1], hue=df["status"], palette="coolwarm", ax=ax)
            ax.set_title("PCA 2D")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=df["status"], palette="coolwarm", ax=ax)
            ax.set_title("t-SNE 2D")
            st.pyplot(fig)
    else:
        st.error("❌ לא נמצא דאטה")

# -------------------------
# 🤖 Tab 2 – Models
# -------------------------
with tabs[1]:
    st.header("🤖 Models – Training & Comparison")

    if metrics is not None:
        # KPIs
        st.subheader("🏆 Model KPIs (Test Set)")
        try:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['test']['accuracy']:.2f}")
            col2.metric("Precision", f"{metrics['test']['precision']:.2f}")
            col3.metric("Recall", f"{metrics['test']['recall']:.2f}")
            col4.metric("F1", f"{metrics['test']['f1']:.2f}")
            col5.metric("ROC-AUC", f"{metrics['test']['roc_auc']:.2f}")
        except:
            st.warning("⚠️ No test split found in metrics.json")

        st.markdown("---")

        # Model Zoo
        st.subheader("🧩 Model Zoo")
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH,"r") as f:
                results_all = json.load(f)
            results_df = pd.DataFrame(results_all).T
            st.dataframe(results_df)

            chosen_model = st.selectbox("בחר מודל:", results_df.index)
            if chosen_model and model is not None:
                st.markdown(f"### 🔍 {chosen_model}")
                try:
                    y_pred = model.predict(X_test)
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax, cmap="Blues", colorbar=False)
                    st.pyplot(fig)
                except:
                    st.warning("⚠️ Confusion Matrix לא זמין")
                try:
                    fig, ax = plt.subplots()
                    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
                    st.pyplot(fig)
                except:
                    st.warning("⚠️ ROC Curve לא זמין")
        else:
            st.error("❌ results_all_models.json לא נמצא")
    else:
        st.error("❌ metrics.json לא נמצא")

# -------------------------
# 🔮 Tab 3 – Prediction
# -------------------------
with tabs[2]:
    st.header("🔮 Prediction")

    if model is not None:
        features = [c for c in df.columns if c not in ["status","name"]]

        mode = st.radio("Prediction Mode:", ["ידני", "CSV"])
        green_thr, yellow_thr = 0.3, 0.6

        def classify_risk(prob, thr=0.5):
            if prob < green_thr:
                return "🟢 Low Risk", f"הסתברות {prob*100:.1f}%, מתחת לסף {thr*100:.0f}%."
            elif prob < yellow_thr:
                return "🟡 Medium Risk", f"הסתברות {prob*100:.1f}%, קרוב לסף {thr*100:.0f}%."
            else:
                return "🔴 High Risk", f"הסתברות {prob*100:.1f}%, מעל לסף {thr*100:.0f}%."

        if mode == "ידני":
            user_input = {f: st.number_input(f, value=0.0) for f in features}
            if st.button("בצע חיזוי"):
                sample = pd.DataFrame([user_input])
                pred = model.predict(sample)[0]
                prob = model.predict_proba(sample)[0][1]
                label, explanation = classify_risk(prob)
                st.subheader("🔮 תוצאה")
                st.write(label)
                st.progress(int(prob*100))
                st.info(f"🧾 {explanation}")
                if st.button("הסבר תחזית"):
                    explainer = shap.Explainer(model, df[features])
                    shap_values = explainer(sample)
                    shap.plots.bar(shap_values[0], max_display=10, show=False)
                    st.pyplot(plt.gcf())
        else:
            uploaded = st.file_uploader("העלה CSV", type=["csv"])
            if uploaded:
                data = pd.read_csv(uploaded)
                probs = model.predict_proba(data)[:,1]
                preds = (probs >= 0.5).astype(int)
                labels, texts = [], []
                for p in probs:
                    l, t = classify_risk(p)
                    labels.append(l)
                    texts.append(t)
                data["prediction"] = preds
                data["probability"] = probs
                data["risk_label"] = labels
                data["decision_text"] = texts
                st.dataframe(data.head())
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ CSV", csv, "predictions.csv", "text/csv")
                output = io.BytesIO()
                data.to_excel(output, index=False, engine="openpyxl")
                st.download_button("⬇️ XLSX", output.getvalue(), "predictions.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error("❌ לא נמצא מודל")

# -------------------------
# ⚡ Tab 4 – Train New Model
# -------------------------
with tabs[3]:
    st.header("⚡ Train New Model")

    uploaded_new = st.file_uploader("📂 העלה דאטה חדש (CSV)", type=["csv"])
    if uploaded_new:
        df_new = pd.read_csv(uploaded_new)
        st.write("📊 דגימה מהדאטה:")
        st.dataframe(df_new.head())

        if st.button("🔄 אימון מחדש"):
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression

            X = df_new.drop(columns=["status","name"])
            y = df_new["status"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            new_model = LogisticRegression(max_iter=1000)
            new_model.fit(X_train,y_train)
            y_pred = new_model.predict(X_test)
            y_prob = new_model.predict_proba(X_test)[:,1]

            metrics_new = {
                "accuracy": accuracy_score(y_test,y_pred),
                "precision": precision_score(y_test,y_pred),
                "recall": recall_score(y_test,y_pred),
                "f1": f1_score(y_test,y_pred),
                "roc_auc": roc_auc_score(y_test,y_prob)
            }

            joblib.dump(new_model, ROOT/"models"/"best_model_new.joblib")
            with open(ROOT/"assets"/"metrics_new.json","w") as f: json.dump(metrics_new,f,indent=4)

            st.success("✅ אימון חדש הושלם")

            # עדכון לוג
            new_entry = {"date": datetime.now().isoformat(), "model":"LogReg", **metrics_new}
            pd.DataFrame([new_entry]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

            if st.button("🚀 Promote New Model"):
                import shutil
                shutil.copy(ROOT/"models"/"best_model_new.joblib", MODEL_PATH)
                shutil.copy(ROOT/"assets"/"metrics_new.json", METRICS_PATH)
                st.success("🚀 המודל החדש קודם בהצלחה!")

    # הצגת Training Log
    if LOG_PATH.exists():
        st.subheader("📜 Training Log")
        log_df = pd.read_csv(LOG_PATH)
        st.dataframe(log_df.tail(10))
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ הורד Training Log", csv, "training_log.csv", "text/csv")
