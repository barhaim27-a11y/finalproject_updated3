@echo off
pip install -r requirements.txt
echo 1) Streamlit  2) FastAPI
set /p choice="Enter choice: "
if "%choice%"=="1" (streamlit run app\streamlit_app.py) else (uvicorn app.api:app --reload --host 0.0.0.0 --port 8000)