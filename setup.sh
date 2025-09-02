#!/bin/bash
pip install -r requirements.txt
echo "1) Streamlit, 2) FastAPI"
read choice
if [ "$choice" == "1" ]; then streamlit run app/streamlit_app.py
elif [ "$choice" == "2" ]; then uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
fi