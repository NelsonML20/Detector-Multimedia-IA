#!/bin/bash
# Inicia el backend en segundo plano
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Inicia el frontend Streamlit (usando el puerto asignado por Render)
streamlit run frontend/app_streamlit.py --server.port=$PORT --server.address=0.0.0.0
