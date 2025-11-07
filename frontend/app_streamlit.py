# Interfaz web Streamlit

import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AIShield", page_icon="🛡️")
st.title("🛡️ AIShield - Detección de Contenido Generado por IA")

st.write("Sube una imagen y el sistema te dirá si fue generada por Inteligencia Artificial o es real.")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Imagen subida", use_container_width=True)

    if st.button("🔍 Analizar imagen"):
        # Llamar al backend
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": image_bytes}
        )

        if response.status_code == 200:
            resultado = response.json()["resultado"]
            st.success(resultado)
        else:
            st.error("❌ Error al comunicarse con el servidor.")
