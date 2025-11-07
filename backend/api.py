# API para predicciones (FastAPI) 

from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from modelo.detector import build_model

app = FastAPI(title=" Detector de Contenido multimedia generado por IA")

# Cargar el modelo y sus pesos
model = build_model()
model.load_weights("modelo/model_weights.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y predice si es REAL o GENERADA POR IA.
    """
    contents = await file.read()
    img = tf.image.decode_image(contents, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)[0][0]
    porcentaje = float(pred * 100)

    if pred > 0.5:
        result = f"🧠 Contenido generado por IA ({porcentaje:.2f}%)"
    else:
        result = f"📸 Imagen real ({100 - porcentaje:.2f}%)"

    return {"resultado": result}
