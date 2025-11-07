import os
import tensorflow as tf
import matplotlib.pyplot as plt
from detector import build_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train_and_save_model():
    # 📂 Obtener la ruta base del archivo actual
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 🧭 Construir rutas seguras
    train_dir = os.path.join(BASE_DIR, '..', '..', 'dataset', 'train')
    test_dir = os.path.join(BASE_DIR, '..', '..', 'dataset', 'test')

    # ✅ Verificar carpetas
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"No se encontró la carpeta de entrenamiento: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"No se encontró la carpeta de prueba: {test_dir}")

    print(f"📁 Carpeta de entrenamiento: {train_dir}")
    print(f"📁 Carpeta de prueba: {test_dir}")

    # ⚙️ Parámetros generales
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS_PER_STAGE = 2   # Cuántas épocas por etapa
    STAGES = 3             # Cuántas etapas entrenará
    MAX_IMAGES_PER_STAGE = 5000  # Limita cuántas imágenes usa por etapa

    # 🧠 Crear modelo
    model = build_model()

    # 📦 Ruta para guardar pesos
    weights_path = os.path.join(BASE_DIR, "model_weights.h5")

    # 🔄 Cargar pesos si existen
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
        print("🔄 Cargando pesos previos del modelo...")
        try:
            model.load_weights(weights_path)
        except Exception:
            print("⚠️ Archivo de pesos corrupto, se reiniciará desde cero.")
    else:
        print("🆕 No se encontraron pesos previos. Se entrenará desde cero.")

    # 💾 Callbacks
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    # ⚙️ Verificar GPU disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detectada: {gpus[0].name}")
    else:
        print("💡 No se detectó GPU, se entrenará con CPU.")

    # 🚀 Entrenamiento por etapas
    for stage in range(1, STAGES + 1):
        print(f"\n🚀 Iniciando etapa {stage}/{STAGES}...")

        # Dataset de entrenamiento con límite de imágenes
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset='training',
            seed=stage * 42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary'
        )

        # Limitar la cantidad de lotes (para no usar todas las imágenes)
        max_batches = MAX_IMAGES_PER_STAGE // BATCH_SIZE
        train_ds = train_ds.take(max_batches)

        # Dataset de validación
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset='validation',
            seed=stage * 42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary'
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        # Entrenar por etapa
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_PER_STAGE,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )

    # ✅ Evaluar con dataset de prueba
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n📈 Precisión final en TEST: {test_acc * 100:.2f}%")

    # 📉 Graficar resultados
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_save_model()


