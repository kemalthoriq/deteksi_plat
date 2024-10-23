import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Spesifikasikan path ke model yang sudah dilatih
model_path = os.path.join('segmentation', 'model.h5')

# Fungsi untuk memuat model
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        print("Model berhasil dimuat.")
        return model
    except OSError as e:
        print(f"Kesalahan saat memuat model: {e}")
        return None

# Fungsi untuk memprediksi segmentasi plat
def predict_plate_segmentation(model, image_path):
    # Membaca dan memproses gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gambar tidak ditemukan di path: {image_path}")
        return None
    
    resized_image = cv2.resize(image, (128, 128))  # Mengubah ukuran gambar sesuai input model
    resized_image = np.expand_dims(resized_image, axis=0)  # Menambahkan dimensi batch

    # Melakukan prediksi
    prediction = model.predict(resized_image)
    return prediction
