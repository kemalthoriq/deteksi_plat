# segmentation/utils.py
import cv2
import numpy as np
from keras.models import load_model

# Load the model once to save time
model = load_model('segmentation/model.h5')

def predict_license_plate(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))  # Sesuaikan ukuran ini dengan model Anda
    img_array = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_array)
    # Proses lebih lanjut untuk mendeteksi plat nomor dari prediksi
    detected_plate = "XYZ 1234"  # Ganti dengan logika deteksi yang sebenarnya
    
    return detected_plate
