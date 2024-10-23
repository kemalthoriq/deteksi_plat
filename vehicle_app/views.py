from django.shortcuts import render, redirect, get_object_or_404
from .models import VehicleImage
from segmentation.predict import predict_plate_segmentation
from ocr.ocr_engine import extract_license_plate_text
from segmentation.predict import load_trained_model
import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detection'))
from roboflow_inference import detect_plate_via_roboflow

model_path = os.path.join('segmentation', 'model.h5')
model = load_trained_model(model_path)

def draw_bounding_box(image_path, prediction):
    image = cv2.imread(image_path)

    # Ambil informasi bounding box dari prediksi
    x = int(prediction['x'])
    y = int(prediction['y'])
    width = int(prediction['width'])
    height = int(prediction['height'])

    # Hitung koordinat bounding box
    top_left = (x - width // 2, y - height // 2)
    bottom_right = (x + width // 2, y + height // 2)

    # Gambar bounding box di sekitar plat nomor
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Pastikan folder untuk menyimpan gambar ada
    processed_folder = os.path.join('media', 'processed_plates')
    os.makedirs(processed_folder, exist_ok=True)

    # Simpan gambar dengan bounding box
    processed_image_path = os.path.join(processed_folder, os.path.basename(image_path))
    cv2.imwrite(processed_image_path, image)

    return processed_image_path

def upload_image(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image']
        vehicle_image = VehicleImage.objects.create(image=uploaded_image)

        try:
            image_path = vehicle_image.image.path
            result = detect_plate_via_roboflow(image_path)

            if result and 'predictions' in result:
                predictions = result['predictions']
                
                if predictions:
                    first_prediction = predictions[0]
                    print(f"Prediksi Roboflow: {first_prediction}")

                    detected_class = first_prediction.get('class', 'Unknown')
                    confidence = first_prediction.get('confidence', 0)
                    vehicle_image.detected_plate = f"{detected_class} (Confidence: {confidence:.2f})"

                    processed_image_path = draw_bounding_box(image_path, first_prediction)

                    # Ekstraksi teks plat nomor
                    extracted_text = extract_license_plate_text(processed_image_path)
                    print(f"Teks yang diekstrak: {extracted_text}")  # Log untuk debugging
                    vehicle_image.detected_plate = extracted_text if extracted_text else "Tidak terdeteksi"

                    vehicle_image.processed_image = os.path.join('processed_plates', os.path.basename(processed_image_path))
                    vehicle_image.save()

                    return redirect('results', image_id=vehicle_image.id)

        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
            return render(request, 'upload.html', {'error': 'Kesalahan saat memproses gambar.'})

    return render(request, 'upload.html')


def results(request, image_id):
    vehicle_image = get_object_or_404(VehicleImage, id=image_id)
    context = {
        'image': vehicle_image.image.url,
        'processed_image': vehicle_image.processed_image.url if vehicle_image.processed_image else None,
        'detected_plate': vehicle_image.detected_plate,
    }
    return render(request, 'results.html', context)
