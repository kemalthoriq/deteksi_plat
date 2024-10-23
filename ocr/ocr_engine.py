import pytesseract  # Pastikan Anda mengimpor pustaka yang diperlukan
import cv2

def extract_license_plate_text(image_path):
    print(f"Mengolah gambar: {image_path}")  # Debugging

    # Contoh memuat gambar menggunakan OpenCV atau PIL
    # Gantilah dengan cara yang sesuai untuk memuat gambar
    image = cv2.imread(image_path)  # Misalnya menggunakan OpenCV

    # Gunakan Tesseract untuk mendeteksi teks dari gambar
    plate_text = pytesseract.image_to_string(image, config='--psm 6')
    print(f"Teks yang terdeteksi: {plate_text}")  # Debugging
    return plate_text.strip()
