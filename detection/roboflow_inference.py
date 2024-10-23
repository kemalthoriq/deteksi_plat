from inference_sdk import InferenceHTTPClient

def detect_plate_via_roboflow(image_path):
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="LmkAH7VZRUvlzj8KKgsC"
    )

    try:
        result = client.infer(image_path, model_id="dataset-plat-nomor/5")
        print(f"Hasil inferensi dari Roboflow: {result}")
        return result
    except Exception as e:
        print(f"Terjadi kesalahan saat inferensi: {e}")
        return None
