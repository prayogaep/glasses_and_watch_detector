import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model  pre-trained
model_path = 'models/glasses_and_watch.pt'  # Path ke model lokal
model = YOLO(model_path)  # Masukan ke model yolo

# Fungsi untuk mendeteksi objek dari gambar
def detect_objects(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi objek
    results = model(img_rgb)

    # Tampilkan hasil
    for result in results:
        # Menampilkan gambar dengan bounding box
        result.show()
        
        # Menyimpan hasil deteksi ke folder "runs/detect/"
        result.save()

# Uji deteksi objek pada gambar input
if __name__ == "__main__":
    # Path gambar yang ingin diuji
    image_path = 'datasets/train/images/1f24dc1a8a956b3614df291522182290.jpg'  # Ganti dengan gambar yang sesuai
    detect_objects(image_path)