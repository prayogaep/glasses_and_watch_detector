import cv2
import numpy as np
from ultralytics import YOLO

# Load model YOLOv5 yang telah dilatih
model_path = 'models/glasses_and_watch.pt'  # Ganti dengan path model yang dilatih
model = YOLO(model_path)  # Load model yang telah dilatih

# Daftar nama kelas sesuai dengan model yang dilatih
class_names = ['WATCH', 'GLASSES']  # Ganti dengan daftar nama kelas Anda

# Fungsi untuk mendeteksi objek dari frame
def detect_objects(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi objek dengan threshold confidence
    results = model(img_rgb, conf=0.5)  # Hanya menampilkan deteksi dengan confidence > 0.5

    # Proses hasil deteksi
    for result in results:
        # Ambil bounding boxes dan label
        boxes = result.boxes.xyxy.numpy()  # Bounding boxes
        scores = result.boxes.conf.numpy()  # Confidence scores
        classes = result.boxes.cls.numpy()  # Class IDs

        # Gambar bounding box dan label pada gambar
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            label = f'{class_names[int(cls)]}, Conf: {score:.2f}'
            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Gambar label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Uji deteksi objek pada aliran video dari kamera
if __name__ == "__main__":
    # Buka kamera (0 untuk webcam default)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka.")
        exit()

    while True:
        # Baca frame dari kamera
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        # Deteksi objek pada frame
        frame_with_detections = detect_objects(frame)

        # Tampilkan hasil
        cv2.imshow('Deteksi Objek', frame_with_detections)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()