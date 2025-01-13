from ultralytics import YOLO

# Load model
model = YOLO("models/yolov5nu.pt") 
# Train the model on your custom dataset
results = model.train(
    data="src/custom_dataset.yaml",     # Path to the data.yaml file
    epochs=100,                         # Number of training epochs
    imgsz=640,                          # Image size (resolution)
    batch=32,                           # Batch size
    name="watch_and_glasses_detector",  # Name for the trained model
    save=True,                          # Save the trained model
    workers=16,                         # Number of data-loading threads
)