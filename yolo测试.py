from sympy import false
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=r"C:\Users\52953\Desktop\yolo_dataset\data.yaml", epochs=100, imgsz=512)