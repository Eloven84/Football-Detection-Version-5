from ultralytics import YOLO

# DATASET = "Football Dataset with Augmentation Version 1 (YOLO)/data.yaml"
DATASET = "E:/FOLDER TUGAS AKHIR 2/Football2/Dataset/Football Dataset with Augmentation Version 1 (YOLO)/data.yaml"

# Load model YOLO
# model = YOLO("yolo12m.pt")
model = YOLO("yolo12l.pt")

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="dataset_football_image/data.yaml", epochs=100, imgsz=640)

# # Run inference with the YOLO12n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")

model.train(
    data=DATASET,
    epochs=150,
    imgsz=640,
    batch=4,
    workers=0,
    device=0,
    #amp=False,
    # name="yolov12l-dataset2-600",
    project="models/YOLO Result Dataset With Augmentation Version 1",
)