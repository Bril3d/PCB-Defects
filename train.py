from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='PCB-Defects--6/data.yaml',
    epochs=50,
    imgsz=640,
    name='circuitfix_ai'
)
print("✅ Training started!")

