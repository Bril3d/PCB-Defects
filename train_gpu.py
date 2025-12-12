from ultralytics import YOLO
import torch

def main():
    print("🔥 STARTING GPU TRAINING ON RTX 4080!")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model = YOLO('yolov8n.pt')

    # Train with GPU optimization - FIXED FOR WINDOWS
    model.train(
        data='PCB-Defects--6/data.yaml',
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        workers=0,  # ← SET TO 0 FOR WINDOWS FIX!
        patience=10,
        save=True,
        name='circuitfix_ai_gpu',
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        cos_lr=True,
        amp=True,
        cache=False,
    )

    print("✅ GPU TRAINING COMPLETED!")

if __name__ == '__main__':
    main()