# 🔍 CircuitFix — AI-Powered PCB Defect Detection & Repair Platform

CircuitFix is an intelligent web application that uses **YOLOv8/YOLOv11** deep learning models to automatically detect and classify defects on Printed Circuit Boards (PCBs). Beyond detection, it provides step-by-step repair tutorials and a community forum for hardware maintainers.

---

## ✨ Key Features

### 🤖 AI-Powered Defect Detection
- **Real-time PCB analysis** using custom-trained YOLOv8 and YOLOv11 object detection models
- **6 defect categories** detected: Missing Hole, Open Circuit, Short Circuit, Mouse Bite, Spur, Spurious Copper
- **Confidence scoring** with adjustable thresholds (conf=0.5, IoU=0.45)
- **Bounding box visualization** on uploaded PCB images

### 🛠️ Guided Repair System
- **Step-by-step repair tutorials** for each detected defect type
- **Severity classification** (Low / Medium / High / Critical)
- **Tool requirements**, time estimates, and cost estimates for each repair
- **Knowledge base** with detailed repair procedures

### 💬 Community Forum
- Discussion threads for hardware maintainers
- Category-based post organization
- Reply system with view tracking

### 📊 Analytics Dashboard
- Defect analysis history tracking
- Per-user analysis records
- Exportable PDF reports

### 🔐 User Management
- Secure authentication (registration/login)
- Password hashing with Werkzeug
- Session-based access control

---

## 🏗️ Architecture

```
PCB-Defects/
├── app.py                 # Flask application (routes, API, AI inference)
├── train.py               # YOLOv8 training script (CPU)
├── train_gpu.py            # YOLOv8 training script (GPU - RTX optimized)
├── download.py             # Roboflow dataset downloader
├── circuitfix.db           # SQLite database
├── yolov8n.pt              # YOLOv8 Nano pre-trained weights
├── yolo11n.pt              # YOLOv11 Nano pre-trained weights
└── templates/
    ├── index.html          # Landing page
    ├── dashboard.html      # Main dashboard with upload & detection
    ├── analysis.html       # Detailed analysis results
    ├── analytics.html      # Analytics & statistics
    ├── knowledge.html      # Repair knowledge base
    ├── forum.html          # Community forum
    ├── forum_new.html      # New forum post
    ├── forum_thread.html   # Forum thread view
    ├── login.html          # Authentication
    ├── register.html       # User registration
    └── profile.html        # User profile
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Bril3d/PCB-Defects.git
cd PCB-Defects
```

### 2. Install dependencies
```bash
pip install flask opencv-python numpy ultralytics fpdf werkzeug roboflow
```

### 3. Download the dataset (optional — for training)
```bash
python download.py
```

### 4. Train the model (optional)
```bash
# CPU training
python train.py

# GPU training (NVIDIA GPU required)
python train_gpu.py
```

### 5. Run the application
```bash
python app.py
```
Open your browser at `http://localhost:5000`

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Base Model** | YOLOv8 Nano / YOLOv11 Nano |
| **Dataset** | [PCB Defects (Roboflow)](https://universe.roboflow.com/university-2xdiy/pcb-defects-chi1b) |
| **Training Epochs** | 50 |
| **Image Size** | 640×640 |
| **Batch Size** | 32 (GPU) |
| **Defect Classes** | 6 (missing_hole, open_circuit, short, mouse_bite, spur, spurious_copper) |

---

## 🔧 Tech Stack

- **AI/ML:** Ultralytics YOLOv8/v11, OpenCV, NumPy
- **Backend:** Flask, SQLite, Werkzeug
- **Frontend:** HTML/CSS/JavaScript (Jinja2 templates)
- **Dataset:** Roboflow
- **Reports:** FPDF (PDF generation)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
