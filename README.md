# 🧠 YOLO Coral Pipeline

Train any YOLO model, convert to TFLite, quantize to INT8, and deploy on one or more Coral Edge TPUs — with real-time tracking, dual-inference threading, and a live web stream.

Supports all YOLO architectures (v5, v8, custom), and scales from Edge Dev Boards to multi-USB Coral setups.

---

## 🔧 Features

- Train any YOLO model using Ultralytics
- Quantize to TFLite INT8 using real images
- Deploy on Coral Edge TPU with dual interpreter threading
- Stream live webcam feed via Flask with tracking overlay
- FPS boost through parallel inference on two TPUs
- Lightweight tracker using MOSSE + KLT + NMS
- Built-in frame-skipping to control inference load

---

## 🖥️ Requirements

> Dependencies are grouped by use case.

### Training
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.1.14
