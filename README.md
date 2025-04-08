# YOLO Coral Pipeline

Train any YOLO model, convert to TFLite, quantize to INT8, and deploy on one or more Coral Edge TPUs — with real-time tracking, dual-inference threading, and a live web stream.

Supports all YOLO architectures (v5, v8, v11, custom), and scales from Edge Dev Boards to multi-USB Coral setups.

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

## Requirements

- A version of PyTorch that supports your hardware (ensure you have the proper backend, e.g. CUDA or ROCm)
- a relatively recent version of tensorflow
- The basic Ultralytics package (the auto-updater will fetch other required components)
- The Edge TPU Compiler installation instructions are available [here](https://coral.ai/docs/edgetpu/compiler/)

---

## Step-by-Step Instructions

1. **Configure the Training Script:**  
   Edit the training script (`training/train.py`) by setting the dataset paths and defining your project and iteration name. This is an all-in-one solution, no need to modify files outside of the script. (If you plan to train a multi-class network, set `single_cls` to `False`.)
   

2. **Run Training:**  
   to initiate your model's training, execute the training script named training.py in the folder training, with the following command: python3 training/train.py
   
3. **Export the Model**

   a. **Navigate to the weights folder:**  
      Open a terminal and change directory to your weights folder. For example:
      ```bash
      cd training/weights
      ```

   b. **Open a Python Shell:**  
      Launch a Python shell by typing:
      ```bash
      python3
      ```
      
   c. **Run the Export Commands:**  
      Execute these commands to enter the python shell and export the model:
      ```python
      python3
      from ultralytics import YOLO
      model = YOLO('best.pt')  
      model.export(format='tflite', imgsz=512)  # Adjust imgsz based on your trained model's image size
      ```

   
