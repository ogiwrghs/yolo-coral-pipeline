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
   Edit the training script (`training/training.py`) by setting the dataset paths and defining your project and iteration name. This is an all-in-one solution, no need to modify files outside of the script. (If you plan to train a multi-class network, set `single_cls` to `False`.)
   

2. **Run Training:**  
   to initiate your model's training, execute the training script named training.py in the folder training, with the following command:
   ```bash
   python3 training/training.py
   ```
   
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

   d. **Configure the Quantization Script**
      Edit the quantization script (`quantization/int8quantizer.py`) by selecting the dataset you with to use for sampling during the quantization
   (usually the **validation** dataset used during training is selected),
    along with the paths to your **trained model**, and the folder you wish your output to be in (usually the source weights folder)

   e. **Quantize the Model**
      run the following command to quantize:
   ```python
   python3 quantization/int8quantizer.py
    ```

   f. **Run the Quantized Model through the Edgetpu Compiler**
   ```python
   python3 edgetpu_compiler weights/best_int8.tflite
    ```
   
   
4. **Perform Inference on the Finished Model**
   after transferring the recently converted model, as well as the inference script on your device of choice activate the inference script, the terminal output will guide you from there.

   When you examine the output from the Edge TPU Compiler, you may observe that certain operations in the model **"fallback"** from      the TPU to the CPU. Since these fallback operations are executed in a **single-threaded manner**, they can create a performance       bottleneck. To alleviate this issue, the inference script launches two instances of the model simultaneously. These instances     process incoming frames in an **alternating fashion**, thereby partially mitigating the single-thread bottleneck and improving        overall inference speed on Coral devices.

   In addition, to maintain continuous tracking between these less frequent inference calls, the script employs a combination of     **MOSSE**-based correlation filters and **KLT** optical flow. On frames where the neural network is not invoked, the MOSSE tracker        quickly predicts object positions, while KLT optical flow refines bounding box alignment by tracking feature points within        each region. This technique bridges the performance gap left by single-threaded fallback, ensuring smoother and more              consistent object tracking throughout the live stream.

   The bounding box estimation can be disabled within the script, this will also disable the non-inference state, meaning only frames that went through inference will be shown. As a result the framerate wil be lower. 

   
   
