import os

# trains yolo from scratch so it learns relu 
# silu kills the edge tpu completely

DATAPATH = '/path/to/your/dataset/data.yaml'
EPOCHS = 100

cmd = f"python yolov5/train.py --img 416 --batch 16 --epochs {EPOCHS} --data {DATAPATH} --cfg yolov5n.yaml --weights '' --name v5native"

print("starting training run...")
os.system(cmd)
