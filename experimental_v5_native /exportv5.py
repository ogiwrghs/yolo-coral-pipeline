import os


# remember to run this from inside the yolov5 clone folder
# requires edgetpu compiler installed on the machine

MODELPATH = 'runs/train/v5native/weights/best.pt'
DATAPATH = '/path/to/your/dataset/data.yaml'


cmd = f"python export.py --weights {MODELPATH} --include edgetpu --int8 --img 416 --data {DATAPATH}"

print("exporting to tflite and compiling...")
os.system(cmd)
