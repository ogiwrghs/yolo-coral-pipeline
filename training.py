from ultralytics import YOLO
import os

# === Set your dataset paths (change these to your actual paths) ===
train_images_path = "/path/to/train/images"
val_images_path   = "/path/to/val/images"

# === Generate YOLO data config ===
data_yaml_path = "data_config.yaml"
data_yaml_content = f"""
path: .
train: {train_images_path}
val: {val_images_path}
names:
  0: cat
"""
with open(data_yaml_path, "w") as f:
    f.write(data_yaml_content.strip())

# === Build checkpoint path dynamically from project and name ===
checkpoint_path = os.path.join(project, name, "weights", "last.pt")

# === Control resume training with a simple flag ===
RESUME_TRAINING = True
resume_mode = RESUME_TRAINING and os.path.exists(checkpoint_path)

# === Load model: resume from checkpoint if available, otherwise start fresh ===
model_path = checkpoint_path if resume_mode else "yolov8n.pt"  # Replace with your preferred model file if needed
model = YOLO(model_path)

# === Train the model ===
model.train(
    data=data_yaml_path,
    epochs=100,
    imgsz=512,
    batch=32,
    workers=8,
    patience=0,           # Set to 0 to disable early stopping (beware, patience only looks at accuracy, and is unaware of any improvements in loss)
    cache=False,
    single_cls=True,
    mosaic=1.0,
    mixup=0.2,
    degrees=6.0,          # Small rotations (in degrees) for augmentation
    translate=0.1,        # Shifts objects up to 10% of the image dimensions
    scale=0.5,            
    shear=0.1,            # Apply a shear transformation with a factor of 0.1
    flipud=0.1,           
    fliplr=0.5,           # Horizontal flip probability
    close_mosaic=30,      # Disables mosaic augmentation 30 epochs before training ends (with epochs=100, mosaic stops at epoch 70)
    multi_scale=True,     # Enable multi-scale training for better detection of objects of varied sizes
    project="cat_finder_project",
    name="v11n512",
    resume=resume_mode,
)
