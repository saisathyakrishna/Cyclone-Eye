from ultralytics import YOLO

# 1. Load a pre-trained model
# We use a pre-trained model (yolov8n.pt) for transfer learning.
# This is much faster and more effective than training from scratch.
# 'yolov8n.pt' is the smallest and fastest version, great for starting.
model = YOLO('yolov8n.pt')

# 2. Train the model on your custom dataset
# The 'data' argument points to your data.yaml file, which tells YOLO
# where to find your training and validation images and labels.
# 'epochs' is the number of times the model will see the entire dataset.
# 'imgsz' is the size images will be resized to for training.
results = model.train(data='data.yaml', epochs=50, imgsz=640)

# 3. Print a confirmation message
print("\n✅ Training complete.")
print("Your trained model and results are saved in the 'runs/detect/' directory.")