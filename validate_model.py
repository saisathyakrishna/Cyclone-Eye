from ultralytics import YOLO

# Load your custom-trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run validation on the dataset specified in your data.yaml file
# The split='val' argument ensures it uses the validation set.
metrics = model.val(split='val')

# The 'metrics' object contains all the performance data
print("\n--- Validation Metrics ---")
# mAP at IoU threshold 0.5 (the most common one)
print(f"  - mAP50: {metrics.box.map50:.4f}")
# Precision for your 'eye' class
print(f"  - Precision: {metrics.box.p[0]:.4f}")
# Recall for your 'eye' class
print(f"  - Recall: {metrics.box.r[0]:.4f}")