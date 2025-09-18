from ultralytics import YOLO

def main():
    # --- LOAD YOUR TRAINED MODEL ---
    # Make sure the path points to your 'best.pt' file from the training run.
    model_path = 'runs/detect/yolov8s_cyclone_run1/weights/best.pt'
    model = YOLO(model_path)

    # --- EVALUATE THE MODEL ON THE TEST SET ---
    print(f"Loading model from {model_path}")
    print("Evaluating model performance on the test set...")
    
    # The 'split='test'' argument tells the model to use the test set
    # defined in your 'cyclone_dataset.yaml' file.
    metrics = model.val(split='test')

    # --- PRINT THE KEY METRICS ---
    print("\n--- Evaluation Metrics ---")
    print(f"mAP50-95 (Box): {metrics.box.map:.4f}") # Mean Average Precision
    print(f"mAP50 (Box): {metrics.box.map50:.4f}")   # mAP at IoU threshold of 0.50
    print(f"Precision (Box): {metrics.box.p[0]:.4f}")
    print(f"Recall (Box): {metrics.box.r[0]:.4f}")
    print("------------------------")

if __name__ == '__main__':
    main()