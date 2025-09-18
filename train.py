from ultralytics import YOLO

def main():
    # --- CHOOSE A PRE-TRAINED MODEL ---
    # YOLOv8 comes in different sizes: n, s, m, l, x
    # 'yolov8s.pt' (small) is a great starting point - fast and effective.
    # The library will automatically download it on the first run.
    model = YOLO('yolov8s.pt')

    # --- TRAIN THE MODEL ---
    # The magic happens here!
    # This will start the training process using your configuration file.
    results = model.train(
        data='cyclone_dataset.yaml',  # Path to your dataset .yaml file
        epochs=50,                   # Number of training cycles. Start with 50, you can increase later.
        imgsz=128,                   # Image size. Your images are 128x128.
        batch=16,                    # Number of images to process at once. 16 is a safe start for your Mac.
        name='yolov8s_cyclone_run1'  # A custom name for this training run's output folder
    )

    print("Training finished!")
    print("Results saved to:", results.save_dir)

if __name__ == '__main__':
    main()