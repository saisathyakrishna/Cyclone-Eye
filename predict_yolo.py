from ultralytics import YOLO

# --- Configuration ---
# Path to your newly trained model weights
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# Path to a sample image to test (e.g., from your validation set)
TEST_IMAGE_PATH = 'datasets/cyclone-eye/valid/images/110806_US_JPG.rf.ca1278322bd809ae3573389db74b4864.jpg'

# --- Main Script ---
if __name__ == '__main__':
    print(f"Loading model from: {MODEL_PATH}")
    # Load your custom-trained model
    model = YOLO(MODEL_PATH)

    print(f"Running prediction on: {TEST_IMAGE_PATH}")
    # Run prediction on the test image
    results = model(TEST_IMAGE_PATH)

    # The results object contains the detections.
    # We can save the output image with the bounding boxes drawn on it.
    output_filename = 'prediction.jpg'

    # Access the first result object and save it
    if results:
        results[0].save(filename=output_filename)
        print(f"\n✅ Prediction complete. Output saved to '{output_filename}'")
    else:
        print("No results were generated.")