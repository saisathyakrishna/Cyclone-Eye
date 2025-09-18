from ultralytics import YOLO
import cv2

# --- Configuration ---
# Path to your newly trained model weights
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# Path to the video file you want to process
VIDEO_PATH = 'video.mp4'

# --- Main Script ---
if __name__ == '__main__':
    print(f"Loading model from: {MODEL_PATH}")
    # Load your custom-trained model
    model = YOLO(MODEL_PATH)

    print(f"Opening video file: {VIDEO_PATH}")
    # Run prediction on the video, with streaming for frame-by-frame processing
    # The 'stream=True' argument is crucial for video processing
    results = model(VIDEO_PATH, stream=True)

    # Loop through the results generator
    for result in results:
        # The 'plot()' method automatically draws the bounding boxes on the frame
        annotated_frame = result.plot()
        
        # Display the annotated frame in a window
        cv2.imshow("Cyclone Eye Prediction", annotated_frame)
        
        # Wait for 1 millisecond, and break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the window resources
    cv2.destroyAllWindows()
    print("\n✅ Video processing complete.")