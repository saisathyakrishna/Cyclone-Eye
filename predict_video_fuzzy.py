import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO
import cv2

# --- 1. Helper function to extract features ---
def get_features(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    eye_region = image[y1:y2, x1:x2]
    eyewall_radius_inner = max((x2 - x1) // 2, (y2 - y1) // 2)
    eyewall_radius_outer = int(eyewall_radius_inner * 1.8)
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), eyewall_radius_outer, 255, -1)
    cv2.circle(mask, (center_x, center_y), eyewall_radius_inner, 0, -1)
    eyewall_region = image[mask == 255]

    if eye_region.size == 0 or eyewall_region.size == 0: return None, None

    contrast = np.mean(eye_region) - np.mean(eyewall_region)
    eyewall_definition = 1.0 / (1.0 + np.std(eyewall_region))
    
    return contrast, eyewall_definition

# --- 2. Define the Fuzzy Logic System ---
contrast = ctrl.Antecedent(np.arange(-50, 101, 1), 'contrast')
eyewall_def = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'eyewall_def')
threat = ctrl.Consequent(np.arange(0, 11, 1), 'threat')

contrast.automf(names=['Low', 'Medium', 'High'])
eyewall_def.automf(names=['Poor', 'Good', 'Excellent'])
threat.automf(names=['Low', 'Moderate', 'High'])

rule1 = ctrl.Rule(contrast['Low'] | eyewall_def['Poor'], threat['Low'])
rule2 = ctrl.Rule(contrast['Medium'] & eyewall_def['Good'], threat['Moderate'])
rule3 = ctrl.Rule(contrast['High'] & eyewall_def['Excellent'], threat['High'])

threat_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
threat_simulation = ctrl.ControlSystemSimulation(threat_ctrl)

# --- 3. Main Video Processing Script ---
if __name__ == '__main__':
    MODEL_PATH = 'runs/detect/train/weights/best.pt'
    VIDEO_PATH = 'video.mp4' # UPDATE THIS

    model = YOLO(MODEL_PATH)
    results = model(VIDEO_PATH, stream=True)

    for result in results:
        annotated_frame = result.plot()
        threat_text = "Threat Level: N/A"

        # Check if an eye was detected in the current frame
        if len(result.boxes) > 0:
            first_box = result.boxes[0]
            infrared_image = result.orig_img[:, :, 0]

            # Extract features and run fuzzy logic
            eye_contrast, eyewall_definition = get_features(infrared_image, first_box)
            if eye_contrast is not None:
                threat_simulation.input['contrast'] = eye_contrast
                threat_simulation.input['eyewall_def'] = eyewall_definition
                threat_simulation.compute()
                fuzzy_threat_level = threat_simulation.output['threat']
                threat_text = f"Threat Level: {fuzzy_threat_level:.2f} / 10"

        # Add the threat level text to the frame using OpenCV
        cv2.putText(
            annotated_frame,
            threat_text,
            (10, 30), # Position (top-left corner)
            cv2.FONT_HERSHEY_SIMPLEX,
            1, # Font scale
            (0, 255, 0), # Text color in BGR (green)
            2 # Thickness
        )

        cv2.imshow("Fuzzy Cyclone Prediction", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    print("\n✅ Video processing complete.")