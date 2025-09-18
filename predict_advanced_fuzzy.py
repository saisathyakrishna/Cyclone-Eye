import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO
import cv2

def get_features(image, box):
    """Extracts advanced features from the image based on the bounding box."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 1. Define Eye and Eyewall Regions
    eye_region = image[y1:y2, x1:x2]
    
    # Define an eyewall region as a "donut" around the eye
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    eyewall_radius_inner = max((x2 - x1) // 2, (y2 - y1) // 2)
    eyewall_radius_outer = int(eyewall_radius_inner * 1.8) # Eyewall is a band outside the eye
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), eyewall_radius_outer, 255, -1)
    cv2.circle(mask, (center_x, center_y), eyewall_radius_inner, 0, -1)
    eyewall_region = image[mask == 255]

    if eye_region.size == 0 or eyewall_region.size == 0:
        return None, None

    # 2. Calculate Features
    avg_eye_temp = np.mean(eye_region)
    avg_eyewall_temp = np.mean(eyewall_region)
    std_dev_eyewall = np.std(eyewall_region)
    
    # Feature 1: Contrast (warm eye vs. cold eyewall)
    contrast = avg_eye_temp - avg_eyewall_temp
    
    # Feature 2: Eyewall Definition (lower std dev is better)
    # We invert it so a higher number is "better" for the fuzzy system
    eyewall_definition = 1.0 / (1.0 + std_dev_eyewall)
    
    return contrast, eyewall_definition

# --- Define the NEW Fuzzy Logic System ---
contrast = ctrl.Antecedent(np.arange(-50, 101, 1), 'contrast') # Range of pixel value differences
eyewall_def = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'eyewall_def')
threat = ctrl.Consequent(np.arange(0, 11, 1), 'threat')

# Membership Functions
contrast['Low'] = fuzz.trimf(contrast.universe, [-50, -50, 25])
contrast['Medium'] = fuzz.trimf(contrast.universe, [10, 35, 60])
contrast['High'] = fuzz.trimf(contrast.universe, [40, 100, 100])

eyewall_def['Poor'] = fuzz.trimf(eyewall_def.universe, [0, 0, 0.5])
eyewall_def['Good'] = fuzz.trimf(eyewall_def.universe, [0.2, 0.8, 1])
eyewall_def['Excellent'] = fuzz.trimf(eyewall_def.universe, [0.6, 1, 1])

threat['Low'] = fuzz.trimf(threat.universe, [0, 0, 4])
threat['Moderate'] = fuzz.trimf(threat.universe, [2, 5, 8])
threat['High'] = fuzz.trimf(threat.universe, [6, 10, 10])

# Fuzzy Rules
rule1 = ctrl.Rule(contrast['Low'] | eyewall_def['Poor'], threat['Low'])
rule2 = ctrl.Rule(contrast['Medium'] & eyewall_def['Good'], threat['Moderate'])
rule3 = ctrl.Rule(contrast['High'] & eyewall_def['Excellent'], threat['High'])
rule4 = ctrl.Rule(contrast['High'] | eyewall_def['Excellent'], threat['High'])

threat_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
threat_simulation = ctrl.ControlSystemSimulation(threat_ctrl)

# --- Main Prediction Script ---
if __name__ == '__main__':
    MODEL_PATH = 'runs/detect/train/weights/best.pt'
    TEST_IMAGE_PATH = 'datasets/cyclone-eye/valid/images/20122021_06_goc_JPG.rf.924ebcfbf3bf270e42ff7e62df1128ae.jpg' # UPDATE THIS
    
    model = YOLO(MODEL_PATH)
    results = model(TEST_IMAGE_PATH)

    if results and len(results[0].boxes) > 0:
        first_box = results[0].boxes[0]
        
        # We need the original image data for pixel analysis
        original_image_data = results[0].orig_img
        # Use the infrared channel for temperature analysis
        infrared_image = original_image_data[:, :, 0] 

        # Extract our new, smarter features
        eye_contrast, eyewall_definition = get_features(infrared_image, first_box)
        
        if eye_contrast is not None:
            # Pass new inputs to the fuzzy control system
            threat_simulation.input['contrast'] = eye_contrast
            threat_simulation.input['eyewall_def'] = eyewall_definition
            threat_simulation.compute()
            fuzzy_threat_level = threat_simulation.output['threat']

            print("\n--- YOLOv8 Prediction ---")
            print(f"  - Eye Detected: Yes (Confidence: {float(first_box.conf[0]):.2f})")
            
            print("\n--- Advanced Feature Analysis ---")
            print(f"  - Eye-Eyewall Contrast: {eye_contrast:.2f}")
            print(f"  - Eyewall Definition Score: {eyewall_definition:.2f}")

            print("\n--- Fuzzy Logic Analysis ---")
            print(f"  - Calculated Threat Level: {fuzzy_threat_level:.2f} / 10")
            
            results[0].save(filename='advanced_fuzzy_prediction.jpg')
            print("\n✅ Prediction image saved to 'advanced_fuzzy_prediction.jpg'")
        else:
            print("Could not analyze features for the detected box.")
    else:
        print("\nNo cyclone eye detected by YOLOv8 model.")