import numpy as np
import h5py
from PIL import Image
import os

# --- Configuration ---
# 1. DEFINE FILE PATHS
H5_FILE_PATH = 'Cyclone_Images_copy.h5' # IMPORTANT: Change this to your .h5 file path
NPY_FILE_PATH = 'Cyclone_Labels_copyh5.npy'  # IMPORTANT: Change this to your .npy file path

# 2. DEFINE OUTPUT FOLDER
OUTPUT_FOLDER = 'cyclone_dataset'

# --- Main Script ---

# Create the output directory if it doesn't exist
print(f"Creating output directory at: {OUTPUT_FOLDER}")
os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'labels'), exist_ok=True)

print("Loading data... This might take a moment for large files.")

# --- Load the H5 file (Images) ---
with h5py.File(H5_FILE_PATH, 'r') as hf:
    # --- Data Inspection (Optional but Recommended) ---
    # Use this part to find the correct key for your images
    print("Keys available in the H5 file:", list(hf.keys()))
    # Let's assume the key is 'images'. If it's different, change it below.
    # For example, if the key is 'X_train', change 'images' to 'X_train'
    IMAGE_DATASET_KEY = 'Images' # <--- CHANGE THIS IF YOUR KEY IS DIFFERENT
    
    images = hf[IMAGE_DATASET_KEY][:] # Use [:] to load all data into memory as a NumPy array

# --- Load the NPY file (Labels) ---
labels = np.load(NPY_FILE_PATH, allow_pickle=True)

# --- Sanity Check ---
# Ensure the number of images matches the number of labels
num_images = images.shape[0]
num_labels = labels.shape[0]

print(f"Loaded {num_images} images with shape {images.shape}")
print(f"Loaded {num_labels} labels with shape {labels.shape}")

if num_images != num_labels:
    print("FATAL ERROR: The number of images and labels do not match!")
    exit()

print("\nStarting extraction process...")

# --- Loop through each image and its corresponding label ---
for i in range(num_images):
    # Get the i-th image and i-th label
    image_array = images[i]
    label_data = labels[i]

    # --- Process and Save the Image ---
    # Check if image data is in float format (0.0 to 1.0) and convert to uint8 (0 to 255)
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        # Assuming float data is normalized between 0 and 1
        image_array = (image_array * 255).astype(np.uint8)

    # Convert the NumPy array to a Pillow Image object
    pil_image = Image.fromarray(image_array)
    
    # Define the filename for the image (e.g., image_0000.png, image_0001.png, etc.)
    # Using zfill or f-string padding helps in keeping files sorted correctly
    image_filename = f"cyclone_{i:05d}.png"
    image_save_path = os.path.join(OUTPUT_FOLDER, 'images', image_filename)
    
    # Save the image
    pil_image.save(image_save_path)

    # --- Save the Corresponding Label ---
    # We will save the label as a simple text file with the same name.
    # This is a common practice.
    label_filename = f"cyclone_{i:05d}.txt"
    label_save_path = os.path.join(OUTPUT_FOLDER, 'labels', label_filename)
    
    # Save the label data. Assuming label_data is a NumPy array like [x, y]
    # We'll save it as comma-separated values.
    # # Convert all items in the label array to strings
    string_label_data = [str(item) for item in label_data]
    # # Join them together with commas
    final_label_string = ",".join(string_label_data)

    with open(label_save_path, 'w') as f:
        f.write(final_label_string)
    # Print progress
    if (i + 1) % 100 == 0:
        print(f"Processed and saved {i + 1}/{num_images} files.")

print("\nExtraction complete!")
print(f"All images are saved in: {os.path.join(OUTPUT_FOLDER, 'images')}")
print(f"All labels are saved in: {os.path.join(OUTPUT_FOLDER, 'labels')}")