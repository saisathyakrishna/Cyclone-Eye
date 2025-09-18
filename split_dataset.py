import os
import glob
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
# The main directory containing your 'train' folder
DATA_DIR = 'datasets/cyclone-eye'
# The percentage of data to move to the validation set (e.g., 0.2 for 20%)
VALIDATION_SPLIT = 0.2

# --- Main Script ---
if __name__ == '__main__':
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'valid')

    # Create the validation directories
    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)

    # Get a list of all image files in the training directory
    # The glob pattern '*.jpg' will find all files ending with .jpg
    image_files = glob.glob(os.path.join(train_dir, 'images', '*.jpg'))

    # Shuffle the list for a random split
    random.shuffle(image_files)

    # Calculate the number of files to move
    num_to_move = int(len(image_files) * VALIDATION_SPLIT)

    # Select the files to move
    files_to_move = image_files[:num_to_move]

    print(f"Found {len(image_files)} total images.")
    print(f"Moving {len(files_to_move)} images (and their labels) to the validation set...")

    # Move the selected image and its corresponding label file
    for img_path in tqdm(files_to_move, desc="Moving files"):
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Construct the corresponding label file path
        label_filename = base_filename + '.txt'
        label_path = os.path.join(train_dir, 'labels', label_filename)

        # Move the image
        shutil.move(img_path, os.path.join(valid_dir, 'images'))

        # Move the label file, if it exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(valid_dir, 'labels'))

    print("\n✅ Dataset splitting complete.")