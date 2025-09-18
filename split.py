import os
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
base_dir = 'datasets/cyclone-eye/'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
split_ratio = 0.20 # 20% for validation

# --- Create validation directories ---
os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)

# --- Get all image files and shuffle them ---
image_files = [f for f in os.listdir(os.path.join(train_dir, 'images')) if f.endswith(('.png', '.jpg'))]
random.shuffle(image_files)

# --- Calculate the split index ---
split_index = int(len(image_files) * split_ratio)
files_to_move = image_files[:split_index]

print(f"Total images: {len(image_files)}")
print(f"Moving {len(files_to_move)} images to the validation set...")

# --- Move the files ---
for filename in tqdm(files_to_move, desc="Moving files"):
    # Move the image
    src_img = os.path.join(train_dir, 'images', filename)
    dst_img = os.path.join(valid_dir, 'images', filename)
    shutil.move(src_img, dst_img)

    # Move the corresponding label file
    label_filename = os.path.splitext(filename)[0] + '.txt'
    src_lbl = os.path.join(train_dir, 'labels', label_filename)
    dst_lbl = os.path.join(valid_dir, 'labels', label_filename)

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print("\n✅ Dataset split complete.")