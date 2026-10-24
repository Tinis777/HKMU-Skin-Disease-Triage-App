import os
import csv
import random
import shutil

# 1. Set paths based on the exact folder structure
source_img_folders = [
    "data/HAM10000/HAM10000_images_part_1",
    "data/HAM10000/HAM10000_images_part_2"
]
csv_path = "data/HAM10000/HAM10000_metadata.csv"
dest_folder = "data/batch_test/test_images"
sample_size = 50  # Sample 50 images per class (7 classes = 350 total)

# 2. Read CSV and group images by class (dx)
class_dict = {}
with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_id = row['image_id']
        label = row['dx'].upper()  # Convert labels to uppercase (e.g., 'BKL', 'MEL')
        
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(img_id)

os.makedirs(dest_folder, exist_ok=True)

# 3. Perform random sampling and distribute into class folders
for class_name, img_list in class_dict.items():
    # Create target class directory (e.g., data/batch_test/test_images/BKL)
    class_dir = os.path.join(dest_folder, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Randomly sample (if a class has fewer than 40 images, take all available)
    sampled = random.sample(img_list, min(sample_size, len(img_list)))
    
    count = 0
    for img_id in sampled:
        # Append .jpg extension automatically
        img_name = f"{img_id}.jpg"
        
        # Determine whether the image is in part_1 or part_2 folder
        src_path = None
        for folder in source_img_folders:
            potential_path = os.path.join(folder, img_name)
            if os.path.exists(potential_path):
                src_path = potential_path
                break
        
        # Copy the file if found
        if src_path:
            dst_path = os.path.join(class_dir, img_name)
            shutil.copy(src_path, dst_path)
            count += 1
            
    print(f"Successfully sampled and copied {count} images for class {class_name}")

print(f"\nAutomated sampling completed! Test images are saved in {dest_folder}")