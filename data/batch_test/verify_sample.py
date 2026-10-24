import os
import csv

# 1. Set paths (ensure these match your actual directory structure)
csv_path = "data/HAM10000/HAM10000_metadata.csv"
test_folder = "data/batch_test/test_images"

# 2. Read CSV to build a dictionary of ground truth labels
ground_truth = {}
with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row['image_id']] = row['dx'].upper()

# 3. Scan and verify all images in the test folder
error_count = 0
class_counts = {}

print("Starting image classification verification...\n")

for folder_name in os.listdir(test_folder):
    folder_path = os.path.join(test_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue
        
    images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
    class_counts[folder_name] = len(images)
    
    for img_name in images:
        # Remove .jpg extension to check the ID against the CSV
        img_id = img_name.replace('.jpg', '')
        
        actual_label = ground_truth.get(img_id)
        
        # Core check: Does the folder name match the actual CSV label?
        if actual_label != folder_name:
            print(f"[ERROR] Image {img_name} actually belongs to {actual_label}, but was placed in the {folder_name} folder.")
            error_count += 1

# 4. Output the final verification report
print("--- Final Verification Report ---")
for cls, count in class_counts.items():
    print(f"Class {cls:<5} : Successfully loaded {count} images")

if error_count == 0:
    print(f"\n[SUCCESS] Sampling classification is 100% correct with no misplaced images. You can safely run the batch test!")
else:
    print(f"\n[WARNING] Found {error_count} classification errors. Please review the sampling script.")