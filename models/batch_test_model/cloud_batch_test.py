import os
import csv
import psutil
from ultralytics import YOLO

# 1. Load the trained PyTorch model
# Assuming the script is executed from the 'models' directory
model = YOLO("models/batch_test_model/weights/best.pt")

# 2. Define input image directory and output CSV file paths
test_folder = "data/batch_test/test_images"
output_csv = "data/batch_test/cloud_results.csv"

# 3. Initialize process tracking for memory usage
process = psutil.Process(os.getpid())
peak_memory_mb = 0.0

# 4. Prepare CSV writer for Cloud results
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Image_ID", "Actual_Class", "Predicted_Class", "Correct?", "Cloud_PyTorch_Confidence(%)", "Cloud_PyTorch_Inference_Time(ms)"])

    correct_count = 0
    total_count = 0
    total_inference_time = 0.0

    # 5. Iterate through subdirectories
    for actual_class in os.listdir(test_folder):
        class_dir = os.path.join(test_folder, actual_class)
        
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                
                # Run PyTorch inference
                results = model(img_path, verbose=False)
                result = results[0]

                # Track peak memory during inference
                current_mem = process.memory_info().rss / (1024 * 1024)
                if current_mem > peak_memory_mb:
                    peak_memory_mb = current_mem

                # Extract prediction metrics
                top1_index = result.probs.top1
                predicted_class = result.names[top1_index]
                confidence = result.probs.top1conf.item() * 100
                inference_time = result.speed['inference']
                
                total_inference_time += inference_time

                # Evaluate correctness
                is_correct = "Yes" if actual_class.upper() == predicted_class.upper() else "No"
                if is_correct == "Yes":
                    correct_count += 1
                total_count += 1

                # Log results to CSV
                writer.writerow([img_name, actual_class.upper(), predicted_class.upper(), is_correct, f"{confidence:.2f}", f"{inference_time:.2f}"])
                print(f"PyTorch Processed {img_name} -> Actual: {actual_class.upper():<5} | Predicted: {predicted_class.upper():<5} | Correct: {is_correct:<3} | Latency: {inference_time:.2f}ms")

    # 6. Calculate and log overall PyTorch metrics
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        avg_inference_time = total_inference_time / total_count

        print(f"\nCloud-side (PyTorch) batch testing completed successfully!")
        print(f"Total images tested: {total_count} | Correct predictions: {correct_count}")
        print(f"Overall PyTorch Accuracy: {accuracy:.2f}%")
        print(f"Total Inference Time: {total_inference_time:.2f} ms")
        print(f"Average Time per Image: {avg_inference_time:.2f} ms")
        print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
        
        writer.writerow([])
        writer.writerow(["--- SUMMARY METRICS ---"])
        writer.writerow(["Total Images", total_count])
        writer.writerow(["Overall PyTorch Accuracy(%)", f"{accuracy:.2f}"])
        writer.writerow(["Total Inference Time(ms)", f"{total_inference_time:.2f}"])
        writer.writerow(["Average Time per Image(ms)", f"{avg_inference_time:.2f}"])
        writer.writerow(["Peak Memory Usage(MB)", f"{peak_memory_mb:.2f}"])
    else:
        print("No images found in the test folder.")