import os
import csv
from ultralytics import YOLO

# 1. Load the exported ONNX model to simulate Client-side (Edge) inference
# Assuming the script is executed from the 'models' directory
model = YOLO("weights/best.onnx")

# 2. Define input image directory and output CSV file paths
test_folder = "../data/batch_test/test_images"
output_csv = "../data/batch_test/client_results.csv"

# 3. Prepare CSV writer with actual class and correctness columns
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Image_ID", "Actual_Class", "Predicted_Class", "Correct?", "Client_ONNX_Confidence(%)", "Client_ONNX_Inference_Time(ms)"])

    correct_count = 0
    total_count = 0

    # 4. Iterate through subdirectories (Folder name = Ground Truth Class)
    for actual_class in os.listdir(test_folder):
        class_dir = os.path.join(test_folder, actual_class)
        
        # Ensure it is a directory
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                
                # Run ONNX inference (verbose=False hides default YOLO console output)
                results = model(img_path, verbose=False)
                result = results[0]

                # Extract prediction metrics
                top1_index = result.probs.top1
                predicted_class = result.names[top1_index]
                confidence = result.probs.top1conf.item() * 100
                inference_time = result.speed['inference']

                # Evaluate correctness
                is_correct = "Yes" if actual_class.upper() == predicted_class.upper() else "No"
                if is_correct == "Yes":
                    correct_count += 1
                total_count += 1

                # Log results to CSV
                writer.writerow([img_name, actual_class.upper(), predicted_class.upper(), is_correct, f"{confidence:.2f}", f"{inference_time:.2f}"])
                
                # Print progress to the terminal
                print(f"ONNX Processed {img_name} -> Actual: {actual_class.upper():<5} | Predicted: {predicted_class.upper():<5} | Correct: {is_correct:<3} | Confidence: {confidence:.2f}% | Latency: {inference_time:.2f}ms")

    # 5. Calculate and log overall ONNX accuracy
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\nClient-side (ONNX) batch testing completed successfully!")
        print(f"Total images tested: {total_count} | Correct predictions: {correct_count}")
        print(f"Overall ONNX Accuracy: {accuracy:.2f}%")
        
        # Append summary to the bottom of the CSV
        writer.writerow([])
        writer.writerow(["Total Images", total_count, "Overall ONNX Accuracy", f"{accuracy:.2f}%"])
    else:
        print("No images found. Please ensure images are placed inside class-named subfolders within 'test_images'.")