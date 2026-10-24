import os
import time
import psutil
import numpy as np
from PIL import Image
import onnxruntime as ort

# Initialize process tracking for memory usage
process = psutil.Process(os.getpid())
peak_memory_mb = 0.0
test_folder = "data/batch_test/test_images"

# Load the pure ONNX engine (No PyTorch/Ultralytics dependency)
session = ort.InferenceSession("models/batch_test_model/weights/best.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Determine image size from model input (default to 224 if dynamic)
img_size = input_shape[2] if isinstance(input_shape[2], int) else 224 

total_inference_time = 0.0
total_count = 0

print("Starting pure ONNX batch testing...")

for actual_class in os.listdir(test_folder):
    class_dir = os.path.join(test_folder, actual_class)
    if not os.path.isdir(class_dir):
        continue
        
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_name)
            
            # Manual Image Preprocessing (Simulating YOLOv8 behavior)
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            img_data = np.array(img).astype(np.float32) / 255.0  # Normalize
            img_data = np.transpose(img_data, (2, 0, 1))         # HWC to CHW
            img_data = np.expand_dims(img_data, axis=0)          # Add batch dimension
            
            # Measure pure inference latency
            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: img_data})
            inference_time = (time.perf_counter() - start_time) * 1000
            
            total_inference_time += inference_time
            total_count += 1
            
            # Track peak RAM usage
            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_memory_mb:
                peak_memory_mb = current_mem

print("\n--- Client-Side (Pure ONNX) Performance Results ---")
print(f"Total Images Processed: {total_count}")
print(f"Average Inference Latency: {total_inference_time / total_count:.2f} ms")
print(f"Peak Memory Usage (RAM): {peak_memory_mb:.2f} MB")