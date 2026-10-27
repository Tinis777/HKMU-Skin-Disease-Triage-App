# HKMU Skin Disease Triage App (FYP)

This repository contains the source code and evaluation scripts for the preliminary benchmarking phase of an AI-powered Skin Disease Triage Application, developed for a Final Year Project (FYP) at HKMU[cite: 7]. 

The current core research focus is a **Comparative Feasibility Study** evaluating the computational efficiency, latency, and resource consumption between **Cloud-side inference (PyTorch)** and **Client-side edge inference (ONNX)**, using a baseline YOLOv8 architecture and the HAM10000 dataset[cite: 7].

## Core Research Objectives
* **Accuracy Retention:** Verify whether converting the model from PyTorch to ONNX format causes any significant degradation in accuracy benchmarks and confidence scores[cite: 7].
* **Computational Efficiency:** Compare the two inference engines based on Total Inference Time and Average Time per Image[cite: 7].
* **Edge Computing Feasibility:** Demonstrate that the Client-side (ONNX) solution delivers highly competitive inference speeds while maintaining zero server computing costs and guaranteeing medical data privacy[cite: 7].

## Current Progress & Milestones
* **Baseline Model Export (Demo):** Conducted a preliminary training run to export a demo YOLOv8 model (`best.pt`) to a lightweight `best.onnx` format[cite: 7]. This serves strictly as an experimental baseline to validate frontend edge inference feasibility before training the final model.
* **Automated Unbiased Sampling:** Developed `random_sample.py` to automatically extract 50 random images for each of the 7 disease classes (350 images in total) to create a standard test set, completely eliminating human cherry-picking bias[cite: 7].
* **Data Integrity Verification:** Created `verify_sample.py` to strictly cross-check the test images against the official CSV Ground Truth labels, ensuring 100% classification accuracy with zero misplaced files[cite: 7].
* **Performance Testing Upgrade:** Upgraded `cloud_batch_test.py` and `client_batch_test.py` to track Total Inference Time and Average Time per Image (ms), alongside accuracy and confidence metrics, with automated CSV export capabilities[cite: 7].
* **Professional Version Control:** Configured `.gitignore` to block massive raw image datasets while successfully force-pushing core evaluation scripts and CSV results to GitHub, maintaining a lightweight and academic repository structure[cite: 7].

## Next Steps
* Train the final, fully optimized YOLOv8 model tailored specifically for the medical triage architecture.
* Execute the final batch inference tests to generate the complete comparative CSV data for Cloud vs. Client[cite: 7].
* Transform the CSV latency and resource consumption data into visual charts for the later FYP report and presentation[cite: 7].

## Project Structure & Scripts

### 1. Data Sampling & Verification
* **`data/batch_test/random_sample.py`**: Automates unbiased random sampling from the original dataset[cite: 7].
* **`data/batch_test/verify_sample.py`**: Validates the sampling results to ensure data integrity[cite: 7].

### 2. Batch Inference Testing (Feasibility Study)
* **`models/batch_test_model/cloud_batch_test.py`**: Simulates Cloud-side server environment using `best.pt` (PyTorch)[cite: 7].
* **`models/batch_test_model/client_batch_test.py`**: Simulates Client-side edge environment using `best.onnx` (ONNX Runtime)[cite: 7].
* **`models/batch_test_model/check_onnx_ram.py`**: A dedicated performance benchmarking script for validating Client-Side (Edge) inference feasibility[cite: 7]. 
  * **Purpose:** Isolates the exported `.onnx` model entirely from the heavy PyTorch/Ultralytics backend, accurately simulating a lightweight browser-based deployment environment[cite: 7].
  * **Metrics Tracked:** Processes the unbiased test images using pure `onnxruntime` and `psutil` to extract the true **Peak Memory Usage (RAM in MB)** and **Average Inference Latency (ms)** without framework overhead[cite: 7].
  * **Significance:** The empirical data generated directly justifies our architectural pivot to edge computing, proving the system can run on resource-constrained smartphones with zero server dependency[cite: 7].
*(Test results are automatically exported to `cloud_results.csv` and `client_results.csv` for chart analysis[cite: 7].)*

### 3. Demo Model Training & Export
* **`data/batch_test/trained_model/Trained model.ipynb`**: Contains the preliminary YOLOv8 training pipeline, hyperparameter configurations, and the code used to export the experimental PyTorch model (`best.pt`) into the edge-optimized ONNX format (`best.onnx`)[cite: 7].
* Training evaluation metrics (e.g., confusion matrix, validation results) are also stored in this directory for academic reference[cite: 7].
* **Weight Placement**: To execute the batch test scripts, ensure the generated `best.pt` and `best.onnx` weights are placed inside the `models/batch_test_model/weights/` directory[cite: 7].

## Dataset Notice
To comply with GitHub's file size limits and maintain a lightweight repository, the `.gitignore` file is configured to block all massive raw image data (including the full HAM10000 dataset and the generated `test_images` folder)[cite: 7].
* Only source code and CSV test results are stored on GitHub[cite: 7].
* To reproduce the benchmark tests, please obtain the authorized `test_images.zip`, extract it, and place the images in the `data/batch_test/` directory before running the evaluation scripts[cite: 7].

## Execution Flow
Ensure you are in a Python 3.12 virtual environment at the project root (`HKMU-Skin-Disease-Triage-App`)[cite: 7]:

1. **Generate Test Samples**: 
   `python data/batch_test/random_sample.py`[cite: 7]
2. **Verify Sample Integrity**: 
   `python data/batch_test/verify_sample.py`[cite: 7]
3. **Run Cloud-side Benchmark**: 
   `python models/batch_test_model/cloud_batch_test.py`[cite: 7]
4. **Run Client-side Benchmark**: 
   `python models/batch_test_model/client_batch_test.py`[cite: 7]