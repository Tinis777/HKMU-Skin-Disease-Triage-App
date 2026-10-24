# HKMU Skin Disease Triage App (FYP)

This repository contains the source code and evaluation scripts for an AI-powered Skin Disease Triage Application, developed for a Final Year Project (FYP) at HKMU. 

The core research focus is a **Comparative Study** evaluating the computational efficiency, latency, and resource consumption between **Cloud-side inference (PyTorch)** and **Client-side edge inference (ONNX)**, using the YOLOv8 architecture and the HAM10000 dataset.

## Core Research Objectives
* **Accuracy Retention:** Verify whether converting the model from PyTorch to ONNX format causes any significant degradation in accuracy benchmarks and confidence scores.
* **Computational Efficiency:** Compare the two inference engines based on Total Inference Time and Average Time per Image.
* **Edge Computing Feasibility:** Demonstrate that the Client-side (ONNX) solution delivers highly competitive inference speeds while maintaining zero server computing costs and guaranteeing medical data privacy.

## Current Progress & Milestones
* **Model Training & Conversion:** Successfully trained the YOLOv8 model using the HAM10000 dataset and exported `best.pt` to a lightweight `best.onnx` format to support frontend edge inference.
* **Automated Unbiased Sampling:** Developed `random_sample.py` to automatically extract 40 random images for each of the 7 disease classes (280 images in total) to create a standard test set, completely eliminating human cherry-picking bias.
* **Data Integrity Verification:** Created `verify_sample.py` to strictly cross-check the test images against the official CSV Ground Truth labels, ensuring 100% classification accuracy with zero misplaced files.
* **Performance Testing Upgrade:** Upgraded `cloud_batch_test.py` and `client_batch_test.py` to track Total Inference Time and Average Time per Image (ms), alongside accuracy and confidence metrics, with automated CSV export capabilities.
* **Professional Version Control:** Configured `.gitignore` to block massive raw image datasets while successfully force-pushing core evaluation scripts and CSV results to GitHub, maintaining a lightweight and academic repository structure.

## Next Steps
* Execute the final batch inference tests to generate the complete comparative CSV data for Cloud vs. Client.
* Transform the CSV latency and resource consumption data into visual charts for the final FYP report and presentation.

## Project Structure & Scripts

### 1. Data Sampling & Verification
* **`data/batch_test/random_sample.py`**: Automates unbiased random sampling from the original dataset.
* **`data/batch_test/verify_sample.py`**: Validates the sampling results to ensure data integrity.

### 2. Batch Inference Testing
* **`models/batch_test_model/cloud_batch_test.py`**: Simulates Cloud-side server environment using `best.pt` (PyTorch).
* **`models/batch_test_model/client_batch_test.py`**: Simulates Client-side edge environment using `best.onnx` (ONNX Runtime).
*(Test results are automatically exported to `cloud_results.csv` and `client_results.csv` for chart analysis.)*

### 3. Model Training & Export
* **`data/batch_test/trained_model/Trained model.ipynb`**: Contains the complete YOLOv8 training pipeline, hyperparameter configurations, and the code used to export the trained PyTorch model (`best.pt`) into the edge-optimized ONNX format (`best.onnx`).
* Training evaluation metrics (e.g., confusion matrix, validation results) are also stored in this directory for academic reference.
* **Weight Placement**: To execute the batch test scripts, ensure the generated `best.pt` and `best.onnx` weights are placed inside the `models/batch_test_model/weights/` directory.

## Dataset Notice
To comply with GitHub's file size limits and maintain a lightweight repository, the `.gitignore` file is configured to block all massive raw image data (including the full HAM10000 dataset and the generated `test_images` folder).
* Only source code and CSV test results are stored on GitHub.
* To reproduce the benchmark tests, please obtain the authorized `test_images.zip`, extract it, and place the images in the `data/batch_test/` directory before running the evaluation scripts.

## Execution Flow
Ensure you are in a Python 3.12 virtual environment at the project root (`HKMU-Skin-Disease-Triage-App`):

1. **Generate Test Samples**: 
   `python data/batch_test/random_sample.py`
2. **Verify Sample Integrity**: 
   `python data/batch_test/verify_sample.py`
3. **Run Cloud-side Benchmark**: 
   `python models/batch_test_model/cloud_batch_test.py`
4. **Run Client-side Benchmark**: 
   `python models/batch_test_model/client_batch_test.py`