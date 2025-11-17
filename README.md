# 🚀 PneumoSynthAI: Modern, Explainable Pneumonia Detection System

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20214036.jpg" width="700">
  <br>
  <b>Intelligent Data Flow from Patient Input to Diagnosis</b>
</div>

---

## 🩺 Overview

**PneumoSynthAI** is a modern AI pipeline for pneumonia detection and reporting from chest X-rays that fuses high-accuracy deep neural networks, reinforcement learning, and explainability, building an adaptive, trustworthy diagnosis/reporting engine for clinicians.

---

## 🌟 What Makes This Project Unique?

- **Hybrid AI:** Deep CNNs (CheXNet/VGG19) + RL for adaptive, automated, explainable reporting.
- **Full Transparency:** All decisions traceable through Grad-CAM overlays.
- **Complete Automation:** Upload to result, all steps are visible and robustly engineered.

---

## 📈 Data Flow Architecture

### Level 0: System Context

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20214036.jpg" width="600">
  <br>
  <i>Level 0: Overall context — user to output</i>
</div>

### Level 1: High-level Pipeline

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20215013.jpg" width="900">
  <br>
  <i>Level 1: Entire processing flow from image to report</i>
</div>

---

## 🧩 Module-by-Module Flows

### Preprocessing

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20220207.jpg" width="700">
  <br>
  <i>Resize, normalize, augment for robust X-ray input</i>
</div>

### CNN Feature Extraction

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20220554.jpg" width="700">
  <br>
  <i>CNN module: how AI "sees" and processes chest X-rays</i>
</div>

### Grad-CAM Explainability

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20222848.jpg" width="700">
  <br>
  <i>Visualizing the model's decision focus</i>
</div>

### RL Agent Module

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20223804.jpg" width="700">
  <br>
  <i>Reinforcement learning workflow: adaptive, feedback-driven diagnosis</i>
</div>

### Diagnosis & Report

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20221441.jpg" width="700">
  <br>
  <i>Bringing outputs and visual explanations together for clinicians</i>
</div>

---

## 🤖 RL Agent — Core Logic and Output

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20232832.jpg" width="520">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20232859.jpg" width="420">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20232910.jpg" width="500">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20232926.jpg" width="550">
  <br>
  <i>RL agent flow, learning, and performance metrics</i>
</div>

---

## 📊 Output & Visualizations

### Sample Output: Disease Probabilities

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20214219.jpg" width="370">
  <img src="SCREENSHOTS/Screenshot%202025-11-17%20215136.jpg" width="370">
</div>

### Model Performance: Accuracy & Loss

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20105037.jpg" width="340">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20094017.jpg" width="340">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20065835.jpg" width="340">
</div>

### Classwise Predictions & Grad-CAM Overlays

<div align="center">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20065436.jpg" width="420">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20103931.jpg" width="420">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20065412.jpg" width="400">
  <img src="SCREENSHOTS/Screenshot%202025-11-15%20070443.jpg" width="400">
</div>

---

## 📁 Dataset

- **NIH ChestX-ray14** ([Official Download](https://nihcc.app.box.com/v/ChestXray-NIHCC))

---

## 🚦 How to Run

1. Place CXR images as required, put all screenshots in `/SCREENSHOTS/` using exact filenames.
2. Run provided scripts for preprocessing, training, RL agent, etc.
3. Test uploads and see all output/visualizations drop into place.

---

## 💡 Contact & Contribution

Questions or suggestions? Open an issue or PR here on GitHub.
Give us a ⭐ if our work helps your research!

---

<em>All screenshots above are generated from the actual pipeline and can be viewed in the SCREENSHOTS directory.</em>
