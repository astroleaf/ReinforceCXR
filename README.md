🚀 PneumoSynthAI: Modern, Explainable Pneumonia Detection System
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-214036.jpg" width="700"> <br> <b>Intelligent Data Flow from Patient Input to Diagnosis</b> </div>
🩺 Overview
PneumoSynthAI is a modern AI-driven pipeline for pneumonia detection and reporting from chest X-rays. This system fuses high-accuracy neural networks, reinforcement learning, and state-of-the-art explainability, building an adaptive, trustworthy diagnosis and reporting engine for clinicians.

🦾 What Makes This Project Unique?
Hybrid Deep Learning + RL: Combines CNNs (CheXNet/VGG19) and policy-gradient RL for adaptive, automated reporting and clinical guidance.

Explainable by Design: Every decision is traceable through Grad-CAM overlays.

Full-stack Workflow: Raw images to advanced AI outputs, with all steps transparent.

Rich Visual Documentation: See the data flow and outputs at every stage below.

📈 Data Flow Architecture
Level 0: System Context
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-214036.jpg" width="600"> <br> <i>Level 0: End-to-end context</i> </div>
Level 1: High-level Pipeline
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-215013.jpg" width="850"> <br> <i>Level 1: From upload to model decision</i> </div>
🧩 Module Breakdown
Preprocessing Module:
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-220207.jpg" width="680"> <br> <i>Resize, normalize, and diverse augmentations for robust input</i> </div>
CNN Feature Extraction:
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-220554.jpg" width="700"> <br> <i>How the neural network "sees" your CXR</i> </div>
Grad-CAM Explainability:
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-222848.jpg" width="700"> <br> <i>Insight into the AI decision process</i> </div>
RL Agent Decision Engine:
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-223804.jpg" width="670"> <br> <i>Adaptive, feedback-driven report and decision logic</i> </div>
Diagnosis & Report Delivery:
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-221441.jpg" width="700"> <br> <i>Bringing it all together for the end-user</i> </div>
🤖 RL Agent — Under the Hood
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-232832.jpg" width="550"> <img src="SCREENSHOTS/Screenshot-2025-11-17-232859.jpg" width="450"><br> <img src="SCREENSHOTS/Screenshot-2025-11-17-232910.jpg" width="550"><br> <img src="SCREENSHOTS/Screenshot-2025-11-17-232926.jpg" width="550"> <br> <i>Training, batch evaluation, learning curves, and RL reward reporting</i> </div>
📊 Outputs
Model Predictions
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-214219.jpg" width="400"> <img src="SCREENSHOTS/Screenshot-2025-11-17-215136.jpg" width="400"> <br> <i>Sample disease probability outputs (per image)</i> </div>
Accuracy/Loss, Per-class Results
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-15-105037.jpg" width="340"> <img src="SCREENSHOTS/Screenshot-2025-11-15-094017.jpg" width="340"> <img src="SCREENSHOTS/Screenshot-2025-11-15-065835.jpg" width="340"> <br> <i>Performance metrics throughout training/testing</i> </div>
Visualizations (Per-Class, Grad-CAM)
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-15-065436.jpg" width="450"> <img src="SCREENSHOTS/Screenshot-2025-11-15-103931.jpg" width="450"><br> <img src="SCREENSHOTS/Screenshot-2025-11-15-065412.jpg" width="340"> <img src="SCREENSHOTS/Screenshot-2025-11-15-070443.jpg" width="340"> <br> <i>Class breakdown, visual "attention" overlays, original/ranked heatmaps</i> </div>
📁 Dataset
NIH ChestX-ray14 (Download/official).
