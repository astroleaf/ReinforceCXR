🚀 PneumoSynthAI: Modern, Explainable Pneumonia Detection System
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-214036.jpg" width="700"><br> <b>Intelligent Data Flow from Patient Input to Diagnosis</b> </div>
🩺 Project Overview
PneumoSynthAI is a cutting-edge, AI-driven pipeline for pneumonia detection and reporting from chest X-rays. The system blends high-accuracy deep neural networks with reinforcement learning and visual explainability, creating a transparent and adaptive diagnosis assistant for clinical use.

🌟 What Makes This Project Unique?
Hybrid AI: Combines proven CNNs (CheXNet/VGG19) for visual diagnosis and RL for adaptive report generation.

Full Transparency: Grad-CAM heatmaps pinpoint what the AI "sees" in each X-ray.

End-to-End Automation: From upload, through preprocessing, deep inference, RL-enhanced reporting, and result delivery.

Comprehensive Visualization: Every phase explained visually (see diagrams below).

🖇️ Data Flow Architecture
Level 0: Context Diagram
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-214036.jpg" width="600"><br> <em>Level 0 Dataflow — system context overview</em> </div>
Level 1: Overall System Pipeline
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-215013.jpg" width="850"><br> <em>Level 1 DFD — high-level system architecture</em> </div>
🔎 Module-by-Module Breakdown
Preprocessing Pipeline
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-220207.jpg" width="700"><br> <em>Resizing, normalization, and augmentation flow</em> </div>
Deep Feature Extraction (CNN Module)
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-220554.jpg" width="700"><br> <em>How the CNN analyzes X-ray structures</em> </div>
Grad-CAM Explainability
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-222848.jpg" width="700"><br> <em>Grad-CAM module highlights model reasoning</em> </div>
Reinforcement Learning Agent
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-223804.jpg" width="700"><br> <em>RL Agent: Adaptive diagnostic decision-making</em> </div>
Diagnosis & Report Delivery
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-221441.jpg" width="700"><br> <em>Final aggregation and delivery to user</em> </div>
🤖 RL Agent — Deep Dive
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-232832.jpg" width="650"><br> <img src="SCREENSHOTS/Screenshot-2025-11-17-232859.jpg" width="600"><br> <img src="SCREENSHOTS/Screenshot-2025-11-17-232910.jpg" width="600"><br> <img src="SCREENSHOTS/Screenshot-2025-11-17-232926.jpg" width="650"><br> <em>RL state processing, agent logic, and performance summary</em> </div>
📊 Output & Interpretability
Sample Output: Prediction Table
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-17-214219.jpg" width="450"> <img src="SCREENSHOTS/Screenshot-2025-11-17-215136.jpg" width="450"><br> </div>
Accuracy & Loss Curves
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-15-105037.jpg" width="450"> <img src="SCREENSHOTS/Screenshot-2025-11-15-094017.jpg" width="450"> <img src="SCREENSHOTS/Screenshot-2025-11-15-065835.jpg" width="450"><br> </div>
Per-class Prediction Visualization
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-15-065436.jpg" width="450"> <img src="SCREENSHOTS/Screenshot-2025-11-15-103931.jpg" width="450"><br> </div>
Grad-CAM Overlays
<div align="center"> <img src="SCREENSHOTS/Screenshot-2025-11-15-065412.jpg" width="400"> <img src="SCREENSHOTS/Screenshot-2025-11-15-070443.jpg" width="400"><br> </div>
📁 Dataset
NIH ChestX-ray14 (Dataset Link)

Organized and processed for robust AI analysis.
