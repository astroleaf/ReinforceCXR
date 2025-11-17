PneumoSynthAI: Intelligent & Explainable Pneumonia Detection
![Banner](Screenshot-2025-11-17-214

PneumoSynthAI is a next-generation, explainable AI system for automated pneumonia detection and report generation from chest X-ray images. By combining deep convolutional neural networks (CNNs) with reinforcement learning (RL), this project delivers accurate diagnosis, visual explanations, and RL-driven adaptive reporting—offering medical professionals reliable, actionable, and transparent insights.

Key Contributions & Uniqueness
Hybrid Deep Learning & RL: Merges state-of-the-art CNNs (CheXNet/VGG19) with a reinforcement learning agent for adaptive decision-making and report generation.

Explainable AI: Integrates Grad-CAM for interpretability, highlighting critical regions that drive model predictions.

Automated Workflow: Offers robust preprocessing, feature extraction, disease classification, visualization, and a smart RL diagnosis/report engine.

Comprehensive Visual Documentation: Full data flow diagrams and system outputs illustrate every phase for clarity and reproducibility.

Data Flow Architecture
Level 0 Data Flow Model
![Level 0 DFD](Screenshot-2025-11-17-214 1 Data Flow Diagram

![Level 1 DFD](Screenshot-2025-11-17-215 Module Flows

Preprocessing Module
![Preprocessing DFD](Screenshot-2025-11-17-220 Feature Extraction Module

![CNN Module DFD](Screenshot-2025-11-17-220-CAM Explainability Module

![Grad-CAM Module DFD](Screenshot-2025-11-17-222 Agent Module

![RL Agent DFD](Screenshot-2025-11-17-223 & Report Delivery Module

![Diagnosis Module DFD](Screenshot-2025-11-17-221 Agent Core Workflow

![RL Agent Core](Screenshot-2025-11-17-232Screenshot-2025-11-17-232Screenshot-2025-11-17-232Screenshot-2025-11-17-232 Output Visualizations

Model Outputs
Diagnosis Probability Table
![Sample Output 1](Screenshot-2025-11-17-214Screenshot-2025-11-17-215**
![Bar Chart](Screenshot-2025-11-15-105 Epochs**
![Accuracy Graph](Screenshot-2025-11-15-094 Epochs**
![Loss Graph](Screenshot-2025-11-15-065ability

Grad-CAM Overlays
![Grad-CAM Single](Screenshot-2025-11-15-065Screenshot-2025-11-15-065Screenshot-2025-11-15-103 Details

Dataset: NIH ChestX-ray14 (official link)

Technologies: Python, PyTorch, Grad-CAM, RL (policy gradient methods), Matplotlib, OpenCV

Key Models: CheXNet, VGG19, Custom RL Agent

Outputs: Disease probabilities, interpretability overlays, automated reports

Evaluation: Accuracy, loss, per-class metrics, and RL agent performance
