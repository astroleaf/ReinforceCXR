# 🚀 PneumoSynthAI: Explainable Pneumonia Detection with Reinforcement Learning

## Overview

PneumoSynthAI is a state-of-the-art AI system designed for accurate and interpretable pneumonia diagnosis using chest X-ray images. This advanced framework integrates powerful convolutional neural networks (CNNs) to extract rich visual features and multi-label disease predictions with reinforcement learning (RL) to adaptively generate diagnostic reports and clinical decision support. The system incorporates Grad-CAM to provide transparent visual explanations of the model’s focus areas, aiding clinical trust and interpretability.

---

## Key Features and Innovations

- **Hybrid Learning Architecture:** Combines CNN-based feature extraction (using models like CheXNet and VGG19) with reinforcement learning agents to optimize diagnostic reporting.
- **Explainability:** Uses Grad-CAM overlays to visualize image regions influential in disease detection, offering insights beyond typical “black box” AI methods.
- **Automated End-to-End Pipeline:** From preprocessing input images, through model inference, to generating interpretable diagnostic reports and feedback-driven refinement via RL.
- **Clinically Relevant Adaptive Reporting:** RL agent learns to refine and personalize reports with feedback signals that mirror clinical accuracy needs.
- **Robust Evaluation:** Corroborated by multiple performance metrics including accuracy, precision, recall, F1 score, and RL reward metrics.

![Screenshot 2025-11-15 065412](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-15%20065412.png)

![Screenshot 2025-11-15 065436](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-15%20065436.png)

![Screenshot 2025-11-15 070443](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-15%20070443.png)

![Screenshot 2025-11-15 094050](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-15%20094050.png)


---

## System Architecture

- **Preprocessing Module:** Handles image resizing, normalization, and augmentation to improve model robustness.
- **CNN Feature Extraction:** Processes preprocessed images to generate multi-class disease probabilities.
- **Grad-CAM Explainability:** Generates heatmaps illustrating model decision focus for each diagnosis.
- **Reinforcement Learning Agent:** Utilizes CNN features as state inputs to adaptively generate diagnostic reports and decisions based on reward feedback.
- **Report and Output Delivery:** Compiles all model outputs and visual explanations into human-readable formats for clinicians.

![Screenshot 2025-11-17 214219](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20214219.png)

![Screenshot 2025-11-17 215013](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20215013.png)

![Screenshot 2025-11-17 220207](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20220207.png)

![Screenshot 2025-11-17 220554](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20220554.png)

![Screenshot 2025-11-17 221441](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20221441.png)

![Screenshot 2025-11-17 222848](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20222848.png)

![Screenshot 2025-11-17 223804](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20223804.png)

---

## Dataset

- Utilizes the public NIH ChestX-ray14 dataset, comprising over 100,000 images with 14 labeled thoracic diseases, enabling comprehensive model training and validation.  
  [Dataset link](https://nihcc.app.box.com/v/ChestXray-NIHCC)

![Screenshot 2025-11-17 232832](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20232832.png)

![Screenshot 2025-11-17 232859](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20232859.png)

![Screenshot 2025-11-17 232910](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20232910.png)

![Screenshot 2025-11-17 232926](https://github.com/astroleaf/ReinforceCXR/blob/main/SCREENSHOTS/Screenshot%202025-11-17%20232926.png)


---

## Getting Started

1. **Clone the repository and ensure all dependencies are installed.**
2. **Download and prepare the NIH ChestX-ray14 dataset as specified.**
3. **Run preprocessing scripts to cleanse and augment images properly.**
4. **Train the CNN model or load a pretrained baseline like CheXNet.**
5. **Train and evaluate the reinforcement learning agent on CNN features.**
6. **Use the inference pipeline to input new images and obtain explainable diagnosis reports.**

---

## Usage Insights

- Designed for clinicians and researchers to streamline pneumonia detection workflows.
- Hybrid RL methods elevate automated reporting by incorporating adaptive feedback.
- Grad-CAM visualizations serve as valuable educational and diagnostic tools.

---

## Contribution & Support

Contributions, bug reports, and feature requests are welcomed via GitHub issues and pull requests.  
Please support this project by starring the repository ⭐ if you find it useful!

---

*This project exemplifies the future of AI-driven healthcare, advancing transparency, adaptability, and reliability in medical image analysis.*



