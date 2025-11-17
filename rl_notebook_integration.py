# ===============================================
# RL + CheXNet Integration for Jupyter Notebook
# Copy these cells into Chest_Xray_work (1).ipynb
# ===============================================

# CELL 1: Setup - Add project to path and import modules
"""
# === RL + CheXNet Integration Setup ===
import sys
from pathlib import Path
import os

# Point this to your CheXNet RL project root
project_root = Path(r"E:\CheXNet")
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from read_data import ChestXrayDataSet
from rl_feature_extractor import CheXNetFeatureExtractor
from rl_environment import ChestXrayRLEnvironment
from rl_agent import DiagnosisAgent
from rl_training import PPOTrainer
from rl_explainability import (
    RLAgentEvaluator,
    visualize_decision_process,
    GradCAM,
)

print("✓ RL modules imported successfully")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
"""

# CELL 2: Load Feature Extractor and Dataset
"""
# === Load CheXNet Feature Extractor and ChestX-ray14 Dataset ===

print("Loading CheXNet feature extractor...")
extractor = CheXNetFeatureExtractor(
    checkpoint_path=str(project_root / "model.pth.tar")
)
print(f"Feature dimension: {extractor.feature_dim}")

# Dataset paths
DATA_DIR = project_root / "ChestX-ray14" / "images"
TRAIN_IMAGE_LIST = project_root / "ChestX-ray14" / "labels" / "train_list.txt"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

print("Loading dataset...")
dataset = ChestXrayDataSet(
    data_dir=str(DATA_DIR),
    image_list_file=str(TRAIN_IMAGE_LIST),
    transform=transform,
)

print(f"Dataset size: {len(dataset)}")
"""

# CELL 3: Create RL Environment and Agent
"""
# === Create RL Environment and Agent ===

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Creating RL environment...")
env = ChestXrayRLEnvironment(
    feature_extractor=extractor.to(device),
    dataset=dataset,
    device=device,
)

print(f"State dimension: {env.get_state_dim()}")
print(f"Action dimension: {env.get_action_dim()}")
print(f"Disease classes: {', '.join(env.get_class_names()[:3])}...")

print("\nCreating RL agent...")
agent = DiagnosisAgent(
    state_dim=extractor.feature_dim,
    action_dim=14,
    hidden_dims=[512, 256],
).to(device)

total_params = sum(p.numel() for p in agent.parameters())
print(f"Total agent parameters: {total_params:,}")

# Quick sanity check
print("\nTesting environment interaction...")
state = env.reset()
action, _ = agent.get_action(state)
_, reward, _, info = env.step(action)

print(f"Reward: {reward:.4f}")
print(f"Accuracy: {info['accuracy']:.4f}")
print(f"False negatives: {info['false_negatives']}")
print(f"False positives: {info['false_positives']}")
print("\n✓ RL environment and agent ready!")
"""

# CELL 4: Evaluate Agent and Generate Visualizations
"""
# === Evaluate Agent and Generate Visualizations ===

print("="*60)
print("RL Agent Evaluation")
print("="*60)

evaluator = RLAgentEvaluator(agent, env)

# Evaluate on episodes
print("\nEvaluating agent on 50 episodes...")
metrics = evaluator.evaluate(num_episodes=50)

# Print metrics
print("\nEvaluation Results:")
evaluator.print_metrics(metrics)

# Create visualization directory
viz_dir = Path("rl_notebook_visualizations")
os.makedirs(viz_dir, exist_ok=True)

print(f"\nGenerating visualizations in: {viz_dir}")

# 1) Reward and accuracy histograms
print("1. Creating reward and accuracy histograms...")
evaluator.visualize_distributions(
    metrics,
    save_path=str(viz_dir / "eval_distributions.png"),
)

# 2) Per-class performance bar chart
print("2. Creating per-class performance chart...")
evaluator.visualize_per_class_performance(
    metrics,
    save_path=str(viz_dir / "per_class_performance.png"),
)

print("\n✓ Evaluation visualizations saved!")
"""

# CELL 5: Generate Grad-CAM and Prediction Visualizations
"""
# === Generate Grad-CAM and Prediction Visualizations ===

print("="*60)
print("Grad-CAM and Prediction Visualization")
print("="*60)

# Choose images to visualize (you can change these indices)
image_indices = [0, 10, 20]

for image_index in image_indices:
    print(f"\nGenerating visualizations for image {image_index}...")
    
    visualize_decision_process(
        agent=agent,
        env=env,
        image_index=image_index,
        save_dir=str(viz_dir),
    )

print("\n✓ All visualizations generated!")
print(f"Check {viz_dir} for output files:")
print("  - predictions_image_*.png (prediction vs ground truth)")
print("  - gradcam_image_*.png (Grad-CAM overlays)")
print("  - eval_distributions.png (reward/accuracy histograms)")
print("  - per_class_performance.png (per-class metrics)")
"""

# CELL 6: Display Visualizations in Notebook
"""
# === Display Visualizations in Notebook ===

from IPython.display import Image, display
import matplotlib.pyplot as plt

print("Displaying visualizations...")
print("\n1. Reward and Accuracy Distributions:")
display(Image(filename=str(viz_dir / "eval_distributions.png")))

print("\n2. Per-Class Performance Metrics:")
display(Image(filename=str(viz_dir / "per_class_performance.png")))

print("\n3. Sample Predictions and Grad-CAM (Image 0):")
display(Image(filename=str(viz_dir / "predictions_image_0.png")))

# Display all Grad-CAM files for image 0
import glob
gradcam_files = glob.glob(str(viz_dir / "gradcam_image_0_*.png"))
for gradcam_file in gradcam_files[:1]:  # Show first one
    print(f"\nGrad-CAM Overlay:")
    display(Image(filename=gradcam_file))
"""

# CELL 7 (Optional): Training Curves from PPO
"""
# === (Optional) Display Training Curves ===
# Only run this if you have trained a model and saved training_history.json

history_path = project_root / "demo_checkpoints" / "training_history.json"

if history_path.exists():
    print("Visualizing training curves...")
    evaluator.visualize_training_curves(
        history_path=str(history_path),
        save_path=str(viz_dir / "training_curves.png"),
    )
    print("Training curves saved!")
    display(Image(filename=str(viz_dir / "training_curves.png")))
else:
    print(f"Training history not found at: {history_path}")
    print("Run training first to generate this visualization.")
"""

# CELL 8: Summary Statistics
"""
# === Summary Statistics ===

print("="*60)
print("Summary Statistics")
print("="*60)

print(f"\nDataset Statistics:")
print(f"  Total images: {len(dataset)}")
print(f"  Disease classes: {env.get_action_dim()}")
print(f"  Feature dimension: {env.get_state_dim()}")

print(f"\nAgent Architecture:")
print(f"  State dim: {agent.state_dim}")
print(f"  Action dim: {agent.action_dim}")
print(f"  Hidden dims: {agent.hidden_dims}")
print(f"  Total parameters: {sum(p.numel() for p in agent.parameters()):,}")

print(f"\nEvaluation Results:")
print(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")

print(f"\nTop 5 Diseases by Accuracy:")
class_names = metrics['class_names']
accuracies = metrics['per_class_accuracy']
sorted_classes = sorted(zip(class_names, accuracies), key=lambda x: x[1], reverse=True)
for name, acc in sorted_classes[:5]:
    print(f"  {name:<20} {acc:.3f}")

print(f"\nVisualization outputs saved to: {viz_dir.absolute()}")
"""

# END OF CELLS
# ===============================================
# Instructions:
# 1. Open Chest_Xray_work (1).ipynb in Jupyter
# 2. Add new cells at the end of the notebook
# 3. Copy the content between triple quotes from each CELL above
# 4. Run the cells in order
# ===============================================
