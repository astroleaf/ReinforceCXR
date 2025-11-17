# encoding: utf-8

"""Generate all RL + CNN visualizations in a single folder.

Outputs (in ./all_visualizations):
- label_distribution.png          : dataset label frequencies
- eval_distributions.png          : reward & accuracy histograms over episodes
- per_class_performance.png       : per-disease Acc/Prec/Rec/F1 bar chart
- predictions_image_*.png         : per-image prediction vs ground-truth bars
- gradcam_image_*_feature_*.png   : Grad-CAM overlays for sample images
- training_curves.png             : (if training_history.json exists)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Workaround for Windows OpenMP duplicate runtime issue that can occur
# when using PyTorch + matplotlib in the same process.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from read_data import ChestXrayDataSet
from rl_feature_extractor import CheXNetFeatureExtractor
from rl_environment import ChestXrayRLEnvironment
from rl_agent import DiagnosisAgent
from rl_explainability import RLAgentEvaluator, visualize_decision_process


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_dataset(project_root: Path) -> ChestXrayDataSet:
    data_dir = project_root / "ChestX-ray14" / "images"
    image_list = project_root / "ChestX-ray14" / "labels" / "train_list.txt"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ChestXrayDataSet(
        data_dir=str(data_dir),
        image_list_file=str(image_list),
        transform=transform,
    )
    return dataset


def plot_label_distribution(dataset: ChestXrayDataSet, save_path: Path) -> None:
    """Plot per-class label counts from ChestXrayDataSet."""
    labels = np.array(dataset.labels)
    counts = labels.sum(axis=0)

    # Class names inferred from RL environment ordering
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia',
    ]

    x = np.arange(len(class_names))
    plt.figure(figsize=(14, 6))
    plt.bar(x, counts, color='slateblue', alpha=0.85)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylabel('Number of positive labels')
    plt.title('ChestX-ray14 Label Distribution (Train)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved label distribution to {save_path}")


def plot_random_image_grid(dataset: ChestXrayDataSet, save_path: Path, num_images: int = 9, seed: int = 42) -> None:
    """Plot a 3x3 grid of random images from the dataset."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=num_images, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, idx in zip(axes.flat, indices):
        img_path = dataset.image_names[idx]
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img, cmap='gray')
        ax.set_title(f"idx {idx}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved random image grid to {save_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    viz_dir = ensure_dir(project_root / "all_visualizations")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Feature extractor and dataset
    print("\n1) Loading feature extractor and dataset...")
    extractor = CheXNetFeatureExtractor(checkpoint_path=str(project_root / "model.pth.tar"))
    dataset = build_dataset(project_root)
    print(f"   Dataset size: {len(dataset)}")

    # 2. Label distribution plot
    print("\n2) Plotting label distribution...")
    plot_label_distribution(dataset, viz_dir / "label_distribution.png")

    # 2b. Random image grid (9 samples used by the model)
    print("\n2b) Plotting 3x3 grid of random dataset images...")
    plot_random_image_grid(dataset, viz_dir / "random_image_grid.png")

    # 3. Environment and agent
    print("\n3) Creating RL environment and agent...")
    env = ChestXrayRLEnvironment(extractor.to(device), dataset, device=device)
    agent = DiagnosisAgent(state_dim=extractor.feature_dim, action_dim=14, hidden_dims=[512, 256]).to(device)
    print(f"   State dim: {env.get_state_dim()}, Action dim: {env.get_action_dim()}")

    # 4. Evaluation and global visualizations
    print("\n4) Evaluating agent on 50 episodes and generating histograms/graphs...")
    evaluator = RLAgentEvaluator(agent, env, device=device)
    metrics = evaluator.evaluate(num_episodes=50)
    evaluator.print_metrics(metrics)

    evaluator.visualize_distributions(
        metrics,
        save_path=str(viz_dir / "eval_distributions.png"),
    )
    evaluator.visualize_per_class_performance(
        metrics,
        save_path=str(viz_dir / "per_class_performance.png"),
    )

    # 5. Per-image visualizations (predictions + Grad-CAM)
    print("\n5) Generating per-image prediction and Grad-CAM visualizations...")
    sample_indices = [0, 10, 20]
    for idx in sample_indices:
        print(f"   Image index {idx}...")
        visualize_decision_process(agent, env, image_index=idx, save_dir=str(viz_dir))

    # 6. Training curves (if history exists)
    print("\n6) Looking for training_history.json to plot training curves...")
    from rl_explainability import RLAgentEvaluator as _Evaluator  # reuse evaluator

    history_candidates = [
        project_root / "demo_checkpoints" / "training_history.json",
        project_root / "checkpoints" / "training_history.json",
    ]

    history_path = None
    for cand in history_candidates:
        if cand.exists():
            history_path = cand
            break

    if history_path is not None:
        print(f"   Found training history at {history_path}, plotting curves...")
        _Evaluator(agent, env).visualize_training_curves(
            history_path=str(history_path),
            save_path=str(viz_dir / "training_curves.png"),
        )
    else:
        print("   No training_history.json found; skipping training curves.")

    print("\nAll visualizations saved under:")
    print(f"  {viz_dir}")


if __name__ == "__main__":
    main()
