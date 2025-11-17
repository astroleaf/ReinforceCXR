# encoding: utf-8

"""
Explainability and Evaluation Tools for RL Agents
Implements Grad-CAM, attention visualization, and performance metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which regions of the image the model focuses on.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Neural network model
            target_layer: Layer to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            class_idx: Target class index (if None, use predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        if len(output.shape) == 2:  # Multi-label case
            target = output[0, class_idx]
        else:
            target = output[0]
        target.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to NumPy and apply percentile-based contrast stretching to
        # avoid flat/mostly-blue heatmaps.
        cam_np = cam.cpu().numpy()
        # Focus on the top 1% activations for better contrast.
        p = np.percentile(cam_np, 99)
        if p > 1e-8:
            cam_np = np.clip(cam_np / p, 0.0, 1.0)
        else:
            cam_np = np.clip(cam_np, 0.0, 1.0)
        
        return cam_np
    
    def visualize(self, image_path, class_idx=None, save_path=None):
        """
        Visualize Grad-CAM overlay on original image.
        
        Args:
            image_path: Path to original image
            class_idx: Target class index
            save_path: Path to save visualization
        """
        # Load and preprocess image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Load original image for overlay
        image_np = np.array(image.resize((224, 224)))
        
        # Overlay heatmap
        overlay = heatmap * 0.4 + image_np * 0.6
        overlay = overlay.astype(np.uint8)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


class RLAgentEvaluator:
    """
    Evaluator for RL agents with comprehensive metrics.
    """
    
    def __init__(self, agent, env, device='cpu'):
        """
        Args:
            agent: Trained RL agent
            env: RL environment
            device: 'cpu' or 'cuda'
        """
        self.agent = agent
        self.env = env
        self.device = device
        self.agent.eval()
    
    def evaluate(self, num_episodes=100):
        """
        Evaluate agent on multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        all_rewards = []
        all_accuracies = []
        per_class_predictions = []
        per_class_labels = []
        per_class_confidences = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            
            # Get deterministic action
            action, _ = self.agent.get_action(state, deterministic=True)
            
            # Execute action
            _, reward, _, info = self.env.step(action)
            
            all_rewards.append(reward)
            all_accuracies.append(info['accuracy'])
            per_class_predictions.append(info['predictions'])
            per_class_labels.append(info['true_labels'])
            # Store continuous confidence scores from environment, if available
            if 'confidence' in info:
                per_class_confidences.append(info['confidence'])
        
        # Compute metrics
        per_class_predictions = np.array(per_class_predictions)
        per_class_labels = np.array(per_class_labels)
        per_class_confidences = np.array(per_class_confidences) if len(per_class_confidences) > 0 else None
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(self.env.n_classes):
            correct = (per_class_predictions[:, i] == per_class_labels[:, i]).mean()
            per_class_acc.append(correct)
        
        # Precision, Recall, F1 per class
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            per_class_labels, per_class_predictions, average=None, zero_division=0
        )
        
        metrics = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_accuracy': float(np.mean(all_accuracies)),
            'std_accuracy': float(np.std(all_accuracies)),
            'per_class_accuracy': [float(x) for x in per_class_acc],
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'class_names': self.env.get_class_names(),
            'rewards': list(all_rewards),
            'accuracies': list(all_accuracies),
            'per_class_confidences': per_class_confidences.tolist() if per_class_confidences is not None else None,
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics in readable format."""
        print("="*60)
        print("RL Agent Evaluation Results")
        print("="*60)
        print(f"Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        print("\nPer-Class Performance:")
        print("-"*60)
        print(f"{'Disease':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-"*60)
        
        for i, name in enumerate(metrics['class_names']):
            print(f"{name:<20} {metrics['per_class_accuracy'][i]:<8.3f} "
                  f"{metrics['per_class_precision'][i]:<8.3f} "
                  f"{metrics['per_class_recall'][i]:<8.3f} "
                  f"{metrics['per_class_f1'][i]:<8.3f}")
        print("="*60)
    
    def visualize_distributions(self, metrics, save_path=None):
        """Plot histograms of rewards and accuracies over evaluation episodes."""
        rewards = metrics.get('rewards', [])
        accuracies = metrics.get('accuracies', [])
        
        if not rewards and not accuracies:
            print("No reward/accuracy data available to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if rewards:
            axes[0].hist(rewards, bins=20, color='steelblue', alpha=0.8)
            axes[0].set_title('Reward Distribution')
            axes[0].set_xlabel('Reward')
            axes[0].set_ylabel('Count')
            axes[0].grid(alpha=0.3)
        else:
            axes[0].axis('off')
        
        if accuracies:
            axes[1].hist(accuracies, bins=20, color='seagreen', alpha=0.8)
            axes[1].set_title('Accuracy Distribution')
            axes[1].set_xlabel('Accuracy')
            axes[1].set_ylabel('Count')
            axes[1].grid(alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved evaluation distributions to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_per_class_performance(self, metrics, save_path=None):
        """Bar charts for per-class accuracy, precision, recall, and F1."""
        class_names = metrics['class_names']
        acc = metrics['per_class_accuracy']
        prec = metrics['per_class_precision']
        rec = metrics['per_class_recall']
        f1 = metrics['per_class_f1']
        
        x = np.arange(len(class_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - 1.5 * width, acc, width, label='Accuracy')
        ax.bar(x - 0.5 * width, prec, width, label='Precision')
        ax.bar(x + 0.5 * width, rec, width, label='Recall')
        ax.bar(x + 1.5 * width, f1, width, label='F1')
        
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved per-class performance plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_training_curves(self, history_path, save_path=None):
        """
        Visualize training curves from training history.
        
        Args:
            history_path: Path to training_history.json
            save_path: Path to save plot
        """
        import json
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        rewards = history['rewards']
        accuracies = history['accuracies']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Reward curve
        axes[0].plot(rewards, alpha=0.6, label='Reward')
        # Moving average
        window = 50
        if len(rewards) > window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                        linewidth=2, label='Moving Avg')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Reward')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(accuracies, alpha=0.6, label='Accuracy')
        if len(accuracies) > window:
            moving_avg = np.convolve(accuracies, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(accuracies)), moving_avg, 
                        linewidth=2, label='Moving Avg')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_decision_process(agent, env, image_index, save_dir='./visualizations'):
    """
    Visualize complete decision-making process of RL agent.
    
    Args:
        agent: Trained RL agent
        env: RL environment
        image_index: Index of image to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Reset to specific image
    state = env.reset(index=image_index)
    
    # Get agent's action and probabilities
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = agent(state_tensor).squeeze(0).cpu().numpy()
    
    # Ground truth
    true_labels = env.current_labels
    
    # Visualize predictions vs ground truth (per-image bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(env.class_names))
    width = 0.35
    
    ax.bar(x - width/2, action_probs, width, label='Predicted Prob', alpha=0.8)
    ax.bar(x + width/2, true_labels, width, label='Ground Truth', alpha=0.8)
    
    ax.set_xlabel('Disease')
    ax.set_ylabel('Probability / Label')
    ax.set_title(f'RL Agent Predictions vs Ground Truth (Image {image_index})')
    ax.set_xticks(x)
    ax.set_xticklabels(env.class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'predictions_image_{image_index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction visualization to {save_path}")
    plt.close()
    
    # Additionally, generate a Grad-CAM visualization for this image if possible
    try:
        if hasattr(env, 'feature_extractor') and hasattr(env, 'dataset') \
                and hasattr(env.dataset, 'image_names'):
            image_path = env.dataset.image_names[image_index]
            # Use the most confident predicted class index as target
            top_idx = int(np.argmax(action_probs))
            gradcam = GradCAM(model=env.feature_extractor,
                              target_layer=env.feature_extractor.features[-1])
            gradcam_save_path = os.path.join(
                save_dir,
                f'gradcam_image_{image_index}_feature_{top_idx}.png'
            )
            gradcam.visualize(
                image_path=image_path,
                class_idx=top_idx,
                save_path=gradcam_save_path,
            )
    except Exception as e:
        print(f"Grad-CAM visualization failed: {e}")


if __name__ == '__main__':
    """Test explainability tools."""
    print("Explainability and evaluation tools loaded successfully!")
    print("\nAvailable tools:")
    print("1. GradCAM - Visualize attention with Grad-CAM")
    print("2. RLAgentEvaluator - Comprehensive evaluation metrics")
    print("3. visualize_decision_process - Decision visualization")
