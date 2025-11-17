# Reinforcement Learning Integration with CheXNet

This module integrates Reinforcement Learning (RL) with CheXNet for chest X-ray diagnosis, enabling adaptive decision-making and report generation.

## Overview

The RL framework transforms the CheXNet CNN into an interactive system where:
- **State**: CNN-extracted features from chest X-rays (1024-dim DenseNet121 features)
- **Action**: Multi-label disease predictions (14 diseases) or report generation
- **Reward**: Based on clinical accuracy, with penalties for false negatives

## Module Structure

```
CheXNet/
├── rl_feature_extractor.py   # Extract CNN features as RL states
├── rl_environment.py          # RL environment (state, action, reward)
├── rl_agent.py                # RL agent architectures (PPO, REINFORCE)
├── rl_training.py             # Training loops with PPO/REINFORCE
├── rl_explainability.py       # Grad-CAM and evaluation tools
└── RL_README.md               # This file
```

## Installation

Required packages (add to your environment):
```bash
pip install opencv-python matplotlib tqdm
```

All other dependencies (torch, torchvision, numpy, sklearn, pillow) are already installed.

## Quick Start

### 1. Extract Features from CheXNet

```python
from rl_feature_extractor import CheXNetFeatureExtractor

# Initialize feature extractor
extractor = CheXNetFeatureExtractor(checkpoint_path='model.pth.tar')

# Extract features from an image
features = extractor.extract_from_image('path/to/xray.png')
print(f"Feature shape: {features.shape}")  # (1024,)
```

### 2. Create RL Environment

```python
from rl_environment import ChestXrayRLEnvironment
from read_data import ChestXrayDataSet
import torchvision.transforms as transforms

# Load dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ChestXrayDataSet(
    data_dir='./ChestX-ray14/images',
    image_list_file='./ChestX-ray14/labels/train_list.txt',
    transform=transform
)

# Create environment
env = ChestXrayRLEnvironment(extractor, dataset)

# Test environment
state = env.reset()
action = np.random.rand(14)  # Random action
next_state, reward, done, info = env.step(action)
print(f"Reward: {reward:.4f}, Accuracy: {info['accuracy']:.4f}")
```

### 3. Train RL Agent

```python
from rl_agent import DiagnosisAgent
from rl_training import PPOTrainer

# Create agent
agent = DiagnosisAgent(state_dim=1024, action_dim=14)

# Create trainer
trainer = PPOTrainer(agent, env, lr=3e-4)

# Train
trainer.train(num_iterations=1000, episodes_per_iter=100)
```

### 4. Evaluate and Visualize

```python
from rl_explainability import RLAgentEvaluator, visualize_decision_process

# Evaluate agent
evaluator = RLAgentEvaluator(agent, env)
metrics = evaluator.evaluate(num_episodes=100)
evaluator.print_metrics(metrics)

# Visualize decision-making
visualize_decision_process(agent, env, image_index=0, save_dir='./visualizations')

# Visualize training curves
evaluator.visualize_training_curves(
    history_path='./checkpoints/training_history.json',
    save_path='./training_curves.png'
)
```

## RL Algorithms Implemented

### 1. PPO (Proximal Policy Optimization)
- **Best for**: Stable training with continuous improvement
- **Use case**: Multi-label disease classification
- **Features**: Clipped objective, value function, entropy bonus

### 2. REINFORCE (Vanilla Policy Gradient)
- **Best for**: Quick prototyping and baseline comparison
- **Use case**: Simple episodic tasks
- **Features**: Monte Carlo policy gradient

### 3. DQN (Deep Q-Network)
- **Best for**: Discrete action spaces
- **Use case**: Binary classification per disease
- **Features**: Experience replay, target network

## Reward Function Design

The reward function balances accuracy with clinical priorities:

```python
reward = accuracy - 2.0 * false_negatives - 0.5 * false_positives + confidence_bonus
```

**Key design choices:**
- **False negatives** (missing diseases): Heavy penalty (-2.0)
- **False positives** (false alarms): Light penalty (-0.5)
- **Confidence bonus**: Rewards high-confidence correct predictions
- **Normalized**: Scaled to [-1, 1] range for stability

## RL Problem Formulations

### Classification Task
- **State**: 1024-dim CNN features
- **Action**: 14-dim probability vector for diseases
- **Episode**: Single image → single prediction
- **Reward**: Clinical accuracy metric

### Report Generation Task
- **State**: CNN features + partial report context
- **Action**: Next token in medical report
- **Episode**: Generate complete report (max 100 tokens)
- **Reward**: BLEU score or clinical accuracy

## Explainability Tools

### Grad-CAM Visualization
```python
from rl_explainability import GradCAM

# Create Grad-CAM
gradcam = GradCAM(model=feature_extractor, target_layer=feature_extractor.features[-1])

# Visualize
gradcam.visualize(
    image_path='./ChestX-ray14/images/00000001_000.png',
    class_idx=0,  # Atelectasis
    save_path='./gradcam_atelectasis.png'
)
```

### Performance Metrics
- **Per-class accuracy, precision, recall, F1**
- **Confusion matrices**
- **AUROC curves** (when continuous predictions available)
- **Reward and accuracy curves over training**

## Training Tips

### Hyperparameter Recommendations

**For PPO:**
```python
trainer = PPOTrainer(
    agent, env,
    lr=3e-4,              # Learning rate
    gamma=0.99,           # Discount factor
    clip_epsilon=0.2,     # PPO clipping
    value_coef=0.5,       # Value loss weight
    entropy_coef=0.01,    # Exploration bonus
    max_grad_norm=0.5     # Gradient clipping
)
```

**Training schedule:**
- Start with 100 episodes per iteration
- Train for 500-1000 iterations
- Save checkpoints every 50 iterations
- Monitor reward and accuracy curves

### Common Issues

**1. Reward not improving:**
- Reduce learning rate (try 1e-4)
- Increase entropy coefficient for more exploration
- Check reward function alignment with task

**2. Training unstable:**
- Reduce clip_epsilon (try 0.1)
- Increase gradient clipping (try 1.0)
- Use smaller batch sizes

**3. Low accuracy:**
- Verify CNN features are meaningful (test feature extractor)
- Check reward function weights
- Increase training iterations

## Advanced Usage

### Custom Reward Functions

```python
# Example: Add disease prevalence weighting
class WeightedRewardEnvironment(ChestXrayRLEnvironment):
    def compute_reward(self, action_binary, action_probs):
        # Your custom reward logic
        disease_weights = [1.0, 2.0, ...]  # Weight by prevalence
        weighted_accuracy = (correct * disease_weights).mean()
        return weighted_accuracy, info
```

### Multi-Task Learning

Train agent on multiple tasks simultaneously:
```python
# Combine classification + report generation
# Use multi-headed policy network
# Share CNN feature encoder
```

### Transfer Learning

```python
# Load pretrained agent
checkpoint = torch.load('pretrained_agent.pth')
agent.load_state_dict(checkpoint['agent_state_dict'])

# Fine-tune on new dataset
trainer = PPOTrainer(agent, new_env, lr=1e-4)  # Lower LR
trainer.train(num_iterations=200)
```

## Comparison: RL vs Supervised Learning

| Aspect | Supervised (CheXNet) | RL Integration |
|--------|---------------------|----------------|
| Training | Fixed labels | Reward-driven |
| Adaptivity | Static | Dynamic policy |
| Clinical focus | Equal weight | Prioritize FN > FP |
| Explainability | Grad-CAM | + Decision process |
| Sample efficiency | High | Lower (needs exploration) |

## Future Extensions

### 1. Active Learning
- Agent selects most informative samples
- Reduces annotation cost

### 2. Human-in-the-loop
- Incorporate radiologist feedback as reward
- Online learning from corrections

### 3. Multi-Agent Systems
- Multiple agents specialize in different diseases
- Consensus-based diagnosis

### 4. Attention Mechanisms
- Visual attention over image regions
- Align with Grad-CAM heatmaps

## References

- **CheXNet**: Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms"
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"

## Citation

If you use this RL framework, please cite:
```
@misc{chexnet-rl,
  title={Reinforcement Learning Integration for CheXNet},
  author={Your Name},
  year={2025}
}
```

## License

Same as original CheXNet repository.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
