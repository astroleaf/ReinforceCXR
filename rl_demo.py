# encoding: utf-8

"""
RL Demo Script for CheXNet
Demonstrates complete workflow: feature extraction → training → evaluation
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from read_data import ChestXrayDataSet
import os

from rl_feature_extractor import CheXNetFeatureExtractor
from rl_environment import ChestXrayRLEnvironment
from rl_agent import DiagnosisAgent
from rl_training import PPOTrainer, REINFORCETrainer
from rl_explainability import RLAgentEvaluator, visualize_decision_process

def demo_feature_extraction():
    """Demo: Extract features from CheXNet."""
    print("="*60)
    print("DEMO 1: Feature Extraction")
    print("="*60)
    
    # Initialize feature extractor
    print("\n1. Initializing CheXNet feature extractor...")
    extractor = CheXNetFeatureExtractor(checkpoint_path='model.pth.tar')
    print(f"   Feature dimension: {extractor.feature_dim}")
    
    # Test with dummy input
    print("\n2. Testing with dummy input...")
    dummy_input = torch.randn(4, 3, 224, 224)
    features = extractor(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output features shape: {features.shape}")
    print(f"   Feature statistics: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
    
    print("\n✓ Feature extraction working correctly!\n")
    return extractor


def demo_environment(extractor):
    """Demo: Create and test RL environment."""
    print("="*60)
    print("DEMO 2: RL Environment")
    print("="*60)
    
    # Setup dataset
    print("\n1. Loading dataset...")
    DATA_DIR = './ChestX-ray14/images'
    TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = ChestXrayDataSet(
            data_dir=DATA_DIR,
            image_list_file=TRAIN_IMAGE_LIST,
            transform=transform
        )
        print(f"   Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"   Warning: Could not load full dataset ({e})")
        print("   Creating minimal test environment...")
        dataset = None
        return None, None
    
    # Create environment
    print("\n2. Creating RL environment...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = ChestXrayRLEnvironment(extractor, dataset, device=device)
    print(f"   State dimension: {env.get_state_dim()}")
    print(f"   Action dimension: {env.get_action_dim()}")
    print(f"   Disease classes: {', '.join(env.get_class_names()[:3])}...")
    
    # Test environment
    print("\n3. Testing environment interaction...")
    state = env.reset()
    print(f"   Initial state shape: {state.shape}")
    
    # Random action
    action = np.random.rand(env.get_action_dim())
    next_state, reward, done, info = env.step(action)
    
    print(f"   Reward: {reward:.4f}")
    print(f"   Accuracy: {info['accuracy']:.4f}")
    print(f"   False negatives: {info['false_negatives']}")
    print(f"   False positives: {info['false_positives']}")
    
    print("\n✓ Environment working correctly!\n")
    return env, dataset


def demo_agent():
    """Demo: Create and test RL agent."""
    print("="*60)
    print("DEMO 3: RL Agent")
    print("="*60)
    
    print("\n1. Creating DiagnosisAgent...")
    agent = DiagnosisAgent(state_dim=1024, action_dim=14, hidden_dims=[512, 256])
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print("\n2. Testing agent forward pass...")
    dummy_state = np.random.randn(1024)
    action, log_prob = agent.get_action(dummy_state)
    print(f"   Action shape: {action.shape}")
    print(f"   Action probabilities (first 5): {action[:5]}")
    print(f"   Log probability: {log_prob:.4f}")
    
    print("\n3. Testing batch evaluation...")
    batch_states = torch.randn(32, 1024)
    batch_actions = torch.randint(0, 2, (32, 14)).float()
    log_probs, values, entropy = agent.evaluate_actions(batch_states, batch_actions)
    print(f"   Log probs shape: {log_probs.shape}")
    print(f"   Values shape: {values.shape}")
    print(f"   Entropy shape: {entropy.shape}")
    
    print("\n✓ Agent working correctly!\n")
    return agent


def demo_quick_training(agent, env):
    """Demo: Quick training run."""
    print("="*60)
    print("DEMO 4: Quick Training (10 iterations)")
    print("="*60)
    
    if env is None:
        print("Skipping training demo (dataset not available)")
        return
    
    print("\n1. Creating PPO trainer...")
    trainer = PPOTrainer(agent, env, lr=3e-4)
    
    print("\n2. Running training for 10 iterations...")
    print("   (This is just a demo - real training needs 500-1000 iterations)")
    
    try:
        trainer.train(num_iterations=10, episodes_per_iter=20, save_dir='./demo_checkpoints')
        
        print("\n3. Training summary:")
        print(f"   Final reward: {trainer.episode_rewards[-1]:.4f}")
        print(f"   Final accuracy: {trainer.episode_accuracies[-1]:.4f}")
        
        print("\n✓ Training demo completed!\n")
    except Exception as e:
        print(f"   Training error: {e}")
        print("   (This is expected if dataset is not fully loaded)")


def demo_evaluation(agent, env):
    """Demo: Agent evaluation."""
    print("="*60)
    print("DEMO 5: Agent Evaluation")
    print("="*60)
    
    if env is None:
        print("Skipping evaluation demo (dataset not available)")
        return
    
    print("\n1. Creating evaluator...")
    evaluator = RLAgentEvaluator(agent, env)
    
    print("\n2. Evaluating agent on 50 episodes...")
    try:
        metrics = evaluator.evaluate(num_episodes=50)
        
        print("\n3. Evaluation results:")
        evaluator.print_metrics(metrics)
        
        # Create directory for demo visualizations if needed
        os.makedirs('./demo_visualizations', exist_ok=True)
        
        print("\n4. Saving evaluation visualizations...")
        evaluator.visualize_distributions(
            metrics,
            save_path='./demo_visualizations/eval_distributions.png',
        )
        evaluator.visualize_per_class_performance(
            metrics,
            save_path='./demo_visualizations/per_class_performance.png',
        )
        
        print("\n✓ Evaluation completed!\n")
    except Exception as e:
        print(f"   Evaluation error: {e}")


def demo_visualization(agent, env):
    """Demo: Visualization tools."""
    print("="*60)
    print("DEMO 6: Visualization")
    print("="*60)
    
    if env is None:
        print("Skipping visualization demo (dataset not available)")
        return
    
    print("\n1. Visualizing agent decision-making...")
    try:
        visualize_decision_process(agent, env, image_index=0, save_dir='./demo_visualizations')
        print("   ✓ Saved visualization to ./demo_visualizations/")
    except Exception as e:
        print(f"   Visualization error: {e}")
    
    print("\n✓ Visualization demo completed!\n")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("CheXNet RL Integration - Complete Demo")
    print("="*60 + "\n")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Demo 1: Feature extraction
    extractor = demo_feature_extraction()
    
    # Demo 2: Environment
    env, dataset = demo_environment(extractor)
    
    # Demo 3: Agent
    agent = demo_agent()
    
    # Demo 4: Training (quick)
    demo_quick_training(agent, env)
    
    # Demo 5: Evaluation
    demo_evaluation(agent, env)
    
    # Demo 6: Visualization
    demo_visualization(agent, env)
    
    # Summary
    print("="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Download ChestX-ray14 dataset to ./ChestX-ray14/images/")
    print("2. Run full training: python rl_training.py")
    print("3. Evaluate trained agent with rl_explainability.py")
    print("4. Customize reward function in rl_environment.py")
    print("5. Experiment with different agent architectures in rl_agent.py")
    print("\nSee RL_README.md for detailed documentation.\n")


if __name__ == '__main__':
    main()
