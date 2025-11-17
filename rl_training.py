# encoding: utf-8

"""
RL Training Loop with PPO (Proximal Policy Optimization)
Trains RL agents for chest X-ray diagnosis using PPO algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json

from rl_feature_extractor import CheXNetFeatureExtractor
from rl_environment import ChestXrayRLEnvironment
from rl_agent import DiagnosisAgent
from read_data import ChestXrayDataSet


class PPOTrainer:
    """
    PPO trainer for RL agents.
    
    Implements Proximal Policy Optimization for stable policy learning.
    """
    
    def __init__(self, agent, env, lr=3e-4, gamma=0.99, clip_epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """
        Args:
            agent: RL agent (DiagnosisAgent)
            env: RL environment (ChestXrayRLEnvironment)
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
        """
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_accuracies = []
    
    def collect_trajectories(self, num_episodes):
        """
        Collect trajectories by interacting with environment.
        
        Args:
            num_episodes: Number of episodes to collect
        
        Returns:
            batch: Dictionary containing states, actions, rewards, etc.
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        infos = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            states.append(state)
            
            # Get action from policy
            action, log_prob = self.agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store trajectory data
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            # Get value estimate
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                value = self.agent.value_net(state_tensor).item()
            values.append(value)
        
        batch = {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'dones': np.array(dones),
            'infos': infos
        }
        
        return batch
    
    def compute_advantages(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # For episodic tasks, advantage is simply reward - value
        advantages = rewards - values
        returns = rewards
        
        return advantages, returns
    
    def update_policy(self, batch, num_epochs=4, batch_size=64):
        """
        Update policy using PPO.
        
        Args:
            batch: Trajectory batch
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size
        
        Returns:
            loss_info: Dictionary with loss information
        """
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        old_log_probs = torch.FloatTensor(batch['log_probs'])
        
        # Compute advantages
        advantages, returns = self.compute_advantages(
            batch['rewards'], batch['values'], batch['dones']
        )
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs of optimization
        for _ in range(num_epochs):
            # Mini-batch updates
            num_samples = len(states)
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                
                # Get current policy evaluation
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 
                                   1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss - 
                       self.entropy_coef * entropy.mean())
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        loss_info = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return loss_info
    
    def train(self, num_iterations, episodes_per_iter=100, save_dir='./checkpoints'):
        """
        Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iter: Episodes per iteration
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for iteration in range(num_iterations):
            # Collect trajectories
            batch = self.collect_trajectories(episodes_per_iter)
            
            # Update policy
            loss_info = self.update_policy(batch)
            
            # Track statistics
            mean_reward = batch['rewards'].mean()
            mean_accuracy = np.mean([info['accuracy'] for info in batch['infos']])
            
            self.episode_rewards.append(mean_reward)
            self.episode_accuracies.append(mean_accuracy)
            
            # Log progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{num_iterations}")
                print(f"  Mean Reward: {mean_reward:.4f}")
                print(f"  Mean Accuracy: {mean_accuracy:.4f}")
                print(f"  Policy Loss: {loss_info['policy_loss']:.4f}")
                print(f"  Value Loss: {loss_info['value_loss']:.4f}")
                print(f"  Entropy: {loss_info['entropy']:.4f}")
            
            # Save checkpoint
            if iteration % 50 == 0:
                checkpoint_path = os.path.join(save_dir, f'agent_iter_{iteration}.pth')
                torch.save({
                    'iteration': iteration,
                    'agent_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mean_reward': mean_reward,
                    'mean_accuracy': mean_accuracy
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'rewards': self.episode_rewards,
                'accuracies': self.episode_accuracies
            }, f)


class REINFORCETrainer:
    """
    REINFORCE (vanilla policy gradient) trainer.
    Simpler alternative to PPO for baseline comparison.
    """
    
    def __init__(self, agent, env, lr=1e-3, gamma=0.99):
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    def train_episode(self):
        """Train on single episode."""
        state = self.env.reset()
        
        # Get action
        action, log_prob = self.agent.get_action(state)
        
        # Execute action
        _, reward, done, info = self.env.step(action)
        
        # Policy gradient update
        loss = -log_prob * reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return reward, info['accuracy']
    
    def train(self, num_episodes):
        """Train for multiple episodes."""
        rewards = []
        accuracies = []
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            reward, accuracy = self.train_episode()
            rewards.append(reward)
            accuracies.append(accuracy)
            
            if episode % 100 == 0:
                mean_reward = np.mean(rewards[-100:])
                mean_accuracy = np.mean(accuracies[-100:])
                print(f"Episode {episode}: Reward={mean_reward:.4f}, Accuracy={mean_accuracy:.4f}")
        
        return rewards, accuracies


def main():
    """
    Main training script.
    """
    print("Initializing RL training for CheXNet...")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize feature extractor
    print("Loading feature extractor...")
    feature_extractor = CheXNetFeatureExtractor(checkpoint_path='model.pth.tar')
    feature_extractor = feature_extractor.to(device)
    
    # Load dataset
    print("Loading dataset...")
    DATA_DIR = './ChestX-ray14/images'
    TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR,
        image_list_file=TRAIN_IMAGE_LIST,
        transform=transform
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create environment
    print("Creating RL environment...")
    env = ChestXrayRLEnvironment(feature_extractor, train_dataset, device=device)
    
    # Create agent
    print("Creating RL agent...")
    agent = DiagnosisAgent(
        state_dim=feature_extractor.feature_dim,
        action_dim=14,
        hidden_dims=[512, 256]
    ).to(device)
    
    # Create trainer
    print("Creating PPO trainer...")
    trainer = PPOTrainer(agent, env, lr=3e-4)
    
    # Train
    print("Starting training...")
    trainer.train(num_iterations=1000, episodes_per_iter=100)
    
    print("Training complete!")


if __name__ == '__main__':
    # Import F for training
    import torch.nn.functional as F
    
    main()
