# encoding: utf-8

"""
Reinforcement Learning Environment for Chest X-ray Diagnosis
Defines state (CNN features), action space (disease predictions), and reward functions.
"""

import torch
import numpy as np
from rl_feature_extractor import CheXNetFeatureExtractor


class ChestXrayRLEnvironment:
    """
    RL Environment for Chest X-ray Diagnosis.
    
    State: CNN feature vector (1024-dim from DenseNet121)
    Action: Multi-label disease predictions (14 diseases)
    Reward: Based on prediction accuracy and clinical utility
    """
    
    def __init__(self, feature_extractor, dataset, device='cpu'):
        """
        Args:
            feature_extractor: CheXNetFeatureExtractor instance
            dataset: ChestXrayDataSet instance for images and labels
            device: 'cpu' or 'cuda'
        """
        self.feature_extractor = feature_extractor.to(device)
        self.dataset = dataset
        self.device = device
        
        # Environment parameters
        self.n_classes = 14
        self.feature_dim = feature_extractor.feature_dim
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        
        # Current episode state
        self.current_index = 0
        self.current_state = None
        self.current_labels = None
        self.done = False
        
    def reset(self, index=None):
        """
        Reset environment to a new image.
        
        Args:
            index: Specific dataset index, or None for random
        
        Returns:
            state: Feature vector (feature_dim,)
        """
        if index is None:
            self.current_index = np.random.randint(0, len(self.dataset))
        else:
            self.current_index = index
        
        # Get image and labels
        image, labels = self.dataset[self.current_index]
        self.current_labels = labels.numpy()
        
        # Extract features as state
        if len(image.shape) == 4:  # TenCrop or batch
            # Use center crop for single state representation
            image = image[0] if image.shape[0] > 1 else image.squeeze(0)
        
        image_batch = image.unsqueeze(0).to(self.device)
        # Detach from graph before converting to NumPy to avoid runtime error
        self.current_state = (
            self.feature_extractor(image_batch)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        
        self.done = False
        return self.current_state.copy()
    
    def step(self, action):
        """
        Execute action in environment.
        
        Args:
            action: Disease probability predictions (n_classes,)
        
        Returns:
            next_state: Next state (None for episodic task)
            reward: Reward signal
            done: Whether episode is finished
            info: Additional information
        """
        # Convert action to binary predictions
        action_binary = (action > 0.5).astype(np.float32)
        
        # Calculate reward
        reward, info = self.compute_reward(action_binary, action)
        
        # Episode ends after one prediction (episodic task)
        self.done = True
        next_state = None
        
        return next_state, reward, self.done, info
    
    def compute_reward(self, action_binary, action_probs):
        """
        Compute reward based on prediction accuracy and clinical utility.
        
        Args:
            action_binary: Binary predictions (n_classes,)
            action_probs: Probability predictions (n_classes,)
        
        Returns:
            reward: Scalar reward
            info: Dictionary with detailed metrics
        """
        # Accuracy-based reward
        correct = (action_binary == self.current_labels).astype(np.float32)
        accuracy = correct.mean()
        
        # Per-class accuracy
        per_class_accuracy = correct
        
        # Weighted reward: penalize false negatives more (clinical importance)
        # False negative: predict 0 when label is 1 (miss a disease)
        # False positive: predict 1 when label is 0 (false alarm)
        false_negatives = ((action_binary == 0) & (self.current_labels == 1)).astype(np.float32)
        false_positives = ((action_binary == 1) & (self.current_labels == 0)).astype(np.float32)
        
        fn_penalty = -2.0  # Higher penalty for missing diseases
        fp_penalty = -0.5  # Lower penalty for false alarms
        
        reward = accuracy + (false_negatives.sum() * fn_penalty) + (false_positives.sum() * fp_penalty)
        
        # Normalize reward to reasonable range
        reward = reward / self.n_classes
        
        # Bonus for high confidence correct predictions
        confidence_correct = np.where(correct, np.abs(action_probs - 0.5), 0)
        confidence_bonus = confidence_correct.mean() * 0.5
        reward += confidence_bonus
        
        info = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy,
            'false_negatives': false_negatives.sum(),
            'false_positives': false_positives.sum(),
            'true_labels': self.current_labels.copy(),
            'predictions': action_binary.copy(),
            'confidence': action_probs.copy()
        }
        
        return reward, info
    
    def get_state_dim(self):
        """Return state dimension."""
        return self.feature_dim
    
    def get_action_dim(self):
        """Return action dimension."""
        return self.n_classes
    
    def get_class_names(self):
        """Return disease class names."""
        return self.class_names


class ReportGenerationEnvironment:
    """
    RL Environment for Medical Report Generation.
    
    State: CNN features + partial report embedding
    Action: Next word/token in report
    Reward: Report quality metrics (BLEU, clinical accuracy)
    """
    
    def __init__(self, feature_extractor, dataset, vocab, max_length=100, device='cpu'):
        """
        Args:
            feature_extractor: CheXNetFeatureExtractor instance
            dataset: ChestXrayDataSet with reports
            vocab: Vocabulary for tokenization
            max_length: Maximum report length
            device: 'cpu' or 'cuda'
        """
        self.feature_extractor = feature_extractor.to(device)
        self.dataset = dataset
        self.vocab = vocab
        self.max_length = max_length
        self.device = device
        
        self.feature_dim = feature_extractor.feature_dim
        self.vocab_size = len(vocab)
        
        # Current episode state
        self.current_image_features = None
        self.current_report_tokens = []
        self.target_report_tokens = []
        self.timestep = 0
        self.done = False
    
    def reset(self, index=None):
        """
        Reset to new image and target report.
        
        Returns:
            state: Combined image features and report context
        """
        if index is None:
            self.current_index = np.random.randint(0, len(self.dataset))
        else:
            self.current_index = index
        
        # Extract image features
        image, report_text = self.dataset[self.current_index]
        if len(image.shape) == 4:
            image = image[0] if image.shape[0] > 1 else image.squeeze(0)
        
        image_batch = image.unsqueeze(0).to(self.device)
        # Detach from graph before converting to NumPy
        self.current_image_features = (
            self.feature_extractor(image_batch)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        
        # Initialize report generation
        self.current_report_tokens = [self.vocab['<START>']]
        self.target_report_tokens = self._tokenize_report(report_text)
        self.timestep = 0
        self.done = False
        
        return self._get_state()
    
    def step(self, action):
        """
        Generate next token.
        
        Args:
            action: Token ID to append
        
        Returns:
            next_state, reward, done, info
        """
        # Append token to report
        self.current_report_tokens.append(action)
        self.timestep += 1
        
        # Check if done
        if action == self.vocab['<END>'] or self.timestep >= self.max_length:
            self.done = True
            reward = self._compute_report_reward()
        else:
            # Intermediate reward for correct token
            if self.timestep < len(self.target_report_tokens):
                reward = 1.0 if action == self.target_report_tokens[self.timestep] else -0.1
            else:
                reward = -0.1  # Penalty for exceeding target length
        
        next_state = self._get_state() if not self.done else None
        
        info = {
            'generated_report': self.current_report_tokens.copy(),
            'target_report': self.target_report_tokens.copy(),
            'timestep': self.timestep
        }
        
        return next_state, reward, self.done, info
    
    def _get_state(self):
        """Combine image features with partial report context."""
        # For simplicity, concatenate image features with last token embedding
        # In practice, use more sophisticated context (LSTM hidden state, etc.)
        return self.current_image_features
    
    def _tokenize_report(self, report_text):
        """Convert report text to token IDs."""
        # Placeholder: implement actual tokenization
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in report_text.split()]
    
    def _compute_report_reward(self):
        """Compute final reward based on report quality."""
        # Placeholder: implement BLEU, METEOR, or clinical accuracy metrics
        # For now, use simple sequence matching
        correct_tokens = sum(1 for g, t in zip(self.current_report_tokens, self.target_report_tokens) if g == t)
        reward = correct_tokens / max(len(self.target_report_tokens), 1)
        return reward
    
    def get_state_dim(self):
        return self.feature_dim
    
    def get_action_dim(self):
        return self.vocab_size


if __name__ == '__main__':
    """Test RL environment."""
    print("RL Environment module loaded successfully")
    print("ChestXrayRLEnvironment: Multi-label disease classification")
    print("ReportGenerationEnvironment: Medical report generation")
