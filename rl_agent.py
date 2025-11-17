# encoding: utf-8

"""
RL Agent Architectures for Chest X-ray Analysis
Implements policy and value networks for disease classification and report generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiagnosisAgent(nn.Module):
    """
    Policy network for multi-label disease classification.
    
    Takes CNN features as input and outputs disease probabilities.
    Can be trained with REINFORCE, PPO, or other policy gradient methods.
    """
    
    def __init__(self, state_dim=1024, action_dim=14, hidden_dims=[512, 256]):
        """
        Args:
            state_dim: Dimension of state (CNN features)
            action_dim: Number of diseases to predict
            hidden_dims: List of hidden layer dimensions
        """
        super(DiagnosisAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build policy network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_net = nn.Sequential(*layers)
        
        # Value network for actor-critic methods
        value_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
    
    def forward(self, state):
        """
        Forward pass through policy network.
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            action_probs: Action probabilities (batch_size, action_dim)
        """
        logits = self.policy_net(state)
        action_probs = torch.sigmoid(logits)  # Multi-label classification
        return action_probs
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: State tensor or numpy array
            deterministic: If True, return argmax action
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.forward(state)
        
        if deterministic:
            action = (action_probs > 0.5).float()
        else:
            # Sample from Bernoulli distribution for each disease
            action = torch.bernoulli(action_probs)
        
        # Compute log probability
        log_prob = self._compute_log_prob(action_probs, action)
        
        return action.squeeze(0).cpu().numpy(), log_prob.item()
    
    def _compute_log_prob(self, probs, actions):
        """Compute log probability for multi-label binary actions."""
        # For Bernoulli: log P(a|s) = sum_i [a_i * log(p_i) + (1-a_i) * log(1-p_i)]
        eps = 1e-8
        log_prob = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
        return log_prob.sum(dim=-1)
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for training.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size, action_dim)
        
        Returns:
            log_probs: Log probabilities (batch_size,)
            values: State values (batch_size,)
            entropy: Action entropy for exploration (batch_size,)
        """
        action_probs = self.forward(states)
        values = self.value_net(states).squeeze(-1)
        log_probs = self._compute_log_prob(action_probs, actions)
        
        # Entropy for exploration bonus
        eps = 1e-8
        entropy = -(action_probs * torch.log(action_probs + eps) + 
                    (1 - action_probs) * torch.log(1 - action_probs + eps)).sum(dim=-1)
        
        return log_probs, values, entropy


class ReportGenerationAgent(nn.Module):
    """
    Policy network for medical report generation.
    
    Uses LSTM to generate reports token-by-token conditioned on image features.
    """
    
    def __init__(self, state_dim=1024, vocab_size=5000, embed_dim=256, 
                 hidden_dim=512, num_layers=2):
        """
        Args:
            state_dim: Dimension of image features
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
        """
        super(ReportGenerationAgent, self).__init__()
        
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Project image features to LSTM input
        self.feature_projection = nn.Linear(state_dim, hidden_dim)
        
        # LSTM for sequential generation
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Value network for actor-critic
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, image_features, token_ids, hidden=None):
        """
        Forward pass for report generation.
        
        Args:
            image_features: Image feature tensor (batch_size, state_dim)
            token_ids: Token IDs (batch_size, seq_len)
            hidden: LSTM hidden state
        
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            hidden: Updated LSTM hidden state
        """
        batch_size = token_ids.size(0)
        
        # Initialize hidden state with image features if not provided
        if hidden is None:
            h0 = self.feature_projection(image_features).unsqueeze(0)
            h0 = h0.repeat(self.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
        
        # Embed tokens
        embeddings = self.embedding(token_ids)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(embeddings, hidden)
        
        # Project to vocabulary
        logits = self.output_projection(lstm_out)
        
        return logits, hidden
    
    def generate_token(self, image_features, prev_token, hidden=None, temperature=1.0):
        """
        Generate next token.
        
        Args:
            image_features: Image features (state_dim,)
            prev_token: Previous token ID
            hidden: LSTM hidden state
            temperature: Sampling temperature
        
        Returns:
            next_token: Sampled token ID
            log_prob: Log probability
            hidden: Updated hidden state
        """
        if isinstance(image_features, np.ndarray):
            image_features = torch.FloatTensor(image_features)
        
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        
        prev_token = torch.LongTensor([[prev_token]])
        
        logits, hidden = self.forward(image_features, prev_token, hidden)
        logits = logits.squeeze(1) / temperature
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        log_prob = F.log_softmax(logits, dim=-1)[0, next_token].item()
        
        return next_token, log_prob, hidden
    
    def evaluate_sequence(self, image_features, token_ids):
        """
        Evaluate token sequence for training.
        
        Args:
            image_features: Batch of image features (batch_size, state_dim)
            token_ids: Batch of token sequences (batch_size, seq_len)
        
        Returns:
            log_probs: Log probabilities (batch_size, seq_len)
            values: State values (batch_size,)
        """
        logits, hidden = self.forward(image_features, token_ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of actual tokens
        token_log_probs = log_probs.gather(2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Compute values
        lstm_final = hidden[0][-1]  # Last layer final hidden state
        value_input = torch.cat([image_features, lstm_final], dim=-1)
        values = self.value_net(value_input).squeeze(-1)
        
        return token_log_probs, values


class DQNAgent(nn.Module):
    """
    Deep Q-Network for discrete action spaces.
    Can be used for simpler RL tasks like binary classification per disease.
    """
    
    def __init__(self, state_dim=1024, action_dim=2, hidden_dims=[512, 256]):
        """
        Args:
            state_dim: State dimension
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
        """
        super(DQNAgent, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.q_network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Return Q-values for all actions."""
        return self.q_network(state)
    
    def get_action(self, state, epsilon=0.0):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.q_network[-1].out_features)
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
        
        return q_values.argmax().item()


if __name__ == '__main__':
    """Test agent architectures."""
    
    # Test DiagnosisAgent
    print("Testing DiagnosisAgent...")
    agent = DiagnosisAgent(state_dim=1024, action_dim=14)
    dummy_state = torch.randn(4, 1024)
    action_probs = agent(dummy_state)
    print(f"Action probabilities shape: {action_probs.shape}")
    
    action, log_prob = agent.get_action(dummy_state[0].numpy())
    print(f"Sampled action shape: {action.shape}, Log prob: {log_prob:.4f}")
    
    # Test ReportGenerationAgent
    print("\nTesting ReportGenerationAgent...")
    report_agent = ReportGenerationAgent(state_dim=1024, vocab_size=1000)
    image_features = torch.randn(2, 1024)
    tokens = torch.randint(0, 1000, (2, 10))
    logits, hidden = report_agent(image_features, tokens)
    print(f"Report generation logits shape: {logits.shape}")
    
    print("\nAll agent architectures initialized successfully!")
