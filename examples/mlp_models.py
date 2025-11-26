"""
MLP Models for Action Prediction

This module contains the MLP architectures used for action prediction:
- MLP: Deterministic action prediction
- GaussianMLP: Stochastic action prediction with distribution outputs
"""

import torch
from torch import nn
import math


class MLP(nn.Module):
    """
    Simple deterministic MLP for action prediction.
    
    Architecture:
        Input → Linear(512) + ReLU → Linear(256) + ReLU → Linear(out_dim) → Output
    
    Args:
        in_dim (int): Input dimension (latent state dimension)
        out_dim (int): Output dimension (action dimension)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, in_dim]
        Returns:
            action: Predicted action [B, out_dim]
        """
        return self.layers(x)


class GaussianMLP(nn.Module):
    """
    Gaussian MLP for stochastic action prediction with uncertainty.
    
    Outputs a Gaussian distribution over actions by predicting both mean and log std.
    Uses a shared feature extractor followed by separate MLP heads.
    
    Architecture:
        Shared Backbone:
            Input → Linear(hidden_dim) + ReLU + Dropout 
                 → Linear(feature_dim) + ReLU + Dropout 
                 → Shared Features
        
        Mean Head:
            Shared Features → Linear(head_hidden_dim) + ReLU + Dropout
                           → Linear(out_dim) → Mean
        
        Log Std Head:
            Shared Features → Linear(head_hidden_dim) + ReLU + Dropout
                           → Linear(out_dim) → Log Std
    
    Args:
        in_dim (int): Input dimension (latent state dimension)
        out_dim (int): Output dimension (action dimension)
        dropout_prob (float): Dropout probability for regularization
        hidden_dim (int): Hidden dimension in shared backbone
        feature_dim (int): Dimension of shared feature representation
        head_hidden_dim (int): Hidden dimension in each head
    """
    def __init__(self, in_dim, out_dim, dropout_prob=0.1, hidden_dim=512, 
                 feature_dim=256, head_hidden_dim=128):
        super().__init__()
        self.out_dim = out_dim
        
        # Shared feature extractor backbone
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        
        # Separate MLP head for mean
        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(head_hidden_dim, out_dim)
        )
        
        # Separate MLP head for log_std
        self.log_std_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(head_hidden_dim, out_dim)
        )
        
        # Initialize log_std head output to reasonable values (log(0.5) ≈ -0.69)
        nn.init.constant_(self.log_std_head[-1].bias, -0.5)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input features [B, in_dim]
            
        Returns:
            mean: Predicted mean of action distribution [B, out_dim]
            log_std: Predicted log standard deviation [B, out_dim]
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Pass through separate MLP heads
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-10, max=2)
        
        return mean, log_std
    
    def log_prob(self, mean, log_std, action):
        """
        Compute log probability of action under the predicted Gaussian distribution.
        
        This is used for computing the negative log likelihood loss during training.
        
        Args:
            mean: Predicted mean [B, out_dim]
            log_std: Predicted log std [B, out_dim]
            action: Ground truth action [B, out_dim]
            
        Returns:
            log_prob: Log probability of the action [B]
        """
        std = torch.exp(log_std)
        var = std ** 2
        
        # Gaussian log probability: -0.5 * [(x-μ)²/σ² + log(2πσ²)]
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var +
            2 * log_std +
            math.log(2 * math.pi)
        )
        
        # Sum over action dimensions to get scalar log probability
        return log_prob.sum(dim=-1)
    
    def sample(self, mean, log_std):
        """
        Sample an action from the predicted Gaussian distribution.
        
        Uses the reparameterization trick: a = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mean: Predicted mean [B, out_dim]
            log_std: Predicted log std [B, out_dim]
            
        Returns:
            action: Sampled action [B, out_dim]
        """
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        return mean + eps * std
    
    def get_std(self, log_std):
        """
        Convert log std to std.
        
        Args:
            log_std: Predicted log std [B, out_dim]
            
        Returns:
            std: Standard deviation [B, out_dim]
        """
        return torch.exp(log_std)


# Helper function to create models
def create_mlp(latent_dim, action_dim, model_type='mlp', **kwargs):
    """
    Factory function to create MLP models.
    
    Args:
        latent_dim (int): Input dimension
        action_dim (int): Output dimension
        model_type (str): Either 'mlp' or 'gaussian'
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        model: Instantiated model
        
    Example:
        >>> mlp = create_mlp(1192, 2, model_type='mlp')
        >>> gaussian_mlp = create_mlp(1192, 2, model_type='gaussian', dropout_prob=0.15)
    """
    if model_type.lower() == 'mlp':
        return MLP(latent_dim, action_dim)
    elif model_type.lower() == 'gaussian':
        return GaussianMLP(latent_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'mlp' or 'gaussian'.")


if __name__ == "__main__":
    # Quick test
    print("Testing MLP models...")
    
    latent_dim = 1192
    action_dim = 2
    batch_size = 32
    
    # Test deterministic MLP
    mlp = MLP(latent_dim, action_dim)
    x = torch.randn(batch_size, latent_dim)
    action = mlp(x)
    print(f"✓ MLP output shape: {action.shape}")
    assert action.shape == (batch_size, action_dim)
    
    # Test Gaussian MLP
    gaussian_mlp = GaussianMLP(latent_dim, action_dim, dropout_prob=0.1)
    gaussian_mlp.eval()  # Set to eval mode for consistent dropout
    mean, log_std = gaussian_mlp(x)
    print(f"✓ Gaussian MLP mean shape: {mean.shape}")
    print(f"✓ Gaussian MLP log_std shape: {log_std.shape}")
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)
    
    # Test sampling
    sampled = gaussian_mlp.sample(mean, log_std)
    print(f"✓ Sampled action shape: {sampled.shape}")
    assert sampled.shape == (batch_size, action_dim)
    
    # Test log probability
    log_prob = gaussian_mlp.log_prob(mean, log_std, sampled)
    print(f"✓ Log probability shape: {log_prob.shape}")
    assert log_prob.shape == (batch_size,)
    
    print("\n✓ All tests passed!")
    
    # Print parameter counts
    mlp_params = sum(p.numel() for p in mlp.parameters())
    gaussian_params = sum(p.numel() for p in gaussian_mlp.parameters())
    print(f"\nParameter counts:")
    print(f"  MLP: {mlp_params:,} parameters")
    print(f"  Gaussian MLP: {gaussian_params:,} parameters")

