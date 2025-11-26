#!/usr/bin/env python3
"""
Test MLP Policies in PushT Environment

This script is self-contained with all model definitions included.
Run after training models in train_mlp_v2.ipynb
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import contextlib
from typing import Dict, Any, Optional
import argparse

import stable_worldmodel as swm
from stable_worldmodel.wm.dinowm import DINOWM


# ============================================================================
# Model Definitions (same as in training notebook)
# ============================================================================

class MLP(nn.Module):
    """Simple Deterministic MLP"""
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
        return self.layers(x)


class GaussianMLP(nn.Module):
    """Gaussian MLP with separate heads for mean and log_std"""
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
        
        nn.init.constant_(self.log_std_head[-1].bias, -0.5)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-10, max=2)
        return mean, log_std
    
    def sample(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        return mean + eps * std


# ============================================================================
# Policy Wrapper Classes
# ============================================================================

class MLPPolicy(swm.policy.Policy):
    """Wrapper to use trained MLP as a policy in the environment."""
    
    def __init__(self, mlp_model, dinowm, device, use_actions=True, num_steps=2):
        super().__init__()
        self.mlp = mlp_model
        self.dinowm = dinowm
        self.device = device
        self.use_actions = use_actions
        self.num_steps = num_steps
        
        # History for temporal encoding
        self.action_history = []
        self.pixel_history = []
        self.proprio_history = []
        
    def reset(self, seed: Optional[int] = None):
        """Reset policy state."""
        self.action_history = []
        self.pixel_history = []
        self.proprio_history = []
    
    def _encode_state(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Encode current observation into latent features."""
        pixels = obs['pixels']
        proprio = obs['proprio']
        goal_pixels = obs.get('goal', pixels)
        
        # Add batch dimension if needed
        if len(pixels.shape) == 3:
            pixels = pixels.unsqueeze(0)
            proprio = proprio.unsqueeze(0)
            goal_pixels = goal_pixels.unsqueeze(0)
        
        # Build history
        self.pixel_history.append(pixels)
        self.proprio_history.append(proprio)
        
        if len(self.pixel_history) > self.num_steps:
            self.pixel_history.pop(0)
            self.proprio_history.pop(0)
        
        while len(self.pixel_history) < self.num_steps:
            self.pixel_history.insert(0, pixels.clone())
            self.proprio_history.insert(0, proprio.clone())
        
        # Stack into sequences
        pixels_seq = torch.cat(self.pixel_history + [goal_pixels], dim=0)
        proprio_seq = torch.cat(self.proprio_history + [proprio], dim=0)
        
        # Add batch dimension
        pixels_seq = pixels_seq.unsqueeze(0)  # [1, T+1, C, H, W]
        proprio_seq = proprio_seq.unsqueeze(0)  # [1, T+1, D]
        
        # Prepare actions if needed
        actions = None
        if self.use_actions and len(self.action_history) > 0:
            action_hist = list(self.action_history[-self.num_steps:])
            while len(action_hist) < self.num_steps:
                action_hist.insert(0, torch.zeros(1, 2, device=self.device))
            actions = torch.stack(action_hist + [torch.zeros(1, 2, device=self.device)], dim=1)
        
        # Encode with DINOWM
        data = {
            "pixels": pixels_seq.to(self.device),
            "proprio": proprio_seq.to(self.device),
        }
        if actions is not None:
            data["action"] = actions.to(self.device)
        
        with torch.no_grad():
            context = (torch.autocast(device_type=self.device.type, dtype=torch.float16) 
                      if self.device.type in ("cuda", "mps") else contextlib.nullcontext())
            with context:
                out = self.dinowm.encode(
                    data,
                    target="embed",
                    pixels_key="pixels",
                    proprio_key="proprio",
                    action_key=("action" if actions is not None else None),
                )
        
        # Extract and combine features
        pix_out = out["pixels_embed"].mean(dim=2).float()
        prp_out = out["proprio_embed"].float()
        
        z_pix, z_gpix = pix_out[:, :-1], pix_out[:, -1]
        z_prp, z_gprp = prp_out[:, :-1], prp_out[:, -1]
        
        if self.use_actions and actions is not None:
            z_act = out["action_embed"][:, :-2].float()
            z_hist = torch.cat([z_pix[:, :-1], z_prp[:, :-1], z_act], dim=2)
        else:
            z_hist = torch.cat([z_pix[:, :-1], z_prp[:, :-1]], dim=2)
        
        z_hist = torch.flatten(z_hist, start_dim=1, end_dim=2)
        z_cur = torch.cat([z_pix[:, -1], z_prp[:, -1], z_gpix, z_gprp], dim=1)
        z = torch.cat([z_hist, z_cur], dim=1)
        
        return z
    
    def __call__(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict action from observation."""
        z = self._encode_state(obs)
        
        with torch.no_grad():
            action = self.mlp(z)
        
        self.action_history.append(action.clone())
        if len(self.action_history) > self.num_steps:
            self.action_history.pop(0)
        
        return action.cpu().numpy()


class GaussianMLPPolicy(MLPPolicy):
    """Wrapper to use trained Gaussian MLP as a policy."""
    
    def __init__(self, gaussian_mlp, dinowm, device, use_actions=True, 
                 num_steps=2, stochastic=True):
        super().__init__(gaussian_mlp, dinowm, device, use_actions, num_steps)
        self.stochastic = stochastic
    
    def __call__(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict action from observation."""
        z = self._encode_state(obs)
        
        with torch.no_grad():
            mean, log_std = self.mlp(z)
            action = self.mlp.sample(mean, log_std) if self.stochastic else mean
        
        self.action_history.append(action.clone())
        if len(self.action_history) > self.num_steps:
            self.action_history.pop(0)
        
        return action.cpu().numpy()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test MLP policies in PushT environment')
    parser.add_argument('--mlp-path', type=str, default='mlp_action_head.pt',
                       help='Path to saved MLP model')
    parser.add_argument('--gaussian-mlp-path', type=str, default='gaussian_mlp_action_head.pt',
                       help='Path to saved Gaussian MLP model')
    parser.add_argument('--checkpoint', type=str, default='dinowm_pusht_object.ckpt',
                       help='DINOWM checkpoint name')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=2,
                       help='Number of history steps')
    parser.add_argument('--use-actions', action='store_true', default=True,
                       help='Use action history in encoding')
    parser.add_argument('--record-video', action='store_true',
                       help='Record video of best policy')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load DINOWM
    print(f"\nLoading DINOWM checkpoint: {args.checkpoint}")
    cache_dir = swm.data.get_cache_dir()
    checkpoint_path = cache_dir / args.checkpoint
    dinowm = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dinowm = dinowm.to(device).eval()
    
    for param in dinowm.parameters():
        param.requires_grad_(False)
    
    print(f"Loaded DINOWM from: {checkpoint_path}")
    
    # Calculate dimensions
    d_pixel = dinowm.backbone.config.hidden_size
    d_proprio = dinowm.proprio_encoder.emb_dim
    d_action = dinowm.action_encoder.emb_dim
    
    LATENT_DIM = ((d_pixel + d_proprio) * (args.num_steps + 1) + 
                  (d_action if args.use_actions else 0) * (args.num_steps - 1))
    ACTION_DIM = 2
    
    print(f"Latent dim: {LATENT_DIM}, Action dim: {ACTION_DIM}")
    
    # Load MLP models
    mlp_available = False
    gaussian_mlp_available = False
    
    try:
        mlp_model = MLP(LATENT_DIM, ACTION_DIM).to(device)
        mlp_model.load_state_dict(torch.load(args.mlp_path, map_location=device))
        mlp_model.eval()
        print(f"✓ Loaded MLP model from: {args.mlp_path}")
        mlp_available = True
    except FileNotFoundError:
        print(f"✗ MLP model not found at {args.mlp_path}")
    
    try:
        gaussian_mlp_model = GaussianMLP(LATENT_DIM, ACTION_DIM, dropout_prob=0.15).to(device)
        gaussian_mlp_model.load_state_dict(torch.load(args.gaussian_mlp_path, map_location=device))
        gaussian_mlp_model.eval()
        print(f"✓ Loaded Gaussian MLP model from: {args.gaussian_mlp_path}")
        gaussian_mlp_available = True
    except FileNotFoundError:
        print(f"✗ Gaussian MLP model not found at {args.gaussian_mlp_path}")
    
    if not mlp_available and not gaussian_mlp_available:
        print("\nNo models available. Train models first in train_mlp_v2.ipynb")
        return
    
    # Create PushT environment
    print(f"\nCreating PushT environment with {args.num_envs} parallel environments...")
    world = swm.World(
        "swm/PushT-v1",
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=300,
        render_mode="rgb_array",
    )
    print(f"Environment created successfully")
    
    # Evaluation results storage
    results = {}
    
    # Evaluate Deterministic MLP
    if mlp_available:
        print("\n" + "="*70)
        print("Evaluating Deterministic MLP Policy")
        print("="*70)
        
        mlp_policy = MLPPolicy(
            mlp_model=mlp_model,
            dinowm=dinowm,
            device=device,
            use_actions=args.use_actions,
            num_steps=args.num_steps
        )
        
        world.set_policy(mlp_policy)
        mlp_results = world.evaluate(episodes=args.episodes, seed=args.seed)
        results['MLP'] = mlp_results
        
        print(f"\nResults:")
        print(f"  Mean Reward: {mlp_results['mean_reward']:.4f} ± {mlp_results['std_reward']:.4f}")
        print(f"  Mean Episode Length: {mlp_results['mean_episode_length']:.2f}")
    
    # Evaluate Gaussian MLP (Deterministic)
    if gaussian_mlp_available:
        print("\n" + "="*70)
        print("Evaluating Gaussian MLP Policy (Deterministic - Using Mean)")
        print("="*70)
        
        gaussian_policy_det = GaussianMLPPolicy(
            gaussian_mlp=gaussian_mlp_model,
            dinowm=dinowm,
            device=device,
            use_actions=args.use_actions,
            num_steps=args.num_steps,
            stochastic=False
        )
        
        world.set_policy(gaussian_policy_det)
        gaussian_det_results = world.evaluate(episodes=args.episodes, seed=args.seed)
        results['Gaussian MLP (Mean)'] = gaussian_det_results
        
        print(f"\nResults:")
        print(f"  Mean Reward: {gaussian_det_results['mean_reward']:.4f} ± {gaussian_det_results['std_reward']:.4f}")
        print(f"  Mean Episode Length: {gaussian_det_results['mean_episode_length']:.2f}")
    
    # Evaluate Gaussian MLP (Stochastic)
    if gaussian_mlp_available:
        print("\n" + "="*70)
        print("Evaluating Gaussian MLP Policy (Stochastic - Sampling)")
        print("="*70)
        
        gaussian_policy_stoch = GaussianMLPPolicy(
            gaussian_mlp=gaussian_mlp_model,
            dinowm=dinowm,
            device=device,
            use_actions=args.use_actions,
            num_steps=args.num_steps,
            stochastic=True
        )
        
        world.set_policy(gaussian_policy_stoch)
        gaussian_stoch_results = world.evaluate(episodes=args.episodes, seed=args.seed)
        results['Gaussian MLP (Sample)'] = gaussian_stoch_results
        
        print(f"\nResults:")
        print(f"  Mean Reward: {gaussian_stoch_results['mean_reward']:.4f} ± {gaussian_stoch_results['std_reward']:.4f}")
        print(f"  Mean Episode Length: {gaussian_stoch_results['mean_episode_length']:.2f}")
    
    # Print comparison summary
    if results:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Policy':<25} {'Mean Reward':<15} {'Std Reward':<15} {'Episode Length':<15}")
        print("-"*70)
        for policy_name, result in results.items():
            print(f"{policy_name:<25} {result['mean_reward']:<15.4f} "
                  f"{result['std_reward']:<15.4f} {result['mean_episode_length']:<15.2f}")
        print("="*70)
    
    # Record video if requested
    if args.record_video and results:
        print("\nRecording video of best performing policy...")
        
        best_policy_name = max(results.keys(), key=lambda k: results[k]['mean_reward'])
        print(f"Best policy: {best_policy_name}")
        
        if best_policy_name == 'MLP':
            world.set_policy(mlp_policy)
        elif best_policy_name == 'Gaussian MLP (Mean)':
            world.set_policy(gaussian_policy_det)
        else:
            world.set_policy(gaussian_policy_stoch)
        
        dataset_name = f"eval_{best_policy_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        world.record_dataset(dataset_name, episodes=3, seed=args.seed)
        world.record_video_from_dataset("./", dataset_name, episode_idx=[0, 1, 2])
        print(f"Video saved for {best_policy_name}!")


if __name__ == "__main__":
    main()
