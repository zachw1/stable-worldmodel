#!/usr/bin/env python3
"""
Standalone script for recording MPC rollouts using a trained DINO-WM model.
"""
import stable_worldmodel as swm
import torch
import numpy as np
import time
from pathlib import Path



CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"

# Recording parameters
MPC_EPISODES = 10           # Quick test run
MPC_SEED = 42               # Different seed for variety
MPC_DATASET_NAME = "pusht-mpc-10ep"

# MPC planning parameters 
MPC_HORIZON = 8             # Shorter horizon = faster
MPC_RECEDING_HORIZON = 8    # Match horizon
MPC_ACTION_BLOCK = 5        # Must match training frameskip (5 for dinowm_pusht)
MPC_NUM_SAMPLES = 128       # Fewer samples = much faster
MPC_N_STEPS = 15            # Fewer CEM iterations
MPC_TOPK = 16               # Top-k elites
MPC_VAR_SCALE = 0.5         # Lower variance = more exploitation

# Environment parameters
MAX_EPISODE_STEPS = 185     # Shorter episodes for faster testing
IMAGE_SIZE = 224


# ============================================================================
# Main Recording Function
# ============================================================================

def record_mpc_rollouts(
    device=None,
    checkpoint_name=None,
    dataset_name=None,
    episodes=None,
    seed=None
):
    """
    Record MPC rollouts using trained DINO-WM model.
    
    Args:
        device: torch device (auto-detected if None)
        checkpoint_name: Name of DINO-WM checkpoint
        dataset_name: Name for output dataset
        episodes: Number of episodes to record
        seed: Random seed
    """
    import datasets
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    
    # Use defaults
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_name = checkpoint_name or CHECKPOINT_NAME
    dataset_name = dataset_name or MPC_DATASET_NAME
    episodes = episodes or MPC_EPISODES
    seed = seed or MPC_SEED
    
    print("="*70)
    print("MPC Rollout Recording")
    print("="*70)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Output dataset: {dataset_name}")
    print(f"Episodes: {episodes}, Seed: {seed}")
    
    start_time = time.perf_counter()
    
    # Create world
    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode="rgb_array",
    )
    
    # Load dataset for fitting preprocessors
    cache_dir = swm.data.get_cache_dir()
    
    try:
        random_ds = datasets.load_from_disk(str(cache_dir / "example-pusht"))
        print("Loaded random dataset for preprocessors")
    except:
        raise FileNotFoundError(
            "Could not find 'example-pusht' dataset for fitting preprocessors.\n"
            "Run: python generate_random_pusht.py first"
        )
    
    # Fit preprocessors (normalizers)
    print("Fitting preprocessors...")
    action_process = preprocessing.StandardScaler()
    action_process.fit(random_ds["action"][:])
    
    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(random_ds["proprio"][:])
    
    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }
    print("✓ Fitted preprocessors")
    
    # Setup image transforms
    def img_transform():
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(size=IMAGE_SIZE, antialias=True),
            transforms.CenterCrop(size=IMAGE_SIZE),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }
    
    # Load world model
    print(f"\nLoading DINO-WM checkpoint: {checkpoint_name}...")
    model = swm.policy.AutoCostModel(
        checkpoint_name.replace("_object.ckpt", "")
    ).to(device)
    
    # Create MPC policy
    print("Creating MPC policy...")
    config = swm.PlanConfig(
        horizon=MPC_HORIZON,
        receding_horizon=MPC_RECEDING_HORIZON,
        action_block=MPC_ACTION_BLOCK
    )
    
    solver = swm.solver.CEMSolver(
        model,
        num_samples=MPC_NUM_SAMPLES,
        var_scale=MPC_VAR_SCALE,
        n_steps=MPC_N_STEPS,
        topk=MPC_TOPK,
        device=str(device)
    )
    
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform
    )
    
    print(f"  Horizon: {MPC_HORIZON}, Receding: {MPC_RECEDING_HORIZON}")
    print(f"  Samples: {MPC_NUM_SAMPLES}, Steps: {MPC_N_STEPS}, TopK: {MPC_TOPK}")
    
    # Record ALL episodes in one call (so they're saved together)
    print(f"\nRecording {episodes} episodes...")
    print("="*70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    world.set_policy(policy)
    
    # Record all episodes at once
    world.record_dataset(
        dataset_name,
        episodes=episodes,
        seed=seed,
        options=None,
    )
    
    total_time = time.perf_counter() - start_time
    
    # Load dataset and compute stats
    final_ds = datasets.load_from_disk(str(cache_dir / dataset_name))
    total_transitions = len(final_ds)
    
    # Compute per-episode stats using REAL game success criteria
    # Success = position diff < 20 AND angle diff < π/9 (same as pusht.py eval_state)
    episode_indices = sorted(set(final_ds['episode_idx']))
    episode_successes = []
    episode_rewards = []
    episode_lengths = []
    
    for ep_idx in episode_indices:
        ep_data = [row for row in final_ds if row['episode_idx'] == ep_idx]
        ep_data = sorted(ep_data, key=lambda x: x['step_idx'])
        
        total_reward = sum(row['reward'] for row in ep_data if not np.isnan(row['reward']))
        episode_rewards.append(total_reward)
        episode_lengths.append(len(ep_data))
        
        # Check if terminated (environment's success signal)
        final_terminated = ep_data[-1]['terminated'] if ep_data else False
        episode_successes.append(final_terminated)
        
        # Print per-episode summary
        status = "SUCCESS" if final_terminated else "✗ FAILED"
        print(f"Episode {ep_idx}: {status} | Steps: {len(ep_data)} | Reward: {total_reward:.1f}")
    
    final_success_rate = (sum(episode_successes) / len(episode_successes)) * 100 if episode_successes else 0
    
    print()
    
    print("="*70)
    print("RECORDING COMPLETE")
    print("="*70)
    print(f"Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Episodes: {len(episode_indices)}")
    print(f"Total Transitions: {total_transitions:,}")
    print(f"Avg Steps/Episode: {np.mean(episode_lengths):.0f}")
    print(f"Avg Time/Episode: {total_time/len(episode_indices):.1f}s")
    print(f"Success Rate: {final_success_rate:.1f}% ({sum(episode_successes)}/{len(episode_indices)})")
    print(f"Average Reward: {np.mean(episode_rewards):.1f}")
    print(f"Avg Reward/Step: {np.mean(episode_rewards)/np.mean(episode_lengths):.1f}")
    print("="*70)
    print(f"Dataset saved to: {cache_dir / dataset_name}")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Sync dataset to local: rsync -avz ... ")
    print("  2. Split into train/val: python split_dataset.py --input", dataset_name)
    print("  3. Train action head: python example_actionhead.py")


if __name__ == "__main__":
    record_mpc_rollouts()

