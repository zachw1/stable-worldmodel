#!/usr/bin/env python3
"""
Generate expert PushT datasets for action head training.

This script uses MPC with DINO-WM to generate expert demonstrations,
then splits them into training and validation sets.
"""

import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import datasets

# Parameters
CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"
TRAIN_EPISODES = 100
VAL_EPISODES = 20
SEED = 2347

# MPC parameters
MPC_HORIZON = 5
MPC_RECEDING_HORIZON = 5
MPC_ACTION_BLOCK = 5
MPC_NUM_SAMPLES = 300
MPC_N_STEPS = 30
MPC_TOPK = 30


def generate_expert_dataset(dataset_name, episodes, seed_offset=0):
    """Generate expert demonstrations using MPC policy."""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\nGenerating {dataset_name} with {episodes} episodes...")
    print(f"Device: {device}")
    
    # Create world
    world = swm.World(
        "swm/PushT-v1",
        num_envs=2,
        image_shape=(224, 224),
        max_episode_steps=25,
        render_mode="rgb_array",
    )
    
    # Load random dataset for fitting preprocessors
    cache_dir = swm.data.get_cache_dir()
    
    try:
        random_ds = datasets.load_from_disk(str(cache_dir / "example-pusht"))
        print("Loaded example-pusht for preprocessors")
    except:
        raise FileNotFoundError(
            "Could not find 'example-pusht' dataset. "
            "Please run example_pusht.py first to generate it."
        )
    
    # Fit preprocessors
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
    
    # Setup transforms
    def img_transform():
        return transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }
    
    # Load DINO-WM and create MPC policy
    print(f"Loading DINO-WM checkpoint: {CHECKPOINT_NAME}...")
    model = swm.policy.AutoCostModel(CHECKPOINT_NAME.replace("_object.ckpt", "")).to(device)
    
    print("Creating MPC policy...")
    config = swm.PlanConfig(
        horizon=MPC_HORIZON,
        receding_horizon=MPC_RECEDING_HORIZON,
        action_block=MPC_ACTION_BLOCK
    )
    
    solver = swm.solver.CEMSolver(
        model,
        num_samples=MPC_NUM_SAMPLES,
        var_scale=1.0,
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
    
    print(f"  MPC config - Horizon: {MPC_HORIZON}, Samples: {MPC_NUM_SAMPLES}, Steps: {MPC_N_STEPS}")
    
    # Record dataset
    print(f"Recording {episodes} expert episodes...")
    world.set_policy(policy)
    world.record_dataset(
        dataset_name,
        episodes=episodes,
        seed=SEED + seed_offset,
        options=None,
    )
    
    print(f"✓ Generated {dataset_name} at: {cache_dir / dataset_name}")
    return cache_dir / dataset_name


if __name__ == "__main__":
    print("="*70)
    print("EXPERT DATASET GENERATION")
    print("="*70)
    print(f"Checkpoint: {CHECKPOINT_NAME}")
    print(f"Train episodes: {TRAIN_EPISODES}")
    print(f"Val episodes: {VAL_EPISODES}")
    print(f"Seed: {SEED}")
    print("="*70)
    
    # Generate training set
    train_path = generate_expert_dataset(
        "pusht_expert_dataset_train",
        TRAIN_EPISODES,
        seed_offset=0
    )
    
    # Generate validation set (different seed)
    val_path = generate_expert_dataset(
        "pusht_expert_dataset_val",
        VAL_EPISODES,
        seed_offset=1000
    )
    
    print("\n" + "="*70)
    print("✓ EXPERT DATASETS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"Training set: {train_path}")
    print(f"Validation set: {val_path}")
    print("\nYou can now run: sbatch run_actionhead_oscar.sh")
    print("="*70)

