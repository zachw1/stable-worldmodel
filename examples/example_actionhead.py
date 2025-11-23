#!/usr/bin/env python3
from datasets import Dataset
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.data import StepsDataset, dataset_info
from stable_worldmodel.policy import AutoCostModel
from stable_worldmodel.wm.dinowm import DINOWM

import stable_pretraining as spt

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import math, time, contextlib

# TODO: add goals + better evals


# ============================================================================
# Parameters
# ============================================================================

NUM_WORKERS = 6
NUM_STEPS = 2 # T
BATCH_SIZE = 256 # B
FRAMESKIP = 5 # S
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_ACTIONS = True

# file paths
TRAIN_DIR = "pusht_expert_dataset_train"
VAL_DIR = "pusht_expert_dataset_val"
CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"

# MPC recording parameters
MPC_EPISODES = 10
MPC_SEED = 2347
MPC_DATASET_NAME = "example-pusht-mpc"
MPC_HORIZON = 5
MPC_RECEDING_HORIZON = 5
MPC_ACTION_BLOCK = 5
MPC_NUM_SAMPLES = 100  # Reduced from 300 for GPU memory
MPC_N_STEPS = 20       # Reduced from 30 for GPU memory
MPC_TOPK = 15          # Reduced from 30 for GPU memory


# Simple MLP
class MLP(nn.Module):
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


# ============================================================================
# Helpers
# ============================================================================

# Transform function: normalize + reshape
def step_transform(num_steps):
    transforms = []
    for t in range(num_steps):
        key = f"pixels.{t}"
        transforms.append(
            spt.data.transforms.Compose(
                spt.data.transforms.ToImage(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    source=key,
                    target=key,
                ),
                spt.data.transforms.Resize(224, source=key, target=key),
                spt.data.transforms.CenterCrop(224, source=key, target=key),
            )
        )
    return spt.data.transforms.Compose(*transforms)

# Preprocessing function
def load_dataset(dir, num_steps=1, frameskip=5):
    transform = step_transform(num_steps)

    dataset = StepsDataset(dir, num_steps=num_steps, transform=transform, frameskip=frameskip) # frameskip?
    # hacky
    dataset.data_dir = dataset.data_dir.parent
    attach_goal(dataset)

    return dataset

def attach_goal(steps: StepsDataset):
    data = steps.dataset
    if "goal" in data.column_names:
        return
    data = data.with_format("python")

    goal_pixels = [None] * len(data)
    goal_proprio = [None] * len(data)

    for ep, idx_slice in steps.episode_slices.items():
        final_idx = int(idx_slice[-1])
        goal_px = data["pixels"][final_idx]
        goal_pr = data["proprio"][final_idx]
        for i in idx_slice:
            goal_pixels[int(i)] = goal_px
            goal_proprio[int(i)] = goal_pr

    data = data.add_column("goal", goal_pixels)
    data = data.add_column("goal_proprio", goal_proprio)
    # print('done!')
    steps.dataset = data.with_format("torch")



def get_loaders(train_data, val_data, batch_size, device, num_workers):
    # optionally pin_memory on CUDA, not Mac
    train_loader = DataLoader(
        train_data,
            batch_size=batch_size,
        shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=(device.type=='cuda')
    )

    val_loader = DataLoader(
        val_data,
            batch_size=batch_size,
        shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=(device.type=='cuda')
        )
    return train_loader, val_loader

# load model checkpoint from cache_dir in inference mode
def load_checkpoint(checkpoint_name, device):
    cache_dir = swm.data.get_cache_dir()
    checkpoint_path = cache_dir / checkpoint_name
    model = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only false
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad_(False)
    return model

# Encoder func: call dinowm encoders
@torch.inference_mode()
def encode(batch, dinowm, device, use_actions=False):
    if device.type in ("cuda", "mps"):
        context = torch.autocast(device_type=device.type, dtype=torch.float16) # float32?
    else:
        context = contextlib.nullcontext()
    
    # no pad, we use frameskip=5
    # actions = batch["action"][::-1]
    # print(actions.shape)
    # padded_actions = pad_actions(actions)
    # repeated_actions = repeat_actions(actions)
    # print(repeated_actions.shape)

    data = {
        "pixels": batch["pixels"].to(device, non_blocking=True),
        "proprio": batch["proprio"].to(device, non_blocking=True),
        "action": (batch["action"].to(device, non_blocking=True) if use_actions else None),
    }

    with context:
        out = dinowm.encode(
            data,
            target="embed",
            pixels_key="pixels",
            proprio_key="proprio",
            action_key=("action" if use_actions else None),
        )

    z_pixels = out["pixels_embed"].mean(dim=2).float() # B x T x d_pixels (pooled by patch)
    z_proprio = out["proprio_embed"].float() # B x T x d_proprio
    z_actions = None
    if use_actions:
        z_actions = out["action_embed"][:,:-1].float() # B x (T - 1) * d_actions_effective := (d_actions * frame_skip)
        
    return z_pixels, z_proprio, z_actions

def to_feature(z_pixels, z_proprio, z_actions):
    parts = [z_pixels[:,:-1], z_proprio[:,:-1]]
    if z_actions is not None:
        parts.append(z_actions)

    # history latents (block of S actions + state := (pixel + proprio embeddings))
    z_hist = torch.cat(parts, dim=2) # B x (T-1) x d_embed := [d_pixels + d_proprio (+ d_actions_effective)]
    z_hist = torch.flatten(z_hist, start_dim=1, end_dim=2) # B x ((T-1) * d_embed)
    
    # current latents (just pixel + proprio embeedings)
    z_cur = torch.cat((z_pixels[:,-1], z_proprio[:,-1]), dim=1) # B x 1 x (d_pixels + d_proprio)

    # concat
    z = torch.cat((z_hist, z_cur), dim=1)
    return z


def record_mpc_rollouts(device=None, checkpoint_name=None, dataset_name=None, episodes=None, seed=None):
    """
    Args:
        device: torch device (auto-detected if None)
        checkpoint_name: Name of DINO-WM checkpoint (uses CHECKPOINT_NAME if None)
        dataset_name: Name for output dataset (uses MPC_DATASET_NAME if None)
        episodes: Number of episodes to record (uses MPC_EPISODES if None)
        seed: Random seed (uses MPC_SEED if None)
    """
    
    # Use defaults if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    checkpoint_name = checkpoint_name or CHECKPOINT_NAME
    dataset_name = dataset_name or MPC_DATASET_NAME
    episodes = episodes or MPC_EPISODES
    seed = seed or MPC_SEED
    
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Output dataset: {dataset_name}")
    print(f"Episodes: {episodes}, Seed: {seed}")
    
    # Create world
    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,  # Reduced from 2 to save GPU memory
        image_shape=(224, 224),
        max_episode_steps=25,
        render_mode="rgb_array",
    )
    
    # Load random dataset for preprocessors (or use existing expert dataset)
    import datasets
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    
    cache_dir = swm.data.get_cache_dir()
    
    try:
        random_ds = datasets.load_from_disk(str(cache_dir / "example-pusht"))
        print("Loaded random dataset for preprocessors")
    except:
        # Fallback to expert dataset if random doesn't exist
        try:
            random_ds = datasets.load_from_disk(str(cache_dir / TRAIN_DIR))
            print("using expert dataset for preprocessors")
        except:
            raise FileNotFoundError(
                "Could not find dataset for fitting preprocessors. "
                "Need either 'example-pusht' or expert dataset."
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
    print("Fitted preprocessors")
    
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
    print(f"\nLoading DINO-WM checkpoint: {checkpoint_name}...")
    model = swm.policy.AutoCostModel(checkpoint_name.replace("_object.ckpt", "")).to(device)
    
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
    print(f"  Horizon: {MPC_HORIZON}, Receding: {MPC_RECEDING_HORIZON}")
    print(f"  Samples: {MPC_NUM_SAMPLES}, Steps: {MPC_N_STEPS}, TopK: {MPC_TOPK}")
    
    # Record dataset
    print(f"\nRecording {episodes} episodes with MPC.")
    
    world.set_policy(policy)
    world.record_dataset(
        dataset_name,
        episodes=episodes,
        seed=seed,
        options=None,
    )
    
    print(f"\n✓ Recorded MPC dataset to: {cache_dir / dataset_name}")
    print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def run():
    # find device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize()
        if device.type=='mps':
            torch.mps.synchronize()

    print("device:", device)

    # load datasets
    train_data, val_data = load_dataset(TRAIN_DIR, NUM_STEPS), load_dataset(VAL_DIR, NUM_STEPS)
    print(f"Loaded datasets: train size={len(train_data)}, val size={len(val_data)}")

    # build loaders
    train_loader, val_loader = get_loaders(train_data, val_data, BATCH_SIZE, device, NUM_WORKERS)


    # load DINO-WM
    dinowm = load_checkpoint(CHECKPOINT_NAME, device)
    assert isinstance(dinowm, DINOWM)
    print(f"Loaded DINO-WM from checkpoint: '{CHECKPOINT_NAME}'")
        
    # calculate dims
    d_pixel = dinowm.backbone.config.hidden_size
    d_proprio = dinowm.proprio_encoder.emb_dim
    d_action = dinowm.action_encoder.emb_dim
    LATENT_DIM = (d_pixel + d_proprio) * NUM_STEPS + (d_action if USE_ACTIONS else 0) * (NUM_STEPS - 1)
    ACTION_DIM = 2 # predict actual action, not latent
    print(f'latent_dim={LATENT_DIM}, action_dim={ACTION_DIM}')

    # train action head
    action_head = MLP(LATENT_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.AdamW(action_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def report_stats():
        sync()
        elapsed = time.perf_counter() - t0
        sps = i / max(1e-9, elapsed)
        bps = n / max(1e-9, elapsed)
        eta = (num_batches - i) / max(1e-9, sps)
        print(
            f"Epoch: {epoch} Step: {i}/{len(train_loader)} "
            f"Loss = {loss.item():.4f} "
            f"steps / sec = {sps:.1f}, batches / sec = {bps:.1f} "
            f"ETA = {eta / 60.0:.1f} min"
        )

    for epoch in range(1, EPOCHS + 1):
        action_head.train()

        t0 = time.perf_counter()
        n = 0
        num_batches = len(train_loader) # could be outside loop

        for i, batch in enumerate(train_loader):

            z_pix, z_prp, z_act = encode(batch, dinowm, device, USE_ACTIONS)
            z = to_feature(z_pix, z_prp, z_act)
            action = batch['action'][:,-1,:2].to(device) # first action from the last (current) step

            pred = action_head(z)

            loss = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += BATCH_SIZE
            if i % 100 == 0:
                report_stats()
    
        # eval
        action_head.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                z_pix, z_prp, z_act = encode(batch, dinowm, device, USE_ACTIONS)
                z = to_feature(z_pix, z_prp, z_act)
                action = batch['action'][:,-1,:2].to(device)

                pred = action_head(z)
                val_loss += F.mse_loss(pred, action)
        val_rmse = math.sqrt(val_loss / len(val_data))
        print(f'epoch {epoch}: RMSE: {val_rmse:.6f}')

if __name__ == "__main__":
    import sys
    
    # check if user wants to record MPC rollouts
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        print("="*70)
        print("MPC Rollout Recording Mode")
        print("="*70)
        record_mpc_rollouts()
    else:
        # default
        run()

