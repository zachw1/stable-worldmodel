#!/usr/bin/env python3
from datasets import Dataset
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.data import StepsDataset
from stable_worldmodel.wm.dinowm import DINOWM

import stable_pretraining as spt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import math, time, contextlib, functools

# TODO:
#   - Lightning/spt callbacks?
#   - split into cached and uncached
#   - add attentive pooler


# ============================================================================
# Parameters
# ============================================================================

# DATA PARAMS:
NUM_STEPS = 2       # T
FRAMESKIP = 5       # FIXED

# LOAD PARAMS:
BATCH_SIZE = 1024   # B
NUM_WORKERS = 16
PREFETCH_FACTOR = 4

# TRAINING PARAMS:
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_ACTIONS = True

# PATHS:
TRAIN_DIR = "pusht_expert_dataset_train"
VAL_DIR = "pusht_expert_dataset_val"
CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"


# ============================================================================
# Model to train
# ============================================================================

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
def make_transform(keys):
    transforms = []
    for key in keys:
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
def load_dataset(dir, num_steps=NUM_STEPS, frameskip=FRAMESKIP):
    transform = make_transform([f'pixels.{i}' for i in range(num_steps)])
    dataset = StepsDataset(dir, num_steps=num_steps, transform=transform, frameskip=frameskip)
    # cache_dir=swm.data.get_cache_dir()?
    dataset.data_dir = dataset.data_dir.parent

    # add goal column if not there
    goals = cache_goals(dataset)

    return dataset, goals 

def cache_goals(steps: StepsDataset):
    data = steps.dataset.with_format("python")
    columns = set(data.column_names)

    transform = make_transform(['goal_pixels'])
    goals = {}  # {episode -> {goal pixel, goal_proprio}}

    for ep, indices in steps.episode_slices.items():
        end = indices[-1]
        goal_img_path = steps.data_dir / data["goal" if "goal" in columns else "pixels"][end]

        with Image.open(goal_img_path) as img:
            info = {'goal_pixels': img}
            transform(info)
            goal_pixels = info['goal_pixels']

        goal_proprio = data["goal_proprio" if "goal_proprio" in columns else "proprio"][end]
        goal_proprio = torch.as_tensor(goal_proprio, dtype=torch.float32).clone()

        goals[ep] = {
            "goal_pixels": goal_pixels,
            "goal_proprio": goal_proprio,
        }

    steps.dataset = data.with_format("torch")
    return goals

def attach_goals(batch, goals):
    goal_pixels = []
    goal_proprios = []

    for eps in batch["episode_idx"].tolist():
        goal = goals[eps[0]]
        goal_pixels.append(goal["goal_pixels"])
        goal_proprios.append(goal["goal_proprio"])

    batch["goal_pixels"] = torch.stack(goal_pixels).unsqueeze(1)
    batch["goal_proprio"] = torch.stack(goal_proprios).unsqueeze(1)
    return batch

def get_loader(data,
               device,
               batch_size=BATCH_SIZE,
               num_workers=NUM_WORKERS,
               prefetch_factor=PREFETCH_FACTOR,
               shuffle=True):
    
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=(device.type=='cuda')
    )
    return loader

# Load model checkpoint from cache_dir in inference mode
def load_checkpoint(checkpoint_name, device):
    cache_dir = swm.data.get_cache_dir()
    checkpoint_path = cache_dir / checkpoint_name
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad_(False)
    return model

# Encoder func: call dinowm encoders
@torch.inference_mode()
def encode(batch, dinowm, device, use_actions=USE_ACTIONS):
    # this is hacky
    pixels = torch.cat((batch["pixels"],
                        batch['goal_pixels']), dim=1).to(device, non_blocking=True)
    proprio = torch.cat((batch["proprio"],
                         batch["goal_proprio"]), dim=1).to(device, non_blocking=True)
    actions = (
        torch.cat(
            (batch["action"], torch.zeros_like(batch['action'][:, :1])),
            dim=1,
        ).to(device, non_blocking=True) if use_actions else None
    ) # pad with a single step of 0s

    data = {
        "pixels": pixels,
        "proprio": proprio,
        "action": actions,
    }

    context = torch.autocast(device_type=device.type, dtype=torch.float16) if device.type in ("cuda", "mps") else contextlib.nullcontext()
    with context:
        out = dinowm.encode(
            data,
            target="embed",
            pixels_key="pixels",
            proprio_key="proprio",
            action_key=("action" if use_actions else None),
        )

    # attach attention pooler here
    pix_out = out["pixels_embed"].mean(dim=2).float()
    prp_out = out["proprio_embed"].float()

    # detach goal pixels + proprio
    z_pix, z_gpix = pix_out[:,:-1], pix_out[:,-1] # B x T x d_pixels (pooled by patch), B x d_pixels
    z_prp, z_gprp = prp_out[:,:-1], prp_out[:,-1] # B x T x d_proprio, B x d_proprio
    
    z_act = None
    if use_actions:
        z_act = out["action_embed"][:,:-2].float() # B x (T - 1) * d_actions_effective := (d_actions * frame_skip)
    
    return z_pix, z_prp, z_act, z_gpix, z_gprp

def to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp):
    parts = [z_pix[:,:-1], z_prp[:,:-1]]
    if z_act is not None:
        parts.append(z_act)

    # history latents (block of S actions + state := (pixel + proprio embeddings))
    z_hist = torch.cat(parts, dim=2) # B x (T-1) x d_embed := [d_pixels + d_proprio (+ d_actions_effective)]
    z_hist = torch.flatten(z_hist, start_dim=1, end_dim=2) # B x ((T-1) * d_embed)
    
    # current + goal latents (just pixel + proprio embeedings)
    z_cur = torch.cat((z_pix[:,-1], z_prp[:,-1], z_gpix, z_gprp), dim=1) # B x 2 * (d_pixels + d_proprio)

    # concat
    z = torch.cat((z_hist, z_cur), dim=1)
    return z


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
    train_data, train_goals = load_dataset(TRAIN_DIR)
    val_data, val_goals = load_dataset(VAL_DIR)
    print(f"Loaded datasets from {TRAIN_DIR}, {VAL_DIR}:\ntrain size={len(train_data)}, val size={len(val_data)}")

    # build loaders
    train_loader, val_loader = get_loader(train_data, device), get_loader(val_data, device, shuffle=False)

    # load DINO-WM
    dinowm = load_checkpoint(CHECKPOINT_NAME, device)
    assert isinstance(dinowm, DINOWM)
    print(f"Loaded DINO-WM from checkpoint: '{CHECKPOINT_NAME}'")
        
    # calculate dims
    d_pixel = dinowm.backbone.config.hidden_size
    d_proprio = dinowm.proprio_encoder.emb_dim
    d_action = dinowm.action_encoder.emb_dim
    
    LATENT_DIM = (d_pixel + d_proprio) * (NUM_STEPS + 1) + (d_action if USE_ACTIONS else 0) * (NUM_STEPS - 1)
    ACTION_DIM = 2
    print(f'latent_dim={LATENT_DIM}, action_dim={ACTION_DIM}')

    # setup training loop
    action_head = MLP(LATENT_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.AdamW(action_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def report_stats():
        sync()
        elapsed = time.perf_counter() - t0
        sps = i / max(1e-9, elapsed)
        bps = n / max(1e-9, elapsed)
        eta = (num_batches - i) / max(1e-9, sps)
        print(
            f"Epoch {epoch}: step {i}/{len(train_loader)} "
            f"Loss = {loss.item():.4f} "
            f"steps / sec = {sps:.1f}, batches / sec = {bps:.1f} "
            f"ETA = {eta / 60.0:.1f} min"
        )

    num_batches = len(train_loader)
    for epoch in range(1, EPOCHS + 1):
        action_head.train()
        t0 = time.perf_counter()
        n = 0

        # train
        for i, batch in enumerate(train_loader):
            attach_goals(batch, train_goals)
            z_pix, z_prp, z_act, z_gpix, z_gprp = encode(batch, dinowm, device)
            z = to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
            action = batch['action'][:,-1,:2].to(device) # first action from the last (current) step

            pred = action_head(z)

            loss = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += BATCH_SIZE # partial batches this breaks but whatever
            if i % 100 == 0:
                report_stats()
        
        # eval
        action_head.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                attach_goals(batch, val_goals)
                z_pix, z_prp, z_act, z_gpix, z_gprp = encode(batch, dinowm, device)
                z = to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
                action = batch['action'][:,-1,:2].to(device)

                pred = action_head(z)
                loss = F.mse_loss(pred, action)
                batch_size = batch['action'].shape[0]
                val_loss += loss.item() * batch_size

        val_rmse = math.sqrt(val_loss / len(val_data))
        print(f'epoch {epoch}: RMSE: {val_rmse:.6f}')

if __name__ == "__main__":
    run()
