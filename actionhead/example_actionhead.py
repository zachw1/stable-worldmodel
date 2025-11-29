#!/usr/bin/env python3
from typing import Any
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

import math, time, contextlib
import wandb

# TODO:
#   - Lightning/spt callbacks?
#   - move encode logic into model
#   - better OOP
#   - validate wandb + checkpoints
#   - add attentive pooler
#   X split into cached and uncached


# ============================================================================
# Parameters
# ============================================================================

# DATA PARAMS:
NUM_STEPS = 2       # T
FRAMESKIP = 5       # FIXED

# LOAD PARAMS:
BATCH_SIZE = 2048   # B
NUM_WORKERS = 16
PREFETCH_FACTOR = 2 # lower if low VRAM

# TRAINING PARAMS:
EPOCHS = 25
LEARNING_RATE = 5e-4    # 5e-4, 3e-5
WEIGHT_DECAY = 1e-4     # 1e-5
USE_ACTIONS = True

# PATHS:
TRAIN_DIR = "/dev/shm/data/pusht_expert_dataset_train"
VAL_DIR = "/dev/shm/data/pusht_expert_dataset_val"
CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"
OUTPUT_DIR = "checkpoints"


# ============================================================================
# Model to train
# ============================================================================

# Simple MLP
class MLP(nn.Module):
    '''Simple MLP action head predicting raw action from DINO-WM latents

    Inputs:
    - pixel latents: (NUM_STEPS, d_pixels)
        - d_pixels = 384 with mean pooling
    - proprio latents: (NUM_STEPS, d_proprio)
        - d_proprio = 10
    - action_latents: (NUM_STEPS - 1, d_latent)
        - d_latent = 10
        - each latent represents FRAMESKIP block of actions; t, t+1, t+2, ... t+FRAMESKIP-1
    - goal pixel latents: (d_pixels)
    - goal proprio latents: (d_proprio)

    Outputs:
    - action: (2)
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


# ============================================================================
# Helpers
# ============================================================================

def make_transform(keys):
    '''SPT transform for pixels: normalize and to tensor
    assumes 224x224 images -> removed resize/center'''
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
                # tv_tensors.Image -> Tensor
                spt.data.transforms.WrapTorchTransform(
                    transform=lambda t: t.as_subclass(torch.Tensor),
                    source=key,
                    target=key
                )
            )
        )
    return spt.data.transforms.Compose(*transforms)

def load_dataset(path, num_steps=NUM_STEPS, frameskip=FRAMESKIP):
    '''Preprocess dataset + cache goals'''
    if not Path(path).exists():
        print(f"[WARNING] Path {path} not found")
    
    transform = make_transform([f'pixels.{i}' for i in range(num_steps)])
    dataset = StepsDataset(path, num_steps=num_steps, transform=transform, frameskip=frameskip)
    print(f"Loaded dataset from {path}: size={len(dataset)}")
    # cache_dir=swm.data.get_cache_dir()?
    dataset.data_dir = dataset.data_dir.parent

    # add goal column if not there
    goals = cache_goals(dataset)

    return dataset, goals 

def cache_goals(steps: StepsDataset):
    '''Cache goal pixel + proprio into CPU-side tensors'''
    data = steps.dataset.with_format("python")
    cols = set(data.column_names)
    num_eps = len(steps.episodes)

    # pre-alloc tensors to store all goal pixels + proprio
    # assumes episodes are 1-indexed consecutively i..e 1,2,...n
    goal_pixels_tensor = torch.zeros((num_eps + 1, 3, 224, 224), dtype=torch.float32)
    goal_proprio_tensor = torch.zeros((num_eps + 1, 4), dtype=torch.float32)

    transform = make_transform(['goal_pixels'])
    for ep, indices in steps.episode_slices.items():
        last = int(indices[-1])
        
        # cache goal pixels
        goal_img_path = steps.data_dir / data["goal" if "goal" in cols else "pixels"][last]
        with Image.open(goal_img_path) as goal_img:
            info = {'goal_pixels': goal_img}
            transform(info)
            goal_pixels = info['goal_pixels']
        goal_pixels_tensor[ep] = goal_pixels

        # cache goal proprio
        goal_proprio = data["goal_proprio" if "goal_proprio" in cols else "proprio"][last]
        goal_proprio_tensor[ep] = torch.as_tensor(goal_proprio, dtype=torch.float32)

    steps.dataset = data.with_format("torch")
    return {
        "pixels": goal_pixels_tensor,
        "proprio": goal_proprio_tensor
    }

def attach_goals(batch, goals):
    """Vectorized version of old attach_goals"""
    # index into goal tensors
    ep_indices = batch["episode_idx"][:, 0].long()
    batch["goal_pixels"] = goals["pixels"][ep_indices].unsqueeze(1)     # B x 3 x 224 x 224
    batch["goal_proprio"] = goals["proprio"][ep_indices].unsqueeze(1)   # B x 10
    return batch

def load_checkpoint(checkpoint_name, device):
    '''Load model checkpoint from cache_dir in inference mode'''
    cache_dir = swm.data.get_cache_dir()
    checkpoint_path = cache_dir / checkpoint_name
    dinowm = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dinowm = dinowm.to(device).eval()
    assert isinstance(dinowm, DINOWM)
    for param in dinowm.parameters():
        param.requires_grad_(False)
    
    # optimization
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            dinowm.backbone = torch.compile(dinowm.backbone)
        except Exception as e:
            print(f"[WARNING] torch.compile failed, continuing without compile: {e}")
    
    # free VRAM
    del dinowm.predictor
    del dinowm.decoder
    torch.cuda.empty_cache()

    print(f"Loaded DINO-WM from checkpoint: '{checkpoint_name}'")
    return dinowm

@torch.inference_mode()
def encode(batch, dinowm, device, use_actions=USE_ACTIONS):
    '''Encode to latents via DINO-WM'''
    pixels = torch.cat((batch["pixels"], batch['goal_pixels']), dim=1)
    proprio = torch.cat((batch["proprio"], batch["goal_proprio"]), dim=1)
    actions = (
        torch.cat((batch["action"], torch.zeros_like(batch['action'][:, :1])), dim=1,) if use_actions else None
    ) # pad with a single step of 0s

    data = {
        "pixels": pixels,
        "proprio": proprio,
        "action": actions,
    }

    # optimization
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    context = torch.autocast(device_type=device.type, dtype=dtype) if device.type in ("cuda", "mps") else contextlib.nullcontext()
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
    '''Concatenate latents into feature'''
    parts = [z_pix[:,:-1], z_prp[:,:-1]]
    if z_act is not None:
        parts.append(z_act)

    # history latents (block of S actions + state := (pixel + proprio embeddings))
    z_hist = torch.cat(parts, dim=2) # B x (T-1) x d_embed := [d_pixels + d_proprio (+ d_actions_effective)]
    z_hist = torch.flatten(z_hist, start_dim=1, end_dim=2) # B x ((T-1) * d_embed)
    
    # current + goal latents (just pixel + proprio embeedings)
    z_cur = torch.cat((z_pix[:,-1], z_prp[:,-1], z_gpix, z_gprp), dim=1) # B x 2 * (d_pixels + d_proprio)

    # end feature
    z = torch.cat((z_hist, z_cur), dim=1)
    return z

# ============================================================================
# Loops
# ============================================================================

def train_epoch(action_head, device, dinowm, loader, goals, optimizer, epoch):
    '''Train for one epoch'''
    action_head.train()
    t0 = t_prev = time.perf_counter()
    num_batches = len(loader)

    for i, batch in enumerate(loader):
        # move CPU -> GPU
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        
        # attach goals
        attach_goals(batch, goals)
        start_event = torch.cuda.Event(enable_timing=True)
        encode_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # encode to feature
        start_event.record()
        z_pix, z_prp, z_act, z_gpix, z_gprp = encode(batch, dinowm, device)

        encode_event.record()
        z = to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
        action = batch['action'][:,-1,:2] # first action from the last (current) step

        pred = action_head(z)

        loss = F.mse_loss(pred, action)
        optimizer.zero_grad(set_to_none=True)   # optimization
        loss.backward()
        optimizer.step()

        end_event.record()

        if device.type == "cuda": torch.cuda.synchronize()
        dino_time = start_event.elapsed_time(encode_event)
        mlp_time = encode_event.elapsed_time(end_event)

        # logging
        if (i + 1) % 50 == 0:
            if device.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()

            bps_cum = (i + 1) / (t - t0)
            bps_cur = 50 / (t - t_prev)
            ips_cum = bps_cum * BATCH_SIZE
            ips_cur = bps_cur * BATCH_SIZE
            t_prev = t
            
            eta = (num_batches - (i + 1)) / (bps_cum * 60.0)

            print(f'Epoch {epoch} [{i + 1}/{num_batches}] | loss: {loss.item():.4f}, batch/s: {bps_cum:.1f}({bps_cur:.1f}), img/s: {ips_cum:.1f}({ips_cur:.1f}), ETA: {eta:.1f} min')
            print(f"Time Breakdown: DINO={dino_time:.1f}ms, MLP={mlp_time:.1f}ms")

            wandb.log({
                "train/loss": loss.item(),
                "train/rps": ips_cum,
                "train/epoch": epoch
            })

def evaluate(action_head, device, dinowm, loader, goals, epoch):
    action_head.eval()
    val_loss = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            
            attach_goals(batch, goals)
            z_pix, z_prp, z_act, z_gpix, z_gprp = encode(batch, dinowm, device)
            z = to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
            action = batch['action'][:,-1,:2]

            pred = action_head(z)
            loss = F.mse_loss(pred, action)

            batch_size = batch['action'].shape[0]
            val_loss += loss.item() * batch_size
            total += batch_size

    rmse = math.sqrt(val_loss / total)
    print(f'Epoch {epoch} | Val RMSE: {rmse:.6f}')
    wandb.log({"val/rmse": rmse, "epoch": epoch})
    return rmse


# ============================================================================
# Main Entry Point
# ============================================================================

def run():
    '''Trains action head from input datasets'''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # load datasets
    train_data, train_goals = load_dataset(TRAIN_DIR)
    val_data, val_goals = load_dataset(VAL_DIR)
    
    # move goal tensors to device
    train_goals = {k: v.to(device) for k, v in train_goals.items()}
    val_goals = {k: v.to(device) for k, v in val_goals.items()}

    # build loaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=(device.type == 'cuda')
    )

    # load DINO-WM
    dinowm = load_checkpoint(CHECKPOINT_NAME, device)
        
    # calculate dims
    d_pixel = dinowm.backbone.config.hidden_size
    d_proprio = dinowm.proprio_encoder.emb_dim
    d_action = dinowm.action_encoder.emb_dim
    
    LATENT_DIM = (d_pixel + d_proprio) * (NUM_STEPS + 1) + \
        (d_action if USE_ACTIONS else 0) * (NUM_STEPS - 1)
    ACTION_DIM = 2
    print(f'latent_dim={LATENT_DIM}, action_dim={ACTION_DIM}')

    # setup training loop
    action_head = MLP(LATENT_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.AdamW(
        action_head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True  # optimization
    )

    # logging
    wandb.init(project="pusht-actionhead", config={
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "num_steps": NUM_STEPS
    })
    wandb.watch(action_head, log_freq=100)

    # checkpoint dir
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # share memory for workers
    # train_goals["pixels"] = train_goals["pixels"].share_memory_()
    # train_goals["proprio"] = train_goals["proprio"].share_memory_()

    for epoch in range(1, EPOCHS + 1):
        # train + evaluate
        train_epoch(action_head, device, dinowm, train_loader, train_goals, optimizer, epoch)
        rmse = evaluate(action_head, device, dinowm, val_loader, val_goals, epoch)
        
        checkpoint_path = out_dir / f"epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': action_head.state_dict(),
            'val_rmse': rmse
        }, checkpoint_path)
        print(f"Saved to {checkpoint_path}")
    
    wandb.finish()
    

if __name__ == "__main__":
    run()