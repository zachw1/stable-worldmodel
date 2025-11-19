#!/usr/bin/env python3
from datasets import Dataset, Features, Sequence, Value

import stable_worldmodel as swm
from stable_worldmodel.data import StepsDataset
from stable_worldmodel.wm.dinowm import DINOWM

import stable_pretraining as spt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import contextlib

# ============================================================================
# Parameters
# ============================================================================

# CONSTANTS:
NUM_STEPS = 1   # T
FRAMESKIP = 5   # S

# HYPERPARAMETERS:
SHARD_SIZE = 10000
NUM_WORKERS = 16
BATCH_SIZE = 1024    # B
PREFETCH_FACTOR = 4

# FILE PATHS:
TRAIN_DIR = "pusht_expert_dataset_train"
VAL_DIR = "pusht_expert_dataset_val"
TRAIN_OUT = "latents_train"
VAL_OUT = "latents_val"

CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"

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

    return dataset 

def get_loader(data,
               device,
               batch_size=BATCH_SIZE,
               num_workers=NUM_WORKERS,
               prefetch_factor=PREFETCH_FACTOR):
    
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=(device.type=='cuda')
    )
    return loader

# load model checkpoint from cache_dir in inference mode
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
def encode(batch, dinowm, device):
    data = {
        "pixels":  batch["pixels"].to(device, non_blocking=True),
        "proprio": batch["proprio"].to(device, non_blocking=True),
        "action": batch["action"].to(device, non_blocking=True),
    }

    context = torch.autocast(device_type=device.type, dtype=torch.float16) if device.type in ("cuda", "mps") else contextlib.nullcontext()
    with context:
        out = dinowm.encode(
            data,
            target="embed",
            pixels_key="pixels",
            proprio_key="proprio",
            action_key="action",
        )

    # mean-pool
    z_pixels = out["pixels_embed"].mean(dim=2).float()
    z_proprio = out["proprio_embed"].float()
    z_action = out["action_embed"].float()

    return z_pixels, z_proprio, z_action

# ============================================================================
# Main
# ============================================================================

def cache_latents(in_path, out_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # load dataset
    dataset = load_dataset(in_path)
    print(f"Loaded dataset from {in_path}: size={len(dataset)}")

    # build loader
    loader = get_loader(dataset, device)

    # load DINO-WM
    dinowm = load_checkpoint(CHECKPOINT_NAME, device)
    assert isinstance(dinowm, DINOWM)
    print(f"Loaded DINO-WM from checkpoint: '{CHECKPOINT_NAME}'")
        
    # calculate dims
    d_pixels = dinowm.backbone.config.hidden_size
    d_proprio = dinowm.proprio_encoder.emb_dim
    d_action = dinowm.action_encoder.emb_dim
    action_dim = dataset[0]["action"].shape[-1]
    print(f'pixel dim={d_pixels}, proprio dim={d_proprio}, action dim={d_action}, action raw dim={action_dim}')

    features = Features({
        "episode_idx": Value("int32"),
        "step_idx": Value("int32"),
        "pixels_embed": Sequence(Sequence(Value("float32"), length=d_pixels), length=NUM_STEPS),
        "proprio_embed": Sequence(Sequence(Value("float32"), length=d_proprio), length=NUM_STEPS),
        "action_embed": Sequence(Sequence(Value("float32"), length=d_action), length=NUM_STEPS),
        "action": Sequence(Sequence(Value("float32"), length=action_dim), length=NUM_STEPS),
    })

    cache_dir = swm.data.get_cache_dir()
    out_dir = cache_dir / out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    cur_shards = [item for item in out_dir.glob('shard_*') if item.is_dir()]
    cur_shards.sort() # lexsort

    num_shards = len(cur_shards)
    num_rows = num_shards * SHARD_SIZE
    if num_rows:
        print(f'{num_shards} already exist: skipping {num_rows} rows')

    # cache
    num_skip = num_rows
    row_buffer = []
    for batch in tqdm(loader, total=len(loader), desc=f"caching from {in_path} to {out_path}"):
        batch_size = batch['action'].shape[0]
        
        # resuming logic
        if num_skip > 0:
            if batch_size <= num_skip:
                num_skip -= batch_size
                continue

            for key in batch.keys():
                batch[key] = batch[key][num_skip:]
            batch_size -= num_skip
            num_skip = 0

        # to cpu for buffer
        z_pixels, z_proprio, z_action = encode(batch, dinowm, device)
        z_pixels = z_pixels.cpu()
        z_proprio = z_proprio.cpu()
        z_action = z_action.cpu()
        actions = batch['action'].cpu().tolist()
        ep_idx = batch['episode_idx'][:,0].cpu().tolist()
        step_idx = batch['step_idx'][:,0].cpu().tolist()

        for i in range(batch_size):
            row_buffer.append({
                'episode_idx': ep_idx[i],
                'step_idx': step_idx[i],
                'pixels_embed': z_pixels[i].tolist(),
                'proprio_embed': z_proprio[i].tolist(),
                'action_embed': z_action[i].tolist(),
                'action': actions[i],
            })
            if len(row_buffer) == SHARD_SIZE:
                shard_path = out_dir / f'shard_{num_shards:05d}'
                Dataset.from_list(row_buffer, features=features).save_to_disk(str(shard_path))
                print(f'saved {str(shard_path)}')
                num_shards += 1
                row_buffer = []

    if row_buffer:
        shard_path = out_dir / f'shard_{num_shards:05d}'
        Dataset.from_list(row_buffer, features=features).save_to_disk(str(shard_path))

if __name__ == "__main__":
    # mp.set_end_method("spawn", force=True) <- if persistent_workers causes DataLoader crash
    cache_latents(VAL_DIR, VAL_OUT)
    