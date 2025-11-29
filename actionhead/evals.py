from stable_worldmodel.world import World
from stable_worldmodel.policy import BasePolicy, ExpertPolicy, RandomPolicy
import contextlib
from collections import deque
import stable_worldmodel as swm
from stable_worldmodel.wm.dinowm import DINOWM

import stable_pretraining as spt

import torch
from torch import nn

NUM_ENVS=3      # N
NUM_EPISODES=3
MAX_STEPS=25

NUM_STEPS = 2       # T
FRAMESKIP = 5       # FIXED
ACTION_DIM = 2
USE_ACTIONS = True

CHECKPOINT_NAME = "dinowm_pusht_object.ckpt"

class MeanPooler:
    """
    Mean pooling
    """
    def pool(self, x: torch.Tensor):
        return x.mean(dim=2)

class MLP(nn.Module):
    """
    Simple MLP action head predicting raw action from DINO-WM latents

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
    """
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

class Actor(nn.Module):
    """
    Wrapper to transform (encode + combine) features into base model feature
    """
    def __init__(self, dinowm, head=None, pooler=None, use_actions=True):
        super().__init__()
        self.head = head if head is not None else nn.Identity()
        self.dinowm = dinowm
        self.pooler = pooler if pooler is not None else MeanPooler()
        self.use_actions = use_actions

        for p in self.dinowm.parameters():
            p.requires_grad_(False)
        self.dinowm.eval()

    def train(self, mode=True):
        # ensure DINO stays in eval mode even if head is training?
        super().train(mode)
        self.dinowm.eval()
        return self
    
    def set_head(self, head):
        self.head = head
    
    def set_pooler(self, pooler):
        self.pooler = pooler
        
    @torch.inference_mode()
    def encode(self, pixels, proprios, actions, goal_pixel, goal_proprio):
        '''Encode to latents via DINO-WM'''
        # TODO:
        #   - makes assumptions about dims, might break at inference-time?

        pix_in = torch.cat((pixels, goal_pixel), dim=1)
        prp_in = torch.cat((proprios, goal_proprio), dim=1)
        act_in = (
            torch.cat((actions, torch.zeros_like(actions[:, :1])), dim=1,) if self.use_actions else None
        ) # pad with a single step of 0s
        print(act_in.shape, pix_in.shape, prp_in.shape)

        data = {
            "pixels": pix_in,
            "proprio": prp_in,
            "action": act_in,
        }

        device = pixels.device  # infer

        dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        context = torch.autocast(device_type=device.type, dtype=dtype) if device.type in ("cuda", "mps") else contextlib.nullcontext()
        with context:
            out = self.dinowm.encode(
                data,
                target="embed",
                pixels_key="pixels",
                proprio_key="proprio",
                action_key="action",
            )

        # pooler here
        pix_out = self.pooler.pool(out["pixels_embed"]).float()
        prp_out = out["proprio_embed"].float()

        # detach goal pixels + proprio
        z_pix, z_gpix = pix_out[:,:-1], pix_out[:,-1] # B x T x d_pixels (pooled by patch), B x d_pixels
        z_prp, z_gprp = prp_out[:,:-1], prp_out[:,-1] # B x T x d_proprio, B x d_proprio
        
        z_act = None
        if self.use_actions:
            z_act = out["action_embed"][:,:-2].float() # B x (T - 1) * d_actions_effective := (d_actions * frame_skip)
        
        return z_pix, z_prp, z_act, z_gpix, z_gprp
    
    def to_feature(self, z_pix, z_prp, z_act, z_gpix, z_gprp):
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
    
    def forward(self, pixels, proprios, actions, goal_pixel, goal_proprio):
        z_pix, z_prp, z_act, z_gpix, z_gprp = self.encode(pixels, proprios, actions, goal_pixel, goal_proprio)
        z = self.to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
        return self.head(z)

class AmortizedMPCPolicy(BasePolicy):   # ExpertPolicy? not WorldModelPolicy right
    def __init__(self,
                 actor: Actor,
                 transform,
                 num_steps=NUM_STEPS,
                 device='cuda'
                 ):
        super().__init__()
        self.actor = actor
        self.transform = transform
        self.num_steps = num_steps

        self.n_envs = 0
        self.act_hist = None
        self.pix_hist = None
        self.prp_hist = None
        self.device = device
    
    def set_env(self, env):
        self.env = env
        self.n_envs = getattr(env, "num_envs", 1)
        self.act_hist = [deque(maxlen=self.num_steps - 1) for _ in range(self.n_envs)]
        self.pix_hist = [deque(maxlen=self.num_steps) for _ in range(self.n_envs)]
        self.prp_hist = [deque(maxlen=self.num_steps) for _ in range(self.n_envs)]
    
    def reset_hist(self, env_idx, pixels, proprio):
        null_act = torch.zeros(ACTION_DIM * FRAMESKIP, device=self.device, dtype=torch.float32)
        self.act_hist[env_idx] = deque([null_act for _ in range(self.num_steps - 1)])
        self.pix_hist[env_idx] = deque([pixels for _ in range(self.num_steps)])
        self.prp_hist[env_idx] = deque([proprio for _ in range(self.num_steps)])

    def get_action(self, obs, **kwargs):
        steps = obs['step_idx']
        batch = {
            'pixels': [],
            'proprios': [],
            'actions': [],
            'goal_pixels': [],
            'goal_proprios': []
        }

        for i in range(self.n_envs):
            pixels = obs['pixels'][i]
            goal_pixels = obs['goal'][i]
            proprio = obs['proprio'][i]
            goal_proprios = obs['goal_proprio'][i]

            transforms = self.transform({'pixels': pixels, 'goal': goal_pixels})
            pixels = transforms['pixels'].to(self.device)
            goal_pixels = transforms['goal'].to(self.device)
            proprio = torch.from_numpy(proprio).float().to(self.device)
            goal_proprios = torch.from_numpy(goal_proprios).float().to(self.device)

            if steps[i] == 0:
                self.reset_hist(i, pixels, proprio)
            else:
                self.pix_hist[i].append(pixels)
                self.prp_hist[i].append(proprio)
            
            batch['pixels'].append(torch.stack(list(self.pix_hist[i])))
            batch['proprios'].append(torch.stack(list(self.prp_hist[i])))
            batch['actions'].append(torch.stack(list(self.act_hist[i])))
            batch['goal_pixels'].append(goal_pixels)
            batch['goal_proprios'].append(goal_proprios)
        
        pixels_in = torch.stack(batch['pixels'])
        proprios_in = torch.stack(batch['proprios'])
        actions_in = torch.stack(batch['actions'])
        goal_pixels_in = torch.stack(batch['goal_pixels']).unsqueeze(1)
        goal_proprios_in = torch.stack(batch['goal_proprios']).unsqueeze(1)

        actions_out = self.actor(pixels_in, proprios_in, actions_in, goal_pixels_in, goal_proprios_in)
        for i in range(self.n_envs):
            self.act_hist[i].append(actions_out[i])

        return actions_out.cpu().numpy()

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

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    world = World(
        env_name="swm/PushT-v1",
        num_envs=NUM_ENVS,
        image_shape=(224,224),
        max_episode_steps=MAX_STEPS,
        render_mode="rgb_array",    #?
    )

    dinowm = load_checkpoint(CHECKPOINT_NAME, device)

    d_pixel = dinowm.backbone.config.hidden_size
    d_proprio = dinowm.proprio_encoder.emb_dim
    d_action = dinowm.action_encoder.emb_dim
    LATENT_DIM = (d_pixel + d_proprio) * (NUM_STEPS + 1) + (d_action if USE_ACTIONS else 0) * (NUM_STEPS - 1)
    mlp = MLP(in_dim=LATENT_DIM, out_dim=ACTION_DIM).to(device)

    actor = Actor(dinowm=dinowm, head=mlp)

    amortized_mpc = AmortizedMPCPolicy(actor=actor, transform=make_transform(["pixels", "goal"]), num_steps=NUM_STEPS, device=device)

    world.set_policy(amortized_mpc)
    world.evaluate(episodes=NUM_EPISODES, seed=42)
    print(world.infos.keys())
    print(world.infos['pixels'].shape)
    print(world.infos['goal'].shape)
    print(world.infos['proprio'].shape)
    print(world.infos['goal_proprio'].shape)