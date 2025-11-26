# Testing MLP Policies in PushT Environment

Complete guide for training and testing MLP policies in the PushT environment.

## Quick Start

### 1. Train Your Models (in `train_mlp_v2.ipynb`)

The notebook already has the deterministic MLP. Add these cells for Gaussian MLP:

**Add this cell after the MLP class:**

```python
# Gaussian MLP with separate heads
class GaussianMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.1, hidden_dim=512, 
                 feature_dim=256, head_hidden_dim=128):
        super().__init__()
        self.out_dim = out_dim
        
        # Shared feature extractor
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
    
    def log_prob(self, mean, log_std, action):
        std = torch.exp(log_std)
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi)
        )
        return log_prob.sum(dim=-1)
    
    def sample(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        return mean + eps * std
```

**Add training cell for Gaussian MLP (after the regular MLP training):**

```python
# Train Gaussian MLP
gaussian_action_head = GaussianMLP(LATENT_DIM, ACTION_DIM, dropout_prob=0.15).to(device)
optimizer_gaussian = torch.optim.AdamW(
    gaussian_action_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

for epoch in range(1, EPOCHS + 1):
    gaussian_action_head.train()
    epoch_nll = 0.0
    num_samples = 0
    
    for i, batch in enumerate(train_loader):
        attach_goals(batch, train_goals)
        z_pix, z_prp, z_act, z_gpix, z_gprp = encode(batch, dinowm, device, USE_ACTIONS)
        z = to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
        action = batch['action'][:,-1,:2].to(device)
        
        mean, log_std = gaussian_action_head(z)
        log_prob = gaussian_action_head.log_prob(mean, log_std, action)
        nll_loss = -log_prob.mean()
        
        optimizer_gaussian.zero_grad()
        nll_loss.backward()
        optimizer_gaussian.step()
        
        batch_size = action.shape[0]
        epoch_nll += nll_loss.item() * batch_size
        num_samples += batch_size
        
        if i % 100 == 0:
            print(f"Epoch {epoch}, step {i}, NLL={nll_loss.item():.4f}")
    
    # Validation
    gaussian_action_head.eval()
    val_nll = 0.0
    val_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            attach_goals(batch, val_goals)
            z_pix, z_prp, z_act, z_gpix, z_gprp = encode(batch, dinowm, device, USE_ACTIONS)
            z = to_feature(z_pix, z_prp, z_act, z_gpix, z_gprp)
            action = batch['action'][:,-1,:2].to(device)
            
            mean, log_std = gaussian_action_head(z)
            log_prob = gaussian_action_head.log_prob(mean, log_std, action)
            nll = -log_prob.mean()
            
            batch_size = action.shape[0]
            val_nll += nll.item() * batch_size
            val_samples += batch_size
    
    print(f'Epoch {epoch}: Train NLL={epoch_nll/num_samples:.6f}, Val NLL={val_nll/val_samples:.6f}')
```

**Add cell to save models:**

```python
# Save trained models
torch.save(action_head.state_dict(), "mlp_action_head.pt")
print("✓ Saved MLP model to: mlp_action_head.pt")

torch.save(gaussian_action_head.state_dict(), "gaussian_mlp_action_head.pt")
print("✓ Saved Gaussian MLP model to: gaussian_mlp_action_head.pt")
```

### 2. Test in Environment

```bash
cd "/Users/anthonywong/Desktop/ml research/stable-worldmodel/examples"
python test_mlp_policies.py
```

## Architecture

### Gaussian MLP Structure

```
Input (latent state)
  ↓
Shared Feature Extractor
  → Linear(1192 → 512) + ReLU + Dropout
  → Linear(512 → 256) + ReLU + Dropout
  ↓
  ├─→ Mean Head: Linear(256 → 128) + ReLU + Dropout → Linear(128 → 2) → μ
  └─→ Log Std Head: Linear(256 → 128) + ReLU + Dropout → Linear(128 → 2) → log(σ)
```

**Key Features:**
- Common feature extractor (shared backbone)
- Splits into two separate MLP heads
- Each head has its own 128-dim hidden layer
- Dropout throughout for regularization
- Outputs Gaussian distribution N(μ, σ²)

## Command Line Options

```bash
# Basic usage
python test_mlp_policies.py

# More episodes for better statistics
python test_mlp_policies.py --episodes 20

# Record video of best policy
python test_mlp_policies.py --record-video

# Custom model paths
python test_mlp_policies.py \
    --mlp-path path/to/mlp.pt \
    --gaussian-mlp-path path/to/gaussian.pt

# All options
python test_mlp_policies.py \
    --mlp-path mlp_action_head.pt \
    --gaussian-mlp-path gaussian_mlp_action_head.pt \
    --episodes 20 \
    --num-envs 4 \
    --record-video \
    --seed 42

# See all options
python test_mlp_policies.py --help
```

## Results Interpretation

The script evaluates 3 policy modes:

1. **Deterministic MLP**: Direct action output
2. **Gaussian MLP (Mean)**: Uses mean of distribution (deterministic)
3. **Gaussian MLP (Sample)**: Samples from distribution (stochastic)

Example output:

```
======================================================================
COMPARISON SUMMARY
======================================================================
Policy                    Mean Reward     Std Reward      Episode Length
----------------------------------------------------------------------
MLP                       0.8543          0.1234          145.30
Gaussian MLP (Mean)       0.8621          0.1156          142.50
Gaussian MLP (Sample)     0.8234          0.1567          158.20
======================================================================
```

**Performance Benchmarks:**
- Expert: 0.95+ reward
- Well-trained: 0.80-0.90 reward
- Random: 0.20-0.30 reward

## Files

- `train_mlp_v2.ipynb` - Training notebook (add Gaussian MLP code here)
- `test_mlp_policies.py` - Self-contained testing script
- `mlp_models.py` - Reference implementation of models
- `README.md` - This file

## Troubleshooting

**"Model not found"**
→ Save models after training (see Step 1)

**"DINOWM checkpoint not found"**
→ Check that `dinowm_pusht_object.ckpt` is in cache directory

**Low performance**
→ Train for more epochs or use more training data

**Dimension mismatch**
→ Ensure NUM_STEPS and USE_ACTIONS match between training and testing

