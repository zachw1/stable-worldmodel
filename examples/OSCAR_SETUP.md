# Running example_actionhead.py on Oscar

This guide walks you through setting up and running the action head training on Brown University's Oscar GPU cluster.

## Prerequisites

- Oscar account with GPU access
- SSH access to Oscar
- Your data files ready to transfer

## Setup Instructions

### 1. Transfer Files to Oscar

From your local machine:

```bash
# Transfer the entire stable-worldmodel directory
rsync -avz "/Users/anthonywong/Desktop/ml research/stable-worldmodel/" <your_username>@ssh.ccv.brown.edu:~/stable-worldmodel/

# Make sure to also transfer your data directories
# Adjust paths as needed for your datasets
rsync -avz "/path/to/pusht_expert_dataset_train/" <your_username>@ssh.ccv.brown.edu:~/stable-worldmodel/data/pusht_expert_dataset_train/
rsync -avz "/path/to/pusht_expert_dataset_val/" <your_username>@ssh.ccv.brown.edu:~/stable-worldmodel/data/pusht_expert_dataset_val/

# Transfer the checkpoint file
rsync -avz "/path/to/dinowm_pusht_object.ckpt" <your_username>@ssh.ccv.brown.edu:~/stable-worldmodel/dinowm_pusht_object.ckpt
```

### 2. SSH into Oscar

```bash
ssh <your_username>@ssh.ccv.brown.edu
```

### 3. Set Up Python Environment

```bash
# Load Python module (check available versions with: module avail python)
module load python/3.11.0

# Create a virtual environment
python -m venv ~/venvs/worldmodel-env
source ~/venvs/worldmodel-env/bin/activate

# Install dependencies
cd ~/stable-worldmodel

# Install the main packages
pip install --upgrade pip
pip install -e .

# Install additional dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install datasets scikit-learn

# If stable-pretraining is needed, install it
cd ~/stable-pretraining  # if you have this
pip install -e .
# OR if it's on PyPI:
# pip install stable-pretraining
```

### 4. Update Data Paths (if needed)

Edit `example_actionhead.py` to use absolute paths if your data is in a different location:

```python
# Change these lines in example_actionhead.py if needed:
TRAIN_DIR = "/users/<your_username>/stable-worldmodel/data/pusht_expert_dataset_train"
VAL_DIR = "/users/<your_username>/stable-worldmodel/data/pusht_expert_dataset_val"
CHECKPOINT_NAME = "/users/<your_username>/stable-worldmodel/dinowm_pusht_object.ckpt"
```

### 5. Make Scripts Executable

```bash
cd ~/stable-worldmodel/examples
chmod +x run_actionhead_oscar.sh
chmod +x run_mpc_recording_oscar.sh
```

## Running Jobs

### Train Action Head

```bash
cd ~/stable-worldmodel/examples
sbatch run_actionhead_oscar.sh
```

**Job Configuration:**
- GPU: 1 GPU
- CPUs: 8 cores
- Memory: 32GB
- Time: 6 hours

### Record MPC Rollouts

```bash
cd ~/stable-worldmodel/examples
sbatch run_mpc_recording_oscar.sh
```

**Job Configuration:**
- GPU: 1 GPU
- CPUs: 4 cores
- Memory: 16GB
- Time: 2 hours

## Monitoring Your Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View all GPU partition jobs
allq gpu

# Get detailed job info
scontrol show job <job_id>
```

### View Logs

```bash
cd ~/stable-worldmodel/examples/logs

# Watch output in real-time
tail -f actionhead_<job_id>.out

# Watch errors in real-time
tail -f actionhead_<job_id>.err

# View completed job logs
cat actionhead_<job_id>.out
```

### Cancel a Job

```bash
scancel <job_id>
```

## Interactive Testing

If you want to test interactively before submitting a batch job:

```bash
# Request interactive GPU node
interact -q gpu -g 1 -n 4 -t 01:00:00 -m 16g

# Load modules
module load python/3.11.0 cuda/11.8

# Activate environment
source ~/venvs/worldmodel-env/bin/activate

# Navigate and run
cd ~/stable-worldmodel/examples
python example_actionhead.py

# Exit when done
exit
```

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors, reduce `BATCH_SIZE` in `example_actionhead.py`:

```python
BATCH_SIZE = 128  # or even 64
```

### Module Not Found Errors

Make sure you're in the virtual environment:

```bash
source ~/venvs/worldmodel-env/bin/activate
```

Check installed packages:

```bash
pip list | grep stable
pip list | grep torch
```

### Data Loading Issues

Make sure paths are correct and data files exist:

```bash
ls -la ~/stable-worldmodel/data/
```

### Check Available GPU Partitions

```bash
# See partition info
sinfo -p gpu

# See available GPUs
allq gpu
```

## Customizing Job Parameters

Edit the `#SBATCH` directives in the `.sh` files:

```bash
#SBATCH -t 12:00:00     # Request 12 hours instead of 6
#SBATCH --mem=64G       # Request 64GB RAM instead of 32GB
#SBATCH --gres=gpu:2    # Request 2 GPUs instead of 1
```

**Note:** The GPU-HE partition has better GPUs but may have longer wait times. To use it:

```bash
#SBATCH -p gpu-he --gres=gpu:1
```

## Additional Resources

- Oscar Documentation: https://docs.ccv.brown.edu/oscar/
- GPU Computing: https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu
- CCV Support: support@ccv.brown.edu

