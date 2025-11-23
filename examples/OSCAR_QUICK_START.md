# Oscar Quick Start Guide

## Every Time You SSH In

```bash
# 1. Load modules
module load python/3.11.0s-ixrhc3q
module load cuda/11.8

# 2. Activate virtual environment
source ~/venvs/worldmodel-env-fixed/bin/activate

# 3. Go to examples directory
cd ~/stable-worldmodel/examples
```

## Submit Training Job

```bash
sbatch run_actionhead_oscar.sh
```

## Submit MPC Recording Job

```bash
sbatch run_mpc_recording_oscar.sh
```

## Monitor Your Jobs

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f logs/actionhead_*.out

# Check errors
tail -f logs/actionhead_*.err

# Check all GPU jobs
allq gpu

# Cancel a job
scancel <job_id>
```

## Automate Setup (Optional)

Add to `~/.bashrc`:

```bash
# Auto-load modules
module load python/3.11.0s-ixrhc3q
module load cuda/11.8

# Auto-activate venv
if [ -f ~/venvs/worldmodel-env-fixed/bin/activate ]; then
    source ~/venvs/worldmodel-env-fixed/bin/activate
fi

# Helpful aliases
alias myjobs='squeue -u $USER'
alias gpu='allq gpu'
```

Then run: `source ~/.bashrc`

## Quick Tests

```bash
# Test Python packages
python -c "import stable_worldmodel; print('✓ stable_worldmodel')"
python -c "import stable_pretraining; print('✓ stable_pretraining')"
python -c "import torch; print('✓ torch, CUDA:', torch.cuda.is_available())"

# Test interactively before submitting batch job
interact -q gpu -g 1 -n 4 -t 01:00:00 -m 16g
python example_actionhead.py
exit
```

## File Transfer from Local Machine

```bash
# Update a file
scp "local/path/to/file.py" awong55@ssh.ccv.brown.edu:~/stable-worldmodel/examples/

# Sync entire directory
rsync -avz "local/directory/" awong55@ssh.ccv.brown.edu:~/remote/directory/
```

## Troubleshooting

**Import errors:**
```bash
pip list | grep stable
python -c "import stable_worldmodel"
```

**Check venv is activated:**
```bash
echo $VIRTUAL_ENV
# Should show: /users/awong55/venvs/worldmodel-env-fixed
```

**Module not loaded:**
```bash
module list
# Should show python/3.11.0s-ixrhc3q and cuda/11.8
```

