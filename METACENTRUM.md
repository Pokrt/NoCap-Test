# Running on MetaCentrum

## Prerequisites

1. **MetaCentrum account** — log in at https://metavo.metacentrum.cz/
2. **SSH access** — `ssh <username>@nympha.metacentrum.cz`
3. **wandb API key** (optional) — get it from https://wandb.ai/authorize

## One-time setup

```bash
# SSH to the cluster
ssh kadlej27@nympha.metacentrum.cz

# Clone the repo
cd ~
git clone https://github.com/Pokrt/NoCap-Test.git

# Store wandb API key securely (outside the repo, not tracked by git)
echo 'YOUR_WANDB_API_KEY' > ~/.wandb_key
chmod 600 ~/.wandb_key

# Download the training data (~9.5GB, persists in home dir)
# This is done automatically by job.sh if data is not present,
# but you can also do it manually:
module load python/3.11.11-gcc-10.2.1-555dlyc
python ~/NoCap-Test/data/cached_fineweb10B.py
```

## Job script (`job.sh`)

The job script lives on the cluster at `~/NoCap-Test/job.sh`:

```bash
#!/bin/bash -l
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=40gb:gpu_cap=compute_80:mem=32gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -N nocap-gpt2
#PBS -o /auto/plzen1/home/kadlej27/NoCap-Test/job_output.log
#PBS -e /auto/plzen1/home/kadlej27/NoCap-Test/job_error.log

module load cuda/12.6.1-gcc-10.2.1-hplxoqp
module load python/3.11.11-gcc-10.2.1-555dlyc

PROJDIR=/auto/plzen1/home/kadlej27/NoCap-Test
cd $PROJDIR || exit 1

export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

# Set up venv on scratch (fast local disk, no quota issues)
VENV=$SCRATCHDIR/venv
python3 -m venv $VENV
source $VENV/bin/activate

pip install --upgrade pip
pip install torch numpy huggingface_hub datasets wandb tiktoken

# Download data if not present (to home dir so it persists across jobs)
if [ ! -d "$PROJDIR/data/fineweb10B" ]; then
    python $PROJDIR/data/cached_fineweb10B.py
fi

cd $PROJDIR
# Load wandb API key from home dir (not in git)
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null)
chmod +x run.sh
./run.sh
```

### Key PBS parameters

| Parameter | Value | Meaning |
|---|---|---|
| `ngpus=1` | 1 GPU | Single GPU training |
| `gpu_mem=40gb` | 40GB VRAM | Gets A100/A40 class GPUs |
| `gpu_cap=compute_80` | Compute 8.0+ | Ampere or newer (A100, A40, RTX 3090/4090) |
| `mem=32gb` | 32GB RAM | System memory |
| `scratch_local=50gb` | 50GB scratch | Local SSD for venv + pip cache |
| `walltime=12:00:00` | 12 hours | Maximum job duration |

### Pinning to a specific node

To get reproducible results, you can pin to a specific node by adding `vnode=<node>` to the select line:

```
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=40gb:gpu_cap=compute_80:mem=32gb:scratch_local=50gb:vnode=luna205
```

This ensures you always get the same GPU hardware. Downside: longer queue wait.

## Submitting a job

```bash
ssh kadlej27@nympha.metacentrum.cz
cd ~/NoCap-Test
qsub job.sh
# Output: 18326102.pbs-m1.metacentrum.cz
```

## Monitoring

```bash
# Check job status (Q=queued, R=running, F=finished)
qstat <job_id>

# Detailed status with resource usage
qstat -f <job_id> | grep -E 'job_state|resources_used|exec_host'

# Check all your jobs
qstat -u kadlej27

# View training output (updates only after job finishes or periodically)
tail -50 ~/NoCap-Test/job_output.log

# View errors
cat ~/NoCap-Test/job_error.log

# Check training progress (validation losses)
grep 'val:' ~/NoCap-Test/pylog124M/*.log

# Cancel a job
qdel <job_id>
```

## Output files

| File | Contents |
|---|---|
| `job_output.log` | stdout — pip install output, training step logs |
| `job_error.log` | stderr — module loading, warnings, errors |
| `pylog124M/<uuid>.log` | Training log — `s:<step> val:<loss>` and `s:<step> trn:<loss>` entries |

## Notes

- **Venv on scratch**: The Python virtual environment is created on `$SCRATCHDIR` (fast local SSD) each job. This avoids home directory quota issues (~915MB for torch alone). Packages are re-downloaded each run.
- **Data persistence**: The 9.5GB training data is stored in `~/NoCap-Test/data/fineweb10B/` (home dir), so it persists across jobs and doesn't need to be re-downloaded.
- **wandb**: If `~/.wandb_key` exists, wandb logging is enabled. Otherwise, `run.sh` will fail unless you remove `--log_wandb` from it or set `export WANDB_MODE=disabled` in job.sh.
- **nvidia-smi**: Added to `run.sh` to log the GPU model at the start of each run.

## Baseline results

### MetaCentrum A100 — Job 18331539 (March 23, 2026)

| Detail | Value |
|---|---|
| **Node** | zia3.cerit-sc.cz |
| **GPU** | NVIDIA A100-SXM4-40GB |
| **CPU** | 128 cores (256 logical) |
| **Job ID** | 18331539.pbs-m1.metacentrum.cz |
| **PBS params** | gpu_mem=40gb, 12h walltime, wandb enabled |
| **Outcome** | Completed (4768/4768 steps) |
| **Step time** | **2810.73 ms/step** avg (~2.81s) |
| **Total training time** | **13,401.57s** (~3.72 hours) |
| **Final val loss** | **3.3805** |
| **Final train loss** | 3.3802 |
| **Peak GPU memory** | 9,877 MiB / 40,960 MiB |
| **wandb run** | `honza-kadlec-ctu-fee/benchmark_gpt2/wvmnqbud` |
| **Python** | CPython 3.11.11 |
| **PyTorch** | with wandb 0.25.1 |

### RTX 4090 reference (from benchmark)

| Detail | Value |
|---|---|
| **GPU** | NVIDIA RTX 4090 |
| **Step time** | ~4.1 seconds/step |
| **Total time** | 5.4 hours (19,444s) for 4768 steps |
| **Final val loss** | **3.3821** |
| **Source** | wandb run `64s1zc1w` included in repo |

### Comparison

| Metric | A100-SXM4-40GB | RTX 4090 |
|---|---|---|
| Step time | 2.81s | ~4.1s |
| Total time | 3.72h (13,402s) | 5.4h (19,444s) |
| Final val loss | **3.3805** | 3.3821 |
| Speedup | **1.45x faster** | baseline |

The A100 achieves a slightly better val loss (3.3805 vs 3.3821) and completes training ~45% faster than the RTX 4090 reference.

---

## Submitting a new experiment

Follow these steps every time you want to run a new experiment from your local machine.

### 1. Create a branch

```bash
git checkout -b <experiment-name>   # e.g. log-freq-bias-init
```

### 2. Make your changes locally

Edit `train_gpt2.py` and/or `run.sh`, then commit and push:

```bash
git add train_gpt2.py run.sh
git commit -m "Short description of experiment"
git push -u origin <experiment-name>
```

### 3. SSH to MetaCentrum interactively

Non-interactive SSH cannot submit jobs (Kerberos ticket required):

```bash
ssh kadlej27@nympha.metacentrum.cz
# Enter SSH password, then Kerberos password when prompted
```

### 4. Pull the branch and submit

```bash
cd ~/NoCap-Test
git fetch
git checkout <experiment-name>
git pull
qsub job.sh
```

To submit multiple runs for benchmarking across random GPUs:

```bash
qsub job.sh && qsub job.sh && qsub job.sh && qsub job.sh
```

### 5. Monitor

```bash
# Check job status (Q=queued, R=running, F=finished)
qstat -u kadlej27

# Watch live output
tail -f ~/NoCap-Test/job_output.log

# Check errors
cat ~/NoCap-Test/job_error.log
```

Results are logged to wandb at `honza-kadlec-ctu-fee/benchmark_gpt2`.

### Notes

- If the cluster has local uncommitted changes on a branch, use `git stash` before `git checkout`
- Jobs run from whatever branch is currently checked out in `~/NoCap-Test` at submission time
- Each job gets a random GPU assigned — submit 3–4 runs for a fair benchmark average
