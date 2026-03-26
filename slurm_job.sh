#!/bin/bash
#SBATCH --job-name=nocap-gpt2
#SBATCH --partition=1day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== JOB INFO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working dir: $(pwd)"

# Show GPU info
nvidia-smi

# Go to project directory
cd ~/NoCap-Test

# Fix CUDA linking: create libcuda.so symlink for torch.compile/triton
# (compute nodes have libcuda.so.1 but not the unversioned symlink)
mkdir -p /tmp/cuda_stubs
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /tmp/cuda_stubs/libcuda.so
export LIBRARY_PATH="/tmp/cuda_stubs:${LIBRARY_PATH:-}"

# Setup pip
export PATH="$HOME/.local/bin:$PATH"
export PIP_BREAK_SYSTEM_PACKAGES=1

# Install pip if not present
if ! command -v pip &>/dev/null; then
    echo "Installing pip..."
    wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
    python3 /tmp/get-pip.py --user --break-system-packages
fi

# Install PyTorch + deps
echo "=== Installing PyTorch ==="
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
pip install --user wandb numpy requests huggingface_hub datasets 2>&1 | tail -3

# Verify torch + CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Download data if not present
if [ ! -f "data/fineweb10B/fineweb_val_000000.bin" ]; then
    echo "=== Downloading training data ==="
    python3 data/cached_fineweb10B.py
else
    echo "=== Data already downloaded ==="
fi

echo "=== Data files ==="
ls -lh data/fineweb10B/ | head -5
echo "... ($(ls data/fineweb10B/*.bin 2>/dev/null | wc -l) bin files total)"

# Disable wandb for now (can enable later with API key)
export WANDB_MODE=disabled

echo "=== Starting training ==="
echo "Start time: $(date)"

torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024

echo "=== Training complete ==="
echo "End time: $(date)"
