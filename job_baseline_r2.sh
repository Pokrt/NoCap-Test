#!/bin/bash -l
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=40gb:gpu_cap=compute_80:mem=32gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -N baseline-r2
#PBS -o /auto/plzen1/home/kadlej27/NoCap-Test/job_baseline_r2_output.log
#PBS -e /auto/plzen1/home/kadlej27/NoCap-Test/job_baseline_r2_error.log

module load cuda/12.6.1-gcc-10.2.1-hplxoqp
module load python/3.11.11-gcc-10.2.1-555dlyc

PROJDIR=/auto/plzen1/home/kadlej27/NoCap-Test
cd $PROJDIR || exit 1

export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

VENV=$SCRATCHDIR/venv
python3 -m venv $VENV
source $VENV/bin/activate

pip install --upgrade pip
pip install torch numpy huggingface_hub datasets wandb tiktoken

if [ ! -d "$PROJDIR/data/fineweb10B" ]; then
    python $PROJDIR/data/cached_fineweb10B.py
fi

cd $PROJDIR
export WANDB_API_KEY=$(cat /auto/plzen1/home/kadlej27/.wandb_key 2>/dev/null)

echo "=== GPU Info ==="
nvidia-smi
echo "=== Processes on GPU at job start ==="
nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader || echo "(no other processes)"

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
  --warmdown_iters 1024 \
  --log_wandb
