#!/bin/bash
# Submits 3 batches × 4 variants = 12 training jobs.
# Each batch contains: baseline, swa-256, swa-128, swa-64
# Usage: bash submit_batches.sh

PROJDIR=/auto/plzen1/home/kadlej27/NoCap-Test

submit_job() {
    local batch=$1       # 1, 2, 3
    local window=$2      # 0 = baseline, else window size

    if [ "$window" -eq 0 ]; then
        local name="b${batch}-baseline"
        local window_arg=""
    else
        local name="b${batch}-swa${window}"
        local window_arg="--window_size ${window}"
    fi

    local outlog="${PROJDIR}/${name}_output.log"
    local errlog="${PROJDIR}/${name}_error.log"

    # Write job script to a temp file and submit it
    local tmpjob=$(mktemp /tmp/job_XXXXXX.sh)
    cat > "$tmpjob" <<JOBEOF
#!/bin/bash -l
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=40gb:gpu_cap=compute_80:mem=32gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -N ${name}
#PBS -o ${outlog}
#PBS -e ${errlog}

module load cuda/12.6.1-gcc-10.2.1-hplxoqp
module load python/3.11.11-gcc-10.2.1-555dlyc

PROJDIR=${PROJDIR}
cd \$PROJDIR || exit 1

export TMPDIR=\$SCRATCHDIR
export PIP_CACHE_DIR=\$SCRATCHDIR/pip_cache

VENV=\$SCRATCHDIR/venv
python3 -m venv \$VENV
source \$VENV/bin/activate

pip install --upgrade pip
pip install torch numpy huggingface_hub datasets wandb tiktoken

if [ ! -d "\$PROJDIR/data/fineweb10B" ]; then
    python \$PROJDIR/data/cached_fineweb10B.py
fi

cd \$PROJDIR
export WANDB_API_KEY=\$(cat /auto/plzen1/home/kadlej27/.wandb_key 2>/dev/null)

echo "=== GPU Info ==="
nvidia-smi
echo "=== Processes on GPU at job start ==="
nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader || echo "(no other processes)"

torchrun --standalone --nproc_per_node=1 train_gpt2.py \\
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \\
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \\
  --output_dir pylog124M \\
  --model d12 \\
  --batch_size 16 \\
  --grad_accumulation_steps 32 \\
  --sequence_length 1024 \\
  --val_loss_every 128 \\
  --val_batch_size 16 \\
  --num_iterations 4768 \\
  --weight_decay 0.1 \\
  --learning_rate 0.0018 \\
  --warmup_iters 256 \\
  --warmdown_iters 1024 \\
  ${window_arg} \\
  --log_wandb
JOBEOF

    local job_id
    job_id=$(qsub "$tmpjob")
    local exit_code=$?
    rm -f "$tmpjob"

    if [ $exit_code -eq 0 ]; then
        echo "Submitted ${name}: ${job_id}"
    else
        echo "FAILED to submit ${name}" >&2
    fi
}

echo "=== Submitting 3 batches × 4 variants ==="
for batch in 1 2 3; do
    echo "--- Batch ${batch} ---"
    submit_job $batch 0    # baseline
    submit_job $batch 256
    submit_job $batch 128
    submit_job $batch 64
done

echo ""
echo "=== Current queue ==="
qstat -u kadlej27
