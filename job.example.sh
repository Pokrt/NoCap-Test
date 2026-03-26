#!/bin/bash -l
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=40gb:gpu_cap=compute_80:mem=32gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -N nocap-gpt2
#PBS -o /auto/plzen1/home/<USERNAME>/NoCap-Test/job_output.log
#PBS -e /auto/plzen1/home/<USERNAME>/NoCap-Test/job_error.log

module load cuda/12.6.1-gcc-10.2.1-hplxoqp
module load python/3.11.11-gcc-10.2.1-555dlyc

PROJDIR=/auto/plzen1/home/<USERNAME>/NoCap-Test
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
# Set your wandb API key (get it from https://wandb.ai/authorize)
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
chmod +x run.sh
./run.sh
