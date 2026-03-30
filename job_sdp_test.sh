#!/bin/bash -l
#PBS -l select=1:ncpus=2:ngpus=1:gpu_cap=compute_80:mem=8gb:scratch_local=20gb
#PBS -l walltime=0:20:00
#PBS -N sdp-backend-test
#PBS -o /auto/plzen1/home/kadlej27/NoCap-Test/job_sdp_test_output.log
#PBS -e /auto/plzen1/home/kadlej27/NoCap-Test/job_sdp_test_error.log

module load cuda/12.6.1-gcc-10.2.1-hplxoqp
module load python/3.11.11-gcc-10.2.1-555dlyc

export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

VENV=$SCRATCHDIR/venv
python3 -m venv $VENV
source $VENV/bin/activate

pip install --quiet torch

python /auto/plzen1/home/kadlej27/NoCap-Test/test_sdp_backend.py
