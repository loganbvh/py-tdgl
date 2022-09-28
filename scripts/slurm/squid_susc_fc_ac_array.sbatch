#!/bin/bash
#SBATCH --job-name=squid_susc_fc_ac_array
#SBATCH --array=0-50
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=12:00:00

set -x

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

outdir=$GROUP_SCRATCH/py-tdgl/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir

pyscript=$HOME/py-tdgl/scripts/squid_susc_fc_ac.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --I_fc 0.1 10.1 $SLURM_ARRAY_TASK_COUNT \
    --index=$SLURM_ARRAY_TASK_ID \
    --squid-type=hypres-small \
    --squid-position 0 0 0.5 \
    --squid-points=3000 \
    --squid-optimesh=50 \
    --squid-iterations=5 \
    --film-radius=15 \
    --film-points=4000 \
    --film-optimesh=100 \
    --cycles=3 \
    --points-per-cycle=15 \
    --d=0.2 \
    --lam=2 \
    --xi=1 \
    --gamma=10 \
    --dt-min=1e-3 \
    --dt-max=1e-2 \
    --total-time=5e3 \
    --pinning=0 \
    --seed-solutions

# Move the stdout log to the results directory
mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir
