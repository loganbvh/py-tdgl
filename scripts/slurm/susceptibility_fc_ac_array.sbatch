#!/bin/bash
#SBATCH --job-name=susc_fc_ac_array
#SBATCH --array=0-50
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=12:00:00

set -x

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

outdir=$GROUP_SCRATCH/py-tdgl/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir

pyscript=$HOME/py-tdgl/scripts/susceptibility_fc_ac.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --I_fc 0.1 10.1 $SLURM_ARRAY_TASK_COUNT \
    --index=$SLURM_ARRAY_TASK_ID \
    --film-radius=20 \
    --cycles=3 \
    --points-per-cycle=25 \
    --r_fc=2.0 \
    --r_pl=0.75 \
    --z0=1.0 \
    --d=0.2 \
    --lam=2 \
    --xi=1 \
    --gamma=10 \
    --min-points=2000 \
    --optimesh-steps=100 \
    --dt=1e-2 \
    --steps=10e3 \
    --pinning=0 \
    --seed-solutions \
    --screening

# Move the stdout log to the results directory
mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir
