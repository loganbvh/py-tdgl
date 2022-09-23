#!/bin/bash
#SBATCH --job-name=susc_fc_dc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -x

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

outdir=$GROUP_SCRATCH/py-tdgl/results/$SLURM_JOB_ID
mkdir -p $outdir

pyscript=$HOME/py-tdgl/scripts/susceptibility_fc_dc.py

# Copy the python script and this shell script to the results directory
cp $pyscript $outdir/
cp $0 $outdir/

python $pyscript \
    --directory=$outdir \
    --I_fc 0.1 5.1 51 \
    --r_fc=2.5 \
    --r_pl=0.75 \
    --z0=0.5 \
    --d=0.1 \
    --lam=2 \
    --xi=0.5 \
    --gamma=10 \
    --min-points=1500 \
    --optimesh-steps=100 \
    --dt=1e-2 \
    --steps=10e3 \
    --pinning=0 \
    --screening

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir
