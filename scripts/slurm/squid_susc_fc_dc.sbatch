#!/bin/bash
#SBATCH --job-name=squid_susc_fc_dc
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
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

pyscript=$HOME/py-tdgl/scripts/squid_susc_fc_dc.py

# Copy the python script and this shell script to the results directory
cp $pyscript $outdir/
cp $0 $outdir/

python $pyscript \
    --directory=$outdir \
    --I_fc 0.1 5.1 51 \
    --squid-type=hypres-small \
    --squid-position 0 0 0.5 \
    --squid-points=3000 \
    --squid-optimesh=50 \
    --squid-iterations=5 \
    --film-radius=20 \
    --film-points=4000 \
    --film-optimesh=100 \
    --d=0.2 \
    --lam=2 \
    --xi=1 \
    --gamma=10 \
    --dt=1e-2 \
    --steps=10e3 \
    --pinning=0 \
    --screening

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir
