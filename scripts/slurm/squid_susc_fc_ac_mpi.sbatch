#!/bin/bash
#SBATCH --job-name=squid_susc_fc_ac_mpi
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=12:00:00

set -x

module load devel openmpi

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1

outdir=$GROUP_SCRATCH/py-tdgl/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir

pyscript=$HOME/py-tdgl/scripts/squid_susc_fc_ac_mpi.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

srun -n $SLURM_NTASKS --mpi=pmix python -m mpi4py.futures $pyscript \
    --mpi \
    --max-loss=0.1 \
    --max-wall-time=11 \
    --directory=$outdir \
    --I_fc 0.1 10.1 \
    --squid-type=hypres-small \
    --squid-position 0 0 0.5 \
    --squid-points=3000 \
    --squid-optimesh=50 \
    --squid-iterations=5 \
    --film-radius=15 \
    --film-points=4000 \
    --film-optimesh=100 \
    --cycles=2 \
    --skip-cycles=1 \
    --points-per-cycle=25 \
    --d=0.2 \
    --lam=1.5 \
    --xi=0.5 \
    --gamma=10 \
    --dt-min=1e-3 \
    --dt-max=5e-2 \
    --total-time=20e3 \
    --pinning=0 \
    --save-every=500 \
    --seed-solutions

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir
