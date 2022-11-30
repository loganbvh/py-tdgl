#!/bin/bash
#SBATCH --job-name=simple-squid
#SBATCH --array=0-200
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=12:00:00

set -x

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

outdir=$GROUP_SCRATCH/py-tdgl/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir

pyscript=$HOME/py-tdgl/docs/scripts/simple_squid_sbatch.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --output=$outdir/$SLURM_ARRAY_TASK_ID \
    --currents 0 100 101 \
    --fields "-200 200 ${SLURM_ARRAY_TASK_COUNT}" \
    --index=$SLURM_ARRAY_TASK_ID \
    --min-points=3000 \
    --lead-angle=60 \
    --solve-time=500 \
    --field-units="mT" \
    --current-units="uA"

# Move the stdout log to the results directory
mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir
