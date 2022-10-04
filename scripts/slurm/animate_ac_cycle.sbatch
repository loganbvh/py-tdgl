#!/bin/bash
#SBATCH --job-name=animate-ac-cycle
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -x

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

slurm_job_id=64043402
slurm_array_id=0
outdir=$GROUP_SCRATCH/py-tdgl/videos/$slurm_job_id
outfile=$outdir/field-${slurm_array_id}.mp4
mkdir -p $outdir

pyscript=$HOME/py-tdgl/scripts/animate_ac_cycle.py

# Copy the python script and this shell script to the results directory
cp $pyscript $outdir/
cp $0 $outdir/

python $pyscript \
    --input=$GROUP_SCRATCH/py-tdgl/results/$slurm_job_id/$slurm_array_id \
    --output=$outfile \
    --fps=30 \
    --dpi=200 \
    --skip=1 \
    --gpu

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir
