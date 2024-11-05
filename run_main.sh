#!/bin/bash -l
#SBATCH --job-name=hexplane-compress
#SBATCH --output=/project/jacobcha/nk643/HexPlane/job_logs/%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=/project/jacobcha/nk643/HexPlane/job_logs/%x.%j.err # prints the error message
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G # Maximum allowable memory per CPU 4G
#SBATCH --qos=standard
#SBATCH --account=jacobcha # Replace PI_ucid with the NJIT UCID of PI
#SBATCH --time=03:00:00  # D-HH:MM:SS
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nk643@njit.edu
set -e

module purge
module load wulver # load slurn, easybuild
module load easybuild
module load GCCcore/11.2.0
module load git
# add any required module loads here, e.g. a specific Python
module load bright
module load cuda11.8/toolkit/11.8.0
module load foss/2021b FFmpeg/4.3.2
module load Anaconda3
module load Mamba

cd /project/jacobcha/nk643/HexPlane
conda activate ./envs

python main.py config=config/dnerf_general.yaml

