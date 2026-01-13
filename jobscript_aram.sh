#!/bin/bash
#SBATCH --output=logs/md-hecto.out #multi-domain-32bs-lr5e-6.out
#SBATCH --error=logs/md-hecto.err #multi-domain-32bs-lr5e-6.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --account=DestE_340_26
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0 # ALLOCATE FULL RAM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --switches=1
#SBATCH --exclude=lrdn[0153,1902,2051,0189,0163,1781,0388,0399,0407,1308,1309,1953,1984,2151,2094,2792,2371,1748]
#SBATCH --time=06:00:00
#SBATCH --job-name=md-hectometric
#SBATCH --exclusive

#module purge
#module load profile/deeplrn
#module load cineca-ai/4.3.0
module load gcc/12.2.0
export PYTHON_HOME=/leonardo_work/DestE_330_25/users/asalihi0/compiled-libraries/python/python-3.11.7-gcc-12.2.0-cmake-3.27.9
export SQLITE3_HOME=/leonardo_work/DestE_330_25/users/asalihi0/compiled-libraries/python/sqlite-3.45-gcc-12.2.0
export PATH=$PYTHON_HOME/bin:$SQLITE3_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_HOME/lib:$SQLITE3_HOME/lib:$LD_LIBRARY_PATH

source /leonardo_work/DestE_340_26/users/sbuurman/multi-domain-training/multi-domain-torch-2.6.0-cu124/bin/activate
CONFIG_NAME=hectometric_finetuning.yaml
CONFIG_DIR=/leonardo_work/DestE_340_26/users/sbuurman/multi-domain-training/anemoi-core/training/src/anemoi/training/config
export NCCL_DEBUG=INFO
srun --mpi=pmix_v3 anemoi-training train --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME

