#!/bin/bash
#SBATCH --job-name=generate-graphs
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err

########## RESOURCE REQUESTS ##########
#SBATCH --nodes=1                    # Example: M=4 nodes
#SBATCH --ntasks-per-node=1           # 4 tasks per node
#SBATCH --gpus-per-node=1             # 4 GPUs per node
#SBATCH --gpus-per-task=1             # 1 GPU per array task
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:15:00

########## OPTIONAL ##########
#SBATCH --exclusive
#SBATCH --switches=1

########################################
#        ENVIRONMENT SETUP
########################################

module load gcc/12.2.0

export PYTHON_HOME=/leonardo_work/DestE_330_25/users/asalihi0/compiled-libraries/python/python-3.11.7-gcc-12.2.0-cmake-3.27.9
export SQLITE3_HOME=/leonardo_work/DestE_330_25/users/asalihi0/compiled-libraries/python/sqlite-3.45-gcc-12.2.0

export PATH=$PYTHON_HOME/bin:$SQLITE3_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_HOME/lib:$SQLITE3_HOME/lib:$LD_LIBRARY_PATH

source /leonardo_work/DestE_330_25/users/sbuurman/multi-domain-training/anemoi-core/multi-domain-torch-2.6.0-cu124/bin/activate


#        RUN PYTHON WITH INJECTED PATH
########################################

python3 process_configs.py

