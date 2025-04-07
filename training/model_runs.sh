#!/bin/bash
#SBATCH --output=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/logs/output.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/logs/output.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --account=project_465000527
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --job-name=multidomain_lr
#SBATCH --cpus-per-task=7

CONFIG=multi_domain_MEPS_AA.yaml
export VIRTUAL_ENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/.venv

# Ensure the virtual environment is loaded inside the container
export PYTHONUSERBASE=$VIRTUAL_ENV
export PATH=$PATH:$VIRTUAL_ENV/bin

PROJECT_DIR=/pfs/lustrep4/scratch/$SLURM_JOB_ACCOUNT
CONTAINER_SCRIPT=$PROJECT_DIR/anemoi/run-pytorch/run-pytorch.sh
CONTAINER=$PROJECT_DIR/anemoi/containers/anemoi-core-pytorch-2.3.1-rocm-6.0-python-3.11.sif

# config file
export ANEMOI_CONFIG_PATH=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/src/anemoi/training/config


module load LUMI/24.03 partition/G

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

# MPI + OpenMP bindings: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding
CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

# run run-pytorch.sh in singularity container like recommended
# in LUMI doc: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch
export ANEMOI_CONFIG_PATH=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/src/anemoi/training/config/

srun --cpu-bind=$CPU_BIND\
    singularity exec -B /pfs:/pfs \
                     -B /var/spool/slurmd \
                     -B /opt/cray \
                     -B /usr/lib64 \
        $CONTAINER $CONTAINER_SCRIPT $CONFIG


