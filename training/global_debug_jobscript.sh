#!/bin/bash
#SBATCH --output=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/logs/output.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/logs/output.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_465000527
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=anemoi
#SBATCH --cpus-per-task=7
#SBATCH --exclusive

export VIRTUAL_ENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/.venv

# Ensure the virtual environment is loaded inside the container
export PYTHONUSERBASE=$VIRTUAL_ENV
export PATH=$PATH:$VIRTUAL_ENV/bin

PROJECT_DIR=/pfs/lustrep4/scratch/$SLURM_JOB_ACCOUNT
CONTAINER_SCRIPT=$PROJECT_DIR/anemoi/run-pytorch/run-pytorch.sh
CONTAINER=$PROJECT_DIR/anemoi/containers/anemoi-core-pytorch-2.3.1-rocm-6.0-python-3.11.sif

# config file
export ANEMOI_CONFIG_PATH=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/src/anemoi/training/config
CONFIG=debug.yaml

module load LUMI/24.03 partition/G

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

# MPI + OpenMP bindings: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"

# run run-pytorch.sh in singularity container like recommended
# in LUMI doc: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch
export ANEMOI_CONFIG_PATH=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/src/anemoi/training/config/

srun \
    singularity exec -B /pfs:/pfs \
                     -B /var/spool/slurmd \
                     -B /opt/cray \
                     -B /usr/lib64 \
        $CONTAINER $CONTAINER_SCRIPT $CONFIG

# --cpu-bind=$CPU_BIND
