#!/bin/bash
#SBATCH --output=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/logs/output.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/logs/output.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=project_465000527
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --job-name=meps_weighted
#SBATCH --cpus-per-task=7



#Change this
CONFIG_NAME=multi_domain_MEPSonly_StageB.yaml  #This file should be located in run-anemoi/lumi

#Should not have to change these
PROJECT_DIR=/pfs/lustrep4/scratch/$SLURM_JOB_ACCOUNT
# CONTAINER_SCRIPT=$PROJECT_DIR/anemoi/run-pytorch/run_pytorch_rocm603.sh
CONTAINER_SCRIPT=$PROJECT_DIR/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/run_pytorch_new.sh
chmod 770 ${CONTAINER_SCRIPT}
CONFIG_DIR=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/anemoi-core/training/src/anemoi/training/config/

# NB! in order to avoid NCCL timeouts it is adviced to use 
# pytorch 2.3.1 or above to have NCCL 2.18.3 version

CONTAINER=$PROJECT_DIR/anemoi/containers/anemoi-core-pytorch-2.3.1-rocm-6.0-python-3.11.sif
# CONTAINER=$PROJECT_DIR/anemoi/containers/anemoi-training-pytorch-2.3.1-rocm-6.0.3-py-3.11.5.sif
VENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/.md
export VIRTUAL_ENV=$VENV

module load LUMI/23.09 partition/G

# see https://docs.lumi-supercomputer.eu/hardware/lumig/
# see https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/

# New bindings see docs above. Correct ordering of cpu affinity
# excludes first and last core since they are not available 
# on GPU-nodes
CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

# run run-pytorch.sh in singularity container like recommended
# in LUMI doc: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch
srun --cpu-bind=${CPU_BIND} \
    singularity exec  \
                     -B /pfs:/pfs \
                     -B /var/spool/slurmd \
                     -B /opt/cray \
                     -B /usr/lib64 \
                     -B /opt/cray/libfabric/1.15.2.0/lib64/libfabric.so.1 \
        $CONTAINER $CONTAINER_SCRIPT $CONFIG_DIR $CONFIG_NAME

#