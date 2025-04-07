#!/bin/bash
#SBATCH --output=logs/slurm.out
#SBATCH --error=logs/slurm.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=project_465000527
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --job-name=anemoi_train
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
module load LUMI/24.03 partition/G

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Optional! Set NCCL debug output to check correct use of aws-ofi-rccl (these are very verbose)
#export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3


# Function to find an unused TCP port starting from a specified port number.
find_unused_port() {
    local port=$1
    while : ; do
        # Check if the port is in use
        if ! ss -tuln | grep -qE ":::$port|0.0.0.0:$port" ; then
            # Port is not in use
            echo $port
            break
        fi
        # Increment port by 1 and check again.
        ((port++))
    done
}

# fetches slurm nodelist
get_master_node() {
    # Get the first item in the node list
    first_nodelist=$(echo $SLURM_NODELIST | cut -d',' -f1)

    if [[ "$first_nodelist" == *'['* ]]; then
        # Split the node list and extract the master node
        base_name=$(echo "$first_nodelist" | cut -d'[' -f1)
        range_part=$(echo "$first_nodelist" | cut -d'[' -f2 | cut -d'-' -f1)
        master_node="${base_name}${range_part}"
    else
        # If no range, the first node is the master node
        master_node="$first_nodelist"
    fi

    echo "$master_node"
}



export MASTER_ADDR=$(get_master_node)
export MASTER_PORT=$(find_unused_port 29500)
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

# CXI stuff
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# Enable verbose hydra error outputs in Anemoi
export HYDRA_FULL_ERROR=1

# Set this virtual environment
export VIRTUAL_ENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/.venv

# Ensure the virtual environment is loaded inside the container
export PYTHONUSERBASE=$VIRTUAL_ENV
export PATH=$PATH:$VIRTUAL_ENV/bin
export ANEMOI_CONFIG_PATH=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/multi-domain-training/temp/anemoi-core/training/src/anemoi/training/config

echo $PATH
srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 \
singularity exec -B /pfs:/pfs \
	         -B /var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
                    /pfs/lustrep4/scratch/project_465000527/anemoi/containers/anemoi-core-pytorch-2.3.1-rocm-6.0-python-3.11.sif \
                    anemoi-training train --config-name=multi_domain.yaml

# --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 