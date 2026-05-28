#!/usr/bin/env bash
set -e

# 1. Prime the environment variables
export TRITON_LIBCUDA_PATH=/nas/longleaf/home/anvu/ToM/cuda_fix
export LD_LIBRARY_PATH=/nas/longleaf/home/anvu/ToM/cuda_fix:/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/nas/longleaf/home/anvu/Avalon/Avalon-Reputation:${PYTHONPATH:-}
export HUGGING_FACE_HUB_TOKEN=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
export HF_HOME=/tmp/anvu_hf_cache
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME"

# 2. Navigate to directory and load Apptainer
cd /nas/longleaf/home/anvu/Avalon/Avalon-Reputation
module load apptainer

# 3. Launch the dual experiment
apptainer exec --nv --writable-tmpfs --bind /dev/shm:/dev/shm \
    /nas/longleaf/home/anvu/ToM/tom_mas.sif \
    python -u run_eval_dual.py \
    --model "Qwen/Qwen2.5-32B-Instruct" \
    --tensor-parallel-size 2 \
    --data-file data/avalon/dev_all_servant.json \
    --start-idx 0 \
    --end-idx 10 \
    --num-repeats 5 \
    --concurrent-games 10 \
    --no-periodic-prediction \
    --exp1-use-pubrep \
    --exp1-suffix "honey_trap_32b" \
    --exp1-personality default default naive deceptive deceptive \
    --exp2-use-pubrep \
    --exp2-suffix "cognitive_warfare_32b" \
    --exp2-personality default default default deceptive naive
