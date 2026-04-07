#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/train.yaml}"

readarray -t TRAIN_SETTINGS < <(
python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

train_mode = str(config.get("train_mode", "single")).lower()
gpu_ids = config.get("gpu_ids", "0")
master_addr = str(config.get("master_addr", "127.0.0.1"))
master_port = str(config.get("master_port", 29500))

if isinstance(gpu_ids, int):
    gpu_list = [str(gpu_ids)]
elif isinstance(gpu_ids, str):
    gpu_list = [x.strip() for x in gpu_ids.split(",") if x.strip()]
elif isinstance(gpu_ids, (list, tuple)):
    gpu_list = [str(x).strip() for x in gpu_ids if str(x).strip()]
else:
    raise ValueError("gpu_ids must be an int, string, or list.")

if not gpu_list:
    raise ValueError("gpu_ids cannot be empty.")

print(train_mode)
print(",".join(gpu_list))
print(len(gpu_list))
print(master_addr)
print(master_port)
PY
)

TRAIN_MODE="${TRAIN_SETTINGS[0]}"
GPU_IDS="${TRAIN_SETTINGS[1]}"
GPU_COUNT="${TRAIN_SETTINGS[2]}"
MASTER_ADDR="${TRAIN_SETTINGS[3]}"
MASTER_PORT="${TRAIN_SETTINGS[4]}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ "${TRAIN_MODE}" == "single" ]]; then
    if [[ "${GPU_COUNT}" != "1" ]]; then
        echo "train_mode=single 时，gpu_ids 必须只包含 1 张卡。当前: ${GPU_IDS}" >&2
        exit 1
    fi
    exec python train.py --config "${CONFIG_PATH}"
fi

if [[ "${TRAIN_MODE}" == "ddp" ]]; then
    if [[ "${GPU_COUNT}" -lt "2" ]]; then
        echo "train_mode=ddp 时，gpu_ids 至少需要 2 张卡。当前: ${GPU_IDS}" >&2
        exit 1
    fi
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${GPU_COUNT}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        train.py \
        --config "${CONFIG_PATH}"
fi

if [[ "${TRAIN_MODE}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" == "1" ]]; then
        exec python train.py --config "${CONFIG_PATH}"
    fi
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${GPU_COUNT}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        train.py \
        --config "${CONFIG_PATH}"
fi

echo "Unsupported train_mode: ${TRAIN_MODE}" >&2
exit 1
