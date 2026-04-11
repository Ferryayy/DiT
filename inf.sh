#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/infer.yaml}"
shift || true
EXTRA_ARGS=("$@")

readarray -t INFER_SETTINGS < <(
python - "$CONFIG_PATH" <<'PY'
import os
import sys

import torch
import yaml


def normalize_gpu_list(gpu_ids):
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, int):
        return [str(gpu_ids)]
    if isinstance(gpu_ids, str):
        gpu_ids = gpu_ids.strip()
        if not gpu_ids or gpu_ids.lower() == "auto":
            return None
        return [x.strip() for x in gpu_ids.split(",") if x.strip()]
    if isinstance(gpu_ids, (list, tuple)):
        return [str(x).strip() for x in gpu_ids if str(x).strip()]
    raise ValueError("gpu_ids must be an int, string, or list.")


config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

infer_mode = str(config.get("infer_mode", "auto")).lower()
gpu_ids_value = config.get("gpu_ids", "auto")
gpu_list = normalize_gpu_list(gpu_ids_value)
master_addr = str(config.get("master_addr", "127.0.0.1"))
master_port = str(config.get("master_port", 29501))
gpu_source = "explicit"

if gpu_list is None:
    gpu_source = "auto"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        gpu_list = [x.strip() for x in visible_devices.split(",") if x.strip()]
    elif torch.cuda.is_available():
        gpu_list = [str(i) for i in range(torch.cuda.device_count())]
    else:
        gpu_list = []

print(infer_mode)
print(",".join(gpu_list))
print(len(gpu_list))
print(master_addr)
print(master_port)
print(gpu_source)
PY
)

INFER_MODE="${INFER_SETTINGS[0]}"
GPU_IDS="${INFER_SETTINGS[1]}"
GPU_COUNT="${INFER_SETTINGS[2]}"
MASTER_ADDR="${INFER_SETTINGS[3]}"
MASTER_PORT="${INFER_SETTINGS[4]}"
GPU_SOURCE="${INFER_SETTINGS[5]}"

if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

if [[ "${INFER_MODE}" == "single" ]]; then
    if [[ "${GPU_COUNT}" -gt "1" ]]; then
        if [[ "${GPU_SOURCE}" == "auto" ]]; then
            FIRST_GPU="${GPU_IDS%%,*}"
            export CUDA_VISIBLE_DEVICES="${FIRST_GPU}"
            exec python inf.py --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
        fi
        echo "infer_mode=single expects at most one visible GPU, but got: ${GPU_IDS}" >&2
        exit 1
    fi
    exec python inf.py --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
fi

if [[ "${INFER_MODE}" == "ddp" ]]; then
    if [[ "${GPU_COUNT}" -lt "2" ]]; then
        echo "infer_mode=ddp requires at least 2 visible GPUs, but got: ${GPU_IDS}" >&2
        exit 1
    fi
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${GPU_COUNT}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        inf.py \
        --config "${CONFIG_PATH}" \
        "${EXTRA_ARGS[@]}"
fi

if [[ "${INFER_MODE}" == "auto" ]]; then
    if [[ "${GPU_COUNT}" -le "1" ]]; then
        exec python inf.py --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
    fi
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${GPU_COUNT}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        inf.py \
        --config "${CONFIG_PATH}" \
        "${EXTRA_ARGS[@]}"
fi

echo "Unsupported infer_mode: ${INFER_MODE}" >&2
exit 1
