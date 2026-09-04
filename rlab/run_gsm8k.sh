#!/bin/bash
# rlab 单命令启动：ref_server + 训练端（rank0 自动 spawn 生成 worker）
#
# 显存卡位（2×H20 96G，每样东西各得其所，勿全挤一卡）：
#   GPU0: ref 模型(~7G) + vLLM 生成 0.35(~33G) + torch gen_logps 副本(~6G) ≈ 46G
#   GPU1: DeepSpeed 训练端 3B ZeRO-0 全态(~60G，含 AdamW fp32 状态)
#
# 用法:
#   bash rlab/run_gsm8k.sh <algo> [model_path] [train.py 参数...]
# 示例:
#   bash rlab/run_gsm8k.sh dapo /root/Qwen2.5-3B
#   bash rlab/run_gsm8k.sh rfpp /root/Qwen2.5-3B        # 自动切 ref_server --mode rfpp
# 卡位可用环境变量覆盖: REF_GPU=0 TRAIN_GPU=1 bash rlab/run_gsm8k.sh ...
set -e
ALGO=${1:?用法: run_gsm8k.sh <algo> [model_path] [train.py 参数...]}
MODEL=${2:-/root/Qwen2.5-3B}
shift; shift || true

REF_GPU=${REF_GPU:-0}
TRAIN_GPU=${TRAIN_GPU:-1}

# 【必带】vLLM 权重同步环境（缺一卡死：apply_model 传输 stall 的教训）
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# 训练机无交互终端 + 常无 WANDB_API_KEY：默认离线记录防 wandb login prompt 卡死；
# 有 key 且想实时上传时 WANDB_MODE=online bash rlab/run_gsm8k.sh ... 覆盖。
export WANDB_MODE=${WANDB_MODE:-offline}

PORT=59875
MODE=passthrough
if [ "$ALGO" = "rfpp" ]; then MODE=rfpp; fi

echo "[run] algo=$ALGO model=$MODEL ref_server_mode=$MODE ref_gpu=$REF_GPU train_gpu=$TRAIN_GPU"

# 退出清理：无论正常结束、训练崩溃还是 Ctrl+C，都杀掉 ref_server，
# 防止孤儿进程占着 ~7G 显存（set -e 直接退出会跳过普通 kill 语句）
REF_PID=""
cleanup() {
  if [ -n "$REF_PID" ] && kill -0 "$REF_PID" 2>/dev/null; then
    echo "[run] stopping ref_server (pid=$REF_PID)"
    kill "$REF_PID" 2>/dev/null || true
    wait "$REF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=$REF_GPU python -m rlab.ref_server --model_path "$MODEL" \
    --port $PORT --mode $MODE &
REF_PID=$!
sleep 15   # 等 ref 模型加载完（首条 /get 会一直 empty，这里只是避免竞态日志）

# CUDA_VISIBLE_DEVICES 限定训练卡；生成 worker 由 train.py spawn 后自行把
# CUDA_VISIBLE_DEVICES 改回 REF_GPU 的物理卡号（rollout.gen_worker 内置）
CUDA_VISIBLE_DEVICES=$TRAIN_GPU python -m rlab.train --algo "$ALGO" \
    --model_path "$MODEL" --port $PORT "$@"
