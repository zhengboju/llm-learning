#!/bin/bash
# rlab 单命令启动：ref_server + 训练端（rank0 自动 spawn 生成 worker）
# 用法:
#   bash rlab/run_gsm8k.sh <algo> [model_path] [-- 其余透传给 train.py]
# 示例:
#   bash rlab/run_gsm8k.sh dapo /root/Qwen2.5-3B
#   bash rlab/run_gsm8k.sh rfpp /root/Qwen2.5-3B        # 自动切 ref_server --mode rfpp
set -e
ALGO=${1:?用法: run_gsm8k.sh <algo> [model_path] [train.py 参数...]}
MODEL=${2:-/root/Qwen2.5-3B}
shift; shift || true

# 【必带】vLLM 权重同步环境（缺一卡死：apply_model 传输 stall 的教训）
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

PORT=59875
MODE=passthrough
if [ "$ALGO" = "rfpp" ]; then MODE=rfpp; fi

echo "[run] algo=$ALGO model=$MODEL ref_server_mode=$MODE"
python -m rlab.ref_server --model_path "$MODEL" --port $PORT --mode $MODE &
REF_PID=$!
sleep 15   # 等 ref 模型加载完（首条 /get 会一直 empty，这里只是避免竞态日志）

python -m rlab.train --algo "$ALGO" --model_path "$MODEL" --port $PORT "$@"
RC=$?
kill $REF_PID 2>/dev/null || true
exit $RC
