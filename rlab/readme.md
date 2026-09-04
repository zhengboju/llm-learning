# rlab — 统一 RL 实验骨架（阶段0）

`docs/RL进阶学习实验规划.md` 阶段0 的交付物：把 `simple_grpo_v1`、`simple-reinforce++`、
根目录评测脚本收敛成一套可插拔骨架。**所有算法差异只落在 `losses.py` 一个函数里。**

## 文件结构

| 文件 | 职责 |
|---|---|
| `config.py` | 全部超参 + 六算法 preset（`get_config("dapo")` 一行切换） |
| `protocol.py` | 生成端↔训练端 batch 契约与字节编解码（algo 感知双布局） |
| `data.py` | 数据加载（GSM8K HF/modelscope 回落 + CPU fixture） |
| `reward.py` | acc/format/overlong 三组件（math_verify 线程安全：timeout=None） |
| `losses.py` | **核心**：advantage 三模式 + 六算法 loss（grpo/dapo/dr_grpo/cispo/gspo/rfpp） |
| `sync.py` | 权重同步（apply_model 优先 + V0 兜底 + fail-fast） |
| `rollout.py` | 生成端 worker（vLLM 采样 + torch gen_logps 副本 + dynamic sampling） |
| `ref_server.py` | 打分中转服务器，双模式：passthrough（GRPO 家族）/ rfpp（macro-batch per-token advantage） |
| `train.py` | DeepSpeed 训练端主程序（ZeRO-0，rank0 spawn 生成端） |
| `eval.py` | 评测入口（委托根目录 eval_vllm.py，协议 N=300 seed=42） |
| `analysis.py` | eval 汇总表（±2pp 噪声地板判定）+ record.jsonl 曲线 |
| `tests/test_smoke_cpu.py` | 41 项 CPU 冒烟测试（losses 解析值/协议/reward/数据） |

## 算法切换对照

| algo | advantage | loss 归一化 | clip | 特有机制 |
|---|---|---|---|---|
| grpo | group_std | sample_mean | 对称 0.2 | k3 KL β=0.04 |
| dapo | group_std | token_mean | 0.2/0.28 | dynamic sampling + overlong shaping |
| dr_grpo | group_mean | token_const(=512) | 对称 | 去 1/std 偏差 + 去长度归一化偏差 |
| cispo | group_std | token_mean | 只截上升 | 截断 token 保留 min(ratio,1+ε) 梯度 |
| gspo | group_std | seq_mean | 序列级对称 | sequence 级 importance ratio |
| rfpp | 服务端 macro-batch 反向 cumsum | token_items | 对称 | 无 KL；eos-mask reward 信用回传 |

## 快速开始（训练机上，2×H20）

**卡位编排（勿全挤一卡，vLLM 启动时 12G free 的 OOM 教训）**：
- GPU0：ref 模型(~7G) + vLLM 生成 0.35(~33G) + torch gen_logps 副本(~6G) ≈ 46G
- GPU1：DeepSpeed 训练端 3B ZeRO-0 全态（~60G，含 AdamW fp32 状态）

```bash
# 方式一：一键（自动按上述卡位隔离，REF_GPU/TRAIN_GPU 可覆盖）
bash rlab/run_gsm8k.sh dapo /root/Qwen2.5-3B

# 方式二：分进程（手动控制卡位）
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
CUDA_VISIBLE_DEVICES=0 python -m rlab.ref_server --model_path /root/Qwen2.5-3B --port 59875
CUDA_VISIBLE_DEVICES=1 python -m rlab.train --algo dapo --model_path /root/Qwen2.5-3B
#   rfpp 需 ref_server --mode rfpp；生成端进程死亡会触发训练端 fail-fast，不会无限等待
```

评测与汇总：

```bash
python -m rlab.eval --models dapo300=./rlab_out/dapo/step_300 --gpus 0
python -m rlab.analysis --eval-json eval_vllm_all.json
python -m rlab.analysis --record rlab_out/record.jsonl
```

CPU 冒烟（本机即可跑，共 81 项）：

```bash
python -m rlab.tests.test_smoke_cpu        # 41 项：losses 解析值/协议/reward/数据 ✅
python -m rlab.tests.test_train_step_cpu   #  9 项：tiny 模型端到端 plen 切片/mask/backward ✅
python -m rlab.tests.test_ref_server_cpu   # 16 项：eos mask/passthrough 布局/rfpp 信用回传数学 ✅
python -m rlab.tests.test_e2e_http         # 15 项：真实 HTTP 双模式服务器 + 6 算法消费闭环 ✅
```

注意：e2e 测试中 bottle 的启动 banner 走 stderr，在 PowerShell 管道里可能显示
NativeCommandError 假象；以退出码为准（stdout/stderr 重定向到文件时 exit=0）。

## 与历史实验的可比性说明

- 超参默认对齐 `grpo_dapo.py` 实测配置（Q_batch_size=1、num_pre_Q=4、300 步、lr 1e-6、ZeRO-0）。
- **reward 口径统一为 2.0×acc + 1.0×fmt − overlong**（与 grpo_dapo 相同）。历史 rf++ 基线用的是
  {2, 0.5, −2} 离散口径——在 advantage 标准化下期望等价，但与 rfpp100=72.3 的绝对数不是严格同口径，
  对比时以 rlab 内部同口径重跑的 rfpp 为准。
- 数据顺序注意：`simple_grpo_v1` 用全随机采样，`Auto_Program` 用顺序遍历；rlab 默认随机采样（与 v1 一致）。

## 阶段2/3 扩展点（先留白，不预埋死代码）

- 多轮工具调用（ReTool）：替换 `rollout.build_prompt` + 在 gen worker 加"代码块检测→沙箱执行→续写"
  循环；`protocol.py` meta 需加工具段 mask 字段（工具返回 token 不进 loss）。
- 检索 RL（Search-R1）：新增 `search_backend.py`（BM25 起步），reward 换 EM；数据在 `data.py` 注册。

## 验收状态（阶段0）

- [x] 骨架代码完成，六算法 loss 数值正确性 41 项 CPU 测试通过
- [x] 协议双布局 roundtrip 通过
- [ ] **待训练机执行**：GRPO 基线复现 ≈74 / DAPO 复现 ≈78（本机无 GPU，见 run_gsm8k.sh）
