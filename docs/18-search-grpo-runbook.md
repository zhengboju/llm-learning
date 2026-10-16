# Search-GRPO 运行手册（docs/16 的落地实现）

> 状态：**代码已落地并通过 CPU 回归（145 项）**，**待 pod 真机验收**。
> 前置：[docs/16-search-grpo-plan.md](16-search-grpo-plan.md)（方案）、
> [docs/09-native-tool-protocol.md](09-native-tool-protocol.md)（原生工具协议）、
> [docs/17-swe-rl-plan.md](17-swe-rl-plan.md)（后续项目，与本档无关）。
> 本文只写「怎么跑、跑之前核什么、看到什么算对」——不重复方案论证。

---

## 0. 一句话架构

```
训练 pod（GPU）                          检索服务器（另机，不占训练机磁盘）
  train.py --algo search_qa   ──HTTP──▶   rlab/search_server.py serve
  只持 --search_endpoint                   SQLite FTS5 索引（Wikipedia 全量可）
```

索引 ~20G **不进训练机**（docs/16 §3 的硬要求）。训练侧启动时 `check_health()`
探活，不通**直接拒训**——不让基础设施故障伪装成「模型全答错」。

---

## 1. 交付清单（本次新增/改动）

| 文件 | 作用 |
|---|---|
| `rlab/search_server.py` | 独立检索服务（标准库 SQLite FTS5 + HTTP，**不 import torch**，可直接跑） |
| `rlab/search_client.py` | 客户端（HTTP `http://` 与本地 `local:<path>` 同语义）+ `RunCode` 同型 runner |
| `rlab/search_qa.py` | QA 打分（SQuAD 口径 EM/F1、多别名取最大）+ 数据装载（held-out 契约） |
| `rlab/prepare_search_qa.py` | 训练/评测池生成（在线拉取或离线导入，含落盘复读自证） |
| `rlab/tests/test_search_qa_cpu.py` | 44 项 CPU 回归（协议/客户端/打分/数据/配置/多轮接线） |
| `rlab/protocol.py` | 原生工具协议**多工具泛化**（`search` 与 `code_interpreter` 并列） |
| `rlab/rollout.py` | 工具名驱动声明/解析/执行器；检索 runner 注入；QA 打分分派 |
| `rlab/config.py` | `search_qa` preset + 检索服务键 + 四处 fail-fast |
| `rlab/train.py` | `--search_endpoint/top_k/timeout/retries/auth_env/native_tool_name`；签名与 run_info |
| `eval_vllm_one.py` | `--eval_task search_qa` 同档评测（同工具声明、同知识库、QA 口径打分） |
| `rlab/probe_difficulty.py` | `--algo search_qa`：难度探针用检索 runner + QA 口径打分 |

**兼容性**：`tool_protocol` 缺省仍是 `fence`、`native_tool_name` 缺省仍是
`code_interpreter`——**p1–p11 与原生 retool_math 的全部行为逐字不变**
（145 项 CPU 测试含历史对照位）。

---

## 2. 上机步骤

### 步骤 1：准备数据（pod 上 HF 不通 → 离线导入）

在有网的机器导出两份 JSONL（NQ-open：`question` + `answer`(list)；
HotpotQA：`question` + `answer`(str)），拷到 pod 后：

```bash
python -m rlab.prepare_search_qa \
    --train_input /root/data/nq_train.jsonl \
    --dev_input   /root/data/nq_dev.jsonl
# 产物：rlab/datasets/search_qa/{train.jsonl,dev.jsonl,manifest.json}
```

有网时可直接 `--dataset nq_open`（HF 优先、modelscope 兜底）。
脚本会：按题面去重（同题多解并入 aliases）→ 剔除与 dev 同题面的行 →
丢弃「答案归一化后为空」的脏行 → 落盘后**复读**再核一遍题面不相交。

> ⚠ **dev 不能省**：缺 dev 直接拒跑（没有 held-out 就无法回答「有没有学到」）。
> dev < 300 题会告警（±1~2pp 的效应读不出来）。

### 步骤 2：建索引并起服务（**另机**）

```bash
# 语料 JSONL：每行 {"id": "...", "title": "...", "text": "..."}
python -m rlab.search_server build-index /data/wiki_corpus.jsonl /data/wiki.sqlite
# ↑ 输出 {"corpus_id": "<sha256>"}；已存在的目标会拒绝（防误删旧索引）

python -m rlab.search_server serve /data/wiki.sqlite --host 0.0.0.0 --port 8000
# 可选鉴权（令牌只留在环境变量里，不进配置/日志）：
#   SEARCH_BEARER_TOKEN=xxx python -m rlab.search_server serve ... --auth-env SEARCH_BEARER_TOKEN
```

接口：`GET /health`、`POST /search {query,top_k}`、`POST /search/batch {queries,top_k}`
（批量**保序**）。空/无效 query 返回空结果（**合法**，不是故障）。

### 步骤 3：pod 上核实连通性（M1 探针第一项）

```bash
curl -s http://<检索机>:8000/health   # 必须返回 {"ok":true,"corpus_id":...,"documents":N}
```

不通就别往下走（内网一般可达，但 pod 的外网限制与内网可达性是两件事）。

### 步骤 4：训练

```bash
bash rlab/run_gsm8k.sh search_qa /root/Qwen3.5-4B \
    --search_endpoint http://<检索机>:8000 \
    --out_dir rlab_out/search_qa_p1 --steps 200 --save_steps 50 \
    --attn_implementation flash_attention_2 --optim_8bit \
    --micro_rows 1 --gen_gpu_mem 0.30 \
    --vllm_gen_logps --vllm_logprobs_n 1 \
    --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
    --seed 42
```

启动 30 秒内必须看到两行自证（缺任一就停下来查）：

```
[train] 偏离签名 signature=...-tnsearch-se<hash>-tpnative...
[rollout] 检索服务已探活: http://<检索机>:8000（top_k=3，超时 10.0s，并发 8）
```

### 步骤 5：评测（held-out dev）

```bash
python -m rlab.eval --algo search_qa --n 300 --seed 42 --val_n 12 \
  --base_path /root/Qwen3.5-4B \
  --search_endpoint http://<检索机>:8000 \
  --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
  --gpu_mem 0.78 \
  --models "step200=./rlab_out/search_qa_p1/step_200" \
  --out eval_search_qa_p1.json
```

---

## 3. 判据（跑之前写死，别事后编）

### 3.1 go/no-go：base 对 `search` 工具的调用率

**这是本方案第一个要看的数**（docs/16 §3 的既有缺口）：Qwen3.5-4B 对
`code_interpreter` 的 base 调用率实测 100%（docs/09 §10.5.1），但**对没见过的
`search` 工具声明未必有同等先验**。

| 观察 | 判定 |
|---|---|
| base 调用率 ≥50% | ✅ 先验够，直接 RL（本方案前提成立） |
| 调用率 20~50% | ⚠ 先验弱：考虑轻量 SFT 或声明措辞强化后再 RL |
| 调用率 <20% | ✗ 「工具先验」假设不成立——这本身是一个值得记录的发现 |

跑法：用 `rlab/native_probe.py` 的同档提示（`--algo search_qa` 的 preset 提示 +
`search` 声明）在 base 上采样几条，看 `parse_assistant(kind="tool")` 的比例；
或直接看训练第 1 波的 `code_used` 分布（`traj_dump=True` 落盘）。

### 3.2 训练中盯三件事

| 指标 | 健康形态 | 警报形态 |
|---|---|---|
| `code_used`（检索次数/轨迹） | 随步数上升或稳定 >0 | **恒 0** = 检索被压灭（先查调用率与奖励口径） |
| `degenerate group`（整组无梯度） | ≈0 | 冲高 = 组内奖励无区分度 |
| `acc`（held-out Average@12） | 单调升 | **平坦** = 系统性签名，当场停 |

⚠ 「**精确 0% / 恒常数 / 平坦**」是本项目认定的系统性 bug 签名（AGENTS.md），
不是「还没学到」。

### 3.3 验收（与 docs/16 M2 对应）

| 判据 | 阈值 |
|---|---|
| vs base 配对 McNemar | **p<0.05**（N≥300、同 seed、同轮） |
| 效应量 | >4~5pp 才可读（单 run 方差可达 ±4pp；1~4pp 排序作废） |
| 条件指标 | 必报 `acc/fmt`（fmt = 交出 `\boxed{}` 的比例）——「准确率涨了」可能全是交卷率涨 |
| OOD | Bamboogle / 2Wiki / MusiQue（M4 起） |

---

## 4. 本次实现里最容易踩的四件事（都是设计里刻意防的）

1. **检索服务不通 = 拒训，不是零分**。客户端在限次重试后抛
   `SearchServiceError`，**绝不**返回「检索为空」——否则基础设施故障会伪装成
   「模型不会答题」，日志里只有丢弃率上涨。`runner.check_health()` 在训练/探针/
   评测三处入口都调了。
2. **工具名错配 = 显形，不是静默空转**。`native_tool_name` 驱动「模板声明 / 解析
   正则 / 执行器」三处；配了 `search` 却渲染 `code_interpreter` 时
   `protocol.native_tool` 直接 `KeyError`；反过来，`search` 调用在 `code` 档下被判
   `invalid`（不是 `answer`）。config 层还有一条 `search_qa` 必须用 `search` 的
   校验。
3. **调用载荷里的 `\boxed{}` 不是答案**。打分域先剥完整调用块、再从未闭合调用的
   开标记处截断（`search_qa._final_answer_text`，标签从 `protocol` 正则派生，
   **零手写标签字面量**）。
4. **held-out 契约落在文件上**。loader 读 train.jsonl 时**真的去读 dev.jsonl
   核对题面不相交**——「已剔除 dev」是声明不是证据（2026-09-09/09-16 两次事故）。

---

## 5. 已知未做（如实声明，别当成已完成）

| 项 | 说明 |
|---|---|
| **真机验收** | 本机无 GPU：**base 调用率、训练增益、显存峰值、检索延迟全部未测**（docs/16 M2 起） |
| 奖励塑形 R1~R3 | 本次只落地 **R0（outcome-only EM ±1）**；检索次数惩罚/中间奖励是后续单变量实验 |
| 多跳强化（M4） | HotpotQA 主训、上下文压缩机制均未做 |
| 检索服务性能 | BM25 单机 FTS5；大规模并发/向量检索（e5）/分片均未做（先用 BM25 起步，docs/16 §6 已列预案） |
| 索引构建工具 | 只吃 `{id,title,text}` JSONL；Wikipedia dump → 该格式的预处理脚本未提供 |

---

## 6. 复现顺序（照着做）

```bash
cd ~/llm-learning && git pull

# ① 数据（离线导入优先）
python -m rlab.prepare_search_qa --train_input .../nq_train.jsonl --dev_input .../nq_dev.jsonl

# ② 检索服务（另机）
python -m rlab.search_server build-index corpus.jsonl wiki.sqlite
python -m rlab.search_server serve wiki.sqlite --host 0.0.0.0 --port 8000

# ③ 连通性
curl -s http://<检索机>:8000/health

# ④ base 调用率（go/no-go，见 §3.1）

# ⑤ 训练（--steps 20 先通闭环，再 200+）
bash rlab/run_gsm8k.sh search_qa /root/Qwen3.5-4B --search_endpoint http://<检索机>:8000 ...

# ⑥ 评测 + 配对检验
python -m rlab.eval --algo search_qa --search_endpoint http://<检索机>:8000 ...
python -m rlab.analysis --pair-...    # 同题配对（McNemar）
```
