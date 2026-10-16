# Search-GRPO：多轮检索推理智能体的 RL 训练（项目方案）

> 状态：规划 | 前置：docs/02（ReTool）、docs/09（工具协议）、docs/15（纯 RL 解题路线）
> 定位：从「单轮/代码交织 RLVR」推进到「多轮、环境交互、稀疏奖励」的 agentic RL，复现并超越 Search-R1 范式。
> **协议定版：原生工具协议**（docs/09 路线，search = 一个原生工具声明），不自造围栏标签。

## 1. 背景与动机

- 社区重心（2025）已从单轮 math RLVR 转向 agentic RL：模型在多轮交互中自主决定**何时检索、查什么、怎么用**，代表工作 Search-R1、R1-Searcher、WebSailor、DeepResearcher。
- 本项目的技术深度（区别于已完成的 math/ReTool 线）：
  1. **长程信用分配**：奖励只在最终答案给出，中间 5~15 轮检索查询质量由 RL 自学。
  2. **噪声环境**：检索返回无关/矛盾文档，模型须学会批判性使用，非 math 的干净环境。
  3. **上下文预算博弈**：多轮检索注入大量文本，context 是稀缺资源，预算策略本身成为被优化对象（复用 `--round_gen_tokens/--max_rounds/--max_context_tokens` 三件套经验）。
  4. **泛化研究**：训练集与 OOD 评测集分离是 agentic RL 论文核心卖点。

## 2. 资产复用

| 现有资产 | 复用方式 |
|---|---|
| **原生工具协议全套**（docs/09：CODE_TOOL 声明、`parse_assistant`、`build_next_prompt` token 增量拼接、`tool_message`、双形态解析） | **search 声明为一个原生工具**（与 code_interpreter 同构的 tool declaration），不自造 `<search>` 围栏——Qwen3.5-4B 实测 function 形态、base 调用率先验 100%（§10.5.1），自造协议要从零教起（p1–p11 失败根因） |
| 六算法实现（GRPO/DAPO/CISPO/GSPO/RF++） | agentic 设定下的算法对照——社区尚无人系统做过，是本项目的净产出之一 |
| retool_math 显存/vLLM 经验 | 多轮检索 rollout 更长，docs/04 公式表直接迁移重算 |
| 评测协议（N≥300、固定 seed、配对检验、条件指标 acc/fmt） | agentic 噪声更大，纪律即差异化优势 |
| pod 2×H20 + 4B 模型 | Search-R1 原版即 3B/7B 级，算力匹配；pod 定版参数（--vllm_model_path / enable_thinking false / --gen_gpu_mem 0.30）沿用 |

## 3. 系统架构

```
prompt + tools=[SEARCH_TOOL] → [LLM 生成] → parse_assistant 判定 kind="tool"（search 调用）？
            ↑ 是 → [检索后端 top-k] → tool_message 消毒注入 → build_next_prompt token 增量续写
            ↓ 否（kind="answer"，含答案）
        [EM/F1 打分] → reward → GRPO 家族更新
```

- **协议（定版：原生工具协议，docs/09 路线）**：
  - `SEARCH_TOOL`：与 `CODE_TOOL` 同构的工具声明 dict（name=`search`，参数=`query`，description 强调 top-k 摘要、每次调用独立），经 `apply_chat_template(tools=[SEARCH_TOOL])` 进 prompt。
  - 解析/回包/拼接**全部复用** `parse_assistant` / `tool_message` / `build_next_prompt`，`--tool_protocol native --native_tool_style function`（Qwen3.5-4B 实测形态，docs/09 §10.5）。**不新增任何自造标签**；通用化方式 = 工具声明与执行器参数化（name → executor 映射），解析层零改动。
  - 基座对 search 的**零先验缺口**需 probe 验证：base 对 code_interpreter 调用率 100%，但对没见过的 `search` 工具声明未必——M1 用 native_probe 同型脚本先测（调用率 <50% 则先做轻量 SFT 或 few-shot 声明注入，这本身是一个可发表的单变量发现）。
- **检索后端**：本地 Wikipedia 2018 dump 索引，首选 flashrag 自带索引（e5/bm25 双档）；pod 无外网，禁止依赖在线搜索 API。索引体积 ~20G 需先核实 pod 磁盘。检索结果在拼接口消毒（sanitize_tool_text 同位）。
- **rollout**：多轮生成与训练逐 token 同一序列（prompt_token_ids 续写）。轮数上限建议 8~15，`--max_context_tokens` 按 docs/04 公式重算后定版。
- **reward 设计空间**（单变量实验）：
  - R0 纯结果奖励：EM 0/1（基线，Search-R1 原版）
  - R1 结果 + 格式惩罚（结构性惩罚注意「精确 0%」签名）
  - R2 结果 + 检索质量中间奖励（如检索命中率/答案支持度）
  - R3 R2 + 检索次数惩罚（抑制无效重复检索，防 reward hacking）

## 4. 数据与评测

- **训练池**：NQ + HotpotQA 混合（Search-R1 配方，~170k，可按池子去重后裁剪）。held-out 契约照旧落盘复读断言（verify_train_pool_clean）。
- **评测集**：in-domain NQ/TriviaQA dev；OOD Bamboogle、2Wiki、MusiQue。
- **协议**：N≥300、固定 seed、同轮配对（rlab.analysis --pair-*）；主指标 EM，辅指标 F1、平均检索轮数、无效检索率；条件指标照报。
- **基线**：无检索直接答 / RAG（强制首轮检索一次）/ SFT-only，三个对照先行，再上 RL。

## 5. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 环境 | Wikipedia 索引落 pod、SEARCH_TOOL 声明接入、search 先验 probe、rollout 链路跑通 | 单 batch 多轮检索 rollout 完成；base 对 search 的调用率有实测数（go/no-go） |
| M2 基线复现 | GRPO+R0 在 NQ 上复现 Search-R1 量级（4B 目标 EM≥35%） | 对照基线显著提升（配对检验 p<0.05） |
| M3 算法对照 | 六算法在 agentic 设定的对照实验 + reward 设计 R0~R3 单变量矩阵 | docs 级结论（含 ±4pp 方差声明、近平局双 seed） |
| M4 深度扩展（选一） | 多跳推理强化（HotpotQA 主训）/ 检索奖励塑形深挖 / 上下文压缩机制 | OOD 泛化增益可复现 |

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| 检索索引体积/内存超 pod 限制（RAM 60G） | 先核实磁盘；索引用 mmap 模式；BM25 优先于稠密向量（省显存） |
| rollout 延迟爆炸（检索串行） | 检索服务常驻+批量接口；先画数据流实际发生次数再优化 |
| reward hacking：重复检索刷格式分 | R3 次数惩罚；监控「平均检索轮数」平坦/异常作 bug 签名 |
| 检索结果注入污染打分域 | 打分只取 assistant 答案段；工具段剥离后评分 |
| 4B 模型多跳能力上限 | M4 前先做 probe：SFT 基线在 HotpotQA 的上限测量，再定 RL 目标 |

## 7. 资源估算

- 沿用 retool_math 定版：4B 纯文本副本 + vLLM 原模型 + enable_thinking false。多轮检索使平均序列变长约 2~3 倍，显存按 docs/04 公式重算，`--gen_gpu_mem` 与 `--micro_rows` 预留下调空间。
- 首轮实验规模：训练池 8~16k 子集（simpleRL-reason 已证 8k 足够出效应）、N=300 评测、单算法单 seed 探路 → 效应 >5pp 再扩矩阵。
