# SWE-RL：仓库级代码修复的 RL 训练（项目方案）

> 状态：规划 | 前置：docs/02（ReTool 沙箱）、docs/16（Search-GRPO，agentic RL 共用多轮框架）
> 定位：以「真实仓库 issue → patch → 单测验证」为环境的 RL，奖励=F2P 单测通过，代表工作 SWE-RL（Meta）、R2E、SWE-Gym。

## 1. 背景与动机

- SWE-bench 系列是代码智能体标准战场；SWE-RL 证明了「在仓库快照上以单测为奖励做 RL」可显著提升修复率且泛化到通用代码任务。
- 技术深度点：
  1. **环境即工程**：每个 instance 一个仓库快照 + 依赖环境 + 测试命令，环境构建成本远超 math/检索，是本方向的核心壁垒。
  2. **稀疏且昂贵的奖励**：一次评测=跑一次（部分）测试套件，秒级到分钟级，reward 成本比 math 高 2~3 个数量级。
  3. **长上下文 diff 动作空间**：输出是 patch（结构化 diff），格式错误率、apply 失败率本身是主要瓶颈。
  4. **能力门槛诚实**：SWE-bench Verified 上 4B 级模型基线极低（<5%），需要诚实的降级路径设计。

## 2. 关键决策：任务难度分层（降级预案内建）

| 层 | 任务 | 适用判定 |
|---|---|---|
| L0 | HumanEvalFix 式合成 bug（单函数、单测现成） | 4B 起步训练池，效应可测 |
| L1 | R2E/SWE-Gym 中「小仓库 + 定位已给出」子集 | L0 收敛后 |
| L2 | 完整 SWE-bench 式（自定位 + 自修复） | 仅当 L1 有显著效应再上 |

**决策规则**：先做 L0 probe（SFT 基线 + GRPO），若 SFT 基线在 L0 上 <20% 修复率，说明 4B 能力不足以支撑 RL 信号密度，项目降级为纯研究性，不投入全量。

## 3. 系统架构

```
issue 描述 + 仓库快照 → [LLM 生成 patch（diff 格式）]
   → [沙箱：git apply 校验 → 跑 F2P 测试] → reward = F2P 全过 0/1（+apply 失败/格式惩罚）
```

- **沙箱**：docker per-instance 是标准做法，但 pod RAM 60G + 无 root 特权需先验证 docker-in-pod 可行性；备选：venv/conda 隔离 + 进程级资源限制（CPU/时间/内存上限），安全性用「仅训练可信数据」兜底。
- **数据来源**：SWE-Gym（~2.4k 带环境 instance）或 R2E；L0 层可用现有 retool_math 代码沙箱直接合成（函数 + 注入 bug + 单测）。
- **协议**：patch 标签协议复用 ReTool 多轮框架；apply 失败单独计数（它是格式指标，不等同能力指标——参照 acc/fmt 拆分原则，报条件修复率 = 修复率/apply 成功率）。
- **奖励成本治理**：测试执行加超时+缓存；先画数据流实际评测次数再决定并发度；同 instance 多采样（group）内去重相同 patch 只测一次。

## 4. 评测

- 主指标：F2P 修复率；辅指标：apply 成功率、条件修复率、平均测试耗时。
- 协议沿用：N≥300（SWE-bench Verified 500 题天然满足）、固定 seed、配对检验；与 SFT 基线、RAG 式检索增强基线对照。
- 「精确 0% / 平坦」照常作为系统性 bug 签名当场停。

## 5. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 环境探针 | pod 沙箱方案定版（docker vs venv）、SWE-Gym 一个 instance 端到端跑通 | 单个 issue→patch→测试闭环，耗时/内存有数据 |
| M2 L0 训练 | 合成 bug 池 + GRPO 训练 | vs SFT 基线配对检验显著 |
| M3 L1 迁移 | SWE-Gym 子集训练 + 评测 | 修复率显著超基线，条件修复率报告 |
| M4 与 Search-GRPO 合流 | 共用 agentic RL 框架抽象（多轮 rollout/环境接口/reward 插件化） | 两项目跑在同一框架上，沉淀 docs 级框架结论 |

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| docker 在 pod 不可用 | venv 隔离降级；或本地不跑、仅作方案设计 + 小规模云验证 |
| 4B 能力不足，L0 基线过低 | 触发第 2 节决策规则，降级为 L0-only 研究；不硬上 L2 |
| 测试环境构建失败率高（SWE-Gym 已知 ~30%） | 构建失败 instance 直接剔除并落盘记录；入池前全量预构建筛一遍 |
| 评测成本拖垮训练吞吐 | 测试超时硬上限 + patch 去重 + 测试并行 worker 池；先量化单 reward 成本再定 batch |

## 7. 与 Search-GRPO 的取舍建议

- **优先 Search-GRPO**：环境轻（本地索引 vs 仓库 docker）、复用度高（协议/沙箱/评测全同构）、4B 能力匹配度好、reward 便宜（毫秒级打分 vs 秒级测试）。
- **SWE-RL 价值在 M1/M4**：沙箱工程探针本身有沉淀价值；M4 的「环境接口抽象」是两项目合流后框架级的净产出。
- 建议顺序：Search-GRPO M1→M3 完成后再启 SWE-RL M1 探针，探针数据决定 SWE-RL 是否全量投入。
