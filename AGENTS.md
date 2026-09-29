# simple_GRPO 项目规则（跨机生效：本地开发机与 k8s 训练 pod 都必须遵守）

RL 教学项目（GRPO/DAPO/CISPO/GSPO/RF++ 六算法 + ReTool 代码交织 + retool_math）。本文件只放「会改变行为」的规则；结论/知识按需查 docs/ 与 MCP 图谱，不在本文件重复记账。

## 代码与同步
- 唯一 git remote 为 llm-learning（zhengboju/llm-learning）：开发完 push，pod 上 git pull 部署；临时诊断脚本不提交，只提交开发产物
- 换模型换代际：torch（训练/ref/gen_logps 副本）与 vLLM 是两端独立实现，必须分别验证加载；探针协议必须与目标模型对齐（thinking 开关、轮数预算；截断率是最先看的信号）
- 显存预算按公式重算（docs/04 公式表：bf16 静态≈params×14B、step 临时≈params×8B、logits O(B·T·V)、SDPA math 回退的 T²）；config 标称≠进程实测，共居卡按 nvidia-smi 核实

## 标签字节铁律（已三次发作）
- 任何含格式标签的代码/探针/正则**绝不手写标签字面量**（显示层与工具调用参数都会改写字节）：从 rlab._FORMAT_RE 导入或自动提取，roundtrip 回查，写文件后复读验证；被会话管道生成/修改过的含标签脚本，其测量结论一律作废，须零标签字面量复测
- 读路径（read/console）会把尖括号标签渲染成无括号普通词：判字节真伪用十进制 ord 逐字符直出

## 评测协议
- N≥300、固定 seed、同轮内配对比较才硬；评测 ±1-2pp、训练 run 间方差可达 ±4pp——单 run 只读 >4~5pp 效应，1~4pp 排序作废，近平局需双 seed
- 左 pad 解码按 Lmax 切片（Li 只校验）；tokenizer.eos ≠ 模型停止词（Qwen 实际为 im_end），不显式传 eos
- math_verify 在非主线程必传 parsing_timeout/timeout_seconds=None（signal.alarm 限制）；「串行对照」若仍在 worker 线程跑则对照无效
- 并发 from_pretrained 必须串行化（load_lock + p.is_meta 校验）；结果标签用路径末3段/显式 name 防同名覆盖
- held-out 契约必须落盘复读断言（verify_train_pool_clean——「已剔除 dev」是声明不是证据）；prepare 切分按题面不按行；n>池子要告警+落盘实际 N；空答案计 0 分入分母
- eval 与 rlab 的 reward 是两套实现（0/1 vs ±1 口径），改打分逻辑两边都查
- 聚合指标改口径（greedy→Average@N）时清点所有按题号索引/取平均/二值判定的消费点；「>100% 的率」是除数错的免费签名；「显著」先问检验的是哪个量

## 训练/实验纪律
- 单变量实验纪律；性能优化先画数据流实际发生次数（随机池上「重复出现才触发」的逻辑先算重现概率）；「过滤器存在且测试全绿」≠真实数据流生效
- 新算法 loss 落地必须配梯度探针测试（只测正向 loss 数值发现不了错误公式/零梯度）
- 「精确 0% / 恒常数 / 平坦」是系统性 bug 签名（fmt 0/300、acc 平坦、code_rate 恒 0、权重指纹不变、HealthMonitor 告警）——当场停，不是跑完再说；阈值区分「死在低位」与「赢在高位」，多条判定规则先特例后一般
- 协议引入新行为（如写代码）先查是否违反既有验证规则的锚定假设（MUST 代码先行 vs 格式 ^ 锚定冲突案例）；结构性惩罚的签名是精确 0%
- 多轮生成与训练必须逐 token 同一条序列：token id 续写（prompt_token_ids），禁止文本续写后再重新 tokenize（BPE 跨段边界合并）
- 打分域=模型自己的文本（assistant 段/剥离代码块后）；工具 token 不进 loss、工具输出不当答案；沙箱 stdout 是注入面，消毒在拼接口做（sanitize_tool_text）
- 重构 dict 键名必须 grep 全仓库旧键零引用；改 import 别名过 pyflakes（运行时才执行的路径 import 冒烟测不到）；上游产 dict 下游按键消费的接缝必须用「上游真实输出喂下游」的测试（require_qa_rows 入口 fail-fast，防 .get() None 兜底塌成同类）

## 真机运行（pod）
- 复现档：VLLM_BATCH_INVARIANT=1 + --attention_backend FLASH_ATTN；告警分档（已修复≠已知坏）
- --vllm_gen_kwargs 是整体替换（加一个键必须重述全部），--vllm_attention_backend 是 merge；eval 读 preset 看不到训练 CLI 覆盖
- expandable_segments 与 CUDA IPC（pidfd_open）互斥：train.py 强制 False、gen_worker True（分层覆盖，勿全局开关）
- DAPO-Math 注意：HF 侧源数据每题重复 100~400 份（1.79M 行 ≈16.7k 题），必须按 (Q,A) 去重后再判「与去重前可比性」；pod HF 不通数据走 modelscope（data.py 带 verification_mode 垫片）；wandb 无 key 勿 login("")（卡死），有 WANDB_API_KEY 才 login
- 4B retool_math 定版：model_path=纯文本抽取副本 + --vllm_model_path 原多模态 + enable_thinking false + --round_gen_tokens 3072 + --gen_gpu_mem 0.30 + --micro_rows 1 + --optim_8bit；pod RAM 60G 上限，offload/zero_stage 已证死路
- 探针与训练必须同预算口径（--round_gen_tokens/--max_rounds/--max_context_tokens 三件套 CLI）

## 文档地图（结论级内容查这里，别重复排查）
- docs/01-loss-variants.md：阶段1 六算法总表与大结论（变体趋同、GRPO 并列最佳、运行方差 ±4pp）；docs/02-retool.md：ReTool 代码交织
- docs/03-qwen35-4b-retool-math-checklist.md：4B 迁移清单；docs/04-qwen35-4b-gpu-memory-oom.md：显存炸点时序+公式表；docs/07-vllm-logprobs-non-determinism-4b.md：可复现档
- docs/08-sft-cold-start.md、09-native-tool-protocol.md、14-native-p3-handover.md、p8-diagnosis.md、p10-diagnosis.md