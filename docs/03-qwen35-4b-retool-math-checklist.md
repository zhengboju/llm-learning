# Qwen3.5-4B 迁移 checklist（retool_math 换基座实验）

> 状态：**待决策清单**（2026-09-10 整理）。依据：agentic-rl-lab/05-retool 实测
> Qwen3.5-4B 下 `degenerate_group_rate≈0`（base correct 0.29 → 全错组 (0.71)^8≈6.6%
> 且随训练下降），而 rlab 的 Qwen2.5-3B base 实测丢弃率 81%（p≈5%）。
> 结论：换强基座是丢弃率的一杆到底解，但不是"换个 model_path"，以下是全部连带项。

## 0. 决策前置：先跑双探针对照（各 ~30 分钟，不要跳过）

```bash
# 3B base 对照（现有路径的难度表，无论如何都要跑）
CUDA_VISIBLE_DEVICES=0 python -m rlab.probe_difficulty \
    --model_path /root/Qwen2.5-3B --k 4 --max_questions 500 \
    --out rlab_out/difficulty_probe_3b.jsonl
# 4B 试验组
CUDA_VISIBLE_DEVICES=0 python -m rlab.probe_difficulty \
    --model_path <Qwen3.5-4B路径> --k 4 --max_questions 500 \
    --out rlab_out/difficulty_probe_4b.jsonl
```

判读（summarize 直接打印）：4B 的可学带（0<pass<1）占比 >50% 且无 boxed 率 <10%
→ 迁移收益是实的；否则先在 3B 上用过滤+调度继续压丢弃率。

## 1. 协议层（决定成败，最优先）

- [x] **`enable_thinking=False`**：`build_prompt` 的 apply_chat_template 必须显式
  关闭思考模式。官方实测 Qwen3 系思考模式下"很少写代码、训练效果差"；不关，
  thinking 长链还会爆 round_gen_tokens=1024 的单轮预算。
  **已落地（2026-09-11）**：`config.chat_template_kwargs`（BASE 默认 None，Qwen2.5
  路径零变化）+ train.py/probe_difficulty.py 的 `--chat_template_kwargs` JSON CLI。
  4B 探针实测未关 thinking：截断 98.9%/无 boxed 99.2%/可学带仅 6/490——协议失败
  非能力失败，**该探针数据作废，必须带开关重探**。
- [ ] **chat template 差异重验**：Qwen3.5 模板会 strip assistant 内容，采样文本含
  `</think>` 时模板把消息重构成 reasoning/content 两段（05-retool 踩坑实录）。
  rlab 是 token id 续写 + 分段拼接，理论上绕开了文本往返，但
  `retool_build_batch` 的"生成序列==训练序列"必须用真 4B tokenizer 重跑
  `_audit_tokenize_probe` 类探针确认（BPE 词表不同，边界合并行为不可外推）。
- [ ] **tokenizer 契约**：Qwen3.5 的 pad/eos id 与 2.5 不同，run_gsm8k.sh 前先
  打印 `tokenizer.pad_token_id / eos_token_id` 核对 protocol 语义。

## 2. 预算重审（3B 时代的行为假设全部失效）

- [x] `round_gen_tokens`：**v4 探针实锤（2026-09-11）必须 3072**。机制：4B 关
  thinking 后推理全走 content 通道、极啰嗦（答"8 选 7"也烧 ~900 tok），第一轮
  1024 被 prose 烧断→无完整代码围栏→命中"本轮无代码即终局"→**max_rounds 是
  虚假预算，有效单轨迹预算=round_gen_tokens**（v3 实测 max_rounds 翻倍截断不动）。
  3072 下：正确轨迹 10.4%→36.7%、全对题 7→80、可学带 99→194(k=4 低估，n=8
  修正后 ≈60%)、完成题正确率 ≈82%。**训练必须与探针同口径**（--round_gen_tokens 3072）。
  备注：4B 样本 code_ok≈0，纯 prose 直接解——TIR 协议对 4B 可能不必要，留观。
- [ ] `max_rounds=3` 保持即可（对 4B 无效预算，见上；若后续 prompt 约束让模型
  写代码，再按参考 6 轮重审）。
- [ ] `max_prompt_length=1024`：4B 下 prompt 变长（模板+工具定义），确认不误杀。
- [ ] 采样配置 temp=1.0/top_p=1.0/top_k=-1 保持（与参考对齐）；eval 端参考用
  top_p=0.7，AIME 评测时对齐。

## 3. 显存与训练结构（2×H20 需重新排卡）

- [ ] GPU1：4B ZeRO-0 全态（bf16 权重+梯度+fp32 AdamW 状态）≈80G 量级，单卡很
  紧——预留方案：ZeRO stage 2 / optimizer offload / micro batch 降 4（单变量
  回退位）。
- [ ] GPU0：vLLM + torch gen_logps 副本 + ref 模型各涨 ~30%，`gen_gpu_mem=0.45`
  重新核算峰值（gen_logps logits 瞬时峰与 4B 词表相关，可能显著变大）。
- [ ] lr 保持 1e-6 全参（与官方全参 actor_lr 一致；参考的 LoRA 4e-5 不可比）。

## 4. 数据与探针（与模型绑定，不可复用）

- [ ] **difficulty_path 必须用 4B 重新探**：通过率表是"某模型在某分布下"的快照，
  换模型沿用 3B 的表会把 4B 能做对的题挡在门外。全量 16.5k×k4 重跑（断点续跑
  支持，中断无损失）。
- [ ] DAPO 模板剥离（`_strip_dapo_template`）已做，无需动。

## 5. 实验方法学

- [ ] **与阶段0-2 的 3B 结果不可比**（基座、格式、行为全变）——开新 out_dir 与
  报告章节，声明"新实验系列"，不与 82-83 天花板同表排列。
- [ ] 固定 seed；验收顺序：双探针 → `probe_retool_gen.py`（4B 代码率/围栏完整率）
  → 20 步验证跑（盯 degenerate/code_calls/correct 三曲线，参考的验证跑形态）
  → 放大到 200 步。
- [ ] 训练期盯 `[健康检查]`：4B 下 fmt 应该高开（instruct），若 fmt 恒低 →
  chat template/思考开关没对齐（第 1 节没做对）。

## 6. 明确不迁移的理由记录（防止反复摇摆）

3B base 路线在过滤+调度落地后预期丢弃率已可压到 20-30%，且保留阶段0-2 的
可比性。若双探针显示 4B 可学带占比并不显著更高（DAPO-Math-17k 对 4B 也可能
偏难），迁移收益不抵第 1-3 节的改动风险，应放弃迁移继续 3B。
