# -*- coding: utf-8 -*-
"""rlab/train.py — DeepSpeed 训练端主程序（loss 可插拔）。

用法（2×H20 推荐部署，三进程）：
    # 1. 打分服务器（0 号卡，与 vLLM 共卡）
    python -m rlab.ref_server --model_path /root/Qwen2.5-3B --port 59875 [--mode rfpp]
    # 2. 训练端（单卡 ZeRO-0；如需双卡训练改 deepspeed --num_gpus 2，stage 建议 0/2）
    deepspeed --num_gpus 1 rlab/train.py --algo dapo --model_path /root/Qwen2.5-3B
    #    训练端 rank0 自动 spawn 生成 worker（共驻 0 号卡，gpu_mem 0.35）

必带环境变量（run_gsm8k.sh 已内置）：
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
"""

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# 【2026-09-11 IPC 坑，4B 首跑死在第 16 步权重同步】run_gsm8k.sh 全局 export 的
# expandable_segments:True 与 CUDA IPC 互斥：gen_update_steps 推 state_dict（CUDA
# bf16 张量）过 mp.Queue 时，expandable 段的跨进程共享要走 pidfd_open 系统调用，
# 容器内核不支持 -> 生成端反序列化直接 RuntimeError（3B 时代不炸是因为当时还没
# 这个全局 export）。训练进程强制普通段分配（GPU1 的碎片治理靠 micro_rows/
# empty_cache，不依赖该开关）；生成端在 gen_worker 入口自行改回 True（GPU0 三方
# 共居的碎片治理仍需要）。两个变量名都覆盖（不同 torch 版本读不同名字）。
for _alloc_k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    os.environ[_alloc_k] = "expandable_segments:False"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlab.config import ds_config, get_config
from rlab.losses import ALGOS, compute_loss, forward_per_token_logps
from rlab.protocol import decode_batch


def get_batch(ref_server):
    import requests
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b"empty":
            return None
    except Exception:
        return None
    return decode_batch(r)


def _git_head() -> str:
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "log", "-1", "--oneline"],
                                       text=True, cwd=root).strip()
    except Exception:
        return "unknown"


def run_signature(cfg: dict) -> str:
    """一行「偏离签名」：把最容易被静默改掉、事后只能靠考古发现的维度压成可读串。

    【2026-09-12 为什么要它】run2 的四个关键偏离散落在四处——trunc_shaping 在
    preset、max_rounds 在 preset 注释里、步数在 CLI、评测口径在 eval 的默认值——
    跑完 6 小时才发现"和参考项目(agentic-rl-lab/05-retool)差了哪几维"要靠翻日志。
    签名同时进 ①wandb run name ②run_info.json ③启动 print，三次冗余。
    字段顺序固定、缺项写 "-"（`g` 格式避免 5e-06 这类尾巴），保证两次 run 可逐段比对。
    """
    if cfg.get("difficulty_path"):
        lo, hi = (cfg.get("difficulty_band") or (0.0, 1.0))
        dtag = f"d{lo:g}-{hi:g}"
    else:
        dtag = "nodiff"
    lr = cfg.get("lr")
    lr_tag = f"{lr:g}" if isinstance(lr, (int, float)) else str(lr)
    ts = float(cfg.get("trunc_shaping") or 0.0)
    return (f"{cfg.get('algo')}-ts{ts:g}-ol{1 if cfg.get('overlong_shaping') else 0}"
            f"-r{cfg.get('max_rounds', 1)}x{cfg.get('round_gen_tokens') or 0}"
            f"-s{cfg.get('all_steps')}x{cfg.get('save_steps')}"
            f"-lr{lr_tag}-{dtag}")


def write_run_info(path: str, cfg: dict) -> None:
    """把 run 身份（git head/开始时刻/完整配置/偏离签名）落成 json——checkpoint 与
    record 从此自证出处。（2026-09-08 教训：out_dir 按 algo 共享，多次 run 会静默覆盖
    step_* 同名 checkpoint，评测可能在测旧 run 的模型而毫不知情。）"""
    info = {"signature": cfg.get("run_signature") or run_signature(cfg),
            "git_head": _git_head(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algo": cfg["algo"], "model_path": cfg["model_path"],
            "all_steps": cfg["all_steps"], "save_steps": cfg.get("save_steps"),
            "seed": cfg.get("seed"),
            "reward_switch_step": cfg.get("reward_switch_step"),
            "round_gen_tokens": cfg.get("round_gen_tokens"),
            "max_rounds": cfg.get("max_rounds"),
            "trunc_shaping": cfg.get("trunc_shaping"),
            "overlong_shaping": cfg.get("overlong_shaping"),
            "difficulty_path": cfg.get("difficulty_path"),
            "difficulty_band": cfg.get("difficulty_band"),
            "lr": cfg.get("lr"),
            "temperature": cfg.get("temperature"),
            # 【2026-09-11】完整配方落盘（lr/beta/grad_clip/overlong/GAS…）：旧版只记
            # 7 个字段，10 小时长跑后无法自证用的是哪套超参——评测到一个 checkpoint
            # 时无从核对它对应哪次 run 的哪个配方（9/8 覆盖事故的同类盲区）。
            "config": {k: v for k, v in cfg.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)


def run_training(cfg, args):
    import deepspeed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    deepspeed.init_distributed()

    # rank0 在 spawn 出 gen worker 之后再加载训练模型，避免 fork 时的 CUDA 上下文污染
    gen_proc = None
    Q = None
    if dist.get_rank() == 0:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        write_run_info(os.path.join(cfg["out_dir"], "run_info.json"), cfg)
        print("\n[train] START vLLM generation worker...\n")
        mp.set_start_method("spawn", force=True)
        Q = mp.Queue()
        gen_proc = mp.Process(target=_spawn_gen, args=(Q, cfg), daemon=True)
        gen_proc.start()

    def _ensure_gen_alive():
        """fail-fast：生成端进程死亡 = 权重/数据链路已断，继续等只会空转
        （vLLM 启动 OOM 等故障曾表现为训练端无限 'waiting for batch'）。"""
        if gen_proc is not None and not gen_proc.is_alive():
            raise RuntimeError(
                "[train] 生成端进程已退出（见其 traceback，常见原因：显存不足/"
                "权重同步失败）-> 训练端中止。检查 run_gsm8k.sh 的卡位与显存编排。")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=torch.bfloat16,
        _attn_implementation=cfg.get("attn_implementation", "sdpa"))
    # 【2026-09-11 4B】8-bit 优化器在 DS initialize 前构建并传入（ds_config 相应
    # 省略 optimizer 段）：bnb AdamW8bit 把 m/v 量化到 8bit（32G->8G），step 全程
    # GPU、无 offload、无 RAM 压力。数值口径声明见 config.optim_8bit 注释。
    user_optimizer = None
    if cfg.get("optim_8bit"):
        import bitsandbytes as bnb
        user_optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=cfg["lr"])
    # 【2026-09-11 4B 显存】backbone 激活检查点：B=8×T~5.4k 的层内激活 ~80G 量级
    # （3B 实测 ~20G × T 3.2× H 1.25×），不开必 OOM；只存层输入、backward 重算。
    # use_reentrant=False 与分块 logps 的 checkpoint 重算兼容。
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False   # 训练不用 KV cache，关掉防 HF 告警/缓存分支
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config(cfg), model=model, model_parameters=model.parameters(),
        optimizer=user_optimizer)
    # 【2026-09-11 4B OOM 根因】from_pretrained 默认 eval 模式，而 transformers
    # 激活检查点在 DecoderLayer.__call__ 里要求 self.training 为真——不进 train
    # 模式则 gradient_checkpointing_enable() 静默失效，backbone 全量激活保留
    # （实测 94.65G OOM；3B 时代激活小从未暴露此坑）。Qwen 系 dropout=0，
    # train 模式无数值影响。
    _m = engine.module if hasattr(engine, "module") else engine
    _m.train()
    print(f"[train] train 模式已启用: training={_m.training} "
          f"激活检查点={getattr(_m.base_model, 'gradient_checkpointing', '?')}")
    pad_id = tokenizer.pad_token_id

    wandb_run = None
    if cfg["use_wandb"] and dist.get_rank() == 0:
        try:
            import wandb
            # 空 key 时 wandb.login 会弹交互式 prompt 把训练卡住：只在给了 key 才 login；
            # 无 key 默认离线记录（WANDB_MODE 可覆盖为 online），事后 wandb sync 补传。
            if os.environ.get("WANDB_API_KEY"):
                wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
            # run name 带偏离签名：wandb 列表页上直接可比"这轮和参考/上轮差在哪几维"
            wandb_run = wandb.init(
                project=cfg["wandb_project"],
                name=f"{cfg['wandb_name']}-{cfg.get('run_signature') or run_signature(cfg)}",
                config={k: v for k, v in cfg.items()},
                mode=os.environ.get("WANDB_MODE", "offline"))
        except Exception as e:
            print(f"[train] wandb 不可用（{e}），继续训练不记录")
    totals = {"num": 0, "acc": 0.0, "fmt": 0.0}
    zero_grad_streak = 0   # 连续全零梯度计数（cispo 零梯度 bug 的直接签名）

    from tqdm import tqdm
    progress = tqdm(range(1, cfg["all_steps"] + 1)) if dist.get_rank() == 0 \
        else range(1, cfg["all_steps"] + 1)

    for step in progress:
        batch = get_batch(cfg["ref_server"])
        while batch is None:
            _ensure_gen_alive()
            if dist.get_rank() == 0:
                print("waiting for batch...")
            time.sleep(3)
            batch = get_batch(cfg["ref_server"])
        _ensure_gen_alive()

        plen = batch["plen"]
        inputs = batch["inputs"].to(engine.device)
        advantages = batch["advantages"].to(engine.device)
        gen_logps = batch["gen_logps"].to(engine.device)
        ref_logps = batch["refs"].to(engine.device)

        # 【2026-09-11 4B 显存】分块 logps + backbone 激活检查点（模型初始化处开启）。
        # micro_rows>0 再按行拆 micro-backward（前向 1 行 -> backward -> 图释放）：
        # DS bf16 优化器全态（fp32 master+m+v ≈ params×12B）把 GPU1 静态逼到
        # ~80G，本机 RAM 60G 上不去 offload（DS pin_memory 还撞容器锁页上限）
        # ——唯一可砍的是"8 行检查点包共存"。sample_mean 归一下
        # Σ chunk_loss×(k/R) 与整批 loss 梯度严格等价（DS 的统一缩放两条路径
        # 同乘相消）；其他 loss_norm 在下面 fail-fast（批内归一跨 chunk 变义）。
        _mmod = engine.module if hasattr(engine, "module") else engine
        if "mask" in batch:
            # 阶段2 retool：mask 由生成端按段边界给出（assistant=1 / 工具返回段=0 / pad=0），
            # 训练端直接采用——工具返回 token 不进 loss 是 TIR 的核心契约，不可用 pad 重算。
            mask = batch["mask"].to(engine.device)
        else:
            mask = (inputs[:, plen:] != pad_id).float()

        micro_rows = int(cfg.get("micro_rows", 0) or 0)
        # 分块前向粒度（0/缺省→1=逐行，历史口径；>1 减少 kernel 次数与调度开销，
        # 数学等价，代价是 logits/激活峰值随 B 线性增长——见 config.fwd_batch_chunk）
        _fbc = max(1, int(cfg.get("fwd_batch_chunk", 1) or 1))
        R = inputs.shape[0]
        if micro_rows and micro_rows < R:
            if cfg.get("loss_norm") != "sample_mean":
                raise RuntimeError(
                    f"[train] micro_rows 拆行仅支持 loss_norm=sample_mean"
                    f"（当前 {cfg.get('loss_norm')}）——批内归一跨 chunk 变义，"
                    "禁止静默改语义")
            loss_total, stats_list = 0.0, []
            for c0 in range(0, R, micro_rows):
                sl = slice(c0, min(c0 + micro_rows, R))
                chunk_logps = forward_per_token_logps(
                    _mmod, inputs[sl], batch_chunk=_fbc,
                    use_checkpoint=True)[:, plen - 1:]
                chunk_loss, chunk_stats = compute_loss(
                    cfg["algo"], chunk_logps, gen_logps[sl], advantages[sl],
                    mask[sl], cfg, ref_logps=ref_logps[sl])
                engine.backward(chunk_loss * (sl.stop - sl.start) / R)
                loss_total += float(chunk_loss.item()) * (sl.stop - sl.start) / R
                stats_list.append(chunk_stats)
            loss = loss_total   # 显示口径 = 整批等价 loss
            stats = {k: sum(s[k] for s in stats_list) / len(stats_list)
                     for k in stats_list[0]}
        else:
            per_token_logps = forward_per_token_logps(
                _mmod, inputs, batch_chunk=_fbc, use_checkpoint=True)[:, plen - 1:]
            loss, stats = compute_loss(
                cfg["algo"], per_token_logps, gen_logps, advantages, mask, cfg,
                ref_logps=ref_logps,
                num_items_in_batch=batch.get("num_items_in_batch"))
            engine.backward(loss)
            loss = float(loss.item())
        # 梯度健康探针：策略梯度全零 = 零梯度 bug 的直接签名（cispo 教训：
        # 300 步 loss 数值"正常"但梯度处处为零，训完评测才发现）。连续 3 次
        # 全零直接 fail-fast，不在废训上继续烧 GPU。
        if step % 10 == 0:
            gsum = sum(float(p.grad.abs().sum()) for p in engine.module.parameters()
                       if p.grad is not None)
            if gsum == 0.0:
                zero_grad_streak += 1
                print(f"[健康检查] 第{step}步策略梯度全零（连续 {zero_grad_streak}/3 次）",
                      flush=True)
                if zero_grad_streak >= 3:
                    raise RuntimeError(
                        "[健康检查] 连续 3 次策略梯度全零 → 零梯度 bug 签名"
                        "（loss 数值可能仍'正常'），训练中止排查 losses.py 梯度路径")
            else:
                zero_grad_streak = 0
        # 【2026-09-11 4B】step 前清缓存：DS fused optimizer 要一次性物化全参数
        # 梯度平坦副本(~8G bf16)，行循环的缓存块尺寸各异不会被它复用——不归还
        # 驱动就在第 N 步顶满卡（实测第 4 步 94.99G OOM 于 80M 分配）。代价仅是
        # 下一步重新分配（对 84s/it 可忽略）。
        torch.cuda.empty_cache()
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss:.6f}")
            # 【2026-09-11】ratio 口径健康：这些数原本只进 wandb（默认离线），终端
            # 看不见——而它们是"gen_logps 与 policy 差多少"的**唯一直接量**。判读：
            # 第 1 步 / 每次权重同步后的第一步，batch 与策略同权重 → ratio 理论上恒 1，
            # 实测 clip_frac/kl 就是两路实现的口径差（torch 副本档应严格 0；
            # vLLM logprobs 档（减法①）属实现层噪声，见 docs/02）。
            # 【2026-09-12 判读口径修正（必读）】
            #  · mean_ratio **恒为 1 是重要度采样恒等式**（E_{t~π_old}[π_new/π_old]=1
            #    对任意远的 π_new 都成立），没有诊断价值——留着只为与历史 run 对齐；
            #  · 真正该看的是 kl（KL(π_old‖π_new) 的采样估计）与 frac_d_gt_0.1
            #    （真的漂了多少比例的 token）；clip_frac 已按 1±clip_lo/hi 计；
            #  · staleness = 本步距该批数据生成时权重的 micro-step 差（gen_version
            #    由生成端随批上传）。旧版只能"假设 16 步周期"，实测 approx_kl 忽
            #    5e-4 忽 9.9e-2、完全对不上周期 → 新鲜度必须测而不是猜。
            if step == 1 or step % 10 == 0:
                _gv = batch.get("gen_version")
                if isinstance(_gv, int):
                    # 【2026-09-12 精确口径】step 是 1-based，而 gen_version 的语义是
                    # "第 v 步 engine.step() 之后的权重"（推送发生在 step 末）：
                    #   训练端在本次 forward 前已应用的更新数 = floor((step−1)/GAS)
                    #   生成端权重对应的更新数           = floor(gen_version/GAS)
                    # 直接写 step−gen_version 会恒多算 1 个 micro-step（step 1 应报 0
                    # 却报 1——真机日志已见），opt-step 那项还会带上 .25 的小数尾巴。
                    _gas = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
                    _upd = (step - 1) // _gas - _gv // _gas
                    _micro = step - 1 - _gv
                    _sx = f"staleness={_micro} micro-step({_upd} opt-step)"
                else:
                    # 只有"训练端裸传 state_dict / 静态 rollout（Q=None）"才会走到这
                    _sx = "staleness=n/a（该批无 gen_version 标签）"
                print(f"[train][口径] step {step}: clip_frac={stats['clip_frac']:.4f} "
                      f"kl={stats['kl']:.2e} approx_kl={stats['approx_kl']:.2e} "
                      f"frac|d|>0.1={stats['frac_d_gt_0.1']:.3f} "
                      f"mean_ratio={stats['mean_ratio']:.4f} "
                      f"| gen_version={_gv} {_sx}", flush=True)
            n = inputs.shape[0]
            totals["num"] += n
            if "acc_scores" in batch:
                totals["acc"] += float((batch["acc_scores"] > 0).sum())
                totals["fmt"] += float((batch["format_scores"] > 0).sum())
            if wandb_run is not None:
                log = {"loss": stats["loss"], "clip_frac": stats["clip_frac"],
                       "approx_kl": stats["approx_kl"], "mean_ratio": stats["mean_ratio"],
                       "kl": stats["kl"], "frac_d_gt_0.1": stats["frac_d_gt_0.1"],
                       "acc_correct_ratio": totals["acc"] / totals["num"],
                       "format_correct_ratio": totals["fmt"] / totals["num"]}
                if isinstance(batch.get("gen_version"), int):
                    _g = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
                    log["staleness_micro_steps"] = step - 1 - batch["gen_version"]
                    log["staleness_opt_steps"] = ((step - 1) // _g
                                                  - batch["gen_version"] // _g)
                wandb_run.log(log, step=step)

        if step % cfg["gen_update_steps"] == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print("[train] sending latest state_dict ...")
                # 【2026-09-12】(version, state_dict)：version = 本次推送对应的
                # train micro-step，生成端把它写进每组数据的 meta，训练端据此算
                # 真实 staleness（旧协议裸传 state_dict，新鲜度不可测）。
                Q.put((step, engine.module.state_dict()))
                print(f"[train] send state_dict ok! (version={step})")
            dist.barrier()

        if step % cfg["save_steps"] == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                save_name = os.path.join(cfg["out_dir"], f"step_{step}")
                os.makedirs(save_name, exist_ok=True)
                sd = engine.module.state_dict()
                sd = type(sd)({k: v.cpu() for k, v in sd.items()})
                engine.module.save_pretrained(save_name, state_dict=sd)
                tokenizer.save_pretrained(save_name)
                write_run_info(os.path.join(save_name, "run_info.json"), cfg)
                print(f"[train] saved -> {save_name}")
            dist.barrier()


def _spawn_gen(Q, cfg):
    """子进程入口：必须走模块顶层可寻址的函数（spawn pickle 约束）。"""
    from rlab.rollout import gen_worker
    gen_worker(Q, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--vllm_model_path", default=None,
                    help="vLLM 生成用的 checkpoint 路径（多模态 Qwen3.5 分裂加载："
                         "model_path=纯文本给 torch，本参数=原多模态给 vLLM）")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 all_steps")
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--gen_update_steps", type=int, default=None)
    ap.add_argument("--num_pre_Q", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gen_device", type=int, default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-log", action="store_true", help="关闭 wandb")
    ap.add_argument("--difficulty_path", default=None,
                    help="probe_difficulty.py 产出的通过率表；设置后训练池只保留"
                         "通过率在 difficulty_band 内的题（离线难度预过滤）")
    ap.add_argument("--seed", type=int, default=None,
                    help="固定训练种子（抽题顺序+生成采样），阶段1 起对比实验必带")
    ap.add_argument("--chat_template_kwargs", default=None,
                    help='JSON dict 透传 apply_chat_template；Qwen3.5 系必传 '
                         '\'{"enable_thinking": false}\'（不关 thinking 会烧穿单轮预算）')
    # 预算三件套（与 probe_difficulty 同名参数同语义：探针与训练必须同口径，
    # 否则难度表描述的是另一个采样分布、过滤失真。4B v4 探针实锤 3072）
    ap.add_argument("--round_gen_tokens", type=int, default=None,
                    help="覆盖单轮生成预算（默认取 preset；须与探针口径一致）")
    ap.add_argument("--max_rounds", type=int, default=None,
                    help="覆盖工具轮数上限（默认取 preset）")
    ap.add_argument("--max_context_tokens", type=int, default=None,
                    help="覆盖总上下文上限（默认取 preset）")
    ap.add_argument("--gen_gpu_mem", type=float, default=None,
                    help="覆盖 vLLM 显存占比（默认 0.45 是 3B 时代标定；Qwen3.5 "
                         "多模态实现实测超支 ~15G，4B 建议 0.30 给 ref/torch 腾位）")
    ap.add_argument("--zero_stage", type=int, default=None,
                    help="DeepSpeed zero stage（默认 0；4B 用 2 = 优化器态 offload "
                         "CPU，GPU1 静态 64G->24G）")
    ap.add_argument("--micro_rows", type=int, default=None,
                    help="训练步按行拆 micro-backward（0=整批；4B 用 1：前向 1 行->"
                         "backward->释放图，动态峰值降到单行；仅支持 sample_mean）")
    ap.add_argument("--optim_8bit", action="store_true",
                    help="bitsandbytes AdamW8bit 优化器（4B 显存：8bit 态留 GPU，"
                         "规避 fused fp32 的 96G 无解与 CPU offload 的 RAM 爆；"
                         "需 pip install bitsandbytes）")
    # 优化超参三件套（2026-09-11 补 CLI 入口）：4B 200 步跑出 +9.7pp 信号后，
    # 放大更新预算不需要改 config 源码——preset 默认值仍是 3B 时代标定。
    ap.add_argument("--lr", type=float, default=None,
                    help="覆盖学习率（preset 默认 1e-6；4B 加杠杆建议 5e-6）")
    ap.add_argument("--beta", type=float, default=None,
                    help="覆盖 KL 锚系数（preset 默认 0.04；放松建议 0.01）")
    ap.add_argument("--overlong_shaping", action="store_true",
                    help="开 DAPO 截断软悬崖惩罚（总长对 overlong_ref_tokens 起坡）。"
                         "【2026-09-12 已修正】参考系 = min(max_rounds×round_gen_tokens, "
                         "max_context_tokens−max_prompt_length)；预算自洽时它必然可达"
                         "（config.validate_retool_budget 会 fail-fast 拦截不自洽配置）")
    ap.add_argument("--trunc_shaping", type=float, default=None,
                    help="末段被轮长上限切断（trunc_final=1）的额外扣分权重（默认取 preset；"
                         "retool_math=0.5，其余算法=0）。这是 prose 轨迹唯一够得到的"
                         "长度反向信号——总长惩罚够不到 clen ≤ round_gen_tokens 的单轮轨迹")
    ap.add_argument("--discard_abort", type=float, default=None,
                    help="窗口丢弃率熔断线（默认 0.90，0=关闭）：超过即 fail-fast，"
                         "防丢弃率爬升到采样空转、训练端无限 waiting for batch 的事故")
    ap.add_argument("--grad_clip", type=float, default=None,
                    help="DeepSpeed 梯度裁剪（默认 0=不裁剪=历史口径；4B 大 lr 长跑建议 1.0）")
    ap.add_argument("--vllm_gen_logps", action="store_true",
                    help="gen_logps 改用逐轮 vLLM 采样 logprobs（省 GPU0 ~8G 的 torch "
                         "副本与每步一次全序列前向；数学同义但 kernel 口径变化，"
                         "首次启用请配 --verify_gen_logps 对拍）")
    ap.add_argument("--verify_gen_logps", type=int, default=None,
                    help="前 N 组同时算两路 gen_logps 并打印最大差，验完释放 torch 副本")
    ap.add_argument("--fwd_batch_chunk", type=int, default=None,
                    help="分块前向每次过 backbone 的行数（默认 1=逐行=历史口径；"
                         ">1 减少 kernel/调度开销，显存峰值线性增长；四处共用同一"
                         "口径，ref_server 由 FWD_BATCH_CHUNK 环境变量同步）")
    ap.add_argument("--attn_implementation", default=None,
                    choices=("sdpa", "flash_attention_2"),
                    help="torch 侧注意力实现（默认 sdpa；flash_attention_2 提速："
                         "head_dim 256 消除 T² math 回退，配合放开 --micro_rows；"
                         "需 pip install flash-attn，ref_server 同步降 bf16）")
    ap.add_argument("--local_rank", type=int, default=0)  # deepspeed 传入
    args = ap.parse_args()

    overrides = {}
    if args.model_path: overrides["model_path"] = args.model_path
    if args.vllm_model_path: overrides["vllm_model_path"] = args.vllm_model_path
    if args.steps: overrides["all_steps"] = args.steps
    if args.save_steps: overrides["save_steps"] = args.save_steps
    if args.gen_update_steps: overrides["gen_update_steps"] = args.gen_update_steps
    if args.num_pre_Q: overrides["num_pre_Q"] = args.num_pre_Q
    if args.out_dir: overrides["out_dir"] = args.out_dir
    if args.gen_device is not None: overrides["gen_device"] = args.gen_device
    if args.port is not None: overrides["ref_server_port"] = args.port
    if args.no_log: overrides["use_wandb"] = False
    if args.difficulty_path: overrides["difficulty_path"] = args.difficulty_path
    if args.seed is not None: overrides["seed"] = args.seed
    if args.chat_template_kwargs:
        overrides["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.round_gen_tokens is not None: overrides["round_gen_tokens"] = args.round_gen_tokens
    if args.max_rounds is not None: overrides["max_rounds"] = args.max_rounds
    if args.max_context_tokens is not None: overrides["max_context_tokens"] = args.max_context_tokens
    if args.gen_gpu_mem is not None: overrides["gen_gpu_mem"] = args.gen_gpu_mem
    if args.zero_stage is not None: overrides["zero_stage"] = args.zero_stage
    if args.micro_rows is not None: overrides["micro_rows"] = args.micro_rows
    if args.optim_8bit: overrides["optim_8bit"] = True
    if args.attn_implementation: overrides["attn_implementation"] = args.attn_implementation
    if args.lr is not None: overrides["lr"] = args.lr
    if args.beta is not None: overrides["beta"] = args.beta
    if args.overlong_shaping: overrides["overlong_shaping"] = True
    if args.trunc_shaping is not None: overrides["trunc_shaping"] = args.trunc_shaping
    if args.discard_abort is not None: overrides["discard_abort"] = args.discard_abort
    if args.grad_clip is not None: overrides["gradient_clipping"] = args.grad_clip
    if args.fwd_batch_chunk is not None: overrides["fwd_batch_chunk"] = args.fwd_batch_chunk
    if args.vllm_gen_logps: overrides["vllm_gen_logps"] = True
    if args.verify_gen_logps is not None: overrides["verify_gen_logps"] = args.verify_gen_logps

    cfg = get_config(args.algo, **overrides)
    # 【2026-09-12】偏离签名先算好再打印/落盘：wandb run name 与 run_info.json 同源，
    # 终端这一行是"和参考项目差了哪几维"的第一现场（grep signature= 即可）。
    cfg["run_signature"] = run_signature(cfg)
    print(f"[train] 偏离签名 signature={cfg['run_signature']}"
          f"（对比参考 agentic-rl-lab/05-retool 与上轮 run 时先看这一行）")
    print("[train] config:", json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
    run_training(cfg, args)


if __name__ == "__main__":
    main()
