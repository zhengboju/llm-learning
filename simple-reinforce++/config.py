# 【batch128单变量实验版】验证假设：RF++过训滑坡源于macro batch=32太小→批基线方差大
# 原版(batch32): Q_batch_size=32, gradient_accumulation_steps=8, all_steps=300, save_steps=100, gen_update_steps=16
# 本版: macro batch = grad_accum(32) × micro(4) = 128；all_steps=1200 使 optimizer 更新数与原版严格配对
#   (1200/32=37.5 次更新 = 原版 300/8；checkpoint step_400/800/1200 对应原版 step_100/200/300 的更新数)
# 布局：GPU0 = ref_server + vLLM生成(共用)，GPU1 = 训练(单卡)
base_config={
  "model_path": "/root/Qwen2.5-3B",
  "gen_device": "0",                      # 必须指向 ref_server 所在的卡
  "train_gpu_num": 1,
  "train_batch_size": 4,                  # micro batch 不变（显存约束，单变量只动 macro）
  "beta": 0.01,
  "all_steps": 1200,                      # micro步数；1200/32=37.5次更新，与原版300步(37.5次)配对
  "Q_batch_size": 128,                    # 【实验变量】32→128，每题仍只采1条(num_pre_Q=1)
  "num_pre_Q": 1,
  "gen_update_steps": 64,                 # 64 micro步=2次更新同步一次，与原版频率等比
  "save_steps": 400,                      # step_400/800/1200 三个checkpoint
  "clip_param": 0.2,
  "port": 51414,
  "ref_server": "http://localhost:51414",
}


ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 32,    # 【实验变量】8→32：macro=4*32=128
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    # stage0全放GPU(~60G<96G)，不offload→不占CPU内存(避免-9)、不pin_memory
    "zero_optimization": {
        "stage": 0
    }
}
