# 【3B单卡适配版】原版是 Qwen2-7B + 2卡训练 + 7B，已改为适配 H20 2卡Pod(60G内存limit)
# 布局：GPU0 = ref_server + vLLM生成(共用)，GPU1 = 训练(单卡)
base_config={
  "model_path": "/root/Qwen2.5-3B",      # 原版 Qwen2-7B → 3B（Pod装不下7B）
  "gen_device": "0",                      # vLLM和ref_server共用物理P0；P1留给训练
  "train_gpu_num": 1,                     # 原版2 → 单卡训练
  "train_batch_size": 4,
  "beta": 0.01,                           # REINFORCE++默认KL权重
  "all_steps": 300,                       # 和GRPO实验对齐，方便对比
  "Q_batch_size": 32,
  "num_pre_Q": 1,                         # REINFORCE++核心：不依赖组
  "gen_update_steps": 16,
  "save_steps": 100,                      # 100的倍数，确保step300也被保存
  "clip_param": 0.2,
  "port": 51414,
  "ref_server": "http://localhost:51414",
}


ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    # 单卡凑 macro_step: train_gpu_num(1) * grad_accum(8) = 8（原版是2*4=8）
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    # 【3B适配】stage0全放GPU(~60G<96G)，不offload→不占CPU内存(避免-9)、不pin_memory
    "zero_optimization": {
        "stage": 0
    }
}
