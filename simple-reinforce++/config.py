# 【KL锚单变量实验版】batch128实验结论：大batch早期更强(74.7@12.5u)但崩得更狠(57.7@37.5u<BASE)，
# 批基线方差不是滑坡主因；下一嫌疑=KL锚太弱(beta=0.01, GRPO用0.04)。本版回到batch32原始几何，只动beta。
# 对照链：batch32/beta0.01(72.3→70.7→67.3) vs batch128/beta0.01(74.7→71.3→57.7) vs batch32/beta0.04(本版)
# 布局：GPU0 = ref_server + vLLM生成(共用)，GPU1 = 训练(单卡)
base_config={
  "model_path": "/root/Qwen2.5-3B",
  "gen_device": "0",                      # 必须指向 ref_server 所在的卡
  "train_gpu_num": 1,
  "train_batch_size": 4,
  "beta": 0.04,                           # 【实验变量】0.01→0.04，与GRPO对齐
  "all_steps": 300,                       # 恢复batch32版几何：300步=37.5次更新
  "Q_batch_size": 32,
  "num_pre_Q": 1,
  "gen_update_steps": 16,                 # 每2次更新同步一次（与batch32成功run等比）
  "save_steps": 100,
  "clip_param": 0.2,
  "port": 51414,
  "ref_server": "http://localhost:51414",
}


ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,     # 恢复batch32版：macro=4*8=32
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
