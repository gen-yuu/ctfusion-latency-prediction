# Experiment Summary: exp004_holdout_ablation_device_only
CTFusion device(GPU)のベクトルのみを使用した提案手法の基本性能をHardwareによる新しいGPUをテストデータとして評価する

## Overall Results (on Holdout Set)

- Overall MAPE (Data-weighted): 18.11%
- Mean MAPE across Test Groups (Simple Avg.): 17.81%
- Std Dev of per-group MAPE: 0.0749
- Best Group: `intel_core_i5_13th_gen__nvidia_geforce_rtx_4070` (11.93%)
- Worst Group: `intel_xeon_gold_6330__nvidia_a100_pcie_40gb` (30.61%)
- Target Column: total_time_ms
- Number of Features: 18

## Experiment Setup

- Validation Strategy: holdout
- Test Groups: ['intel_xeon_gold_6330__nvidia_a100_pcie_40gb', 'intel_core_i5_13th_gen__nvidia_geforce_rtx_4070', 'intel_core_i7_13th_gen__nvidia_geforce_rtx_4070', 'intel_core_i7_9th_gen__nvidia_geforce_rtx_4070', 'intel_xeon_gold_5122__nvidia_geforce_rtx_4070']

## Hyperparameters

```json
{
    "objective":"regression_l1",
    "metric":"mape",
    "n_estimators":10000,
    "learning_rate":0.1,
    "seed":42,
    "n_jobs":-1,
    "importance_type":"gain",
    "num_leaves":15,
    "feature_fraction":0.7,
    "bagging_fraction":0.7
}
```

## Test Group Details

| Group ID | MAPE (%) | Test Group Name |
|:---|:---|:---|
| 1 | 30.61% | `intel_xeon_gold_6330__nvidia_a100_pcie_40gb` |
| 2 | 11.93% | `intel_core_i5_13th_gen__nvidia_geforce_rtx_4070` |
| 3 | 17.89% | `intel_core_i7_13th_gen__nvidia_geforce_rtx_4070` |
| 4 | 15.07% | `intel_core_i7_9th_gen__nvidia_geforce_rtx_4070` |
| 5 | 13.53% | `intel_xeon_gold_5122__nvidia_geforce_rtx_4070` |
## Feature Details

- Feature Columns: ['device_virtual_cnn_block_latency_fp32', 'device_virtual_cnn_block_latency_fp16', 'device_virtual_transformer_block_latency_fp32', 'device_virtual_transformer_block_latency_fp16', 'model_flops_g', 'model_memory_mb', 'num_attention', 'num_batchnorm2d', 'num_conv2d', 'num_gelu', 'num_layernorm', 'num_linear', 'num_relu', 'num_silu', 'batch_size', 'total_frames', 'im_height', 'im_width']

