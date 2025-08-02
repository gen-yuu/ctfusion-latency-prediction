# Experiment Summary: exp005_holdout_ablation_without_interaction
CTFusion 相互作用ベクトルを除いた提案手法の基本性能をHardwareによる新しいGPUをテストデータとして評価する

## Overall Results (on Holdout Set)

- Overall MAPE (Data-weighted): 18.27%
- Mean MAPE across Test Groups (Simple Avg.): 18.02%
- Std Dev of per-group MAPE: 0.0740
- Best Group: `intel_xeon_gold_5122__nvidia_geforce_rtx_4070` (11.05%)
- Worst Group: `intel_xeon_gold_6330__nvidia_a100_pcie_40gb` (28.80%)
- Target Column: total_time_ms
- Number of Features: 29

## Experiment Setup

- Validation Strategy: holdout
- Test Groups: ['intel_xeon_gold_6330__nvidia_a100_pcie_40gb', 'intel_core_i5_13th_gen__nvidia_geforce_rtx_4070', 'intel_core_i7_13th_gen__nvidia_geforce_rtx_4070', 'intel_core_i7_9th_gen__nvidia_geforce_rtx_4070', 'intel_xeon_gold_5122__nvidia_geforce_rtx_4070']

## Hyperparameters

```json
{
    "objective":"regression_l1",
    "metric":"mape",
    "n_estimators":100000,
    "learning_rate":0.05,
    "seed":42,
    "n_jobs":-1,
    "importance_type":"gain",
    "num_leaves":20,
    "feature_fraction":0.5,
    "bagging_fraction":0.5,
    "bagging_freq":1,
    "reg_alpha":1.0,
    "reg_lambda":1.0
}
```

## Test Group Details

| Group ID | MAPE (%) | Test Group Name |
|:---|:---|:---|
| 1 | 28.80% | `intel_xeon_gold_6330__nvidia_a100_pcie_40gb` |
| 2 | 16.14% | `intel_core_i5_13th_gen__nvidia_geforce_rtx_4070` |
| 3 | 22.00% | `intel_core_i7_13th_gen__nvidia_geforce_rtx_4070` |
| 4 | 12.12% | `intel_core_i7_9th_gen__nvidia_geforce_rtx_4070` |
| 5 | 11.05% | `intel_xeon_gold_5122__nvidia_geforce_rtx_4070` |
## Feature Details

- Feature Columns: ['host_compute_Image_Preprocessing_Rate_peak_host_throughput', 'host_compute_Image_Preprocessing_Rate_single_worker_throughput', 'host_compute_Image_Preprocessing_Rate_scalability_factor', 'data_transfer_D2H_PCIe_peak_bandwidth_gbps', 'data_transfer_D2H_PCIe_small_transfer_latency_ms', 'data_transfer_D2H_DMA_peak_bandwidth_gbps', 'data_transfer_D2H_DMA_small_transfer_latency_ms', 'data_transfer_H2D_PCIe_peak_bandwidth_gbps', 'data_transfer_H2D_PCIe_small_transfer_latency_ms', 'data_transfer_H2D_DMA_peak_bandwidth_gbps', 'data_transfer_H2D_DMA_small_transfer_latency_ms', 'device_virtual_cnn_block_latency_fp32', 'device_virtual_cnn_block_latency_fp16', 'device_virtual_transformer_block_latency_fp32', 'device_virtual_transformer_block_latency_fp16', 'model_flops_g', 'model_memory_mb', 'num_attention', 'num_batchnorm2d', 'num_conv2d', 'num_gelu', 'num_layernorm', 'num_linear', 'num_relu', 'num_silu', 'batch_size', 'total_frames', 'im_height', 'im_width']

