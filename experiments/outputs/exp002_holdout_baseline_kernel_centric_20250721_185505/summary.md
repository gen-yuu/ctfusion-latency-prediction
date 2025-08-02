# Experiment Summary: exp002_holdout_baseline_kernel_centric

GPU のカーネル実行時間のみを考慮した推論の基本性能を新しい GPU をテストデータとして評価する

## Overall Results (on Holdout Set)

- Overall MAPE (Data-weighted): 29.09%
- Mean MAPE across Test Groups (Simple Avg.): 28.74%
- Std Dev of per-group MAPE: 0.0919
- Best Group: `intel_core_i5_13th_gen__nvidia_geforce_rtx_4070` (21.76%)
- Worst Group: `intel_xeon_gold_6330__nvidia_a100_pcie_40gb` (43.65%)
- Target Column: total_time_ms
- Number of Features: 1

## Experiment Setup

- Validation Strategy: holdout
- Test Groups: ['intel_xeon_gold_6330__nvidia_a100_pcie_40gb', 'intel_core_i5_13th_gen__nvidia_geforce_rtx_4070', 'intel_core_i7_13th_gen__nvidia_geforce_rtx_4070', 'intel_core_i7_9th_gen__nvidia_geforce_rtx_4070', 'intel_xeon_gold_5122__nvidia_geforce_rtx_4070']

## Hyperparameters

```json
{
  "objective": "regression_l1",
  "metric": "mape",
  "n_estimators": 100000,
  "learning_rate": 0.05,
  "seed": 42,
  "n_jobs": -1,
  "importance_type": "gain"
}
```

## Test Group Details

| Group ID | MAPE (%) | Test Group Name                                   |
| :------- | :------- | :------------------------------------------------ |
| 1        | 43.65%   | `intel_xeon_gold_6330__nvidia_a100_pcie_40gb`     |
| 2        | 21.76%   | `intel_core_i5_13th_gen__nvidia_geforce_rtx_4070` |
| 3        | 22.08%   | `intel_core_i7_13th_gen__nvidia_geforce_rtx_4070` |
| 4        | 24.83%   | `intel_core_i7_9th_gen__nvidia_geforce_rtx_4070`  |
| 5        | 31.36%   | `intel_xeon_gold_5122__nvidia_geforce_rtx_4070`   |

## Feature Details

- Feature Columns: ['gpu_compute_time_ms']
