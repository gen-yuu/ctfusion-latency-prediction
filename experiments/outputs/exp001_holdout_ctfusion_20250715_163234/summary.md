# Experiment Summary: exp024_holdout_interactoin_main

CTFusion (Proposed)

## Overall Results (on Holdout Set)

- Overall MAPE (Data-weighted): 17.59%
- Mean MAPE across Test Groups (Simple Avg.): 17.29%
- Std Dev of per-group MAPE: 0.0756
- Best Group: `intel_xeon_gold_5122__nvidia_geforce_rtx_4070` (11.29%)
- Worst Group: `intel_xeon_gold_6330__nvidia_a100_pcie_40gb` (30.08%)
- Target Column: total_time_ms
- Number of Features: 36

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
  "importance_type": "gain",
  "num_leaves": 30,
  "feature_fraction": 0.7,
  "bagging_fraction": 0.7,
  "bagging_freq": 1,
  "reg_alpha": 0.5,
  "reg_lambda": 0.5
}
```

## Test Group Details

| Group ID | MAPE (%) | Test Group Name                                   |
| :------- | :------- | :------------------------------------------------ |
| 1        | 30.08%   | `intel_xeon_gold_6330__nvidia_a100_pcie_40gb`     |
| 2        | 15.53%   | `intel_core_i5_13th_gen__nvidia_geforce_rtx_4070` |
| 3        | 17.39%   | `intel_core_i7_13th_gen__nvidia_geforce_rtx_4070` |
| 4        | 12.18%   | `intel_core_i7_9th_gen__nvidia_geforce_rtx_4070`  |
| 5        | 11.29%   | `intel_xeon_gold_5122__nvidia_geforce_rtx_4070`   |
