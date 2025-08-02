# ctfusion/tune.py

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold

# 警告を非表示
warnings.filterwarnings("ignore", category=UserWarning)


def run_loocv_for_tuning(df, feature_cols, target_col, validation_config, model_params):
    """チューニングの1試行のためにLOOCVを実行し、平均MAPEを返す関数"""

    X = df[feature_cols]
    y = np.log1p(df[target_col])  # ターゲットの対数変換
    groups = df[validation_config["group_by_col"]]
    gkf = GroupKFold(n_splits=validation_config["k_folds"])

    fold_mape_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mape",
            callbacks=[
                lgb.early_stopping(500, verbose=False)
            ],  # チューニング中はログを非表示に
        )

        preds_log = model.predict(X_val)

        y_original = np.expm1(y_val)
        preds_original = np.expm1(preds_log)

        mape = mean_absolute_percentage_error(y_original, preds_original)
        fold_mape_scores.append(mape)

    return np.mean(fold_mape_scores)


def objective(trial, base_params, search_space, df, feature_cols, validation_config):
    """Optunaが最適化する目的関数"""

    tuning_params = {}
    for param, space in search_space.items():
        if space["type"] == "int":
            low = int(space["low"])
            high = int(space["high"])
            tuning_params[param] = trial.suggest_int(param, low, high)
        elif space["type"] == "float":
            low = float(space["low"])
            high = float(space["high"])
            tuning_params[param] = trial.suggest_float(
                param, low, high, log=space.get("log", False)
            )

    # configの基本パラメータと、Optunaが提案したチューニング用パラメータを結合
    model_params = {**base_params, **tuning_params}

    # LOOCVを実行して平均MAPEを取得
    mean_mape = run_loocv_for_tuning(
        df, feature_cols, "total_time_ms", validation_config, model_params
    )

    return mean_mape


def main(exp_config_path: str, tuning_config_path: str):
    # 実験設定とチューニング設定をロード
    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)
    with open(tuning_config_path, "r") as f:
        tuning_config = yaml.safe_load(f)

    print(f"Base experiment config: {exp_config_path}")
    print(f"Tuning config: {tuning_config_path}")

    # データをロード
    df = pd.read_csv(exp_config["data"]["dataset_path"])

    # 特徴量リストを作成
    feature_cols = []
    for key in ["host_cols", "transfer_cols", "device_cols", "task_cols"]:
        if key in exp_config["features"]:
            feature_cols.extend(exp_config["features"][key])

    # チューニング結果の保存先ディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_name = Path(exp_config_path).stem
    output_dir = Path(f"outputs_tuning/{tuning_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tuning results will be saved to: {output_dir}")

    # OptunaのStudyを作成し、最適化を実行
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            exp_config["model"]["params"],
            tuning_config["search_space"],
            df,
            feature_cols,
            exp_config["validation"],
        ),
        n_trials=tuning_config["n_trials"],
    )

    # --- 結果の保存 ---
    print("\n--- Tuning Finished ---")
    print(f"Best trial MAPE: {study.best_value}")
    print(f"Best params: {study.best_params}")

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "tuning_history.csv", index=False)

    print(f"\nTuning results saved successfully to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--tuning-config",
        type=str,
        required=True,
        help="Path to the tuning config file.",
    )
    args = parser.parse_args()
    main(args.exp_config, args.tuning_config)
