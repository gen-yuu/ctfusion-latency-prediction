import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold, train_test_split

from ctfusion.data.preprocess import scale_features
from ctfusion.utils import save_holdout_summary_report, save_summary_report


def run_holdout_validation(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    validation_config: dict,
    model_params: dict,
    output_path: Path,
):
    """固定されたホールドアウトセットで学習と評価を行う"""

    test_groups = validation_config["test_groups"]
    group_col = validation_config["group_by_col"]

    test_df = df[df[group_col].isin(test_groups)].copy()
    train_val_df = df[~df[group_col].isin(test_groups)]

    print(f"Holdout test set groups: {test_groups}")
    print(f"Train + Validation set size: {len(train_val_df)}")

    X_test = test_df[feature_cols]

    X_train_val = train_val_df[feature_cols]
    y_train_val_log = np.log1p(train_val_df[target_col])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val_log,
        test_size=0.2,
        random_state=model_params.get("seed", 42),
    )

    print("\n--- Training single model for holdout validation ---")
    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mape",
        callbacks=[
            lgb.early_stopping(500, verbose=True),
            lgb.log_evaluation(1000),
        ],
    )
    joblib.dump(model, output_path / "models" / "model_holdout.pkl")

    test_preds_log = model.predict(X_test)
    test_preds_original = np.expm1(test_preds_log)
    test_df["preds"] = test_preds_original

    metric_records = []
    print("\n--- Holdout Test Set Results ---")
    for i, group in enumerate(test_groups):
        group_df = test_df[test_df[group_col] == group]
        group_mape = mean_absolute_percentage_error(
            group_df[target_col], group_df["preds"]
        )
        print(f"MAPE for group '{group}': {group_mape:.4f}")
        metric_records.append(
            {"fold": i + 1, "val_groups": [group], "val_mape": group_mape}
        )
    overall_mape = mean_absolute_percentage_error(test_df[target_col], test_df["preds"])
    print(f"\nOverall Holdout Test MAPE: {overall_mape:.4f}")

    metrics_df = pd.DataFrame(metric_records)

    return metrics_df, test_df, overall_mape


# 交差検証を実行する関数
def run_group_kfold_cv(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    validation_config: dict,
    model_params: dict,
    preprocessing_config: dict,
    output_path: Path,
):
    """
    GroupKFold交差検証を実行し、詳細なfoldごとの結果とOOF予測を返す
    """

    X = df[feature_cols]
    print("Applying log1p transformation to the target variable.")
    y = np.log1p(df[target_col])
    groups = df[validation_config["group_by_col"]]

    gkf = GroupKFold(n_splits=validation_config["k_folds"])

    fold_results = []  # 各foldの結果を格納するリスト
    oof_preds_log = pd.Series([0.0] * len(df), index=df.index, name="oof_preds_log")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"--- Fold {fold+1}/{validation_config['k_folds']} ---")
        print(f"train: {len(train_idx)}, val: {len(val_idx)}")
        val_groups = groups.iloc[val_idx].unique().tolist()
        print(f"Val groups: {val_groups}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if preprocessing_config.get("enabled"):
            print("Applying feature scaling...")
            if preprocessing_config["method"] == "standard_scaler":
                # スケーラーは各foldで学習させ、使い捨てる
                X_train, X_val, _ = scale_features(X_train, X_val)
            else:
                raise ValueError(
                    f"Unknown preprocessing method: {preprocessing_config['method']}"
                )

        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mape",
            callbacks=[
                lgb.early_stopping(500, verbose=True),
                lgb.log_evaluation(100),
            ],
        )

        val_preds_log = model.predict(X_val)
        oof_preds_log.iloc[val_idx] = val_preds_log

        y_val_original = np.expm1(y_val)
        val_preds_original = np.expm1(val_preds_log)
        mape = mean_absolute_percentage_error(y_val_original, val_preds_original)

        # foldの結果を記録
        fold_results.append(
            {
                "fold": fold + 1,
                "val_mape": mape,
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "val_groups": val_groups,  # 検証に使われたsystem_idのリスト
            }
        )

        print(f"Fold {fold+1} MAPE: {mape:.4f}")
        joblib.dump(model, output_path / "models" / f"model_fold_{fold}.pkl")

    oof_preds_original_data = np.expm1(oof_preds_log)
    oof_preds_to_return = pd.Series(
        oof_preds_original_data, index=df.index, name="oof_preds"
    )
    return pd.DataFrame(fold_results), oof_preds_to_return


# 実験全体を管理する関数
def train(config_path: str, exp_output_path: Path):
    """
    設定ファイルに基づき、データの準備と適切な検証方法の実行を管理する
    """
    # 設定ファイルの読み込み
    print("Loading config")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # データの準備
    print("Preparing data")
    df = pd.read_csv(config["data"]["dataset_path"])
    features_config = config.get("features", {})
    feature_cols = (
        (features_config.get("host_cols") or [])
        + (features_config.get("transfer_cols") or [])
        + (features_config.get("device_cols") or [])
        + (features_config.get("task_cols") or [])
    )

    # 検証方法に応じて処理を分岐
    strategy = config["validation"]["strategy"]
    print(f"Starting validation (strategy: {strategy})")

    if strategy == "group_kfold":
        fold_results_df, oof_preds = run_group_kfold_cv(
            df=df,
            feature_cols=feature_cols,
            target_col=config["data"]["target_col"],
            validation_config=config["validation"],
            model_params=config["model"]["params"],
            preprocessing_config=config.get("preprocessing", {}),
            output_path=exp_output_path,
        )
        mean_mape = fold_results_df["val_mape"].mean()
        print(f"\nMean MAPE across all folds: {mean_mape:.4f}")

        # foldごとのメトリクスを保存
        fold_results_df.to_csv(
            exp_output_path / "fold_metrics.csv",
            index=False,
        )

        # 元のデータフレームにOOF予測結果を列として追加
        df["oof_preds"] = oof_preds

        # 誤差とMAPEを計算して、列として追加
        target_col = config["data"]["target_col"]
        df["error"] = df[target_col] - df["oof_preds"]
        df["mape"] = df["error"].abs() / df[target_col]

        # 保存したい列だけを選択
        output_cols = [
            config["validation"]["group_by_col"],  # system_id
            "model_name",
            target_col,  # total_time_ms (or elapsed_time_ms)
            "oof_preds",
            "error",
            "mape",
        ]
        output_df = df[output_cols]

        output_df = output_df.rename(
            columns={
                config["validation"]["group_by_col"]: "system_id",
                target_col: "target",
            }
        )
        output_df.to_csv(
            exp_output_path / "oof_predictions.csv",
            index=False,
        )
        config["config_path"] = config_path
        save_summary_report(
            fold_results_df,
            config,
            feature_cols,
            exp_output_path,
        )
    elif strategy == "holdout":
        results_df, preds_df, overall_mape = run_holdout_validation(
            df=df,
            feature_cols=feature_cols,
            target_col=config["data"]["target_col"],
            validation_config=config["validation"],
            model_params=config["model"]["params"],
            output_path=exp_output_path,
        )
        # ホールドアウト用の簡易的なレポート作成
        results_df.to_csv(exp_output_path / "holdout_metrics.csv", index=False)

        # 予測結果と誤差を計算
        target_col = config["data"]["target_col"]
        output_df = preds_df
        output_df["error"] = output_df[target_col] - output_df["preds"]
        output_df["ape"] = output_df["error"].abs() / output_df[target_col]
        output_cols = [
            config["validation"]["group_by_col"],
            "model_name",
            "batch_size",
            target_col,
            "preds",
            "error",
            "ape",
        ]
        # holdoutの場合、一部カラムが存在しないので、存在するカラムのみ選択
        final_cols = [col for col in output_cols if col in output_df.columns]
        output_df[final_cols].rename(
            columns={
                config["validation"]["group_by_col"]: "system_id",
                target_col: "target",
            }
        ).to_csv(exp_output_path / "predictions.csv", index=False)
        config["config_path"] = config_path

        save_holdout_summary_report(
            results_df, overall_mape, config, feature_cols, exp_output_path
        )
    else:
        raise ValueError(f"Unknown validation strategy: {strategy}")

    print("\n--- Training script finished ---")
