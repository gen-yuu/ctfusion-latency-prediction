from pathlib import Path

import pandas as pd


def save_summary_report(
    fold_results_df: pd.DataFrame,
    config: dict,
    feature_cols: list,
    output_path: Path,
):
    """
    K-Fold交差検証用のMarkdown形式のサマリーレポートを保存する
    """
    exp_name = Path(config["config_path"]).stem
    mean_mape = fold_results_df["val_mape"].mean()
    std_mape = fold_results_df["val_mape"].std()
    best_fold = fold_results_df.loc[fold_results_df["val_mape"].idxmin()]
    worst_fold = fold_results_df.loc[fold_results_df["val_mape"].idxmax()]

    # --- Markdownコンテンツの生成 ---
    report = f"# Experiment Summary: {exp_name}\n"
    report += f"{config.get('description', '')}\n\n"
    report += "## Overall Results\n\n"
    report += f"- Mean MAPE: {mean_mape:.2%}\n"
    report += f"- Std Dev of MAPE: {std_mape:.4f}\n"
    report += f"- Min MAPE (Best Fold): {best_fold['val_mape']:.2%}\
            (Fold {int(best_fold['fold'])})\n"
    report += f"- Max MAPE (Worst Fold): {worst_fold['val_mape']:.2%}\
            (Fold {int(worst_fold['fold'])})\n"
    report += f"- Target Column: {config['data']['target_col']}\n"
    report += f"- Number of Features: {len(feature_cols)}\n\n"

    report += "## Experiment Setup\n\n"
    report += f"- Validation Strategy: {config['validation']['strategy']}\n"
    if config["validation"]["strategy"] == "group_kfold":
        report += f"- K-Folds: {config['validation']['k_folds']}\n"
    report += f"- Group Key: {config['validation']['group_by_col']}\n\n"

    hyperparams = config["model"]["params"]
    report += f"## Hyperparameters\n\n```json\n{pd.Series(hyperparams).to_json(indent=4)}\n```\n\n"

    report += "## Fold Details\n\n"
    report += "| Fold | MAPE (%) | Validation Groups |\n"
    report += "|:---|:---|:---|\n"
    for _, row in fold_results_df.iterrows():
        groups_str = ", ".join(row["val_groups"])
        report += f"| {int(row['fold'])} | {row['val_mape']:.2%} | `{groups_str}` |\n"

    # --- ファイルに書き込み ---
    with open(output_path / "summary.md", "w") as f:
        f.write(report)
    print("Summary report saved to summary.md")


def save_holdout_summary_report(
    metrics_df: pd.DataFrame,
    overall_mape: float,  # <<< 全体のMAPEを直接受け取るように変更
    config: dict,
    feature_cols: list,
    output_path: Path,
):
    """
    ホールドアウト検証用のMarkdown形式のサマリーレポートを保存する
    """
    exp_name = Path(config["config_path"]).stem

    # グループごとのMAPEの統計値は参考情報として計算
    mean_per_group_mape = metrics_df["val_mape"].mean()
    std_mape = metrics_df["val_mape"].std()
    best_group = metrics_df.loc[metrics_df["val_mape"].idxmin()]
    worst_group = metrics_df.loc[metrics_df["val_mape"].idxmax()]

    # --- Markdownコンテンツの生成 ---
    report = f"# Experiment Summary: {exp_name}\n"
    report += f"{config.get('description', '')}\n\n"
    report += "## Overall Results (on Holdout Set)\n\n"

    # ▼▼▼ ここで渡された `overall_mape` を使う ▼▼▼
    report += f"- Overall MAPE (Data-weighted): {overall_mape:.2%}\n"
    report += (
        f"- Mean MAPE across Test Groups (Simple Avg.): {mean_per_group_mape:.2%}\n"
    )
    report += f"- Std Dev of per-group MAPE: {std_mape:.4f}\n"
    report += f"- Best Group: `{best_group['val_groups'][0]}` ({best_group['val_mape']:.2%})\n"
    report += f"- Worst Group: `{worst_group['val_groups'][0]}` ({worst_group['val_mape']:.2%})\n"
    report += f"- Target Column: {config['data']['target_col']}\n"
    report += f"- Number of Features: {len(feature_cols)}\n\n"

    report += "## Experiment Setup\n\n"
    report += f"- Validation Strategy: {config['validation']['strategy']}\n"
    report += f"- Test Groups: {config['validation']['test_groups']}\n\n"

    hyperparams = config["model"]["params"]
    report += f"## Hyperparameters\n\n```json\n{pd.Series(hyperparams).to_json(indent=4)}\n```\n\n"

    report += "## Test Group Details\n\n"
    report += "| Group ID | MAPE (%) | Test Group Name |\n"
    report += "|:---|:---|:---|\n"
    for _, row in metrics_df.iterrows():
        groups_str = ", ".join(row["val_groups"])
        report += f"| {int(row['fold'])} | {row['val_mape']:.2%} | `{groups_str}` |\n"

    report += "## Feature Details\n\n"
    report += f"- Feature Columns: {feature_cols}\n\n"

    # --- ファイルに書き込み ---
    with open(output_path / "summary.md", "w") as f:
        f.write(report)
    print("Summary report saved to summary.md")
