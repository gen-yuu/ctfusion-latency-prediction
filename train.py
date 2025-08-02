# train.py

import argparse
import datetime
from pathlib import Path

import yaml

from ctfusion.modeling.train import train


def main():
    # コマンドラインから設定ファイルのパスを受け取る
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file"
    )
    args = parser.parse_args()

    # 設定ファイルを読み込む
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 実験結果を保存するディレクトリを作成
    # 例: outputs/exp001_cv_main_20250711_1530
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = Path(args.config).stem  # 設定ファイル名から実験名を取得
    exp_output_path = Path(config["output_dir"]) / f"{exp_name}_{now}"
    exp_output_path.mkdir(parents=True, exist_ok=True)
    exp_model_dir = exp_output_path / "models"
    exp_model_dir.mkdir(exist_ok=True)

    print(f"Experiment started: {exp_name}")
    print(f"Output will be saved to: {exp_output_path}")

    # train関数を呼び出して実験を実行
    train(config_path=args.config, exp_output_path=exp_output_path)

    print(f"Experiment finished. Results are in {exp_output_path}")


if __name__ == "__main__":
    main()
