# ctfusion/data/make_dataset.py

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# --- 定数定義 (Constants) ---
# プロジェクトのルートディレクトリを動的に解決
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # インタラクティブな環境（例: Jupyter Notebook）で実行する場合
    ROOT_DIR = Path.cwd()

# 入力ファイルのパス
RAW_DATA_DIR = ROOT_DIR / "data" / "00_raw"
CATALOG_DATA_DIR = ROOT_DIR / "data" / "01_catalog"
OBJ_DET_RESULTS_PATH = RAW_DATA_DIR / "obj_det_results.csv"
BENCHMARK_FEATURES_PATH = RAW_DATA_DIR / "benchmark_parameters.csv"
CPU_SPECS_PATH = CATALOG_DATA_DIR / "cpu_specs.csv"
GPU_SPECS_PATH = CATALOG_DATA_DIR / "gpu_specs.csv"
MODEL_SPECS_PATH = CATALOG_DATA_DIR / "model_specs.csv"

# 出力ファイルのパス
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "02_processed"
FINAL_DATASET_PATH = PROCESSED_DATA_DIR / "stgc_dataset.csv"


def load_data_sources():
    """すべてのデータソースを読み込み、辞書として返す。"""
    print("[1/4] Loading all data sources...")
    paths = {
        "measurements": OBJ_DET_RESULTS_PATH,
        "benchmark": BENCHMARK_FEATURES_PATH,
        "cpu_specs": CPU_SPECS_PATH,
        "gpu_specs": GPU_SPECS_PATH,
        "model_specs": MODEL_SPECS_PATH,
    }
    data_frames = {}
    try:
        for name, path in paths.items():
            print(f"  - Loading {path.name}...")
            data_frames[name] = pd.read_csv(path)
    except FileNotFoundError as e:
        sys.exit(f"\n[ERROR] Required data file not found: {e}. Please ensure all raw data files are in place.")
    
    return data_frames


def merge_base_datasets(data_frames: dict):
    """データソースを結合し、ベースとなる中間データセットを作成する。"""
    print("\n[2/4] Merging base datasets...")
    
    # --- データフレームの準備 ---
    df_measure = data_frames["measurements"].copy()
    df_benchmark = data_frames["benchmark"].copy()
    df_cpu = data_frames["cpu_specs"].copy()
    df_gpu = data_frames["gpu_specs"].copy()
    df_model = data_frames["model_specs"].copy()

    # --- 必要な列の選択とリネーム ---
    # カラム名の衝突を避けるためのリネーム
    df_cpu.rename(columns={"release": "cpu_release"}, inplace=True)
    df_gpu.rename(columns={"release": "gpu_release"}, inplace=True)
    
    # 測定結果から主要な列のみを保持
    cols_to_keep = [
        "system_id", "task_id", "model_name", "video_name", "batch_size",
        "total_time_ms", "total_frames"
    ]
    df_measure = df_measure[[col for col in cols_to_keep if col in df_measure.columns]]
    
    # --- データの結合 ---
    print("  - Merging task measurements with model specifications...")
    df_merged = pd.merge(df_measure, df_model, on="model_name", how="left")
    
    print("  - Merging with hardware benchmark vectors...")
    df_merged = pd.merge(df_merged, df_benchmark, on="system_id", how="left")
    
    # --- CPU/GPUのカタログスペックを結合 ---
    print("  - Merging with CPU/GPU catalog specifications...")
    # system_idからキーを分割 (例: "i5-13_rtx4070" -> "i5-13", "rtx4070")
    if 'system_id' in df_merged.columns:
        df_merged[["cpu_key", "gpu_key"]] = df_merged["system_id"].str.split("__", expand=True, n=1)
        df_merged = pd.merge(df_merged, df_cpu, on="cpu_key", how="left")
        df_merged = pd.merge(df_merged, df_gpu, on="gpu_key", how="left")
    
    # 欠損値のチェック
    if df_merged.isnull().any().any():
        print("\n[WARNING] Found missing values after merging. Review your raw data sources.")
        print("Columns with NaN values:")
        print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])
        
    return df_merged


def generate_ctfusion_features(df: pd.DataFrame):
    """CTFusionで提案された特徴量（仮想ベンチマーク、相互作用特徴量）を生成する。"""
    print("\n[3/4] Generating CTFusion-specific features...")
    df_featured = df.copy()
    epsilon = 1e-9 # ゼロ除算を避けるための微小値

    # --- カテゴリ1: Virtual Benchmark Features (仮想合成ベンチマーク) ---
    print("  - Generating virtual benchmark features...")
    # CNNブロックのレイテンシとスループット
    df_featured["feat_device_virtual_cnn_latency_fp32"] = (
        2 * df_featured["gpu_compute_Conv_Standard_k3s1_fp32_latency"] +
        2 * df_featured["gpu_compute_BatchNorm_Typical_fp32_latency"]
    )
    df_featured["feat_device_virtual_cnn_throughput_fp32"] = 1000 / (
        df_featured["feat_device_virtual_cnn_latency_fp32"] + epsilon
    )

    # Transformerブロックのレイテンシとスループット
    df_featured["feat_device_virtual_transformer_latency_fp32"] = (
        df_featured["gpu_compute_LayerNorm_ViT_fp32_latency"] +
        df_featured["gpu_compute_MatMul_Attention_QK_fp32_latency"] +
        df_featured["gpu_compute_Softmax_Attention_fp32_latency"] +
        df_featured["gpu_compute_MatMul_Batched_512_fp32_latency"]
    )
    df_featured["feat_device_virtual_transformer_throughput_fp32"] = 1000 / (
        df_featured["feat_device_virtual_transformer_latency_fp32"] + epsilon
    )
    
    # --- カテゴリ2: Interaction Features (相互作用特徴量) ---
    print("  - Generating interaction and bottleneck features...")
    # Rooflineモデル（静的）: 演算強度
    df_featured["feat_interact_roofline_static"] = df_featured["model_flops_g"] / (df_featured["gpu_mem_bw_gbps"] + epsilon)
    # L2キャッシュへの負荷（静的）
    df_featured["feat_interact_cache_pressure_static"] = df_featured["model_memory_mb"] / (df_featured["l2_cache_mb"] + epsilon)

    # PCIeバスへの負荷（動的）
    bytes_per_batch = df_featured["im_height"] * df_featured["im_width"] * df_featured["num_channels"] * df_featured["batch_size"]
    df_featured["feat_interact_pcie_demand_dynamic"] = (bytes_per_batch / 1024**3) / (
        df_featured["data_transfer_H2D_PCIe_peak_bandwidth_gbps"] + epsilon
    )
    
    # パイプラインのバランス（動的）
    # モデルアーキテクチャの重みを計算
    total_ops = df_featured["num_conv2d"] + df_featured["num_attention"] + epsilon
    df_featured["arch_weight_cnn"] = df_featured["num_conv2d"] / total_ops
    df_featured["arch_weight_transformer"] = df_featured["num_attention"] / total_ops
    
    # GPUの動的スループット推定値
    df_featured["device_dynamic_throughput_estimate"] = (
        df_featured["arch_weight_cnn"] * df_featured["feat_device_virtual_cnn_throughput_fp32"] +
        df_featured["arch_weight_transformer"] * df_featured["feat_device_virtual_transformer_throughput_fp32"]
    )
    # Host/GPUバランス
    df_featured["feat_interact_pipeline_balance_dynamic"] = df_featured[
        "host_compute_Image_Preprocessing_Rate_peak_host_throughput"
    ] / (df_featured["device_dynamic_throughput_estimate"] + epsilon)

    return df_featured


def main():
    """
    データ生成パイプラインのメイン関数。
    データの読み込み、結合、特徴量生成、保存までを一気通貫で実行する。
    """
    # 1. データの読み込み
    data_frames = load_data_sources()

    # 2. ベースとなるデータセットのマージ
    df_merged = merge_base_datasets(data_frames)

    # 3. 論文で提案した特徴量の生成
    df_final = generate_ctfusion_features(df_merged)

    # 4. 最終データセットの保存
    print("\n[4/4] Saving final processed dataset...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"\n[SUCCESS] Final dataset ({df_final.shape[0]} rows, {df_final.shape[1]} columns) saved to:\n{FINAL_DATASET_PATH}")


if __name__ == "__main__":
    main()