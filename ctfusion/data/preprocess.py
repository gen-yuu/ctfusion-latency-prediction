import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """
    特徴量をStandardScalerで標準化する。
    学習データでfitし、学習データと検証データの両方をtransformする。

    Args:
        train_df (pd.DataFrame): 学習データの特徴量
        val_df (pd.DataFrame): 検証データの特徴量

    Returns:
        tuple: (変換後の学習データ, 変換後の検証データ, 学習済みスケーラー)
    """
    features = train_df.columns
    scaler = StandardScaler()

    # .fit_transform() はNumpy配列を返すため、DataFrameに戻す
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), columns=features, index=train_df.index
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df), columns=features, index=val_df.index
    )

    return train_scaled, val_scaled, scaler
