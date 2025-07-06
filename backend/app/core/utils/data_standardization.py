"""
データ標準化ユーティリティ

backtesting.pyライブラリとの統一を図るためのデータ形式標準化機能
"""

import pandas as pd


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV列名をbacktesting.py標準形式に統一

    Args:
        df: 元のOHLCVデータフレーム

    Returns:
        標準化されたOHLCVデータフレーム

    Raises:
        ValueError: 必要な列が見つからない場合
    """
    if df.empty:
        return df

    # 列名マッピング（小文字 → 大文字）
    column_mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        # 他の可能な形式も対応
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
    }

    # 現在の列名を確認
    current_columns = df.columns.tolist()

    # マッピングを適用
    rename_dict = {}
    for old_col in current_columns:
        if old_col.lower() in column_mapping:
            rename_dict[old_col] = column_mapping[old_col.lower()]

    # 列名を変更
    standardized_df = df.rename(columns=rename_dict)

    # 必要な列が存在するかチェック
    missing_columns = []
    for required_col in ["Open", "High", "Low", "Close"]:
        if required_col not in standardized_df.columns:
            missing_columns.append(required_col)

    if missing_columns:
        raise ValueError(f"OHLCVデータに必要な列が見つかりません: {missing_columns}")

    # Volumeが存在しない場合はデフォルト値を設定
    if "Volume" not in standardized_df.columns:
        standardized_df["Volume"] = 1000  # デフォルト出来高

    return standardized_df
