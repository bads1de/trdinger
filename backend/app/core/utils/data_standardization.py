"""
データ標準化ユーティリティ

backtesting.pyライブラリとの統一を図るためのデータ形式標準化機能
"""

import pandas as pd


# backtesting.py標準のOHLCV列名
STANDARD_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


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


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    OHLCVデータの妥当性をチェック

    Args:
        df: チェック対象のデータフレーム

    Returns:
        データが有効かどうか
    """
    if df.empty:
        return False

    # 必要な列の存在チェック
    required_columns = ["Open", "High", "Low", "Close"]
    for col in required_columns:
        if col not in df.columns:
            return False

    # データ型チェック
    for col in required_columns + ["Volume"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False

    # 価格の論理チェック（High >= Low）
    # NaN値を除外してチェック
    valid_data = df.dropna()
    if len(valid_data) == 0:
        return False

    # 基本的な価格関係のチェック（High >= Low）
    if not (valid_data["High"] >= valid_data["Low"]).all():
        return False

    # 負の価格がないかチェック
    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        if col in valid_data.columns:
            if (valid_data[col] <= 0).any():
                return False

    return True


def prepare_data_for_backtesting(df: pd.DataFrame) -> pd.DataFrame:
    """
    backtesting.py用にデータを準備

    Args:
        df: 元のOHLCVデータ

    Returns:
        backtesting.py用に準備されたデータ

    Raises:
        ValueError: データが無効な場合
    """
    # 列名を標準化
    standardized_df = standardize_ohlcv_columns(df)

    # データの妥当性をチェック
    if not validate_ohlcv_data(standardized_df):
        raise ValueError("OHLCVデータが有効ではありません。")

    # インデックスがDatetimeIndexでない場合は変換
    if not isinstance(standardized_df.index, pd.DatetimeIndex):
        if "timestamp" in standardized_df.columns:
            standardized_df = standardized_df.set_index("timestamp")
        elif "date" in standardized_df.columns:
            standardized_df = standardized_df.set_index("date")
        else:
            # インデックスをDatetimeに変換を試行
            try:
                standardized_df.index = pd.to_datetime(standardized_df.index)
            except ValueError:
                raise ValueError("DataFrameに有効な日時インデックスが見つかりません。")

    # 重複インデックスを削除
    standardized_df = standardized_df[~standardized_df.index.duplicated(keep="first")]

    # ソート
    standardized_df = standardized_df.sort_index()

    # NaN値を削除
    standardized_df = standardized_df.dropna()

    return standardized_df
