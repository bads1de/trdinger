import pandas as pd


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    OHLCVデータの検証を行う

    Args:
        df: OHLCVデータを含むDataFrame

    Returns:
        bool: 検証成功時True

    Raises:
        ValueError: 検証失敗時
    """
    required_columns = ["open", "high", "low", "close", "volume"]

    # 必須カラムの存在確認
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 数値型確認
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} is not numeric")

    # OHLCの関係確認 (low <= open/close <= high)
    if not (
        (df["low"] <= df["open"]).all()
        and (df["open"] <= df["high"]).all()
        and (df["low"] <= df["close"]).all()
        and (df["close"] <= df["high"]).all()
    ):
        raise ValueError("OHLC values do not satisfy low <= open/close <= high")

    # ボリュームは非負
    if (df["volume"] < 0).any():
        raise ValueError("Volume contains negative values")

    return True


def validate_extended_data(df: pd.DataFrame) -> bool:
    """
    拡張データの検証を行う

    Args:
        df: 拡張データを含むDataFrame

    Returns:
        bool: 検証成功時True

    Raises:
        ValueError: 検証失敗時
    """
    # 拡張データの必須カラム (例: fear_greed, funding_rate)
    optional_columns = ["fear_greed", "funding_rate", "open_interest"]

    # 存在するカラムのみ検証
    present_columns = [col for col in optional_columns if col in df.columns]

    if not present_columns:
        # 拡張データがない場合はOK
        return True

    # 数値型確認
    for col in present_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} is not numeric")

    # fear_greed の範囲確認 (通常0-100)
    if "fear_greed" in df.columns:
        if not ((df["fear_greed"] >= 0) & (df["fear_greed"] <= 100)).all():
            raise ValueError("fear_greed values must be between 0 and 100")

    # funding_rate の範囲確認 (通常-1から1程度)
    if "funding_rate" in df.columns:
        invalid_mask = ~((df["funding_rate"] >= -1) & (df["funding_rate"] <= 1))
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            invalid_values = df.loc[invalid_mask, "funding_rate"].head(5).tolist()
            raise ValueError(
                f"funding_rate values must be between -1 and 1. Found {invalid_count} invalid values: {invalid_values}"
            )

    return True


def validate_data_integrity(df: pd.DataFrame) -> bool:
    """
    データ整合性の検証を行う

    Args:
        df: データ整合性を検証するDataFrame

    Returns:
        bool: 検証成功時True

    Raises:
        ValueError: 検証失敗時
    """
    # タイムスタンプカラムの存在確認
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required for integrity check")

    # タイムスタンプの型確認
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("timestamp column must be datetime type")

    # タイムスタンプのソート確認
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be sorted in ascending order")

    # 重複タイムスタンプの確認
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps found")

    # NaN/Null値の確認
    if df.isnull().any().any():
        raise ValueError("Data contains NaN or null values")

    return True
