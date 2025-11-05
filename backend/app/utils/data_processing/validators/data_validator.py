import pandas as pd
from typing import Sequence


def _validate_empty_data(df: pd.DataFrame) -> bool:
    """空データチェックの共通ロジック"""
    return len(df) == 0 or df.empty


def _validate_columns_exist(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """必須カラム存在確認"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """数値型カラム確認"""
    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} is not numeric")


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """OHLCVデータの検証を行う
    
    Args:
        df: OHLCVデータを含むDataFrame
        
    Returns:
        bool: 検証成功時True
        
    Raises:
        ValueError: 検証失敗時
    """
    required_columns = ["open", "high", "low", "close", "volume"]
    
    # 空データチェック
    if _validate_empty_data(df):
        return True
        
    # カラム存在確認と数値型確認
    _validate_columns_exist(df, required_columns)
    _validate_numeric_columns(df, required_columns)
        
    # OHLC関係の検証
    ohlc_data = df[['low', 'open', 'high', 'close']].dropna()
    if len(ohlc_data) > 0 and not (
        (ohlc_data["low"] <= ohlc_data["open"]).all()
        and (ohlc_data["open"] <= ohlc_data["high"]).all()
        and (ohlc_data["low"] <= ohlc_data["close"]).all()
        and (ohlc_data["close"] <= ohlc_data["high"]).all()
    ):
        raise ValueError("OHLC values do not satisfy low <= open/close <= high")
        
    # ボリュームの非負確認
    if (df["volume"] < 0).any():
        raise ValueError("Volume contains negative values")

    return True


def validate_extended_data(df: pd.DataFrame) -> bool:
    """拡張データの検証を行う
    
    Args:
        df: 拡張データを含むDataFrame
        
    Returns:
        bool: 検証成功時True
        
    Raises:
        ValueError: 検証失敗時
    """
    optional_columns = ["funding_rate", "open_interest"]
    
    # 空データチェック
    if _validate_empty_data(df):
        return True
        
    # 存在するカラムのみ数値型確認
    present_columns = [col for col in optional_columns if col in df.columns]
    if present_columns:
        _validate_numeric_columns(df, present_columns)

    # funding_rateの範囲確認
    if "funding_rate" in df.columns:
        funding_rates = df["funding_rate"].dropna()
        if len(funding_rates) > 0:
            invalid_mask = ~((funding_rates >= -1) & (funding_rates <= 1))
            if invalid_mask.any():
                invalid_values = funding_rates[invalid_mask].head(5).tolist()
                raise ValueError(
                    f"funding_rate values must be between -1 and 1. "
                    f"Found {invalid_mask.sum()} invalid values: {invalid_values}"
                )

    return True


def validate_data_integrity(df: pd.DataFrame) -> bool:
    """データ整合性の検証を行う
    
    Args:
        df: データ整合性を検証するDataFrame
        
    Returns:
        bool: 検証成功時True
        
    Raises:
        ValueError: 検証失敗時
    """
    # 空データチェック
    if _validate_empty_data(df):
        return True
        
    # timestamp列がある場合のみ詳細検証を実施
    if "timestamp" in df.columns:
        # タイムスタンプの型確認
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError("timestamp column must be datetime type")

        # タイムスタンプのソート確認
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("timestamp must be sorted in ascending order")

        # 重複タイムスタンプの確認
        if df["timestamp"].duplicated().any():
            raise ValueError("duplicate timestamps found")

    # NaN/Null値は補間で処理されるため警告のみ
    # 補間処理されるため無視（チェックのみ実行）
    has_nulls = df.isnull().values.any()  # Seriesではなくbool値を直接取得
    if has_nulls:
        pass  # 補間処理されるため無視

    return True