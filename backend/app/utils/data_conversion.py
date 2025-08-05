"""
データ変換ユーティリティ

data_converter.py、data_utils.py、data_standardization.py を統合したモジュール。
データ形式変換、型変換、標準化のロジックを統一的に提供します。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataConversionError(Exception):
    """データ変換エラー"""


class OHLCVDataConverter:
    """OHLCV データ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        ohlcv_data: List[List], symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のOHLCVデータをデータベース形式に変換

        Args:
            ohlcv_data: CCXT形式のOHLCVデータ
            symbol: シンボル
            timeframe: 時間軸

        Returns:
            データベース挿入用の辞書リスト
        """
        db_records = []

        for candle in ohlcv_data:
            timestamp_ms, open_price, high, low, close, volume = candle

            # ミリ秒タイムスタンプをdatetimeに変換
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

            db_record = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }

            db_records.append(db_record)

        return db_records

    @staticmethod
    def db_to_api_format(ohlcv_records: List[Any]) -> List[List]:
        """
        データベース形式のOHLCVデータをAPI形式に変換

        Args:
            ohlcv_records: データベースのOHLCVレコード

        Returns:
            API形式のOHLCVデータ
        """
        api_data = []

        for record in ohlcv_records:
            api_data.append(
                [
                    int(
                        record.timestamp.timestamp() * 1000
                    ),  # タイムスタンプ（ミリ秒）
                    record.open,
                    record.high,
                    record.low,
                    record.close,
                    record.volume,
                ]
            )

        return api_data


class FundingRateDataConverter:
    """ファンディングレートデータ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        funding_rate_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のファンディングレートデータをデータベース形式に変換

        Args:
            funding_rate_data: CCXT形式のファンディングレートデータ
            symbol: シンボル

        Returns:
            データベース挿入用の辞書リスト
        """
        db_records = []

        for rate_data in funding_rate_data:
            # データタイムスタンプの処理
            data_timestamp = rate_data.get("datetime")
            if data_timestamp:
                if isinstance(data_timestamp, str):
                    data_timestamp = datetime.fromisoformat(
                        data_timestamp.replace("Z", "+00:00")
                    )
                elif isinstance(data_timestamp, (int, float)):
                    data_timestamp = datetime.fromtimestamp(
                        data_timestamp / 1000, tz=timezone.utc
                    )

            db_record = {
                "symbol": symbol,
                "funding_rate": float(rate_data.get("fundingRate", 0.0)),
                "data_timestamp": data_timestamp,
                "timestamp": datetime.now(timezone.utc),
            }

            # 次回ファンディング時刻の処理
            next_funding = rate_data.get("nextFundingDatetime")
            if next_funding:
                if isinstance(next_funding, str):
                    db_record["next_funding_timestamp"] = datetime.fromisoformat(
                        next_funding.replace("Z", "+00:00")
                    )
                elif isinstance(next_funding, (int, float)):
                    db_record["next_funding_timestamp"] = datetime.fromtimestamp(
                        next_funding / 1000, tz=timezone.utc
                    )

            db_records.append(db_record)

        return db_records


class OpenInterestDataConverter:
    """オープンインタレストデータ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        open_interest_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のオープンインタレストデータをデータベース形式に変換

        Args:
            open_interest_data: CCXT形式のオープンインタレストデータ
            symbol: シンボル

        Returns:
            データベース挿入用の辞書リスト
        """
        db_records = []

        for oi_data in open_interest_data:
            # データタイムスタンプの処理
            data_timestamp = oi_data.get("datetime")
            if data_timestamp:
                if isinstance(data_timestamp, str):
                    data_timestamp = datetime.fromisoformat(
                        data_timestamp.replace("Z", "+00:00")
                    )
                elif isinstance(data_timestamp, (int, float)):
                    data_timestamp = datetime.fromtimestamp(
                        data_timestamp / 1000, tz=timezone.utc
                    )

            # オープンインタレスト値の取得
            open_interest_value = oi_data.get("openInterestAmount") or oi_data.get(
                "openInterest"
            )

            # 値が取得できない場合はスキップ
            if open_interest_value is None:
                logger.warning(
                    f"オープンインタレスト値が取得できませんでした: {oi_data}"
                )
                continue

            logger.info(
                f"オープンインタレストデータを変換中: {oi_data} -> value={open_interest_value}"
            )

            db_record = {
                "symbol": symbol,
                "open_interest_value": float(open_interest_value),
                "data_timestamp": data_timestamp,
                "timestamp": datetime.now(timezone.utc),
            }

            db_records.append(db_record)

        return db_records


class DataSanitizer:
    """データ検証・サニタイズの共通ヘルパークラス（旧DataValidator）"""

    @staticmethod
    def sanitize_ohlcv_data(
        ohlcv_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        OHLCVデータをサニタイズ

        Args:
            ohlcv_records: サニタイズするOHLCVデータのリスト

        Returns:
            サニタイズされたOHLCVデータのリスト
        """
        sanitized_records = []

        try:
            for record in ohlcv_records:
                sanitized_record = {}

                # シンボルの正規化
                sanitized_record["symbol"] = str(record["symbol"]).strip().upper()

                # 時間軸の正規化
                sanitized_record["timeframe"] = str(record["timeframe"]).strip().lower()

                # タイムスタンプの変換
                timestamp = record["timestamp"]
                if isinstance(timestamp, str):
                    sanitized_record["timestamp"] = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                elif isinstance(timestamp, datetime):
                    sanitized_record["timestamp"] = timestamp
                else:
                    sanitized_record["timestamp"] = datetime.fromtimestamp(
                        float(timestamp), tz=timezone.utc
                    )

                # 数値データの変換
                for field in ["open", "high", "low", "close", "volume"]:
                    sanitized_record[field] = float(record[field])

                sanitized_records.append(sanitized_record)

            return sanitized_records

        except Exception as e:
            logger.error(f"OHLCVデータのサニタイズエラー: {e}")
            raise DataConversionError(f"OHLCVデータのサニタイズに失敗しました: {e}")

    @staticmethod
    def validate_ohlcv_record(record: Dict[str, Any]) -> bool:
        """
        単一のOHLCVレコードを検証

        Args:
            record: 検証するOHLCVレコード

        Returns:
            検証結果（True: 有効, False: 無効）
        """
        try:
            # 必須フィールドの存在確認
            required_fields = [
                "symbol",
                "timeframe",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            for field in required_fields:
                if field not in record:
                    logger.warning(f"必須フィールド '{field}' が見つかりません")
                    return False

            # 数値フィールドの検証
            numeric_fields = ["open", "high", "low", "close", "volume"]
            for field in numeric_fields:
                try:
                    float(record[field])
                except (ValueError, TypeError):
                    logger.warning(
                        f"数値フィールド '{field}' が無効です: {record[field]}"
                    )
                    return False

            # 価格の論理的整合性チェック
            high = float(record["high"])
            low = float(record["low"])
            open_price = float(record["open"])
            close = float(record["close"])

            if high < low:
                logger.warning(f"High ({high}) < Low ({low})")
                return False

            if high < open_price or high < close:
                logger.warning("High価格が Open/Close より低い")
                return False

            if low > open_price or low > close:
                logger.warning("Low価格が Open/Close より高い")
                return False

            return True

        except Exception as e:
            logger.error(f"OHLCVレコード検証エラー: {e}")
            return False


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
        standardized_df["Volume"] = 1000

    return standardized_df


def ensure_series(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
    name: Optional[str] = None,
) -> pd.Series:
    """
    データをpandas.Seriesに変換

    Args:
        data: 入力データ（pandas.Series, list, numpy.ndarray, backtesting._Array等）
        raise_on_error: エラー時に例外を発生させるかどうか
        name: 作成するSeriesの名前

    Returns:
        pandas.Series

    Raises:
        DataConversionError: サポートされていないデータ型の場合（raise_on_error=Trueの時）
    """
    try:
        # 既にpandas.Seriesの場合
        if isinstance(data, pd.Series):
            if name is not None and name != data.name:
                result = data.copy()
                result.name = name
                return result
            return data

        # list, numpy.ndarray（valuesより先にチェック）
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data, name=name)

        # backtesting.pyの_Arrayオブジェクト（_data属性を持つ）
        if hasattr(data, "_data"):
            return pd.Series(data._data, name=name)

        # valuesアトリビュートを持つオブジェクト（pandas.DataFrame等、ただし辞書は除外）
        if hasattr(data, "values") and not isinstance(data, dict):
            return pd.Series(data.values, name=name)

        # その他のデータ型（スカラー値等）
        if np.isscalar(data):
            return pd.Series([data], name=name)

        # サポートされていないデータ型
        if raise_on_error:
            raise DataConversionError(f"サポートされていないデータ型です: {type(data)}")
        else:
            logger.warning(f"サポートされていないデータ型: {type(data)}")
            return pd.Series([], name=name)

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"pandas.Seriesへの変換に失敗しました: {e}")
        else:
            logger.warning(f"pandas.Seriesへの変換に失敗: {e}")
            return pd.Series([], name=name)


def ensure_numeric_series(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
    name: Optional[str] = None,
) -> pd.Series:
    """
    データを数値型のpandas.Seriesに変換

    Args:
        data: 入力データ
        raise_on_error: エラー時に例外を発生させるかどうか
        name: 作成するSeriesの名前

    Returns:
        数値型のpandas.Series

    Raises:
        DataConversionError: 変換に失敗した場合（raise_on_error=Trueの時）
    """
    try:
        series = ensure_series(data, raise_on_error=raise_on_error, name=name)

        # 数値型に変換
        numeric_series = pd.to_numeric(
            series, errors="coerce" if not raise_on_error else "raise"
        )

        if raise_on_error and numeric_series.isna().any():
            raise DataConversionError("数値に変換できない値が含まれています")

        return numeric_series

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"数値型pandas.Seriesへの変換に失敗しました: {e}")
        else:
            logger.warning(f"数値型pandas.Seriesへの変換に失敗: {e}")
            return pd.Series([], dtype=float, name=name)


def ensure_array(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
) -> np.ndarray:
    """
    データをnumpy.ndarrayに変換

    Args:
        data: 入力データ
        raise_on_error: エラー時に例外を発生させるかどうか

    Returns:
        numpy.ndarray

    Raises:
        DataConversionError: 変換に失敗した場合（raise_on_error=Trueの時）
    """
    try:
        # 既にnumpy.ndarrayの場合
        if isinstance(data, np.ndarray):
            return data

        # pandas.Seriesの場合
        if isinstance(data, pd.Series):
            return data.values

        # listの場合
        if isinstance(data, list):
            return np.array(data)

        # backtesting.pyの_Arrayオブジェクト（_data属性を持つ）
        if hasattr(data, "_data"):
            return np.array(data._data)

        # valuesアトリビュートを持つオブジェクト
        if hasattr(data, "values"):
            return np.array(data.values)

        # その他のデータ型
        return np.array(data)

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"numpy.ndarrayへの変換に失敗しました: {e}")
        else:
            logger.warning(f"numpy.ndarrayへの変換に失敗: {e}")
            return np.array([])


def ensure_list(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
) -> list:
    """
    データをlistに変換

    Args:
        data: 入力データ
        raise_on_error: エラー時に例外を発生させるかどうか

    Returns:
        list

    Raises:
        DataConversionError: 変換に失敗した場合（raise_on_error=Trueの時）
    """
    try:
        # 既にlistの場合
        if isinstance(data, list):
            return data

        # pandas.Seriesの場合
        if isinstance(data, pd.Series):
            return data.tolist()

        # numpy.ndarrayの場合
        if isinstance(data, np.ndarray):
            return data.tolist()

        # backtesting.pyの_Arrayオブジェクト（_data属性を持つ）
        if hasattr(data, "_data"):
            return list(data._data)

        # valuesアトリビュートを持つオブジェクト
        if hasattr(data, "values"):
            return list(data.values)

        # その他のデータ型
        return list(data)

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"listへの変換に失敗しました: {e}")
        else:
            logger.warning(f"listへの変換に失敗: {e}")
            return []


# 後方互換性のためのエイリアス
DataValidator = DataSanitizer
