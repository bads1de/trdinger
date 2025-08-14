"""
データ変換ユーティリティ（簡素化版）

pandas標準機能を活用し、冗長なカスタム実装を削除。
必要最小限の変換ロジックのみを提供。
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
    OHLCV列名をbacktesting.py標準形式に統一（簡素化版）

    pandas標準のrename()を活用し、シンプルなマッピング処理。
    """
    if df.empty:
        return df

    # 列名マッピング（大文字小文字を統一）
    column_mapping = {
        col: col.capitalize()
        for col in df.columns
        if col.lower()
        in ["open", "high", "low", "close", "volume", "o", "h", "l", "c", "v"]
    }

    # 短縮形の特別マッピング
    short_mapping = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    column_mapping.update(
        {
            col: short_mapping[col.lower()]
            for col in df.columns
            if col.lower() in short_mapping
        }
    )

    # 列名変更
    result_df = df.rename(columns=column_mapping)

    # 必要な列の存在確認
    required_cols = ["Open", "High", "Low", "Close"]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(f"必要な列が見つかりません: {missing_cols}")

    # Volumeがない場合はデフォルト値を設定
    if "Volume" not in result_df.columns:
        result_df["Volume"] = 1000

    return result_df

def ensure_numeric_series(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
    name: Optional[str] = None,
) -> pd.Series:
    """
    データを数値型のpandas.Seriesに変換（簡素化版）

    pandas標準のto_numeric()を直接活用。
    """
    try:
        # pandas.Series の場合はそのまま（必要なら名称を変更）
        if isinstance(data, pd.Series):
            series = data.rename(name) if name is not None else data
        # backtesting._Array の特殊ケース
        elif hasattr(data, "_data"):
            series = pd.Series(data._data, name=name)
        # その他は pandas 標準で処理
        else:
            series = pd.Series(data, name=name)

        return pd.to_numeric(series, errors="raise" if raise_on_error else "coerce")

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"数値型変換に失敗: {e}")
        else:
            logger.warning(f"数値型変換に失敗: {e}")
            return pd.Series([], dtype=float, name=name)


def ensure_array(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
) -> np.ndarray:
    """
    データをnumpy.ndarrayに変換（簡素化版）

    numpy標準のarray()コンストラクタを活用。
    """
    try:
        if isinstance(data, np.ndarray):
            return data

        # pandas.Seriesは.valuesで効率的に変換
        if isinstance(data, pd.Series):
            return data.values

        # backtesting._Arrayの特殊ケース
        if hasattr(data, "_data"):
            return np.array(data._data)

        # その他は全てnumpy標準で処理
        return np.array(data)

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"numpy.ndarray変換に失敗: {e}")
        else:
            logger.warning(f"numpy.ndarray変換に失敗: {e}")
            return np.array([])


def ensure_list(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
) -> list:
    """
    データをlistに変換（簡素化版）

    Python標準のlist()コンストラクタを活用。
    """
    try:
        if isinstance(data, list):
            return data

        # pandas/numpyは.tolist()で効率的に変換
        if hasattr(data, "tolist"):
            return data.tolist()

        # backtesting._Arrayの特殊ケース
        if hasattr(data, "_data"):
            return list(data._data)

        # その他は全てPython標準で処理
        return list(data)

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"list変換に失敗: {e}")
        else:
            logger.warning(f"list変換に失敗: {e}")
            return []


# 後方互換性のためのエイリアス
DataValidator = DataSanitizer
