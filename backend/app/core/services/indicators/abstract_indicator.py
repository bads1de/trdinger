"""
テクニカル指標の基底クラス

全てのテクニカル指標クラスの共通機能を提供します。
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, cast
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
import logging


logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """テクニカル指標の基底クラス"""

    def __init__(self, indicator_type: str, supported_periods: List[int]):
        """
        基底クラスを初期化

        Args:
            indicator_type: 指標タイプ（例: 'SMA', 'RSI'）
            supported_periods: サポートされている期間のリスト
        """
        self.indicator_type = indicator_type
        self.supported_periods = supported_periods

    @abstractmethod
    def calculate(
        self, df: pd.DataFrame, period: int, **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        テクニカル指標を計算（サブクラスで実装）

        Args:
            df: OHLCVデータのDataFrame
            period: 計算期間
            **kwargs: 追加パラメータ

        Returns:
            計算結果（Series または DataFrame）
        """

    def validate_data(self, df: pd.DataFrame, min_periods: int) -> None:
        """
        データの検証

        Args:
            df: 検証するDataFrame
            min_periods: 最小必要期間

        Raises:
            ValueError: データが不十分な場合
        """
        if df is None or df.empty:
            raise ValueError("OHLCVデータが空です")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必要なカラムが不足しています: {missing_columns}")

        if len(df) < min_periods:
            raise ValueError(
                f"データが不十分です。最低{min_periods}期間必要ですが、{len(df)}期間しかありません"
            )

    def validate_parameters(self, period: int, **kwargs) -> None:
        """
        パラメータの検証

        Args:
            period: 計算期間
            **kwargs: 追加パラメータ

        Raises:
            ValueError: パラメータが無効な場合
        """
        if period not in self.supported_periods:
            raise ValueError(
                f"{self.indicator_type}でサポートされていない期間です: {period}. "
                f"サポート対象: {self.supported_periods}"
            )

    async def get_ohlcv_data(
        self, symbol: str, timeframe: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        OHLCVデータを取得してDataFrameに変換

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            limit: 取得件数制限

        Returns:
            OHLCVデータのDataFrame

        Raises:
            ValueError: データが見つからない場合
        """
        if SessionLocal is None or OHLCVRepository is None:
            raise RuntimeError(
                "データベースコンポーネントが利用できません。SessionLocalまたはOHLCVRepositoryがロードされていません。"
            )

        try:
            db = SessionLocal()
            try:
                ohlcv_repository = OHLCVRepository(db)
                ohlcv_data = ohlcv_repository.get_ohlcv_data(
                    symbol=symbol, timeframe=timeframe, limit=limit
                )

                if not ohlcv_data:
                    raise ValueError(
                        f"OHLCVデータが見つかりません: {symbol} {timeframe}"
                    )

                # DataFrameに変換
                df_data = []
                for record in ohlcv_data:
                    df_data.append(
                        {
                            "timestamp": record.timestamp,
                            "open": record.open,
                            "high": record.high,
                            "low": record.low,
                            "close": record.close,
                            "volume": record.volume,
                        }
                    )

                df = pd.DataFrame(df_data)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

                logger.info(f"OHLCVデータ取得成功: {len(df)}件 ({symbol} {timeframe})")
                return df

            finally:
                db.close()

        except Exception as e:
            logger.error(f"OHLCVデータ取得エラー: {e}")
            raise

    def format_single_value_result(
        self, result: pd.Series, symbol: str, timeframe: str, period: int
    ) -> List[Dict[str, Any]]:
        """
        単一値の結果をフォーマット

        Args:
            result: 計算結果のSeries
            symbol: 取引ペア
            timeframe: 時間枠
            period: 期間

        Returns:
            フォーマットされた結果のリスト
        """
        formatted_results = []

        for timestamp, value in result.items():
            if pd.notna(value):  # NaN値をスキップ
                formatted_results.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "indicator_type": self.indicator_type,
                        "period": period,
                        "value": float(value.real),
                        "signal_value": None,
                        "histogram_value": None,
                        "upper_band": None,
                        "lower_band": None,
                        "timestamp": timestamp,
                    }
                )

        return formatted_results

    def format_multi_value_result(
        self,
        result: pd.DataFrame,
        symbol: str,
        timeframe: str,
        period: int,
        value_columns: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        複数値の結果をフォーマット

        Args:
            result: 計算結果のDataFrame
            symbol: 取引ペア
            timeframe: 時間枠
            period: 期間
            value_columns: カラム名のマッピング

        Returns:
            フォーマットされた結果のリスト
        """
        formatted_results = []

        for timestamp in result.index:
            # 全ての値がNaNでないかチェック
            row_values = {}
            all_valid = True

            for key, column in value_columns.items():
                if column in result.columns:
                    value = result.loc[timestamp, column]
                    if pd.notna(value):
                        numeric_value = cast(Union[float, complex], value)
                        row_values[key] = float(numeric_value.real)
                    else:
                        all_valid = False
                        break
                else:
                    row_values[key] = None

            if all_valid and row_values:
                formatted_results.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "indicator_type": self.indicator_type,
                        "period": period,
                        "value": row_values.get("value"),
                        "signal_value": row_values.get("signal_value"),
                        "histogram_value": row_values.get("histogram_value"),
                        "upper_band": row_values.get("upper_band"),
                        "lower_band": row_values.get("lower_band"),
                        "timestamp": timestamp,
                    }
                )

        return formatted_results

    async def calculate_and_format(
        self,
        symbol: str,
        timeframe: str,
        period: int,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        指標を計算してフォーマットされた結果を返す

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            period: 期間
            limit: OHLCVデータの取得件数制限
            **kwargs: 追加パラメータ

        Returns:
            フォーマットされた計算結果のリスト

        Raises:
            ValueError: パラメータが無効な場合
        """
        try:
            logger.info(
                f"テクニカル指標計算開始: {symbol} {timeframe} {self.indicator_type}({period})"
            )

            # パラメータ検証
            self.validate_parameters(period, **kwargs)

            # OHLCVデータを取得
            df = await self.get_ohlcv_data(symbol, timeframe, limit)

            # データ検証
            self.validate_data(df, period)

            # 指標を計算
            result = self.calculate(df, period, **kwargs)

            # 結果をフォーマット（サブクラスで実装される場合はオーバーライド）
            if isinstance(result, pd.Series):
                formatted_result = self.format_single_value_result(
                    result, symbol, timeframe, period
                )
            else:
                # DataFrameの場合はサブクラスでオーバーライドが必要
                raise NotImplementedError(
                    "複数値を返す指標はformat_resultメソッドをオーバーライドしてください"
                )

            logger.info(
                f"テクニカル指標計算完了: {len(formatted_result)}件 "
                f"({symbol} {timeframe} {self.indicator_type}({period}))"
            )
            return formatted_result

        except Exception as e:
            logger.error(f"テクニカル指標計算エラー: {e}")
            raise

    def get_description(self) -> str:
        """
        指標の説明を取得（サブクラスでオーバーライド可能）

        Returns:
            指標の説明
        """
        return f"{self.indicator_type} - 説明なし"
