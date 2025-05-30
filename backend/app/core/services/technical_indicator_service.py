"""
テクニカル指標サービス

pandasの標準機能を使用してテクニカル指標を計算し、
データベースに保存する機能を提供します。
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Union
import logging

from database.connection import SessionLocal
from database.repositories.technical_indicator_repository import (
    TechnicalIndicatorRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標計算サービス"""

    def __init__(self):
        """サービスを初期化"""
        self.supported_indicators = {
            "SMA": {
                "periods": [5, 10, 20, 50, 100, 200],
                "function": self._calculate_sma,
            },
            "EMA": {
                "periods": [5, 10, 20, 50, 100, 200],
                "function": self._calculate_ema,
            },
            "RSI": {"periods": [14, 21, 30], "function": self._calculate_rsi},
        }

    def _validate_parameters(
        self, symbol: str, timeframe: str, indicator_type: str, period: int
    ):
        """
        パラメータの検証

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicator_type: 指標タイプ
            period: 期間

        Raises:
            ValueError: パラメータが無効な場合
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("シンボルは有効な文字列である必要があります")

        if not timeframe or not isinstance(timeframe, str):
            raise ValueError("時間枠は有効な文字列である必要があります")

        if indicator_type not in self.supported_indicators:
            raise ValueError(
                f"サポートされていない指標タイプです: {indicator_type}. "
                f"サポート対象: {list(self.supported_indicators.keys())}"
            )

        if period not in self.supported_indicators[indicator_type]["periods"]:
            raise ValueError(
                f"{indicator_type}でサポートされていない期間です: {period}. "
                f"サポート対象: {self.supported_indicators[indicator_type]['periods']}"
            )

    async def _get_ohlcv_data(
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
        """
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

    def _calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        単純移動平均（SMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            SMA値のSeries
        """
        return df["close"].rolling(window=period, min_periods=period).mean()

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        指数移動平均（EMA）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            EMA値のSeries
        """
        return df["close"].ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        相対力指数（RSI）を計算

        Args:
            df: OHLCVデータのDataFrame
            period: 期間

        Returns:
            RSI値のSeries
        """
        close = df["close"]
        delta = close.diff()

        # 上昇と下降を分離
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 平均上昇と平均下降を計算
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # RSを計算
        rs = avg_gain / avg_loss

        # RSIを計算
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def calculate_technical_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        period: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        テクニカル指標を計算

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicator_type: 指標タイプ（SMA, EMA, RSI）
            period: 期間
            limit: OHLCVデータの取得件数制限

        Returns:
            計算されたテクニカル指標データのリスト

        Raises:
            ValueError: パラメータが無効な場合
            Exception: 計算エラーの場合
        """
        # パラメータの検証
        self._validate_parameters(symbol, timeframe, indicator_type, period)

        try:
            logger.info(
                f"テクニカル指標計算開始: {symbol} {timeframe} {indicator_type}({period})"
            )

            # OHLCVデータを取得
            df = await self._get_ohlcv_data(symbol, timeframe, limit)

            # 指標を計算
            calculate_func = self.supported_indicators[indicator_type]["function"]
            indicator_series = calculate_func(df, period)

            # 結果をリストに変換
            results = []
            for timestamp, value in indicator_series.items():
                if pd.notna(value):  # NaN値をスキップ
                    results.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "indicator_type": indicator_type,
                            "period": period,
                            "value": float(value),
                            "signal_value": None,
                            "histogram_value": None,
                            "timestamp": timestamp,
                        }
                    )

            logger.info(
                f"テクニカル指標計算完了: {len(results)}件 "
                f"({symbol} {timeframe} {indicator_type}({period}))"
            )
            return results

        except Exception as e:
            logger.error(f"テクニカル指標計算エラー: {e}")
            raise

    async def _save_technical_indicator_to_database(
        self,
        technical_indicator_data: List[Dict[str, Any]],
        repository: TechnicalIndicatorRepository,
    ) -> int:
        """
        テクニカル指標データをデータベースに保存

        Args:
            technical_indicator_data: テクニカル指標データのリスト
            repository: テクニカル指標リポジトリ

        Returns:
            保存された件数
        """
        try:
            if not technical_indicator_data:
                logger.warning("保存するテクニカル指標データがありません")
                return 0

            # データベースに保存
            saved_count = repository.insert_technical_indicator_data(
                technical_indicator_data
            )

            logger.info(f"テクニカル指標データ保存完了: {saved_count}件")
            return saved_count

        except Exception as e:
            logger.error(f"テクニカル指標データ保存エラー: {e}")
            raise

    async def calculate_and_save_technical_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        period: int,
        repository: Optional[TechnicalIndicatorRepository] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        テクニカル指標を計算してデータベースに保存

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicator_type: 指標タイプ
            period: 期間
            repository: テクニカル指標リポジトリ（テスト用）
            limit: OHLCVデータの取得件数制限

        Returns:
            計算・保存結果

        Raises:
            ValueError: パラメータが無効な場合
            Exception: データベースエラーの場合
        """
        try:
            # テクニカル指標を計算
            technical_indicator_data = await self.calculate_technical_indicator(
                symbol, timeframe, indicator_type, period, limit
            )

            # データベースに保存
            if repository is None:
                # 実際のデータベースセッションを使用
                db = SessionLocal()
                try:
                    repository = TechnicalIndicatorRepository(db)
                    saved_count = await self._save_technical_indicator_to_database(
                        technical_indicator_data, repository
                    )
                    db.close()
                except Exception as e:
                    db.close()
                    raise
            else:
                # テスト用のリポジトリを使用
                saved_count = await self._save_technical_indicator_to_database(
                    technical_indicator_data, repository
                )

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_type": indicator_type,
                "period": period,
                "calculated_count": len(technical_indicator_data),
                "saved_count": saved_count,
                "success": True,
            }

        except Exception as e:
            logger.error(f"テクニカル指標計算・保存エラー: {e}")
            raise

    async def calculate_and_save_multiple_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        repository: Optional[TechnicalIndicatorRepository] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        複数のテクニカル指標を一括計算・保存

        Args:
            symbol: 取引ペア
            timeframe: 時間枠
            indicators: 指標設定のリスト [{"type": "SMA", "period": 20}, ...]
            repository: テクニカル指標リポジトリ（テスト用）
            limit: OHLCVデータの取得件数制限

        Returns:
            一括計算・保存結果
        """
        try:
            logger.info(
                f"複数テクニカル指標一括計算開始: {symbol} {timeframe} "
                f"({len(indicators)}種類)"
            )

            results = []
            total_calculated = 0
            total_saved = 0
            successful_indicators = 0
            failed_indicators = []

            for indicator_config in indicators:
                try:
                    indicator_type = indicator_config["type"]
                    period = indicator_config["period"]

                    result = await self.calculate_and_save_technical_indicator(
                        symbol=symbol,
                        timeframe=timeframe,
                        indicator_type=indicator_type,
                        period=period,
                        repository=repository,
                        limit=limit,
                    )

                    results.append(result)
                    total_calculated += result["calculated_count"]
                    total_saved += result["saved_count"]
                    successful_indicators += 1

                    logger.info(
                        f"✅ {indicator_type}({period}): "
                        f"{result['calculated_count']}件計算, {result['saved_count']}件保存"
                    )

                    # レート制限対応
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"❌ {indicator_config} 計算エラー: {e}")
                    failed_indicators.append(
                        {"indicator": indicator_config, "error": str(e)}
                    )

            logger.info(
                f"複数テクニカル指標一括計算完了: "
                f"{successful_indicators}/{len(indicators)}成功, "
                f"計算{total_calculated}件, 保存{total_saved}件"
            )

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_indicators": len(indicators),
                "successful_indicators": successful_indicators,
                "failed_indicators": len(failed_indicators),
                "total_calculated": total_calculated,
                "total_saved": total_saved,
                "results": results,
                "failures": failed_indicators,
                "success": successful_indicators > 0,
            }

        except Exception as e:
            logger.error(f"複数テクニカル指標一括計算エラー: {e}")
            raise

    def get_default_indicators(self) -> List[Dict[str, Any]]:
        """
        デフォルトの指標設定を取得

        Returns:
            デフォルト指標設定のリスト
        """
        default_indicators = []

        # SMA: 20, 50期間
        for period in [20, 50]:
            default_indicators.append({"type": "SMA", "period": period})

        # EMA: 20, 50期間
        for period in [20, 50]:
            default_indicators.append({"type": "EMA", "period": period})

        # RSI: 14期間
        default_indicators.append({"type": "RSI", "period": 14})

        return default_indicators

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        Returns:
            サポート指標の情報
        """
        return {
            indicator_type: {
                "periods": config["periods"],
                "description": self._get_indicator_description(indicator_type),
            }
            for indicator_type, config in self.supported_indicators.items()
        }

    def _get_indicator_description(self, indicator_type: str) -> str:
        """
        指標の説明を取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            指標の説明
        """
        descriptions = {
            "SMA": "単純移動平均 - 指定期間の終値の平均値",
            "EMA": "指数移動平均 - 直近の価格により重みを置いた移動平均",
            "RSI": "相対力指数 - 買われすぎ・売られすぎを示すオシレーター（0-100）",
        }
        return descriptions.get(indicator_type, "説明なし")
