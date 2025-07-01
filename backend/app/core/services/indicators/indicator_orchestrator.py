"""
テクニカル指標統合サービス

分割されたテクニカル指標クラスを統合し、既存APIとの互換性を維持します。
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union


from .trend_indicators import get_trend_indicator, TREND_INDICATORS_INFO
from .momentum_indicators import get_momentum_indicator, MOMENTUM_INDICATORS_INFO
from .volatility_indicators import get_volatility_indicator, VOLATILITY_INDICATORS_INFO
from .volume_indicators import get_volume_indicator, VOLUME_INDICATORS_INFO
from .price_transform_indicators import (
    get_price_transform_indicator,
    PRICE_TRANSFORM_INDICATORS_INFO,
)

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標統合サービス"""

    def __init__(self):
        """サービスを初期化"""
        # 全ての指標情報を統合
        self.supported_indicators = {}
        self.supported_indicators.update(TREND_INDICATORS_INFO)
        self.supported_indicators.update(MOMENTUM_INDICATORS_INFO)
        self.supported_indicators.update(VOLATILITY_INDICATORS_INFO)
        self.supported_indicators.update(VOLUME_INDICATORS_INFO)
        self.supported_indicators.update(PRICE_TRANSFORM_INDICATORS_INFO)

        # 指標カテゴリのマッピング
        self.indicator_categories = {
            "trend": ["SMA", "EMA", "MACD"],
            "momentum": [
                "RSI",
                "STOCH",
                "CCI",
                "WILLR",
                "MOM",
                "ROC",
                "ADX",
                "AROON",
                "MFI",
                "STOCHRSI",
                "ULTOSC",
                "CMO",
                "TRIX",
                "BOP",
                "APO",
                "PPO",
                "AROONOSC",
                "DX",
                "ADXR",
                "PLUS_DI",
                "MINUS_DI",
                "ROCP",
                "ROCR",
                "STOCHF",
            ],
            "volatility": ["BB", "ATR"],
            "volume": ["OBV", "AD", "ADOSC"],
            "price_transform": [
                "AVGPRICE",
                "MEDPRICE",
                "TYPPRICE",
                "WCLPRICE",
                "MAMA",
                "APO",
                "HT_DCPERIOD",
                "HT_DCPHASE",
                "HT_PHASOR",
                "HT_SINE",
                "HT_TRENDMODE",
                "FAMA",
                "SAREXT",
                "SAR",
            ],
        }

    def _get_indicator_instance(self, indicator_type: str):
        """
        指標タイプに応じた指標インスタンスを取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            指標インスタンス

        Raises:
            ValueError: サポートされていない指標タイプの場合
        """
        # カテゴリ別に適切なファクトリー関数を呼び出し
        if indicator_type in self.indicator_categories["trend"]:
            return get_trend_indicator(indicator_type)
        elif indicator_type in self.indicator_categories["momentum"]:
            return get_momentum_indicator(indicator_type)
        elif indicator_type in self.indicator_categories["volatility"]:
            return get_volatility_indicator(indicator_type)
        elif indicator_type in self.indicator_categories["volume"]:
            return get_volume_indicator(indicator_type)
        elif indicator_type in self.indicator_categories["price_transform"]:
            return get_price_transform_indicator(indicator_type)
        else:
            raise ValueError(
                f"サポートされていない指標タイプです: {indicator_type}. "
                f"サポート対象: {list(self.supported_indicators.keys())}"
            )

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

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, **kwargs
    ) -> Union[pd.Series, pd.DataFrame, Dict]:
        """
        同期的にテクニカル指標を計算（strategies/indicators.py用）

        Args:
            df: OHLCVデータのDataFrame
            indicator_type: 指標タイプ
            **kwargs: 指標固有のパラメータ

        Returns:
            計算結果（Series、DataFrame、または辞書）

        Raises:
            ValueError: サポートされていない指標タイプの場合
        """
        try:
            # 指標タイプの正規化
            indicator_type = indicator_type.upper()

            # 指標インスタンスの取得と計算
            if indicator_type in self.indicator_categories["trend"]:
                indicator = get_trend_indicator(indicator_type)
            elif indicator_type in self.indicator_categories["momentum"]:
                indicator = get_momentum_indicator(indicator_type)
            elif indicator_type in self.indicator_categories["volatility"]:
                indicator = get_volatility_indicator(indicator_type)
            elif indicator_type in self.indicator_categories["volume"]:
                indicator = get_volume_indicator(indicator_type)
            elif indicator_type in self.indicator_categories["price_transform"]:
                indicator = get_price_transform_indicator(indicator_type)
            else:
                raise ValueError(f"サポートされていない指標タイプ: {indicator_type}")

            # インジケーター計算を実行
            return indicator.calculate(df, **kwargs)

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            raise

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

            # 適切な指標インスタンスを取得
            indicator = self._get_indicator_instance(indicator_type)

            # 指標を計算してフォーマット
            results = await indicator.calculate_and_format(
                symbol=symbol, timeframe=timeframe, period=period, limit=limit
            )

            logger.info(
                f"テクニカル指標計算完了: {len(results)}件 "
                f"({symbol} {timeframe} {indicator_type}({period}))"
            )
            return results

        except Exception as e:
            logger.error(f"テクニカル指標計算エラー: {e}")
            raise

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        Returns:
            サポート指標の情報
        """
        return {
            indicator_type: {
                "periods": config["periods"],
                "description": config["description"],
                "category": config["category"],
            }
            for indicator_type, config in self.supported_indicators.items()
        }
