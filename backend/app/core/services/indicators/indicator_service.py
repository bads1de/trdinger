"""
テクニカル指標統合サービス

分割されたテクニカル指標クラスを統合し、既存APIとの互換性を維持します。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from database.connection import SessionLocal
from database.repositories.technical_indicator_repository import TechnicalIndicatorRepository

from .trend_indicators import get_trend_indicator, TREND_INDICATORS_INFO
from .momentum_indicators import get_momentum_indicator, MOMENTUM_INDICATORS_INFO
from .volatility_indicators import get_volatility_indicator, VOLATILITY_INDICATORS_INFO
from .other_indicators import get_other_indicator, OTHER_INDICATORS_INFO

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
        self.supported_indicators.update(OTHER_INDICATORS_INFO)

        # 指標カテゴリのマッピング
        self.indicator_categories = {
            "trend": ["SMA", "EMA", "MACD"],
            "momentum": ["RSI", "STOCH", "CCI", "WILLR", "MOM", "ROC"],
            "volatility": ["BB", "ATR"],
            "other": ["PSAR"]
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
        elif indicator_type in self.indicator_categories["other"]:
            return get_other_indicator(indicator_type)
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
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                limit=limit
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

        # MACD: 12期間（標準設定）
        default_indicators.append({"type": "MACD", "period": 12})

        # ボリンジャーバンド: 20期間
        default_indicators.append({"type": "BB", "period": 20})

        # ATR: 14期間
        default_indicators.append({"type": "ATR", "period": 14})

        # ストキャスティクス: 14期間
        default_indicators.append({"type": "STOCH", "period": 14})

        # CCI: 20期間
        default_indicators.append({"type": "CCI", "period": 20})

        # Williams %R: 14期間
        default_indicators.append({"type": "WILLR", "period": 14})

        # モメンタム: 10期間
        default_indicators.append({"type": "MOM", "period": 10})

        # ROC: 10期間
        default_indicators.append({"type": "ROC", "period": 10})

        # PSAR: 1期間（固定）
        default_indicators.append({"type": "PSAR", "period": 1})

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
                "description": config["description"],
                "category": config["category"],
            }
            for indicator_type, config in self.supported_indicators.items()
        }
