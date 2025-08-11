"""
ポジションサイジングヘルパー

戦略ファクトリーで使用するポジションサイズ計算のヘルパー機能を提供します。
"""

import logging
from typing import Any, Dict, List

import numpy as np
from app.services.indicators.technical_indicators.volatility import (
    VolatilityIndicators,
)

from ..models.gene_strategy import StrategyGene
from .position_sizing_calculator import PositionSizingCalculatorService

logger = logging.getLogger(__name__)


class PositionSizingHelper:
    """
    ポジションサイジングヘルパー

    戦略ファクトリーで使用するポジションサイズ計算のヘルパー機能を提供します。
    """

    def calculate_position_size(
        self, gene: StrategyGene, account_balance: float, current_price: float, data
    ) -> float:
        """
        ポジションサイズを計算

        Args:
            gene: 戦略遺伝子
            account_balance: 口座残高
            current_price: 現在価格
            data: 市場データ

        Returns:
            計算されたポジションサイズ
        """
        try:
            # ポジションサイジング遺伝子が存在するかチェック
            position_sizing_gene = getattr(gene, "position_sizing_gene", None)

            if not position_sizing_gene or not position_sizing_gene.enabled:
                # ポジションサイジング遺伝子が無効な場合は従来のrisk_managementを使用
                position_size = gene.risk_management.get("position_size", 0.1)
                return max(0.01, min(1.0, position_size))

            # ポジションサイジング計算サービスを使用
            calculator = PositionSizingCalculatorService()

            # 市場データの準備
            market_data = self.prepare_market_data_for_position_sizing(
                data, current_price
            )

            # 取引履歴の準備（簡易版）
            trade_history = self.prepare_trade_history_for_position_sizing()

            # ポジションサイズを計算
            result = calculator.calculate_position_size(
                gene=position_sizing_gene,
                account_balance=account_balance,
                current_price=current_price,
                symbol="BTCUSDT",  # デフォルト
                market_data=market_data,
                trade_history=trade_history,
                use_cache=False,  # バックテスト中はキャッシュを使用しない
            )

            # 計算結果を返す
            return result.position_size

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            # エラー時は従来のrisk_managementにフォールバック
            position_size = gene.risk_management.get("position_size", 0.1)
            # 適切な範囲に制限（上限を大幅に拡大）
            return max(0.01, min(50.0, position_size))

    def prepare_market_data_for_position_sizing(
        self, data, current_price: float
    ) -> Dict[str, Any]:
        """ポジションサイジング用の市場データを準備（改善版）"""
        try:
            market_data = {}

            # ATR計算の改善
            if (
                data is not None
                and hasattr(data, "High")
                and hasattr(data, "Low")
                and hasattr(data, "Close")
            ):
                try:
                    # 実際のATR計算を試行
                    atr_value = self._calculate_atr_from_data(data, period=14)
                    if atr_value > 0:
                        market_data["atr"] = atr_value
                        market_data["atr_source"] = "calculated"
                        market_data["atr_pct"] = (
                            atr_value / current_price if current_price > 0 else 0.04
                        )
                except Exception as e:
                    logger.warning(f"ATR計算失敗: {e}")

            # ATRが計算できない場合の代替指標
            if "atr" not in market_data:
                # 価格ベースの簡易ボラティリティ推定
                estimated_atr = current_price * 0.04  # 4%を仮定
                market_data["atr"] = estimated_atr
                market_data["atr_source"] = "estimated"
                market_data["atr_pct"] = 0.04

            return market_data

        except Exception as e:
            logger.error(f"市場データ準備エラー: {e}")
            return {
                "atr": current_price * 0.02,
                "atr_pct": 0.02,
                "atr_source": "error_fallback",
            }

    def prepare_trade_history_for_position_sizing(self) -> List[Dict[str, Any]]:
        """ポジションサイジング用の取引履歴を準備（簡易版）"""
        try:
            # バックテスト中は取引履歴が利用できないため、
            # ダミーデータまたは空のリストを返す
            # 実際の実装では、過去の取引結果を保存・参照する仕組みが必要
            return []

        except Exception as e:
            logger.error(f"取引履歴準備エラー: {e}")
            return []

    def _calculate_atr_from_data(self, data, period: int = 14) -> float:
        """
        市場データからATRを計算（TA-Lib経由の実装に変更）

        Args:
            data: OHLC市場データ（pandas DataFrame 互換: High/Low/Close カラム）
            period: ATR計算期間

        Returns:
            計算されたATR値（最終要素）
        """
        try:
            if (
                not hasattr(data, "High")
                or not hasattr(data, "Low")
                or not hasattr(data, "Close")
            ):
                return 0.0

            highs = np.asarray(data.High, dtype=np.float64)
            lows = np.asarray(data.Low, dtype=np.float64)
            closes = np.asarray(data.Close, dtype=np.float64)

            if (
                highs.size < max(2, period + 1)
                or lows.size != highs.size
                or closes.size != highs.size
            ):
                return 0.0

            atr_arr = VolatilityIndicators.atr(highs, lows, closes, length=period)
            # TA-Libの出力は先頭にNaNが入ることがあるため、有限値の最後を採用
            finite = atr_arr[~np.isnan(atr_arr)]
            if finite.size == 0:
                return 0.0
            return float(finite[-1])

        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            return 0.0
