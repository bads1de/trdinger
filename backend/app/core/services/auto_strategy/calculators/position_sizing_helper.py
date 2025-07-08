"""
ポジションサイジングヘルパー

戦略ファクトリーで使用するポジションサイズ計算のヘルパー機能を提供します。
"""

import logging
import numpy as np
from typing import Dict, Any, List

from ..models.strategy_gene import StrategyGene
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
        """ポジションサイジング用の市場データを準備"""
        try:
            market_data = {}

            # ATR計算（簡易版）
            if (
                hasattr(data, "High")
                and hasattr(data, "Low")
                and hasattr(data, "Close")
            ):
                # 過去14日のATRを計算
                period = min(14, len(data.Close) - 1)
                if period > 1:
                    high_low = data.High[-period:] - data.Low[-period:]
                    high_close = np.abs(
                        data.High[-period:] - data.Close[-period - 1 : -1]
                    )
                    low_close = np.abs(
                        data.Low[-period:] - data.Close[-period - 1 : -1]
                    )

                    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                    atr_value = np.mean(true_range)

                    market_data["atr"] = atr_value
                    market_data["atr_pct"] = (
                        atr_value / current_price if current_price > 0 else 0.02
                    )
                    market_data["atr_source"] = "calculated"
                else:
                    # データ不足時はデフォルト値
                    market_data["atr"] = current_price * 0.02
                    market_data["atr_pct"] = 0.02
                    market_data["atr_source"] = "default"
            else:
                # データ不足時はデフォルト値
                market_data["atr"] = current_price * 0.02
                market_data["atr_pct"] = 0.02
                market_data["atr_source"] = "default"

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
