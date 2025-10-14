"""
ハーフオプティマルF方式計算クラス
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.config.unified_config import unified_config
from app.utils.error_handler import safe_operation

from .base_calculator import BaseCalculator

logger = logging.getLogger(__name__)


class HalfOptimalFCalculator(BaseCalculator):
    """ハーフオプティマルF方式計算クラス"""

    def calculate(
        self, gene, account_balance: float, current_price: float, **kwargs
    ) -> Dict[str, Any]:
        """ハーフオプティマルF方式の拡張計算"""
        trade_history = kwargs.get("trade_history")
        details: Dict[str, Any] = {"method": "half_optimal_f"}
        warnings: List[str] = []

        if not trade_history or len(trade_history) < 10:
            # データ不足時は簡易版オプティマルF計算を試行
            result = self._calculate_simplified_optimal_f(
                gene, account_balance, current_price, trade_history, warnings, details
            )
        else:
            # 過去データの分析
            result = self._calculate_with_trade_history(
                gene, account_balance, current_price, trade_history, warnings, details
            )

        # 統一された最終処理（重複コード除去）
        return self._apply_size_limits_and_finalize(
            result["position_size"], details, warnings, gene
        )

    def _calculate_simplified_optimal_f(
        self,
        gene,
        account_balance: float,
        current_price: float,
        trade_history: Optional[List[Dict[str, Any]]],
        warnings: List[str],
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """簡易オプティマルF計算"""

        @safe_operation(
            context="簡易オプティマルF計算",
            is_api_call=False,
            default_return={
                "position_size": 0,
                "warnings": ["簡易計算失敗"],
                "details": {"fallback_reason": "simplified_calculation_failed"},
            },
        )
        def _simplified_optimal_f():
            # 統計的仮定値を使用した簡易計算
            assumed_win_rate = unified_config.auto_strategy.assumed_win_rate
            assumed_avg_win = unified_config.auto_strategy.assumed_avg_win
            assumed_avg_loss = unified_config.auto_strategy.assumed_avg_loss

            optimal_f = (
                assumed_win_rate * assumed_avg_win
                - (1 - assumed_win_rate) * assumed_avg_loss
            ) / assumed_avg_win
            half_optimal_f = max(0, min(0.1, optimal_f * gene.optimal_f_multiplier))

            position_amount = account_balance * half_optimal_f
            position_size = self._safe_calculate_with_price_check(
                lambda: position_amount / current_price,
                current_price,
                0,
                "現在価格が無効",
            )

            return {
                "position_size": position_size,
                "warnings": ["取引履歴が不足、簡易版オプティマルF計算を使用"],
                "details": {
                    "fallback_reason": "insufficient_trade_history_simplified",
                    "trade_count": len(trade_history) if trade_history else 0,
                    "assumed_win_rate": assumed_win_rate,
                    "assumed_avg_win": assumed_avg_win,
                    "assumed_avg_loss": assumed_avg_loss,
                    "calculated_optimal_f": optimal_f,
                    "half_optimal_f": half_optimal_f,
                },
            }

        simplified_result = _simplified_optimal_f()
        if isinstance(simplified_result, dict):
            warnings.extend(simplified_result.get("warnings", []))
            details.update(simplified_result.get("details", {}))
            return simplified_result
        else:
            # フォールバック
            position_amount = account_balance * gene.fixed_ratio
            position_size = self._safe_calculate_with_price_check(
                lambda: position_amount / current_price,
                current_price,
                0,
                "取引履歴が不足、固定比率にフォールバック",
                warnings,
            )
            details.update(
                {
                    "fallback_reason": "insufficient_trade_history_to_fixed",
                    "trade_count": len(trade_history) if trade_history else 0,
                    "fallback_ratio": gene.fixed_ratio,
                }
            )
            return {"position_size": position_size}

    def _calculate_with_trade_history(
        self,
        gene,
        account_balance: float,
        current_price: float,
        trade_history: List[Dict[str, Any]],
        warnings: List[str],
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """取引履歴を使用した計算"""
        recent_trades = trade_history[-gene.lookback_period :]

        wins = [t for t in recent_trades if t.get("pnl", 0) > 0]
        losses = [t for t in recent_trades if t.get("pnl", 0) < 0]

        if len(recent_trades) == 0 or len(wins) == 0 or len(losses) == 0:
            position_amount = account_balance * gene.fixed_ratio
            position_size = self._safe_calculate_with_price_check(
                lambda: position_amount / current_price,
                current_price,
                0,
                "有効な取引データなし、固定比率にフォールバック",
                warnings,
            )
            details.update(
                {
                    "fallback_reason": "no_valid_trades",
                    "fallback_ratio": gene.fixed_ratio,
                }
            )
            return {"position_size": position_size}

        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t.get("pnl", 0) for t in wins])
        avg_loss = abs(np.mean([t.get("pnl", 0) for t in losses]))

        # オプティマルF計算
        if avg_win > 0 and avg_loss > 0:
            optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            half_optimal_f = max(0, optimal_f * gene.optimal_f_multiplier)

            # 口座残高に対する比率として適用
            position_amount = account_balance * half_optimal_f
            position_size = self._safe_calculate_with_price_check(
                lambda: position_amount / current_price,
                current_price,
                0,
                "現在価格が無効",
                warnings,
            )

            details.update(
                {
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "optimal_f": optimal_f,
                    "half_optimal_f": half_optimal_f,
                    "trade_count": len(recent_trades),
                    "lookback_period": gene.lookback_period,
                }
            )
            return {"position_size": position_size}
        else:
            # 無効な損益データの場合、ボラティリティベース方式を試行
            fallback_result = self._volatility_fallback(
                gene, account_balance, current_price, warnings, details
            )
            return fallback_result

    def _volatility_fallback(
        self,
        gene,
        account_balance: float,
        current_price: float,
        warnings: List[str],
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """ボラティリティベースフォールバック"""

        @safe_operation(
            context="ボラティリティベースフォールバック",
            is_api_call=False,
            default_return={
                "position_size": (
                    account_balance * gene.fixed_ratio / current_price
                    if current_price > 0
                    else 0
                ),
                "warnings": ["無効な損益データ、固定比率にフォールバック"],
                "details": {
                    "fallback_reason": "invalid_pnl_data_to_fixed",
                    "fallback_ratio": gene.fixed_ratio,
                },
            },
        )
        def _fallback():
            fallback_atr_multiplier = (
                unified_config.auto_strategy.fallback_atr_multiplier
            )
            # 簡易ボラティリティ計算
            atr_value = current_price * fallback_atr_multiplier
            risk_amount = account_balance * gene.risk_per_trade
            volatility_factor = atr_value / current_price if current_price > 0 else 0
            if volatility_factor > 0:
                position_size = risk_amount / (current_price * volatility_factor)
            else:
                position_size = gene.min_position_size

            return {
                "position_size": position_size,
                "warnings": [
                    "無効な損益データ、ボラティリティベース方式にフォールバック"
                ],
                "details": {
                    "fallback_reason": "invalid_pnl_data_to_volatility",
                    "fallback_method": "volatility_based",
                },
            }

        return _fallback()
