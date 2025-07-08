"""
ポジションサイジング計算サービス

PositionSizingGeneに基づいて実際のポジションサイズを計算するサービスです。
市場データ統合、パフォーマンス最適化、キャッシュ機能を提供します。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingResult:
    """ポジションサイジング計算結果"""

    position_size: float
    method_used: str
    calculation_details: Dict[str, Any]
    confidence_score: float
    risk_metrics: Dict[str, float]
    warnings: List[str]
    timestamp: datetime


@dataclass
class MarketDataCache:
    """市場データキャッシュ"""

    atr_values: Dict[str, float]
    volatility_metrics: Dict[str, float]
    price_data: Optional[pd.DataFrame]
    last_updated: datetime

    def is_expired(self, max_age_minutes: int = 5) -> bool:
        """キャッシュが期限切れかチェック"""
        return (
            datetime.now() - self.last_updated
        ).total_seconds() > max_age_minutes * 60


class PositionSizingCalculatorService:
    """
    ポジションサイジング計算サービス

    PositionSizingGeneの設定に基づいて、実際のポジションサイズを計算します。
    市場データの統合、計算結果のキャッシュ、パフォーマンス最適化を提供します。
    """

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self._cache: Optional[MarketDataCache] = None
        self._calculation_history: List[PositionSizingResult] = []

    def calculate_position_size(
        self,
        gene,  # PositionSizingGene
        account_balance: float,
        current_price: float,
        symbol: str = "BTCUSDT",
        market_data: Optional[Dict[str, Any]] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> PositionSizingResult:
        """
        ポジションサイズを計算

        Args:
            gene: ポジションサイジング遺伝子
            account_balance: 口座残高
            current_price: 現在価格
            symbol: 取引ペア
            market_data: 市場データ
            trade_history: 取引履歴
            use_cache: キャッシュを使用するか

        Returns:
            計算結果
        """
        try:
            start_time = datetime.now()
            warnings = []

            # 入力値の検証
            validation_result = self._validate_inputs(
                gene, account_balance, current_price
            )
            if not validation_result["valid"]:
                method_name = (
                    gene.method.value if gene and hasattr(gene, "method") else "unknown"
                )
                return self._create_error_result(
                    validation_result["error"], method_name
                )

            # 市場データの準備
            enhanced_market_data = self._prepare_market_data(
                symbol, current_price, market_data, use_cache
            )

            # 手法別の計算
            if gene.method.value == "half_optimal_f":
                result = self._calculate_half_optimal_f_enhanced(
                    gene, account_balance, current_price, trade_history
                )
            elif gene.method.value == "volatility_based":
                result = self._calculate_volatility_based_enhanced(
                    gene, account_balance, current_price, enhanced_market_data
                )
            elif gene.method.value == "fixed_ratio":
                result = self._calculate_fixed_ratio_enhanced(
                    gene, account_balance, current_price
                )
            elif gene.method.value == "fixed_quantity":
                result = self._calculate_fixed_quantity_enhanced(gene, current_price)
            else:
                # フォールバック
                result = self._calculate_fixed_ratio_enhanced(
                    gene, account_balance, current_price
                )
                warnings.append(
                    f"未知の手法 {gene.method.value}、固定比率にフォールバック"
                )

            # リスクメトリクスの計算
            risk_metrics = self._calculate_risk_metrics(
                result["position_size"],
                account_balance,
                current_price,
                enhanced_market_data,
            )

            # 信頼度スコアの計算
            confidence_score = self._calculate_confidence_score(
                gene, enhanced_market_data, trade_history
            )

            # 結果の作成
            calculation_time = (datetime.now() - start_time).total_seconds()

            final_result = PositionSizingResult(
                position_size=result["position_size"],
                method_used=gene.method.value,
                calculation_details={
                    **result["details"],
                    "calculation_time_seconds": calculation_time,
                    "account_balance": account_balance,
                    "current_price": current_price,
                    "symbol": symbol,
                },
                confidence_score=confidence_score,
                risk_metrics=risk_metrics,
                warnings=warnings + result.get("warnings", []),
                timestamp=datetime.now(),
            )

            # 履歴に追加
            self._calculation_history.append(final_result)
            if len(self._calculation_history) > 1000:
                self._calculation_history = self._calculation_history[-500:]

            return final_result

        except Exception as e:
            self.logger.error(f"ポジションサイズ計算エラー: {e}", exc_info=True)
            return self._create_error_result(
                str(e), gene.method.value if gene else "unknown"
            )

    def _validate_inputs(
        self, gene, account_balance: float, current_price: float
    ) -> Dict[str, Any]:
        """入力値の検証"""
        if not gene:
            return {"valid": False, "error": "遺伝子が指定されていません"}

        if account_balance <= 0:
            return {"valid": False, "error": "口座残高は正の値である必要があります"}

        if current_price <= 0:
            return {"valid": False, "error": "現在価格は正の値である必要があります"}

        # 遺伝子の妥当性チェック
        is_valid, errors = gene.validate()
        if not is_valid:
            return {"valid": False, "error": f"遺伝子が無効です: {', '.join(errors)}"}

        return {"valid": True}

    def _prepare_market_data(
        self,
        symbol: str,
        current_price: float,
        market_data: Optional[Dict[str, Any]],
        use_cache: bool,
    ) -> Dict[str, Any]:
        """市場データの準備と拡張"""
        enhanced_data = market_data.copy() if market_data else {}

        # キャッシュチェック
        if use_cache and self._cache and not self._cache.is_expired():
            enhanced_data.update(self._cache.atr_values)
            enhanced_data.update(self._cache.volatility_metrics)

        # ATR値の確保
        if "atr" not in enhanced_data and "atr_pct" not in enhanced_data:
            # デフォルトATR値を設定（現在価格の2%）
            enhanced_data["atr"] = current_price * 0.02
            enhanced_data["atr_pct"] = 0.02
            enhanced_data["atr_source"] = "default"

        # ボラティリティメトリクスの追加
        if "volatility" not in enhanced_data:
            enhanced_data["volatility"] = enhanced_data.get("atr_pct", 0.02)

        return enhanced_data

    def _calculate_half_optimal_f_enhanced(
        self,
        gene,
        account_balance: float,
        current_price: float,
        trade_history: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """ハーフオプティマルF方式の拡張計算"""
        details: Dict[str, Any] = {"method": "half_optimal_f"}
        warnings = []

        if not trade_history or len(trade_history) < 10:
            # データ不足時は固定比率にフォールバック
            position_size = account_balance * gene.fixed_ratio
            warnings.append("取引履歴が不足、固定比率にフォールバック")
            details.update(
                {
                    "fallback_reason": "insufficient_trade_history",
                    "trade_count": len(trade_history) if trade_history else 0,
                    "fallback_ratio": gene.fixed_ratio,
                }
            )
        else:
            # 過去データの分析
            recent_trades = trade_history[-gene.lookback_period :]

            wins = [t for t in recent_trades if t.get("pnl", 0) > 0]
            losses = [t for t in recent_trades if t.get("pnl", 0) < 0]

            if len(recent_trades) == 0 or len(wins) == 0 or len(losses) == 0:
                position_size = account_balance * gene.fixed_ratio
                warnings.append("有効な取引データなし、固定比率にフォールバック")
                details.update(
                    {
                        "fallback_reason": "no_valid_trades",
                        "fallback_ratio": gene.fixed_ratio,
                    }
                )
            else:
                win_rate = len(wins) / len(recent_trades)
                avg_win = np.mean([t.get("pnl", 0) for t in wins])
                avg_loss = abs(np.mean([t.get("pnl", 0) for t in losses]))

                # オプティマルF計算
                if avg_win > 0 and avg_loss > 0:
                    optimal_f = (
                        win_rate * avg_win - (1 - win_rate) * avg_loss
                    ) / avg_win
                    half_optimal_f = max(0, optimal_f * gene.optimal_f_multiplier)

                    # 口座残高に対する比率として適用
                    position_size = account_balance * half_optimal_f

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
                else:
                    position_size = account_balance * gene.fixed_ratio
                    warnings.append("無効な損益データ、固定比率にフォールバック")
                    details.update(
                        {
                            "fallback_reason": "invalid_pnl_data",
                            "fallback_ratio": gene.fixed_ratio,
                        }
                    )

        # サイズ制限の適用
        position_size = max(
            gene.min_position_size, min(position_size, gene.max_position_size)
        )
        details["final_position_size"] = position_size

        return {
            "position_size": position_size,
            "details": details,
            "warnings": warnings,
        }

    def _calculate_volatility_based_enhanced(
        self,
        gene,
        account_balance: float,
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """ボラティリティベース方式の拡張計算"""
        details: Dict[str, Any] = {"method": "volatility_based"}
        warnings = []

        # ATR値の取得
        atr_value = market_data.get("atr", current_price * 0.02)
        atr_pct = atr_value / current_price if current_price > 0 else 0.02

        # リスク量の計算
        risk_amount = account_balance * gene.risk_per_trade

        # ポジションサイズの計算
        volatility_factor = atr_pct * gene.atr_multiplier
        if volatility_factor > 0:
            position_size = risk_amount / (current_price * volatility_factor)
        else:
            position_size = gene.min_position_size
            warnings.append("ボラティリティが0、最小サイズを使用")

        # サイズ制限の適用
        position_size = max(
            gene.min_position_size, min(position_size, gene.max_position_size)
        )

        details.update(
            {
                "atr_value": atr_value,
                "atr_pct": atr_pct,
                "atr_multiplier": gene.atr_multiplier,
                "risk_per_trade": gene.risk_per_trade,
                "risk_amount": risk_amount,
                "volatility_factor": volatility_factor,
                "final_position_size": position_size,
                "atr_source": market_data.get("atr_source", "provided"),
            }
        )

        return {
            "position_size": position_size,
            "details": details,
            "warnings": warnings,
        }

    def _calculate_fixed_ratio_enhanced(
        self,
        gene,
        account_balance: float,
        current_price: float,
    ) -> Dict[str, Any]:
        """固定比率方式の拡張計算"""
        details: Dict[str, Any] = {"method": "fixed_ratio"}

        # ポジションサイズの計算
        position_size = account_balance * gene.fixed_ratio

        # サイズ制限の適用
        position_size = max(
            gene.min_position_size, min(position_size, gene.max_position_size)
        )

        details.update(
            {
                "fixed_ratio": gene.fixed_ratio,
                "account_balance": account_balance,
                "calculated_size": account_balance * gene.fixed_ratio,
                "final_position_size": position_size,
            }
        )

        return {
            "position_size": position_size,
            "details": details,
            "warnings": [],
        }

    def _calculate_fixed_quantity_enhanced(
        self,
        gene,
        current_price: float,
    ) -> Dict[str, Any]:
        """固定枚数方式の拡張計算"""
        details: Dict[str, Any] = {"method": "fixed_quantity"}

        # ポジションサイズの計算
        position_size = gene.fixed_quantity

        # サイズ制限の適用
        position_size = max(
            gene.min_position_size, min(position_size, gene.max_position_size)
        )

        details.update(
            {
                "fixed_quantity": gene.fixed_quantity,
                "final_position_size": position_size,
            }
        )

        return {
            "position_size": position_size,
            "details": details,
            "warnings": [],
        }

    def _calculate_risk_metrics(
        self,
        position_size: float,
        account_balance: float,
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """リスクメトリクスの計算"""
        try:
            # 基本メトリクス
            position_value = position_size * current_price
            position_ratio = (
                position_value / account_balance if account_balance > 0 else 0
            )

            # ボラティリティベースのリスク
            atr_pct = market_data.get("atr_pct", 0.02)
            potential_loss_1atr = position_value * atr_pct
            potential_loss_ratio = (
                potential_loss_1atr / account_balance if account_balance > 0 else 0
            )

            return {
                "position_value": position_value,
                "position_ratio": position_ratio,
                "potential_loss_1atr": potential_loss_1atr,
                "potential_loss_ratio": potential_loss_ratio,
                "atr_used": atr_pct,
            }
        except Exception as e:
            self.logger.error(f"リスクメトリクス計算エラー: {e}")
            return {
                "position_value": 0.0,
                "position_ratio": 0.0,
                "potential_loss_1atr": 0.0,
                "potential_loss_ratio": 0.0,
                "atr_used": 0.0,
            }

    def _calculate_confidence_score(
        self,
        gene,
        market_data: Dict[str, Any],
        trade_history: Optional[List[Dict[str, Any]]],
    ) -> float:
        """信頼度スコアの計算"""
        try:
            score = 0.5  # ベーススコア

            # データ品質による調整
            if market_data.get("atr_source") == "real":
                score += 0.2
            elif market_data.get("atr_source") == "calculated":
                score += 0.1

            # 取引履歴による調整
            if trade_history and len(trade_history) >= 50:
                score += 0.2
            elif trade_history and len(trade_history) >= 20:
                score += 0.1

            # 遺伝子パラメータの妥当性による調整
            is_valid, _ = gene.validate()
            if is_valid:
                score += 0.1

            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5

    def _create_error_result(
        self, error_message: str, method: str
    ) -> PositionSizingResult:
        """エラー結果の作成"""
        return PositionSizingResult(
            position_size=0.01,  # 最小サイズ
            method_used=method,
            calculation_details={"error": error_message},
            confidence_score=0.0,
            risk_metrics={},
            warnings=[f"計算エラー: {error_message}"],
            timestamp=datetime.now(),
        )

    def get_calculation_history(self, limit: int = 100) -> List[PositionSizingResult]:
        """計算履歴の取得"""
        return self._calculation_history[-limit:]

    def clear_cache(self):
        """キャッシュのクリア"""
        self._cache = None

    def get_cache_status(self) -> Dict[str, Any]:
        """キャッシュ状態の取得"""
        if not self._cache:
            return {"cached": False}

        return {
            "cached": True,
            "last_updated": self._cache.last_updated.isoformat(),
            "expired": self._cache.is_expired(),
            "atr_symbols": list(self._cache.atr_values.keys()),
        }
