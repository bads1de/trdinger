"""
統一ポジションサイジングサービス

PositionSizingGeneに基づいて実際のポジションサイズを計算するサービスです。
市場データ統合、パフォーマンス最適化、キャッシュ機能を提供します。
PositionSizingCalculatorServiceとPositionSizingServiceの機能を統合しています。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config.unified_config import unified_config
from app.utils.error_handler import safe_operation

from .calculators.calculator_factory import CalculatorFactory
from .market_data_handler import MarketDataHandler
from .risk_metrics import calculate_expected_shortfall, calculate_historical_var

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


class PositionSizingService:
    """
    ポジションサイジング計算サービス

    PositionSizingGeneの設定に基づいて、実際のポジションサイズを計算します。
    市場データの統合、計算結果のキャッシュ、パフォーマンス最適化を提供します。
    """

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self._market_data_handler = MarketDataHandler()
        self._calculator_factory = CalculatorFactory()
        self._calculation_history: List[PositionSizingResult] = []

    @safe_operation(context="ポジションサイズ計算", is_api_call=False)
    def calculate_position_size(
        self,
        gene,
        account_balance: float,
        current_price: float,
        symbol: str = "BTC/USDT:USDT",
        market_data: Optional[Dict[str, Any]] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> PositionSizingResult:
        """
        ポジションサイズを計算
        """
        start_time = datetime.now()
        warnings: List[str] = []

        # 1. 入力値の検証
        validation_result = self._validate_inputs(gene, account_balance, current_price)
        if not validation_result["valid"]:
            method_name = "unknown"
            if gene and hasattr(gene, "method"):
                method_name = gene.method.value if hasattr(gene.method, "value") else str(gene.method)
            return self._create_error_result(validation_result["error"], method_name)

        # 2. 市場データの準備
        enhanced_market_data = self._market_data_handler.prepare_market_data(
            symbol, current_price, market_data, use_cache
        )

        # 3. 計算機の実行
        method_val = gene.method.value if hasattr(gene.method, "value") else str(gene.method)
        calculator = self._calculator_factory.create_calculator(method_val)
        result = calculator.calculate(
            gene, account_balance, current_price,
            market_data=enhanced_market_data, trade_history=trade_history
        )

        # 4. リスクと信頼度の計算
        risk_metrics = self._calculate_risk_metrics(
            result["position_size"], account_balance, current_price, enhanced_market_data, gene
        )
        confidence_score = self._calculate_confidence_score(gene, enhanced_market_data, trade_history)

        # 5. 結果の構築
        calculation_time = (datetime.now() - start_time).total_seconds()
        final_result = PositionSizingResult(
            position_size=result["position_size"],
            method_used=method_val,
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

        # 履歴の管理
        self._calculation_history.append(final_result)
        if len(self._calculation_history) > 1000:
            self._calculation_history = self._calculation_history[-500:]

        return final_result

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

    @safe_operation(context="リスクメトリクス計算", is_api_call=False)
    def _calculate_risk_metrics(
        self,
        position_size: float,
        account_balance: float,
        current_price: float,
        market_data: Dict[str, Any],
        gene,
    ) -> Dict[str, float]:
        """リスクメトリクスの計算"""
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

        returns_data = market_data.get("returns")
        returns_sample: List[float] = []
        if returns_data is not None:
            try:
                returns_list = list(returns_data)
                lookback = max(int(getattr(gene, "var_lookback", len(returns_list))), 1)
                returns_sample = returns_list[-lookback:]
            except TypeError:
                returns_sample = []

        var_ratio = calculate_historical_var(
            returns_sample, getattr(gene, "var_confidence", 0.95)
        )
        expected_shortfall_ratio = calculate_expected_shortfall(
            returns_sample, getattr(gene, "var_confidence", 0.95)
        )

        return {
            "position_value": position_value,
            "position_ratio": position_ratio,
            "potential_loss_1atr": potential_loss_1atr,
            "potential_loss_ratio": potential_loss_ratio,
            "atr_used": atr_pct,
            "var": var_ratio,
            "var_loss": position_value * var_ratio,
            "max_var_allowed": account_balance * getattr(gene, "max_var_ratio", 0.0),
            "expected_shortfall": expected_shortfall_ratio,
            "expected_shortfall_loss": position_value * expected_shortfall_ratio,
            "max_expected_shortfall_allowed": account_balance * getattr(gene, "max_expected_shortfall_ratio", 0.0),
            "return_sample_size": len(returns_sample),
        }

    @safe_operation(context="信頼度スコア計算", is_api_call=False)
    def _calculate_confidence_score(
        self,
        gene,
        market_data: Dict[str, Any],
        trade_history: Optional[List[Dict[str, Any]]],
    ) -> float:
        """信頼度スコアの計算"""
        score = 0.5  # ベーススコア

        # データ品質による調整
        atr_source = market_data.get("atr_source")
        if atr_source == "real": score += 0.2
        elif atr_source == "calculated": score += 0.1

        # 取引履歴による調整
        if trade_history:
            if len(trade_history) >= 50: score += 0.2
            elif len(trade_history) >= 20: score += 0.1

        # 遺伝子の妥当性
        is_valid, _ = gene.validate()
        if is_valid: score += 0.1

        return min(1.0, max(0.0, score))

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

    def clear_cache(self):
        """キャッシュのクリア"""
        self._market_data_handler.clear_cache()

    def _create_default_gene(self, **kwargs):
        """デフォルト遺伝子を作成"""
        from ..genes import (
            PositionSizingGene,
            PositionSizingMethod,
        )

        method = kwargs.get("method", "volatility_based")

        # 文字列からenumに変換
        if isinstance(method, str):
            method_map = {
                "half_optimal_f": PositionSizingMethod.HALF_OPTIMAL_F,
                "volatility_based": PositionSizingMethod.VOLATILITY_BASED,
                "fixed_ratio": PositionSizingMethod.FIXED_RATIO,
                "fixed_quantity": PositionSizingMethod.FIXED_QUANTITY,
            }
            method = method_map.get(method, PositionSizingMethod.VOLATILITY_BASED)

        return PositionSizingGene(
            method=method,
            enabled=True,
            risk_per_trade=kwargs.get("risk_per_trade", 0.02),
            fixed_ratio=kwargs.get("fixed_ratio", 0.1),
            fixed_quantity=kwargs.get("fixed_quantity", 0.01),
            atr_multiplier=kwargs.get("atr_multiplier", 2.0),
            optimal_f_multiplier=kwargs.get("optimal_f_multiplier", 0.5),
            lookback_period=kwargs.get("lookback_period", 30),
            min_position_size=kwargs.get("min_position_size", 0.001),
            max_position_size=kwargs.get("max_position_size", 10.0),
        )

    def _calculate_fallback(
        self, account_balance: float, current_price: float
    ) -> PositionSizingResult:
        """フォールバック計算（固定比率）"""
        default_ratio = unified_config.auto_strategy.default_position_ratio
        position_amount = account_balance * default_ratio
        position_size = position_amount / current_price if current_price > 0 else 0.001

        return PositionSizingResult(
            position_size=position_size,
            method_used="fixed_ratio_fallback",
            calculation_details={
                "ratio": default_ratio,
                "position_amount": position_amount,
                "fallback": True,
            },
            confidence_score=0.3,
            risk_metrics={"position_ratio": default_ratio},
            warnings=["フォールバック計算を使用"],
            timestamp=datetime.now(),
        )

    def calculate_position_size_simple(
        self,
        method: str = "volatility_based",
        account_balance: float = 100000.0,
        current_price: float = 50000.0,
        **kwargs,
    ) -> float:
        """
        簡易ポジションサイズ計算（後方互換性用）

        Args:
            method: 計算方式
            account_balance: 口座残高
            current_price: 現在価格
            **kwargs: その他のパラメータ

        Returns:
            ポジションサイズ（数量）
        """

        @safe_operation(
            context="簡易ポジションサイズ計算", is_api_call=False, default_return=0.0
        )
        def _calculate_position_size_simple():
            gene = self._create_default_gene(method=method, **kwargs)
            result = self.calculate_position_size(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                **kwargs,
            )
            return result.position_size

        return _calculate_position_size_simple()

    def calculate_position_size_fast(
        self,
        gene,
        account_balance: float,
        current_price: float,
    ) -> float:
        """
        高速ポジションサイズ計算（バックテスト用）

        詳細なリスクメトリクス計算（VaR、Expected Shortfall等）をスキップし、
        純粋なポジションサイズのみを高速に計算します。
        バックテストループ内での使用に最適化されています。

        Args:
            gene: ポジションサイジング遺伝子
            account_balance: 口座残高
            current_price: 現在価格

        Returns:
            ポジションサイズ（数量）
        """
        try:
            # 入力値の簡易検証（フルバリデーションをスキップ）
            if not gene or account_balance <= 0 or current_price <= 0:
                return 0.01  # デフォルトサイズ

            # 計算機の選択と実行（市場データなしで高速実行）
            method_val = gene.method.value if hasattr(gene.method, "value") else str(gene.method)
            calculator = self._calculator_factory.create_calculator(method_val)
            result = calculator.calculate(
                gene,
                account_balance,
                current_price,
                market_data={},  # 軽量化のため空
                trade_history=None,
            )

            return result["position_size"]

        except Exception as e:
            self.logger.warning(f"高速ポジションサイズ計算エラー: {e}")
            return 0.01  # デフォルトサイズ





