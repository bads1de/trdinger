"""動的ポジションサイジングとリスク管理のテスト"""

from __future__ import annotations

import math

from app.services.auto_strategy.config.constants import PositionSizingMethod
from app.services.auto_strategy.genes.position_sizing import PositionSizingGene
from app.services.auto_strategy.positions.calculators.volatility_based_calculator import (
    VolatilityBasedCalculator,
)
from app.services.auto_strategy.positions.position_sizing_service import (
    PositionSizingService,
)
from app.services.auto_strategy.positions.risk_metrics import (
    calculate_expected_shortfall,
    calculate_historical_var,
)


class TestRiskMetrics:
    """リスク指標ユーティリティのテスト"""

    def test_historical_var_and_es_with_losses(self):
        returns = [-0.20, -0.10, 0.05, 0.10]
        confidence = 0.95

        var = calculate_historical_var(returns, confidence)
        es = calculate_expected_shortfall(returns, confidence)

        assert math.isclose(var, 0.185, rel_tol=1e-9)
        assert math.isclose(es, 0.20, rel_tol=1e-9)

    def test_risk_metrics_with_non_loss_returns(self):
        returns = [0.01, 0.02, 0.03]

        assert calculate_historical_var(returns, 0.99) == 0.0
        assert calculate_expected_shortfall(returns, 0.99) == 0.0


class TestDynamicPositionSizing:
    """動的ポジションサイジング関連のテスト"""

    def _create_gene(self, **overrides) -> PositionSizingGene:
        params = {
            "method": PositionSizingMethod.VOLATILITY_BASED,
            "risk_per_trade": 0.05,
            "atr_multiplier": 1.5,
            "min_position_size": 0.01,
            "max_position_size": 10.0,
            "var_confidence": 0.95,
            "max_var_ratio": 0.01,
            "max_expected_shortfall_ratio": 0.02,
        }
        params.update(overrides)
        return PositionSizingGene(**params)

    def test_volatility_calculator_applies_var_cap(self):
        """VaR制限がポジションサイズに反映されること"""

        calculator = VolatilityBasedCalculator()
        gene = self._create_gene()

        market_data = {
            "atr": 20.0,
            "atr_pct": 0.1,
            "returns": [-0.02, -0.03, -0.015, -0.05, -0.04, 0.01, 0.02],
            "atr_source": "real",
        }

        result = calculator.calculate(
            gene=gene,
            account_balance=10000.0,
            current_price=200.0,
            market_data=market_data,
        )

        risk_controls = result["details"].get("risk_controls", {})
        var_ratio = risk_controls.get("var_ratio")
        expected_cap_size = (10000.0 * gene.max_var_ratio) / (
            max(var_ratio, 1e-12) * 200.0
        )
        es_ratio = risk_controls.get("expected_shortfall")
        expected_shortfall_cap = (10000.0 * gene.max_expected_shortfall_ratio) / (
            max(es_ratio, 1e-12) * 200.0
        )
        expected_size = min(
            expected_cap_size, expected_shortfall_cap, gene.max_position_size
        )

        assert math.isclose(result["position_size"], expected_size, rel_tol=1e-6)

        assert risk_controls.get("var_adjusted") is True
        assert risk_controls.get("expected_shortfall_adjusted") is False
        assert (
            risk_controls.get("var_loss") <= risk_controls.get("max_var_allowed") + 1e-6
        )
        assert var_ratio > 0

    def test_position_sizing_service_reports_var_es(self):
        """サービス結果にVaRとESが含まれること"""

        service = PositionSizingService()
        gene = self._create_gene(max_var_ratio=0.02)

        market_data = {
            "atr": 15.0,
            "atr_pct": 0.05,
            "atr_source": "real",
            "returns": [-0.01, -0.015, -0.03, 0.005, 0.01, -0.025, -0.02],
        }

        result = service.calculate_position_size(
            gene=gene,
            account_balance=5000.0,
            current_price=100.0,
            symbol="BTC/USDT:USDT",
            market_data=market_data,
            use_cache=False,
        )

        risk_metrics = result.risk_metrics
        assert "var" in risk_metrics
        assert "var_loss" in risk_metrics
        assert "expected_shortfall" in risk_metrics
        assert "expected_shortfall_loss" in risk_metrics
        assert risk_metrics["var"] >= 0
        assert risk_metrics["expected_shortfall"] >= 0

    def test_position_sizing_gene_validates_risk_params(self):
        """無効なリスク管理パラメータは検出されること"""

        gene = self._create_gene(var_confidence=0.5, max_var_ratio=0.5)
        is_valid, errors = gene.validate()

        assert is_valid is False
        assert any("var_confidence" in error for error in errors)
        assert any("max_var_ratio" in error for error in errors)

    def test_volatility_calculator_applies_expected_shortfall_cap(self):
        """ES制限が適用されるケースを検証"""

        calculator = VolatilityBasedCalculator()
        gene = self._create_gene(
            atr_multiplier=1.0,
            max_position_size=500.0,
            max_var_ratio=0.5,
            max_expected_shortfall_ratio=0.001,
        )

        market_data = {
            "atr": 2.0,
            "atr_pct": 0.01,
            "returns": [-0.12, -0.15, -0.08, -0.05, -0.03, 0.01, 0.02],
            "atr_source": "real",
        }

        result = calculator.calculate(
            gene=gene,
            account_balance=10000.0,
            current_price=200.0,
            market_data=market_data,
        )

        risk_controls = result["details"].get("risk_controls", {})
        es_ratio = risk_controls.get("expected_shortfall")
        var_ratio = risk_controls.get("var_ratio")
        es_cap = (10000.0 * gene.max_expected_shortfall_ratio) / (
            max(es_ratio, 1e-12) * 200.0
        )
        var_cap = (10000.0 * gene.max_var_ratio) / (max(var_ratio, 1e-12) * 200.0)
        expected_size = min(es_cap, var_cap, gene.max_position_size)
        expected_size = max(expected_size, gene.min_position_size)

        assert math.isclose(result["position_size"], expected_size, rel_tol=1e-6)
        assert risk_controls.get("expected_shortfall_adjusted") is True
        assert risk_controls.get("var_adjusted") is True
        assert (
            risk_controls.get("expected_shortfall_loss")
            <= risk_controls.get("max_expected_shortfall_allowed") + 1e-6
        )

    def test_volatility_calculator_handles_missing_returns(self):
        """リターンデータが無い場合でも安全に計算できること"""

        calculator = VolatilityBasedCalculator()
        gene = self._create_gene()

        market_data = {
            "atr": 10.0,
            "atr_pct": 0.05,
            "atr_source": "synthetic",
            # returnsを意図的に含めない
        }

        result = calculator.calculate(
            gene=gene,
            account_balance=5000.0,
            current_price=100.0,
            market_data=market_data,
        )

        risk_controls = result["details"].get("risk_controls", {})

        assert risk_controls.get("return_sample_size") == 0
        assert risk_controls.get("var_ratio") == 0
        assert risk_controls.get("expected_shortfall") == 0
        assert risk_controls.get("var_adjusted") is False
        assert risk_controls.get("expected_shortfall_adjusted") is False
        assert result["position_size"] >= gene.min_position_size

    def test_volatility_calculator_respects_var_lookback(self):
        """VaR計算が指定したルックバックに従うこと"""

        calculator = VolatilityBasedCalculator()
        gene = self._create_gene(var_lookback=3, var_confidence=0.9, max_var_ratio=0.5)

        returns = [-0.05, -0.02, -0.01, -0.04, -0.06, 0.01]
        market_data = {
            "atr": 5.0,
            "atr_pct": 0.02,
            "returns": returns,
            "atr_source": "real",
        }

        result = calculator.calculate(
            gene=gene,
            account_balance=2000.0,
            current_price=100.0,
            market_data=market_data,
        )

        risk_controls = result["details"].get("risk_controls", {})
        tail_returns = returns[-3:]
        expected_var = calculate_historical_var(tail_returns, gene.var_confidence)

        assert risk_controls.get("return_sample_size") == 3
        assert math.isclose(risk_controls.get("var_ratio"), expected_var, rel_tol=1e-12)
        assert risk_controls.get("var_lookback") == gene.var_lookback

    def test_calculate_position_size_fast_returns_valid_size(self):
        """高速計算メソッドが有効なポジションサイズを返すこと"""

        service = PositionSizingService()
        gene = self._create_gene()

        result = service.calculate_position_size_fast(
            gene=gene,
            account_balance=10000.0,
            current_price=100.0,
        )

        # 戻り値がfloatで、min/maxの範囲内であること
        assert isinstance(result, float)
        assert gene.min_position_size <= result <= gene.max_position_size

    def test_calculate_position_size_fast_handles_invalid_inputs(self):
        """高速計算メソッドが無効な入力時にデフォルト値を返すこと"""

        service = PositionSizingService()
        gene = self._create_gene()

        # 無効な残高
        result1 = service.calculate_position_size_fast(
            gene=gene, account_balance=-100.0, current_price=100.0
        )
        assert result1 == 0.01

        # 無効な価格
        result2 = service.calculate_position_size_fast(
            gene=gene, account_balance=10000.0, current_price=0.0
        )
        assert result2 == 0.01

        # 遺伝子がNone
        result3 = service.calculate_position_size_fast(
            gene=None, account_balance=10000.0, current_price=100.0
        )
        assert result3 == 0.01

    def test_calculate_position_size_fast_is_faster_than_full(self):
        """高速計算がフル計算より高速であること"""
        import time

        service = PositionSizingService()
        gene = self._create_gene()
        iterations = 100

        # 高速版の計測
        start_fast = time.perf_counter()
        for _ in range(iterations):
            service.calculate_position_size_fast(
                gene=gene, account_balance=10000.0, current_price=100.0
            )
        time_fast = time.perf_counter() - start_fast

        # フル版の計測
        start_full = time.perf_counter()
        for _ in range(iterations):
            service.calculate_position_size(
                gene=gene,
                account_balance=10000.0,
                current_price=100.0,
                use_cache=False,  # キャッシュを無効化
            )
        time_full = time.perf_counter() - start_full

        # 高速版がフル版より速いか同等であること
        # （少なくとも大幅に遅くないこと）
        assert (
            time_fast <= time_full * 1.5
        ), f"高速版 ({time_fast:.4f}s) がフル版 ({time_full:.4f}s) より顕著に遅い"
