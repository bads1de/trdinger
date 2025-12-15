import math
from datetime import datetime
from typing import Dict

import pytest

from app.services.auto_strategy.config.enums import PositionSizingMethod
from app.services.auto_strategy.models.position_sizing_gene import PositionSizingGene
from app.services.auto_strategy.positions.calculators.calculator_factory import (
    CalculatorFactory,
)
from app.services.auto_strategy.positions.calculators.fixed_quantity_calculator import (
    FixedQuantityCalculator,
)
from app.services.auto_strategy.positions.calculators.fixed_ratio_calculator import (
    FixedRatioCalculator,
)
from app.services.auto_strategy.positions.calculators.half_optimal_f_calculator import (
    HalfOptimalFCalculator,
)
from app.services.auto_strategy.positions.calculators.volatility_based_calculator import (
    VolatilityBasedCalculator,
)
from app.services.auto_strategy.positions.position_sizing_service import (
    PositionSizingResult,
    PositionSizingService,
)


@pytest.fixture
def base_gene() -> PositionSizingGene:
    # バリデーションに通る安全なデフォルト
    return PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        risk_per_trade=0.02,
        fixed_ratio=0.1,
        fixed_quantity=1.0,
        atr_multiplier=2.0,
        min_position_size=0.001,
        max_position_size=10.0,
        enabled=True,
    )


@pytest.mark.parametrize(
    "balance,price,fixed_ratio,expected",
    [
        (10000.0, 100.0, 0.1, 10.0),
        (5000.0, 250.0, 0.2, 4.0),
        (10000.0, 1000.0, 0.05, 0.5),
    ],
)
def test_fixed_ratio_respects_risk_percentage_and_balance(
    base_gene: PositionSizingGene,
    balance: float,
    price: float,
    fixed_ratio: float,
    expected: float,
) -> None:
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.FIXED_RATIO,
            "fixed_ratio": fixed_ratio,
        }
    )
    calculator = FixedRatioCalculator()

    result: Dict[str, float] = calculator.calculate(gene, balance, price)

    assert math.isclose(result["position_size"], expected, rel_tol=1e-9)
    assert result["details"]["fixed_ratio"] == fixed_ratio
    assert result["details"]["account_balance"] == balance
    assert result["details"]["calculated_amount"] == pytest.approx(
        balance * fixed_ratio
    )


def test_fixed_ratio_invalid_price_uses_safe_minimum(
    base_gene: PositionSizingGene,
) -> None:
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.FIXED_RATIO,
            "fixed_ratio": 0.1,
        }
    )
    calculator = FixedRatioCalculator()

    result_zero = calculator.calculate(gene, 10000.0, 0.0)
    result_negative = calculator.calculate(gene, 10000.0, -100.0)

    # BaseCalculator._safe_calculate_with_price_check の期待に依存:
    # 「無効な価格の場合は下限またはフォールバックを適用する」ことのみを検証
    assert result_zero["position_size"] >= gene.min_position_size
    assert result_negative["position_size"] >= gene.min_position_size


def test_fixed_quantity_returns_constant_size_for_any_input(
    base_gene: PositionSizingGene,
) -> None:
    fixed_qty = 3.0
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.FIXED_QUANTITY,
            "fixed_quantity": fixed_qty,
        }
    )
    calculator = FixedQuantityCalculator()

    for balance, price in [(1000.0, 10.0), (5000.0, 250.0), (100000.0, 50000.0)]:
        result = calculator.calculate(gene, balance, price)
        assert result["position_size"] == pytest.approx(fixed_qty), (
            "固定枚数方式は口座残高や価格に依存せず一定サイズを返すべき"
        )


def test_volatility_based_size_decreases_when_volatility_increases(
    base_gene: PositionSizingGene,
) -> None:
    """
    VolatilityBasedCalculator のビジネス仕様に基づくテスト。

    - position_size ∝ 1 / (atr_pct * atr_multiplier)
    - atr が大きいほど（ボラティリティ上昇） position_size は減少することを期待
    - ただし max_position_size によるクリップが発生すると単調性が崩れるため、
      本テストでは max_position_size に十分大きな値を設定し、クリップが影響しない条件で検証する。
    """
    balance = 10000.0
    price = 100.0
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.VOLATILITY_BASED,
            "risk_per_trade": 0.02,
            "atr_multiplier": 2.0,
            "min_position_size": 0.001,
            "max_position_size": 1e9,  # クリップが発生しないよう十分大きくする
        }
    )
    calculator = VolatilityBasedCalculator()

    low_vol_data = {"atr": 1.0, "returns": [0.0] * 100}
    high_vol_data = {"atr": 5.0, "returns": [0.0] * 100}

    low_vol_result = calculator.calculate(
        gene, balance, price, market_data=low_vol_data
    )
    high_vol_result = calculator.calculate(
        gene, balance, price, market_data=high_vol_data
    )

    assert low_vol_result["position_size"] > high_vol_result["position_size"]
    assert low_vol_result["position_size"] > gene.min_position_size
    assert high_vol_result["position_size"] >= gene.min_position_size


def test_volatility_based_respects_max_position_size(
    base_gene: PositionSizingGene,
) -> None:
    balance = 10_000_000.0
    price = 10.0
    max_size = 1.0
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.VOLATILITY_BASED,
            "risk_per_trade": 0.5,
            "atr_multiplier": 0.1,
            "min_position_size": 0.001,
            "max_position_size": max_size,
        }
    )
    calculator = VolatilityBasedCalculator()

    market_data = {"atr": 0.1, "returns": [0.0] * 100}
    result = calculator.calculate(gene, balance, price, market_data=market_data)

    assert result["position_size"] <= max_size + 1e-12


def test_half_optimal_f_fallback_to_simplified_when_trade_history_insufficient(
    base_gene: PositionSizingGene,
) -> None:
    balance = 10000.0
    price = 100.0
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.HALF_OPTIMAL_F,
            "optimal_f_multiplier": 0.5,
            "fixed_ratio": 0.02,
        }
    )
    calculator = HalfOptimalFCalculator()

    # 不足トレード履歴（< 10 件）: 簡易版計算経由
    short_history = [{"pnl": 100}] * 3
    result = calculator.calculate(gene, balance, price, trade_history=short_history)

    assert result["position_size"] >= gene.min_position_size
    assert "fallback_reason" in result["details"]
    assert (
        "insufficient_trade_history" in result["details"]["fallback_reason"]
        or "simplified_calculation" in result["details"]["fallback_reason"]
    )


def test_half_optimal_f_uses_trade_history_when_sufficient(
    base_gene: PositionSizingGene,
) -> None:
    balance = 10000.0
    price = 100.0
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.HALF_OPTIMAL_F,
            "optimal_f_multiplier": 0.5,
            "lookback_period": 20,
            "fixed_ratio": 0.02,
        }
    )
    calculator = HalfOptimalFCalculator()

    # 20件の履歴 (win: +200, loss: -100) で win_rate > 0.5 を作る
    trade_history = [{"pnl": 200}] * 15 + [{"pnl": -100}] * 5
    result = calculator.calculate(gene, balance, price, trade_history=trade_history)

    assert result["position_size"] > gene.min_position_size
    assert result["position_size"] > 0.0
    # details にオプティマルF関連情報が含まれることを期待
    assert "optimal_f" in result["details"]
    assert "half_optimal_f" in result["details"]


@pytest.mark.parametrize(
    "balance,price",
    [
        (-1000.0, 100.0),
        (0.0, 100.0),
        (1000.0, 0.0),
        (1000.0, -10.0),
    ],
)
def test_position_sizing_service_invalid_inputs_returns_error_result(
    base_gene: PositionSizingGene, balance: float, price: float
) -> None:
    gene = base_gene
    service = PositionSizingService()

    result: PositionSizingResult = service.calculate_position_size(
        gene=gene,
        account_balance=balance,
        current_price=price,
    )

    assert isinstance(result, PositionSizingResult)
    assert result.position_size == pytest.approx(0.01)
    assert result.confidence_score == 0.0
    assert result.warnings
    assert "error" in result.calculation_details


def test_position_sizing_service_integration_fixed_ratio(
    base_gene: PositionSizingGene,
) -> None:
    # サービス経由で FixedRatioCalculator が正しく利用されることを確認
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.FIXED_RATIO,
            "fixed_ratio": 0.1,
        }
    )
    service = PositionSizingService()

    result = service.calculate_position_size(
        gene=gene,
        account_balance=10000.0,
        current_price=100.0,
    )

    assert isinstance(result, PositionSizingResult)
    assert result.method_used == PositionSizingMethod.FIXED_RATIO.value
    assert result.position_size == pytest.approx(10.0)
    assert result.risk_metrics["position_value"] == pytest.approx(1000.0)
    assert result.risk_metrics["position_ratio"] == pytest.approx(0.1)
    assert result.timestamp <= datetime.now()


def test_calculator_factory_creates_expected_calculators() -> None:
    factory = CalculatorFactory()

    assert isinstance(
        factory.create_calculator(PositionSizingMethod.FIXED_RATIO.value),
        FixedRatioCalculator,
    )
    assert isinstance(
        factory.create_calculator(PositionSizingMethod.FIXED_QUANTITY.value),
        FixedQuantityCalculator,
    )
    assert isinstance(
        factory.create_calculator(PositionSizingMethod.VOLATILITY_BASED.value),
        VolatilityBasedCalculator,
    )
    assert isinstance(
        factory.create_calculator(PositionSizingMethod.HALF_OPTIMAL_F.value),
        HalfOptimalFCalculator,
    )
    # 不明なメソッドは FixedRatio にフォールバックする仕様に追随
    assert isinstance(factory.create_calculator("unknown_method"), FixedRatioCalculator)


def test_calculator_raises_or_handles_invalid_inputs_behavior_documented(
    base_gene: PositionSizingGene,
) -> None:
    """
    異常入力に対する挙動をドキュメント化するためのテスト。

    - PositionSizingService 側は負残高・ゼロ価格をエラー扱いし、PositionSizingResult(0.01, confidence_score=0.0) を返す。
    - 各 Calculator は BaseCalculator._safe_calculate_with_price_check に依存するため、
      ここでは「min_position_size 以上の非負サイズを返す」ことのみを保証対象とする。
    """
    service = PositionSizingService()
    gene = PositionSizingGene(
        **{
            **base_gene.__dict__,
            "method": PositionSizingMethod.FIXED_RATIO,
            "fixed_ratio": 0.1,
        }
    )

    # Service レベルの異常系
    res = service.calculate_position_size(
        gene=gene, account_balance=-1000.0, current_price=100.0
    )
    assert res.position_size == pytest.approx(0.01)
    assert res.confidence_score == 0.0

    # Calculator レベルの防御的挙動（仕様を固定しすぎない）
    calc = FixedRatioCalculator()
    raw = calc.calculate(gene, 10000.0, 0.0)
    assert raw["position_size"] >= gene.min_position_size
