"""
EntryDecisionEngine の追加ユニットテスト

既存テストファイルでカバーされていない以下の分岐を網羅します:
- calculate_position_size の各分岐（NaN、データ不足、フォールバックなど）
- calculate_effective_tpsl_prices の各分岐（ATRキャッシュヒット/ミス、固定メソッド）
- tools_block_entry の各分岐（ツールなし、無効化、ツール未登録、ツール例外）
- check_entry_conditions / _get_cached_entry_signal のエッジケース
- determine_entry_direction のステートフル分岐
- execute_entry のマーケット注文パス（ロング・ショート両方）
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.config.constants import (
    EntryType,
    PositionSizingMethod,
    TPSLMethod,
)
from app.services.auto_strategy.genes import (
    Condition,
    PositionSizingGene,
    StrategyGene,
    ToolGene,
    TPSLGene,
)
from app.services.auto_strategy.strategies.entry_decision_engine import (
    EntryDecisionEngine,
)
from app.services.auto_strategy.strategies.runtime_state import StrategyRuntimeState


def _make_position_sizing_gene(
    *,
    enabled: bool = True,
    method: PositionSizingMethod = PositionSizingMethod.FIXED_RATIO,
    min_size: float = 0.001,
    max_size: float = 9999.0,
    lookback: int = 14,
) -> PositionSizingGene:
    return PositionSizingGene(
        enabled=enabled,
        method=method,
        min_position_size=min_size,
        max_position_size=max_size,
        lookback_period=lookback,
    )


def _make_strategy(
    *,
    position_sizing_gene: PositionSizingGene | None = None,
    long_entry: list | None = None,
    short_entry: list | None = None,
    precomputed_signals: dict | None = None,
    precomputed_atr: np.ndarray | None = None,
    precomputed_tpsl_atr: dict | None = None,
    data_length: int = 30,
    tpsl_gene: TPSLGene | None = None,
    entry_gene=None,
    volatility_gate_enabled: bool = False,
    ml_predictor=None,
    tool_genes: list | None = None,
) -> SimpleNamespace:
    """テスト用 strategy モックを構築するヘルパー"""
    high = np.array([105.0 + i * 0.1 for i in range(data_length)])
    low = np.array([95.0 - i * 0.1 for i in range(data_length)])
    close = np.array([100.0 + i * 0.05 for i in range(data_length)])
    volume = np.array([1000.0 + i for i in range(data_length)])

    class _FakeData:
        def __init__(self, length):
            self._length = length
            self.High = high
            self.Low = low
            self.Close = close
            self.Volume = volume
            self.index = list(range(length))

        def __len__(self):
            return self._length

    data = _FakeData(data_length)

    gene = StrategyGene(
        indicators=[],
        long_entry_conditions=long_entry or [],
        short_entry_conditions=short_entry or [],
    )
    if position_sizing_gene is not None:
        gene.position_sizing_gene = position_sizing_gene
    if tool_genes is not None:
        gene.tool_genes = tool_genes

    strategy = SimpleNamespace(
        data=data,
        gene=gene,
        runtime_state=StrategyRuntimeState(),
        ml_predictor=ml_predictor,
        volatility_gate_enabled=volatility_gate_enabled,
        _precomputed_signals=precomputed_signals or {},
        _precomputed_atr=precomputed_atr,
        _precomputed_tpsl_atr=precomputed_tpsl_atr or {},
        _current_bar_index=7,
        stateful_conditions_evaluator=MagicMock(
            get_stateful_entry_direction=MagicMock(return_value=None)
        ),
        entry_executor=MagicMock(calculate_entry_params=MagicMock(return_value={})),
        order_manager=MagicMock(),
        buy=MagicMock(),
        sell=MagicMock(),
        condition_evaluator=MagicMock(),
        tpsl_service=MagicMock(
            calculate_tpsl_prices=MagicMock(return_value=(95.0, 110.0))
        ),
        position_sizing_service=MagicMock(
            calculate_position_size_fast=MagicMock(return_value=0.05)
        ),
        equity=100000.0,
        _ml_allows_entry=MagicMock(return_value=True),
        _get_effective_entry_gene=MagicMock(return_value=entry_gene),
        _get_effective_tpsl_gene=MagicMock(return_value=tpsl_gene),
    )
    return strategy


class TestCheckEntryConditions:
    """check_entry_conditions / _get_cached_entry_signal のテスト"""

    @pytest.fixture
    def strategy(self):
        return _make_strategy(
            long_entry=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
            short_entry=[
                Condition(left_operand="close", operator="<", right_operand=90.0)
            ],
        )

    @pytest.fixture
    def engine(self, strategy):
        return EntryDecisionEngine(strategy)

    def test_uses_cached_pandas_series(self, engine, strategy):
        # data_length 30 に合わせ、index 29 が True
        series = pd.Series([False] * 29 + [True])
        strategy._precomputed_signals = {1.0: series}
        # data_length 30、index = 29
        assert engine.check_entry_conditions(1.0) is True
        strategy.condition_evaluator.evaluate_conditions.assert_not_called()

    def test_uses_cached_ndarray(self, engine, strategy):
        arr = np.array([False] * 29 + [True])
        strategy._precomputed_signals = {1.0: arr}
        assert engine.check_entry_conditions(1.0) is True

    def test_uses_cached_false_signal(self, engine, strategy):
        arr = np.array([False] * 30)
        strategy._precomputed_signals = {1.0: arr}
        assert engine.check_entry_conditions(1.0) is False

    def test_skips_scalar_cached_signal(self, engine, strategy):
        # スカラーは len() がエラーになる
        strategy._precomputed_signals = {1.0: True}
        strategy.condition_evaluator.evaluate_conditions.return_value = False
        assert engine.check_entry_conditions(1.0) is False
        # フォールバックで evaluate_conditions が呼ばれる
        strategy.condition_evaluator.evaluate_conditions.assert_called_once()

    def test_returns_false_when_no_conditions(self, engine, strategy):
        strategy.gene.long_entry_conditions = []
        strategy._precomputed_signals = {}
        assert engine.check_entry_conditions(1.0) is False

    def test_returns_false_when_data_index_out_of_range(self, engine, strategy):
        # cached があるが index が範囲外
        arr = np.array([True])
        strategy._precomputed_signals = {1.0: arr}
        # データ長は 30 だが、シグナル長は 1
        assert engine._get_cached_entry_signal(1.0) is None

    def test_returns_none_when_cache_not_dict(self, engine, strategy):
        strategy._precomputed_signals = "not a dict"
        assert engine._get_cached_entry_signal(1.0) is None

    def test_returns_none_when_direction_missing(self, engine, strategy):
        strategy._precomputed_signals = {2.0: np.array([True])}
        assert engine._get_cached_entry_signal(1.0) is None

    def test_handles_getitem_exception(self, engine, strategy):
        # iloc/__getitem__ で例外を投げるオブジェクト
        class BadSignals:
            def __len__(self):
                return 30

            def __getitem__(self, idx):
                raise IndexError("bad")

        strategy._precomputed_signals = {1.0: BadSignals()}
        assert engine._get_cached_entry_signal(1.0) is None

    def test_falls_back_when_short_direction(self, engine, strategy):
        strategy._precomputed_signals = {-1.0: np.array([False] * 29 + [True])}
        assert engine.check_entry_conditions(-1.0) is True


class TestDetermineEntryDirection:
    """determine_entry_direction のテスト"""

    def test_returns_zero_when_tools_block(self):
        strategy = _make_strategy(
            tool_genes=[ToolGene(tool_name="weekend_filter", enabled=True)]
        )
        engine = EntryDecisionEngine(strategy)
        engine.tools_block_entry = MagicMock(return_value=True)
        engine.check_entry_conditions = MagicMock(return_value=False)
        assert engine.determine_entry_direction() == 0.0
        engine.check_entry_conditions.assert_not_called()

    def test_returns_long_when_long_conditions_match(self):
        strategy = _make_strategy()
        engine = EntryDecisionEngine(strategy)
        engine.tools_block_entry = MagicMock(return_value=False)
        engine.check_entry_conditions = MagicMock(side_effect=[True, False])
        assert engine.determine_entry_direction() == 1.0

    def test_returns_short_when_only_short_matches(self):
        strategy = _make_strategy()
        engine = EntryDecisionEngine(strategy)
        engine.tools_block_entry = MagicMock(return_value=False)
        engine.check_entry_conditions = MagicMock(side_effect=[False, True])
        assert engine.determine_entry_direction() == -1.0

    def test_uses_stateful_fallback(self):
        strategy = _make_strategy()
        strategy.stateful_conditions_evaluator.get_stateful_entry_direction.return_value = -1.0
        engine = EntryDecisionEngine(strategy)
        engine.tools_block_entry = MagicMock(return_value=False)
        engine.check_entry_conditions = MagicMock(side_effect=[False, False])
        assert engine.determine_entry_direction() == -1.0

    def test_returns_zero_when_stateful_returns_none(self):
        strategy = _make_strategy()
        strategy.stateful_conditions_evaluator.get_stateful_entry_direction.return_value = None
        engine = EntryDecisionEngine(strategy)
        engine.tools_block_entry = MagicMock(return_value=False)
        engine.check_entry_conditions = MagicMock(side_effect=[False, False])
        assert engine.determine_entry_direction() == 0.0


class TestExecuteEntry:
    """execute_entry の追加テスト"""

    def test_market_entry_for_short_uses_sell(self):
        strategy = _make_strategy()
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices = MagicMock(return_value=(95.0, 110.0))
        engine.calculate_position_size = MagicMock(return_value=0.5)

        result = engine.execute_entry(-1.0)

        assert result is True
        strategy.sell.assert_called_once_with(size=0.5)
        strategy.buy.assert_not_called()

    def test_market_entry_when_entry_gene_is_none(self):
        # entry_gene が None の場合は market 扱い
        strategy = _make_strategy(entry_gene=None)
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices = MagicMock(return_value=(None, None))
        engine.calculate_position_size = MagicMock(return_value=0.5)

        result = engine.execute_entry(1.0)

        assert result is True
        strategy.buy.assert_called_once_with(size=0.5)

    def test_market_entry_when_entry_gene_disabled(self):
        from app.services.auto_strategy.genes import EntryGene

        gene = EntryGene(entry_type=EntryType.MARKET, enabled=False)
        strategy = _make_strategy(entry_gene=gene)
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices = MagicMock(return_value=(None, None))
        engine.calculate_position_size = MagicMock(return_value=0.5)

        result = engine.execute_entry(1.0)

        assert result is True
        strategy.buy.assert_called_once_with(size=0.5)

    def test_returns_false_when_ml_blocks_long(self):
        ml = MagicMock()
        strategy = _make_strategy(volatility_gate_enabled=True, ml_predictor=ml)
        strategy._ml_allows_entry = MagicMock(return_value=False)
        engine = EntryDecisionEngine(strategy)
        result = engine.execute_entry(1.0)
        assert result is False
        strategy.buy.assert_not_called()
        strategy.order_manager.create_pending_order.assert_not_called()


class TestCalculatePositionSize:
    """calculate_position_size のテスト"""

    def test_returns_default_when_no_position_sizing_gene(self):
        strategy = _make_strategy(position_sizing_gene=None)
        engine = EntryDecisionEngine(strategy)
        assert engine.calculate_position_size() == 0.01

    def test_returns_default_when_position_sizing_disabled(self):
        gene = _make_position_sizing_gene(enabled=False)
        strategy = _make_strategy(position_sizing_gene=gene)
        engine = EntryDecisionEngine(strategy)
        assert engine.calculate_position_size() == 0.01

    def test_uses_precomputed_atr_pct(self):
        gene = _make_position_sizing_gene()
        atr = np.array([5.0] * 30)
        strategy = _make_strategy(position_sizing_gene=gene, precomputed_atr=atr)
        engine = EntryDecisionEngine(strategy)
        size = engine.calculate_position_size()
        # atr/current_price = 5/100 ≈ 0.05
        assert size > 0

    def test_skips_nan_atr_and_computes_manually(self):
        gene = _make_position_sizing_gene(lookback=14)
        atr = np.array([np.nan] * 30)
        strategy = _make_strategy(position_sizing_gene=gene, precomputed_atr=atr)
        engine = EntryDecisionEngine(strategy)
        size = engine.calculate_position_size()
        # 手動計算パスを通る
        assert size > 0

    def test_falls_back_when_atr_pct_not_computable(self):
        # precomputed_atr が None で、データも不足しているケース
        gene = _make_position_sizing_gene(lookback=14)
        strategy = _make_strategy(position_sizing_gene=gene, data_length=5)
        engine = EntryDecisionEngine(strategy)
        # precomputed_atr が None かつ lookback + 1 を超えないのでスキップ
        size = engine.calculate_position_size()
        assert size > 0

    def test_returns_min_fallback_when_position_size_invalid(self):
        gene = _make_position_sizing_gene()
        strategy = _make_strategy(position_sizing_gene=gene)
        strategy.position_sizing_service.calculate_position_size_fast.return_value = (
            float("nan")
        )
        engine = EntryDecisionEngine(strategy)
        assert engine.calculate_position_size() == 0.001

    def test_returns_min_fallback_when_position_size_negative(self):
        gene = _make_position_sizing_gene()
        strategy = _make_strategy(position_sizing_gene=gene)
        strategy.position_sizing_service.calculate_position_size_fast.return_value = (
            -0.5
        )
        engine = EntryDecisionEngine(strategy)
        assert engine.calculate_position_size() == 0.001

    def test_returns_integer_units_when_fraction_over_one(self):
        # size が大きく、fraction >= 1.0 になる場合
        gene = _make_position_sizing_gene(min_size=0.001, max_size=100.0)
        strategy = _make_strategy(position_sizing_gene=gene)
        # position_sizing_service が 50.0 を返す
        strategy.position_sizing_service.calculate_position_size_fast.return_value = (
            50.0
        )
        # current_price=100.0, equity=100000.0
        # fraction = 50 * 100 / 100000 = 0.05 (1 未満) → fraction を返す
        # 別の設定で fraction >= 1 にする
        strategy.equity = 1000.0
        # fraction = 50 * 100 / 1000 = 5.0 (1 以上) → floor(50) = 50
        engine = EntryDecisionEngine(strategy)
        assert engine.calculate_position_size() == 50.0

    def test_handles_invalid_equity(self):
        gene = _make_position_sizing_gene()
        strategy = _make_strategy(position_sizing_gene=gene)
        strategy.position_sizing_service.calculate_position_size_fast.return_value = 0.5
        strategy.equity = "not a number"
        engine = EntryDecisionEngine(strategy)
        # equity が float 変換できない場合はデフォルト 100000.0 扱い
        size = engine.calculate_position_size()
        assert size > 0

    def test_handles_zero_equity(self):
        gene = _make_position_sizing_gene()
        strategy = _make_strategy(position_sizing_gene=gene)
        strategy.position_sizing_service.calculate_position_size_fast.return_value = 0.5
        strategy.equity = 0.0
        engine = EntryDecisionEngine(strategy)
        # equity <= 0 の場合はフォールバック 0.001
        assert engine.calculate_position_size() == 0.001

    def test_handles_data_underflow(self):
        # Close が空の場合のフォールバック（50000.0 がデフォルト）
        gene = _make_position_sizing_gene()
        strategy = _make_strategy(position_sizing_gene=gene)
        strategy.data.Close = []
        engine = EntryDecisionEngine(strategy)
        size = engine.calculate_position_size()
        assert size > 0

    def test_handles_overall_exception(self):
        gene = _make_position_sizing_gene()
        strategy = _make_strategy(position_sizing_gene=gene)
        # attribute error になるよう設定
        strategy.gene.position_sizing_gene = None
        engine = EntryDecisionEngine(strategy)
        assert engine.calculate_position_size() == 0.01

    def test_uses_atr_from_data_when_precomputed_nan(self):
        # precomputed_atr に NaN がある場合は手動計算パスへ
        gene = _make_position_sizing_gene(lookback=10)
        atr = np.array([np.nan] * 30)
        strategy = _make_strategy(
            position_sizing_gene=gene, precomputed_atr=atr, data_length=30
        )
        engine = EntryDecisionEngine(strategy)
        size = engine.calculate_position_size()
        assert size > 0


class TestCalculateEffectiveTpslPrices:
    """calculate_effective_tpsl_prices のテスト"""

    def test_returns_none_when_no_tpsl_gene(self):
        strategy = _make_strategy(tpsl_gene=None)
        engine = EntryDecisionEngine(strategy)
        sl, tp = engine.calculate_effective_tpsl_prices(1.0, 100.0)
        assert sl is None
        assert tp is None

    def test_returns_none_tuple_for_risk_reward_method(self):
        # RISK_REWARD_RATIO は ATR 不要
        gene = TPSLGene(method=TPSLMethod.RISK_REWARD_RATIO, atr_period=14)
        strategy = _make_strategy(tpsl_gene=gene)
        engine = EntryDecisionEngine(strategy)
        sl, tp = engine.calculate_effective_tpsl_prices(1.0, 100.0)
        strategy.tpsl_service.calculate_tpsl_prices.assert_called_once()
        assert sl == 95.0
        assert tp == 110.0

    def test_uses_precomputed_atr_for_volatility_based(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        atr = np.array([2.0] * 30)
        strategy = _make_strategy(
            tpsl_gene=gene, precomputed_tpsl_atr={14: atr}, data_length=30
        )
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices(1.0, 100.0)
        # market_data に atr が含まれているはず
        call_kwargs = strategy.tpsl_service.calculate_tpsl_prices.call_args.kwargs
        assert call_kwargs["market_data"]["atr"] == 2.0

    def test_skips_nan_atr_value(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        atr = np.array([np.nan] * 30)
        strategy = _make_strategy(
            tpsl_gene=gene, precomputed_tpsl_atr={14: atr}, data_length=30
        )
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices(1.0, 100.0)
        # NaN のため ohlc_data パスへ
        call_kwargs = strategy.tpsl_service.calculate_tpsl_prices.call_args.kwargs
        assert "atr" not in call_kwargs["market_data"]
        assert "ohlc_data" in call_kwargs["market_data"]

    def test_uses_ohlc_data_when_atr_cache_missing(self):
        gene = TPSLGene(method=TPSLMethod.ADAPTIVE, atr_period=14)
        strategy = _make_strategy(tpsl_gene=gene, data_length=30)
        # precomputed_tpsl_atr が空 → ohlc_data パスへ
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices(-1.0, 100.0)
        call_kwargs = strategy.tpsl_service.calculate_tpsl_prices.call_args.kwargs
        assert "ohlc_data" in call_kwargs["market_data"]
        assert call_kwargs["position_direction"] == -1.0

    def test_uses_ohlc_data_when_data_too_short_for_atr_cache(self):
        # precomputed_tpsl_atr に該当 period がない
        gene = TPSLGene(method=TPSLMethod.STATISTICAL, atr_period=20)
        strategy = _make_strategy(
            tpsl_gene=gene, precomputed_tpsl_atr={14: np.array([1.0] * 30)}
        )
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices(1.0, 100.0)
        call_kwargs = strategy.tpsl_service.calculate_tpsl_prices.call_args.kwargs
        assert "ohlc_data" in call_kwargs["market_data"]

    def test_skips_ohlc_data_when_data_too_short(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=50)
        strategy = _make_strategy(tpsl_gene=gene, data_length=10)
        engine = EntryDecisionEngine(strategy)
        engine.calculate_effective_tpsl_prices(1.0, 100.0)
        call_kwargs = strategy.tpsl_service.calculate_tpsl_prices.call_args.kwargs
        # データ不足で ohlc_data は設定されない
        assert "ohlc_data" not in call_kwargs["market_data"]


class TestToolsBlockEntry:
    """tools_block_entry のテスト"""

    def test_returns_false_when_no_gene(self):
        strategy = _make_strategy()
        strategy.gene = None
        engine = EntryDecisionEngine(strategy)
        assert engine.tools_block_entry() is False

    def test_returns_false_when_no_tool_genes(self):
        strategy = _make_strategy(tool_genes=[])
        engine = EntryDecisionEngine(strategy)
        assert engine.tools_block_entry() is False

    def test_returns_false_when_tool_genes_attribute_missing(self):
        strategy = _make_strategy()
        # tool_genes 属性を削除
        delattr(strategy.gene, "tool_genes")
        engine = EntryDecisionEngine(strategy)
        assert engine.tools_block_entry() is False

    def test_skips_disabled_tool(self):
        tools = [ToolGene(tool_name="weekend_filter", enabled=False)]
        strategy = _make_strategy(tool_genes=tools)
        engine = EntryDecisionEngine(strategy)
        # 無効化ツールなので False
        assert engine.tools_block_entry() is False

    def test_returns_true_when_tool_blocks(self):
        # ツールをレジストリに登録して動作させる
        from app.services.auto_strategy.tools.base import BaseTool
        from app.services.auto_strategy.tools.registry import tool_registry

        class BlockingTool(BaseTool):
            tool_definition = None  # type: ignore

            @property
            def name(self) -> str:
                return "blocking_test_tool"

            def should_skip_entry(self, context, params):
                return True

        # tool_definition を直接設定
        from app.services.auto_strategy.tools.base import ToolDefinition

        BlockingTool.tool_definition = ToolDefinition(
            name="blocking_test_tool",
            description="",
            default_params={},
            priority="optional",
        )
        tool_registry.register(BlockingTool())

        try:
            tools = [ToolGene(tool_name="blocking_test_tool", enabled=True)]
            strategy = _make_strategy(tool_genes=tools)
            engine = EntryDecisionEngine(strategy)
            assert engine.tools_block_entry() is True
        finally:
            # クリーンアップ
            if "blocking_test_tool" in tool_registry._tools:
                del tool_registry._tools["blocking_test_tool"]

    def test_continues_when_tool_not_in_registry(self):
        tools = [ToolGene(tool_name="nonexistent_tool_xyz", enabled=True)]
        strategy = _make_strategy(tool_genes=tools)
        engine = EntryDecisionEngine(strategy)
        # レジストリにないツールはスキップされ、ブロックしない
        assert engine.tools_block_entry() is False

    def test_returns_false_on_exception(self):
        # data.index[-1] で例外を発生させる
        tools = [ToolGene(tool_name="any_tool", enabled=True)]
        strategy = _make_strategy(tool_genes=tools)
        strategy.data.index = None  # type: ignore[assignment]
        engine = EntryDecisionEngine(strategy)
        # 例外時はフェイルセーフで False
        assert engine.tools_block_entry() is False
