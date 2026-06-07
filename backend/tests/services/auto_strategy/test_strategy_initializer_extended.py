"""
StrategyInitializer の追加ユニットテスト

既存テストファイル (test_strategy_initializer.py) でカバーされていない
事前計算系の分岐（ATR計算、TP/SL ATR計算、信号キャッシュ等）を網羅します。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.config.constants import (
    PositionSizingMethod,
    TPSLMethod,
)
from app.services.auto_strategy.genes import (
    Condition,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)
from app.services.auto_strategy.strategies.strategy_initializer import (
    StrategyInitializer,
)


def _make_strategy(
    *,
    position_sizing_gene: PositionSizingGene | None = None,
    indicators: list | None = None,
    long_entry: list | None = None,
    short_entry: list | None = None,
    long_exit: list | None = None,
    short_exit: list | None = None,
    tpsl_gene_long: TPSLGene | None = None,
    tpsl_gene_short: TPSLGene | None = None,
    data_length: int = 3,
    has_data_df: bool = True,
    calc_vectorized_side_effect: list | None = None,
) -> SimpleNamespace:
    """テスト用 strategy モックを構築するヘルパー"""
    data = SimpleNamespace(
        High=np.array([105.0, 106.0, 107.0]),
        Low=np.array([95.0, 96.0, 97.0]),
        Close=np.array([100.0, 101.0, 102.0]),
    )
    if has_data_df:
        data.df = pd.DataFrame(
            {
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [100.0, 101.0, 102.0],
            }
        )
    data.__len__ = MagicMock(return_value=data_length)
    data.__getitem__ = MagicMock(return_value=data)

    gene = StrategyGene(
        indicators=indicators or [],
        long_entry_conditions=long_entry or [],
        short_entry_conditions=short_entry or [],
        long_exit_conditions=long_exit or [],
        short_exit_conditions=short_exit or [],
    )
    if position_sizing_gene is not None:
        gene.position_sizing_gene = position_sizing_gene

    def get_effective_tpsl(direction: float):
        if direction > 0:
            return tpsl_gene_long
        return tpsl_gene_short

    strategy = SimpleNamespace(
        data=data,
        gene=gene,
        indicator_calculator=SimpleNamespace(init_indicator=MagicMock()),
        ml_filter=SimpleNamespace(precompute_ml_features=MagicMock()),
        condition_evaluator=SimpleNamespace(
            calculate_conditions_vectorized=MagicMock(
                side_effect=calc_vectorized_side_effect or [np.array([True])]
            )
        ),
        volatility_gate_enabled=False,
        ml_predictor=None,
        _precomputed_signals={},
        _precomputed_exit_signals={},
        _precomputed_atr=None,
        _precomputed_tpsl_atr=None,
        _get_effective_tpsl_gene=MagicMock(side_effect=get_effective_tpsl),
    )
    return strategy


class TestInitIndicator:
    """init_indicator のテスト"""

    def test_init_indicator_delegates_to_calculator(self):
        strategy = _make_strategy()
        strategy.indicator_calculator.init_indicator = MagicMock()

        from app.services.auto_strategy.genes import IndicatorGene

        gene = IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        initializer = StrategyInitializer(strategy)

        initializer.init_indicator(gene)

        strategy.indicator_calculator.init_indicator.assert_called_once_with(
            gene, strategy
        )

    def test_init_indicator_raises_when_calculator_fails(self):
        strategy = _make_strategy()
        strategy.indicator_calculator.init_indicator = MagicMock(
            side_effect=Exception("calc error")
        )

        from app.services.auto_strategy.genes import IndicatorGene

        gene = IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        initializer = StrategyInitializer(strategy)

        with pytest.raises(Exception, match="calc error"):
            initializer.init_indicator(gene)

    def test_initialize_returns_early_when_no_gene(self):
        strategy = _make_strategy()
        strategy.gene = None
        initializer = StrategyInitializer(strategy)
        # 例外なく早期 return
        initializer.initialize()


class TestInitializeErrorPath:
    """initialize の例外伝搬テスト"""

    def test_initialize_raises_when_indicator_init_fails(self):
        from app.services.auto_strategy.genes import IndicatorGene

        strategy = _make_strategy(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]
        )
        strategy.indicator_calculator.init_indicator = MagicMock(
            side_effect=Exception("init failed")
        )
        initializer = StrategyInitializer(strategy)
        with pytest.raises(Exception, match="init failed"):
            initializer.initialize()


class TestGetOrCreateSignalCache:
    """_get_or_create_signal_cache のテスト"""

    def test_creates_cache_when_attribute_missing(self):
        strategy = SimpleNamespace()
        # _precomputed_signals は存在しない
        assert not hasattr(strategy, "_precomputed_signals")

        initializer = StrategyInitializer(strategy)
        cache = initializer._get_or_create_signal_cache("_precomputed_signals")

        assert cache == {}
        assert strategy._precomputed_signals is cache

    def test_reuses_existing_dict_cache(self):
        existing = {1.0: "x"}
        strategy = SimpleNamespace(_precomputed_signals=existing)

        initializer = StrategyInitializer(strategy)
        cache = initializer._get_or_create_signal_cache("_precomputed_signals")

        assert cache is existing

    def test_replaces_non_dict_cache(self):
        # 既存の attr が dict でない場合は dict に置き換える
        strategy = SimpleNamespace(_precomputed_signals="not a dict")

        initializer = StrategyInitializer(strategy)
        cache = initializer._get_or_create_signal_cache("_precomputed_signals")

        assert cache == {}
        assert isinstance(strategy._precomputed_signals, dict)


class TestCacheVectorizedSignal:
    """_cache_vectorized_signal のテスト"""

    def test_skips_none(self):
        strategy = _make_strategy()
        initializer = StrategyInitializer(strategy)
        cache: dict = {}
        initializer._cache_vectorized_signal(cache, 1.0, None)
        assert cache == {}

    def test_caches_pandas_series(self):
        strategy = _make_strategy()
        initializer = StrategyInitializer(strategy)
        cache: dict = {}
        series = pd.Series([True, False, True])
        initializer._cache_vectorized_signal(cache, 1.0, series)
        assert cache[1.0] is series

    def test_caches_ndarray(self):
        strategy = _make_strategy()
        initializer = StrategyInitializer(strategy)
        cache: dict = {}
        arr = np.array([True, False, True])
        initializer._cache_vectorized_signal(cache, -1.0, arr)
        assert cache[-1.0] is arr

    def test_skips_scalar_value(self):
        strategy = _make_strategy()
        initializer = StrategyInitializer(strategy)
        cache: dict = {}
        initializer._cache_vectorized_signal(cache, 1.0, True)
        assert cache == {}


class TestPrecomputePositionSizingAtr:
    """_precompute_position_sizing_atr のテスト"""

    def test_skips_when_no_position_sizing_gene(self):
        strategy = _make_strategy(position_sizing_gene=None)
        initializer = StrategyInitializer(strategy)
        initializer._precompute_position_sizing_atr()
        assert strategy._precomputed_atr is None

    def test_skips_when_position_sizing_disabled(self):
        gene = PositionSizingGene(
            enabled=False, method=PositionSizingMethod.VOLATILITY_BASED
        )
        strategy = _make_strategy(position_sizing_gene=gene)
        initializer = StrategyInitializer(strategy)
        initializer._precompute_position_sizing_atr()
        assert strategy._precomputed_atr is None

    def test_computes_atr_with_pandas_ta(self):
        gene = PositionSizingGene(
            enabled=True,
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=14,
        )
        strategy = _make_strategy(position_sizing_gene=gene)

        mock_atr = np.array([1.5, 2.0, 2.5])

        with patch.dict(
            "sys.modules",
            {
                "pandas_ta_classic": MagicMock(
                    atr=MagicMock(return_value=pd.Series(mock_atr))
                )
            },
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_position_sizing_atr()

        np.testing.assert_array_equal(strategy._precomputed_atr, mock_atr)

    def test_handles_pandas_ta_returning_none(self):
        gene = PositionSizingGene(
            enabled=True,
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=14,
        )
        strategy = _make_strategy(position_sizing_gene=gene)

        with patch.dict(
            "sys.modules",
            {"pandas_ta_classic": MagicMock(atr=MagicMock(return_value=None))},
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_position_sizing_atr()

        # ATR 計算が None の場合は _precomputed_atr が None のまま
        assert strategy._precomputed_atr is None

    def test_handles_pandas_ta_import_error(self):
        gene = PositionSizingGene(
            enabled=True,
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=14,
        )
        strategy = _make_strategy(position_sizing_gene=gene)

        # pandas_ta_classic が存在しない状態をシミュレート
        with patch.dict("sys.modules", {"pandas_ta_classic": None}):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_position_sizing_atr()
        # ImportError 発生時もクラッシュせず、_precomputed_atr は None
        assert strategy._precomputed_atr is None

    def test_handles_atr_computation_error(self):
        gene = PositionSizingGene(
            enabled=True,
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=14,
        )
        strategy = _make_strategy(position_sizing_gene=gene)

        with patch.dict(
            "sys.modules",
            {
                "pandas_ta_classic": MagicMock(
                    atr=MagicMock(side_effect=Exception("atr error"))
                )
            },
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_position_sizing_atr()
        # 例外時もクラッシュしない
        assert strategy._precomputed_atr is None


class TestPrecomputeTpslAtr:
    """_precompute_tpsl_atr のテスト"""

    def test_initializes_empty_dict(self):
        strategy = _make_strategy()
        strategy._precomputed_tpsl_atr = "sentinel"
        initializer = StrategyInitializer(strategy)
        initializer._precompute_tpsl_atr()
        assert strategy._precomputed_tpsl_atr == {}

    def test_skips_when_no_tpsl_gene(self):
        strategy = _make_strategy(tpsl_gene_long=None, tpsl_gene_short=None)
        initializer = StrategyInitializer(strategy)
        initializer._precompute_tpsl_atr()
        # tpsl_gene が両方 None なら何もしない
        assert strategy._precomputed_tpsl_atr == {}

    def test_skips_when_method_is_risk_reward(self):
        gene = TPSLGene(method=TPSLMethod.RISK_REWARD_RATIO, atr_period=14)
        strategy = _make_strategy(tpsl_gene_long=gene)
        initializer = StrategyInitializer(strategy)
        initializer._precompute_tpsl_atr()
        # RISK_REWARD_RATIO は ATR 不要
        assert strategy._precomputed_tpsl_atr == {}

    def test_computes_atr_for_volatility_based(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        strategy = _make_strategy(tpsl_gene_long=gene)

        mock_atr = np.array([1.5, 2.0, 2.5])

        with patch.dict(
            "sys.modules",
            {
                "pandas_ta_classic": MagicMock(
                    atr=MagicMock(return_value=pd.Series(mock_atr))
                )
            },
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_tpsl_atr()

        np.testing.assert_array_equal(strategy._precomputed_tpsl_atr[14], mock_atr)

    def test_computes_atr_for_adaptive_method(self):
        gene = TPSLGene(method=TPSLMethod.ADAPTIVE, atr_period=20)
        strategy = _make_strategy(tpsl_gene_short=gene)

        mock_atr = np.array([1.0, 1.5, 2.0])

        with patch.dict(
            "sys.modules",
            {
                "pandas_ta_classic": MagicMock(
                    atr=MagicMock(return_value=pd.Series(mock_atr))
                )
            },
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_tpsl_atr()

        np.testing.assert_array_equal(strategy._precomputed_tpsl_atr[20], mock_atr)

    def test_computes_atr_for_statistical_method(self):
        gene = TPSLGene(method=TPSLMethod.STATISTICAL, atr_period=10)
        strategy = _make_strategy(tpsl_gene_long=gene)

        mock_atr = np.array([0.5, 0.8, 1.0])

        with patch.dict(
            "sys.modules",
            {
                "pandas_ta_classic": MagicMock(
                    atr=MagicMock(return_value=pd.Series(mock_atr))
                )
            },
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_tpsl_atr()

        np.testing.assert_array_equal(strategy._precomputed_tpsl_atr[10], mock_atr)

    def test_reuses_cached_atr_for_same_period(self):
        # 両方向で同じ atr_period の場合、2 回目は計算しない
        gene_long = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        gene_short = TPSLGene(method=TPSLMethod.ADAPTIVE, atr_period=14)
        strategy = _make_strategy(tpsl_gene_long=gene_long, tpsl_gene_short=gene_short)

        mock_atr = np.array([1.5, 2.0, 2.5])
        mock_ta = MagicMock(atr=MagicMock(return_value=pd.Series(mock_atr)))

        with patch.dict("sys.modules", {"pandas_ta_classic": mock_ta}):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_tpsl_atr()

        # atr は 1 回だけ計算される
        assert mock_ta.atr.call_count == 1
        np.testing.assert_array_equal(strategy._precomputed_tpsl_atr[14], mock_atr)

    def test_handles_pandas_ta_returning_none(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        strategy = _make_strategy(tpsl_gene_long=gene)

        with patch.dict(
            "sys.modules",
            {"pandas_ta_classic": MagicMock(atr=MagicMock(return_value=None))},
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_tpsl_atr()
        # ATR が None でもクラッシュしない
        assert 14 not in strategy._precomputed_tpsl_atr

    def test_handles_pandas_ta_exception(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        strategy = _make_strategy(tpsl_gene_long=gene)

        with patch.dict(
            "sys.modules",
            {
                "pandas_ta_classic": MagicMock(
                    atr=MagicMock(side_effect=Exception("boom"))
                )
            },
        ):
            initializer = StrategyInitializer(strategy)
            initializer._precompute_tpsl_atr()
        # 例外時もクラッシュしない
        assert 14 not in strategy._precomputed_tpsl_atr

    def test_skips_when_data_has_no_df_attribute(self):
        gene = TPSLGene(method=TPSLMethod.VOLATILITY_BASED, atr_period=14)
        strategy = _make_strategy(tpsl_gene_long=gene, has_data_df=False)
        initializer = StrategyInitializer(strategy)
        initializer._precompute_tpsl_atr()
        # data.df がない場合は ATR 計算をスキップ
        assert 14 not in strategy._precomputed_tpsl_atr


class TestPrecomputeConditionSignals:
    """_precompute_condition_signals の追加テスト"""

    def test_handles_exception_in_vectorized_calculation(self):
        strategy = _make_strategy(
            long_entry=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
            calc_vectorized_side_effect=Exception("calc error"),
        )
        initializer = StrategyInitializer(strategy)
        # 例外時もクラッシュしない（フォールバック）
        initializer._precompute_condition_signals()

    def test_caches_pandas_series_signals(self):
        signals = [pd.Series([True, False, True]), pd.Series([False, True, False])]
        strategy = _make_strategy(
            long_entry=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
            short_entry=[
                Condition(left_operand="close", operator="<", right_operand=90.0)
            ],
            calc_vectorized_side_effect=signals,
        )
        initializer = StrategyInitializer(strategy)
        initializer._precompute_condition_signals()

        assert 1.0 in strategy._precomputed_signals
        assert -1.0 in strategy._precomputed_signals

    def test_caches_exit_signals(self):
        signals = [
            pd.Series([True, False, True]),  # long entry
            pd.Series([False, True, False]),  # short entry
            pd.Series([True, True, False]),  # long exit
            pd.Series([False, False, True]),  # short exit
        ]
        strategy = _make_strategy(
            long_entry=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
            short_entry=[
                Condition(left_operand="close", operator="<", right_operand=90.0)
            ],
            long_exit=[
                Condition(left_operand="close", operator=">", right_operand=110.0)
            ],
            short_exit=[
                Condition(left_operand="close", operator="<", right_operand=80.0)
            ],
            calc_vectorized_side_effect=signals,
        )
        initializer = StrategyInitializer(strategy)
        initializer._precompute_condition_signals()

        assert 1.0 in strategy._precomputed_signals
        assert -1.0 in strategy._precomputed_signals
        assert 1.0 in strategy._precomputed_exit_signals
        assert -1.0 in strategy._precomputed_exit_signals
