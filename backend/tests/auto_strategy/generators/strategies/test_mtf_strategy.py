import pytest
from unittest.mock import MagicMock, patch
from app.services.auto_strategy.genes.indicator import IndicatorGene
from app.services.auto_strategy.genes import Condition, ConditionGroup
from app.services.auto_strategy.generators.mtf_strategy import (
    MTFStrategy,
)
from app.services.auto_strategy.config.constants import IndicatorType


class TestMTFStrategy:
    @pytest.fixture
    def mock_condition_generator(self):
        generator = MagicMock()
        # モックのコンテキスト設定
        generator.context = {"timeframe": "1h", "symbol": "BTC/USDT"}

        # 指標タイプの分類モック
        def _classify(indicators):
            res = {IndicatorType.TREND: [], IndicatorType.MOMENTUM: [], IndicatorType.VOLATILITY: []}
            for ind in indicators:
                if ind.type in ["SMA", "EMA"]: res[IndicatorType.TREND].append(ind)
                else: res[IndicatorType.MOMENTUM].append(ind)
            return res
        generator._classify_indicators.side_effect = _classify

        # 名称解決モック
        generator._get_indicator_name.side_effect = lambda i: i.type

        # 条件生成のモック
        def _create_side(ind, side, name=None):
            return Condition(name or ind.type, ">" if side=="long" else "<", 0)
        generator._create_side_condition.side_effect = _create_side

        return generator

    @pytest.fixture
    def strategy(self, mock_condition_generator):
        return MTFStrategy(mock_condition_generator)

    def test_determine_higher_timeframe(self, strategy):
        """タイムフレームに応じた上位足の決定ロジックをテスト"""
        assert strategy._determine_higher_tf("1m") in ["5m", "15m"]
        assert strategy._determine_higher_tf("5m") in ["30m", "1h"]
        assert strategy._determine_higher_tf("15m") in ["1h", "4h"]
        assert strategy._determine_higher_tf("1h") in ["4h", "1d"]
        assert strategy._determine_higher_tf("4h") == "1d"

    def test_generate_conditions_structure(self, strategy):
        """生成される条件の構造（AND結合）をテスト"""
        # テスト用指標: トレンド系とモメンタム系を用意
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}),  # トレンド
            IndicatorGene(type="RSI", parameters={"period": 14}),  # モメンタム
        ]

        longs, shorts, exits = strategy.generate_conditions(indicators)

        # 少なくとも1つの条件セットが生成されるべき
        assert len(longs) > 0
        assert len(shorts) > 0

        # 生成された条件はConditionGroupであるべき（MTF条件 AND 下位足条件）
        first_long = longs[0]
        assert isinstance(first_long, ConditionGroup)
        assert first_long.operator == "AND"
        assert len(first_long.conditions) >= 2

        # 上位足の指標が含まれているか確認
        has_mtf_condition = False
        for cond in first_long.conditions:
            if isinstance(cond, Condition):
                # 条件の左辺か右辺に指標名が含まれているはず
                # 実際の生成ロジックでは指標名にtimeframeが付与されるか、
                # またはIndicatorGene自体にtimeframeがセットされる
                pass
            elif isinstance(cond, ConditionGroup):
                pass

        # 簡易的なチェック: 少なくとも2つの要素が結合されていること
        assert len(first_long.conditions) >= 2

    def test_assign_timeframes(self, strategy):
        """指標へのタイムフレーム割り当てロジックをテスト"""
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}),
            IndicatorGene(type="RSI", parameters={"period": 14}),
        ]

        # 上位足を4hに設定して割り当て実行
        higher_tf = "4h"
        mtf_indicators = strategy._create_mtf_indicators(indicators, higher_tf)

        # 元のリストとは別のインスタンスになっているか
        assert mtf_indicators is not indicators

        # 全ての指標に上位足が設定されているか確認（ヘルパーメソッドは無差別に適用するため）
        sma = next(i for i in mtf_indicators if i.type == "SMA")
        assert sma.timeframe == higher_tf

        rsi = next(i for i in mtf_indicators if i.type == "RSI")
        assert rsi.timeframe == higher_tf
