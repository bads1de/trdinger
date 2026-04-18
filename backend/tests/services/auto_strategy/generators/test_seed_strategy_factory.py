"""
SeedStrategyFactory のユニットテスト

シード戦略が正しく生成され、必要な構成要素を持つことを検証します。
"""

import pytest

from app.services.auto_strategy.generators.seed_strategy_factory import (
    SeedStrategyFactory,
    inject_seeds_into_population,
)
from app.services.auto_strategy.genes.conditions import Condition, ConditionGroup
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.genes.tpsl import TPSLGene


def _single_output_ref_name(strategy: StrategyGene, indicator_type: str) -> str:
    indicator = next(i for i in strategy.indicators if i.type == indicator_type)
    return f"{indicator.type}_{indicator.id[:8]}"


def _multi_output_ref_name(
    strategy: StrategyGene, indicator_type: str, output_index: int
) -> str:
    indicator = next(i for i in strategy.indicators if i.type == indicator_type)
    return f"{indicator.type}_{indicator.id[:8]}_{output_index}"


class TestSeedStrategyFactory:
    """SeedStrategyFactory のテストクラス"""

    def test_get_all_seeds_returns_six_strategies(self):
        """get_all_seeds が6つのシード戦略を返すこと"""
        seeds = SeedStrategyFactory.get_all_seeds()

        assert len(seeds) == 6
        assert all(isinstance(s, StrategyGene) for s in seeds)

    def test_get_all_seeds_has_unique_names(self):
        """各シード戦略がユニークな名前を持つこと"""
        seeds = SeedStrategyFactory.get_all_seeds()
        names = [s.metadata.get("seed_strategy") for s in seeds]

        assert len(names) == len(set(names)), "シード戦略名が重複しています"

    def test_get_all_seeds_names(self):
        """シード戦略が期待される名前を持つこと"""
        seeds = SeedStrategyFactory.get_all_seeds()
        expected_names = {
            "dmi_extreme_trend",
            "rsi_momentum",
            "bollinger_breakout",
            "kama_adx_hybrid",
            "wae",
            "trendilo",
        }
        actual_names = {s.metadata.get("seed_strategy") for s in seeds}

        assert actual_names == expected_names

    @pytest.mark.parametrize(
        "name",
        [
            "dmi_extreme",
            "rsi_momentum",
            "bollinger_breakout",
            "kama_adx",
            "wae",
            "trendilo",
        ],
    )
    def test_get_seed_by_name(self, name: str):
        """get_seed_by_name が正しい戦略を返すこと"""
        seed = SeedStrategyFactory.get_seed_by_name(name)

        assert seed is not None
        assert isinstance(seed, StrategyGene)

    def test_get_seed_by_name_returns_none_for_unknown(self):
        """get_seed_by_name が未知の名前に対してNoneを返すこと"""
        seed = SeedStrategyFactory.get_seed_by_name("unknown_strategy")

        assert seed is None


class TestDMIExtremeTrend:
    """DMI Extreme Trend 戦略のテスト"""

    def test_has_adx_indicator(self):
        """ADX指標が含まれること"""
        strategy = SeedStrategyFactory.create_dmi_extreme_trend()

        assert len(strategy.indicators) >= 1
        adx_indicators = [i for i in strategy.indicators if i.type == "ADX"]
        assert len(adx_indicators) == 1

    def test_has_long_conditions(self):
        """ロングエントリー条件が設定されていること"""
        strategy = SeedStrategyFactory.create_dmi_extreme_trend()

        assert len(strategy.long_entry_conditions) > 0

    def test_has_short_conditions(self):
        """ショートエントリー条件が設定されていること"""
        strategy = SeedStrategyFactory.create_dmi_extreme_trend()

        assert len(strategy.short_entry_conditions) > 0

    def test_has_tpsl_gene(self):
        """TPSLGeneが設定されていること"""
        strategy = SeedStrategyFactory.create_dmi_extreme_trend()

        assert strategy.tpsl_gene is not None
        assert isinstance(strategy.tpsl_gene, TPSLGene)
        assert strategy.tpsl_gene.enabled is True

    def test_uses_runtime_indicator_reference_names(self):
        """条件が実行時の指標登録名を参照すること"""
        strategy = SeedStrategyFactory.create_dmi_extreme_trend()
        long_group = strategy.long_entry_conditions[0]
        short_group = strategy.short_entry_conditions[0]

        assert long_group.conditions[0].left_operand["name"] == _multi_output_ref_name(
            strategy, "ADX", 1
        )
        assert long_group.conditions[1].left_operand["name"] == _multi_output_ref_name(
            strategy, "ADX", 0
        )
        assert short_group.conditions[0].left_operand["name"] == _multi_output_ref_name(
            strategy, "ADX", 2
        )
        assert short_group.conditions[1].left_operand["name"] == _multi_output_ref_name(
            strategy, "ADX", 0
        )


class TestRSIMomentum:
    """RSI Momentum 戦略のテスト"""

    def test_has_rsi_indicator(self):
        """RSI指標が含まれること"""
        strategy = SeedStrategyFactory.create_rsi_momentum()

        rsi_indicators = [i for i in strategy.indicators if i.type == "RSI"]
        assert len(rsi_indicators) == 1

    def test_long_condition_threshold(self):
        """ロング条件がRSI > 75であること"""
        strategy = SeedStrategyFactory.create_rsi_momentum()

        assert len(strategy.long_entry_conditions) > 0
        condition = strategy.long_entry_conditions[0]
        assert isinstance(condition, Condition)
        assert condition.right_operand == 75.0
        assert condition.operator == ">"


class TestBollingerBreakout:
    """Bollinger Breakout 戦略のテスト"""

    def test_has_bbands_indicator(self):
        """BBANDS指標が含まれること"""
        strategy = SeedStrategyFactory.create_bollinger_breakout()

        bb_indicators = [i for i in strategy.indicators if i.type == "BBANDS"]
        assert len(bb_indicators) == 1

    def test_has_close_comparison(self):
        """Close価格との比較条件があること"""
        strategy = SeedStrategyFactory.create_bollinger_breakout()

        # ロング条件を確認
        long_cond = strategy.long_entry_conditions[0]
        assert isinstance(long_cond, Condition)
        assert long_cond.left_operand == "close"

    def test_uses_registered_band_names(self):
        """バンド参照が実行時の登録名を使うこと"""
        strategy = SeedStrategyFactory.create_bollinger_breakout()

        assert strategy.long_entry_conditions[0].right_operand[
            "name"
        ] == _multi_output_ref_name(strategy, "BBANDS", 0)
        assert strategy.short_entry_conditions[0].right_operand[
            "name"
        ] == _multi_output_ref_name(strategy, "BBANDS", 2)


class TestKAMAADXHybrid:
    """KAMA-ADX Hybrid 戦略のテスト"""

    def test_has_both_indicators(self):
        """KAMAとADXの両方の指標が含まれること"""
        strategy = SeedStrategyFactory.create_kama_adx_hybrid()

        indicator_types = {i.type for i in strategy.indicators}
        assert "KAMA" in indicator_types
        assert "ADX" in indicator_types

    def test_condition_group_is_and(self):
        """条件がANDグループであること"""
        strategy = SeedStrategyFactory.create_kama_adx_hybrid()

        long_cond = strategy.long_entry_conditions[0]
        assert isinstance(long_cond, ConditionGroup)
        assert long_cond.operator == "AND"
        assert len(long_cond.conditions) >= 3  # Close > KAMA AND DMP > 40 AND ADX > 20


class TestWAE:
    """WAE 戦略のテスト"""

    def test_has_required_indicators(self):
        """MACD, BBANDS, ATRが含まれること"""
        strategy = SeedStrategyFactory.create_wae()

        indicator_types = {i.type for i in strategy.indicators}
        assert "MACD" in indicator_types
        assert "BBANDS" in indicator_types
        assert "ATR" in indicator_types

    def test_uses_registered_macd_names(self):
        """MACD条件が実行時の登録名を使うこと"""
        strategy = SeedStrategyFactory.create_wae()
        long_group = strategy.long_entry_conditions[0]
        short_group = strategy.short_entry_conditions[0]

        assert long_group.conditions[0].left_operand["name"] == _multi_output_ref_name(
            strategy, "MACD", 0
        )
        assert long_group.conditions[0].right_operand["name"] == _multi_output_ref_name(
            strategy, "MACD", 1
        )
        assert short_group.conditions[0].left_operand["name"] == _multi_output_ref_name(
            strategy, "MACD", 0
        )
        assert short_group.conditions[0].right_operand[
            "name"
        ] == _multi_output_ref_name(strategy, "MACD", 1)


class TestTrendilo:
    """Trendilo 戦略のテスト"""

    def test_has_t3_and_adx(self):
        """T3とADXが含まれること"""
        strategy = SeedStrategyFactory.create_trendilo()

        indicator_types = {i.type for i in strategy.indicators}
        assert "T3" in indicator_types
        assert "ADX" in indicator_types


class TestInjectSeedsIntoPopulation:
    """inject_seeds_into_population 関数のテスト"""

    def test_inject_with_zero_rate(self):
        """注入率0の場合、集団が変更されないこと"""
        population = ["dummy1", "dummy2", "dummy3"]
        result = inject_seeds_into_population(population, seed_injection_rate=0.0)

        assert result == population

    def test_inject_replaces_first_elements(self):
        """注入されたシードが集団の先頭を置き換えること"""
        population = [f"dummy_{i}" for i in range(20)]
        result = inject_seeds_into_population(population, seed_injection_rate=0.3)

        # 20 * 0.3 = 6 個注入されるが、シードは6個までしかない
        for i in range(6):
            assert isinstance(result[i], StrategyGene)

        # 残りはそのまま
        for i in range(6, 20):
            assert result[i] == f"dummy_{i}"

    def test_default_injection_rate_is_30_percent(self, monkeypatch):
        """既定のシード注入率が30%であること"""
        population = [f"dummy_{i}" for i in range(10)]
        seeds = [
            StrategyGene(metadata={"seed_strategy": f"seed_{i}"}) for i in range(10)
        ]

        monkeypatch.setattr(
            SeedStrategyFactory,
            "get_all_seeds",
            staticmethod(lambda: seeds),
        )

        result = inject_seeds_into_population(population)

        injected_count = sum(1 for item in result if isinstance(item, StrategyGene))
        assert injected_count == 3

    def test_inject_respects_max_seeds(self):
        """シード数以上は注入されないこと"""
        population = [f"dummy_{i}" for i in range(100)]
        result = inject_seeds_into_population(population, seed_injection_rate=0.5)

        # 100 * 0.5 = 50 個だが、シードは6個しかない
        injected_count = sum(1 for item in result if isinstance(item, StrategyGene))
        assert injected_count == 6

    def test_inject_with_small_population(self):
        """小さな集団でも正しく動作すること"""
        population = ["dummy1", "dummy2"]
        result = inject_seeds_into_population(population, seed_injection_rate=0.5)

        # 2 * 0.5 = 1 個注入
        assert isinstance(result[0], StrategyGene)
        assert result[1] == "dummy2"


class TestStrategyValidation:
    """生成された戦略の妥当性テスト"""

    @pytest.mark.parametrize(
        "factory_method",
        [
            SeedStrategyFactory.create_dmi_extreme_trend,
            SeedStrategyFactory.create_rsi_momentum,
            SeedStrategyFactory.create_bollinger_breakout,
            SeedStrategyFactory.create_kama_adx_hybrid,
            SeedStrategyFactory.create_wae,
            SeedStrategyFactory.create_trendilo,
        ],
    )
    def test_strategy_has_valid_structure(self, factory_method):
        """各戦略が有効な構造を持つこと"""
        strategy = factory_method()

        # 基本構造の検証
        assert strategy.id is not None
        assert len(strategy.indicators) > 0
        assert len(strategy.long_entry_conditions) > 0
        assert len(strategy.short_entry_conditions) > 0
        assert strategy.metadata.get("seed_strategy") is not None
        assert strategy.metadata.get("version") is not None

    @pytest.mark.parametrize(
        "factory_method",
        [
            SeedStrategyFactory.create_dmi_extreme_trend,
            SeedStrategyFactory.create_rsi_momentum,
            SeedStrategyFactory.create_bollinger_breakout,
            SeedStrategyFactory.create_kama_adx_hybrid,
            SeedStrategyFactory.create_wae,
            SeedStrategyFactory.create_trendilo,
        ],
    )
    def test_strategy_can_be_cloned(self, factory_method):
        """各戦略がclone可能であること"""
        strategy = factory_method()
        cloned = strategy.clone()

        assert cloned.id != strategy.id
        assert len(cloned.indicators) == len(strategy.indicators)
        assert cloned.metadata.get("seed_strategy") == strategy.metadata.get(
            "seed_strategy"
        )


class TestHelperMethods:
    """ヘルパーメソッドのテスト"""

    def test_create_volatility_based_tpsl(self):
        """ボラティリティベースTPSL生成ヘルパーのテスト"""
        tpsl = SeedStrategyFactory._create_volatility_based_tpsl(
            sl_pct=0.02,
            tp_pct=0.04,
            atr_sl_multiplier=2.0,
            atr_tp_multiplier=4.0,
            atr_period=14,
        )

        assert isinstance(tpsl, TPSLGene)
        assert tpsl.stop_loss_pct == 0.02
        assert tpsl.take_profit_pct == 0.04
        assert tpsl.atr_multiplier_sl == 2.0
        assert tpsl.atr_multiplier_tp == 4.0
        assert tpsl.atr_period == 14
        assert tpsl.enabled is True

    def test_create_risk_reward_tpsl(self):
        """リスクリワード比率TPSL生成ヘルパーのテスト"""
        tpsl = SeedStrategyFactory._create_risk_reward_tpsl(
            sl_pct=0.03,
            tp_pct=0.06,
            risk_reward_ratio=2.0,
        )

        assert isinstance(tpsl, TPSLGene)
        assert tpsl.stop_loss_pct == 0.03
        assert tpsl.take_profit_pct == 0.06
        assert tpsl.risk_reward_ratio == 2.0
        assert tpsl.enabled is True

    def test_create_simple_condition(self):
        """単一条件生成ヘルパーのテスト"""
        condition = SeedStrategyFactory._create_simple_condition(
            left_operand="close",
            operator=">",
            right_operand=100.0,
            direction="long",
        )

        assert isinstance(condition, Condition)
        assert condition.left_operand == "close"
        assert condition.operator == ">"
        assert condition.right_operand == 100.0
        assert condition.direction == "long"

    def test_create_and_condition_group(self):
        """AND条件グループ生成ヘルパーのテスト"""
        cond1 = SeedStrategyFactory._create_simple_condition(
            left_operand="close",
            operator=">",
            right_operand=100.0,
            direction="long",
        )
        cond2 = SeedStrategyFactory._create_simple_condition(
            left_operand="close",
            operator="<",
            right_operand=200.0,
            direction="long",
        )

        group = SeedStrategyFactory._create_and_condition_group([cond1, cond2])

        assert isinstance(group, ConditionGroup)
        assert group.operator == "AND"
        assert len(group.conditions) == 2


class TestParameterConstants:
    """パラメータ定数クラスのテスト"""

    def test_dmi_extreme_params_exist(self):
        """DMI Extremeパラメータ定数が存在すること"""
        assert hasattr(SeedStrategyFactory, "DMIExtremeParams")
        assert SeedStrategyFactory.DMIExtremeParams.DI_THRESHOLD == 45.0
        assert SeedStrategyFactory.DMIExtremeParams.ADX_THRESHOLD == 45.0
        assert SeedStrategyFactory.DMIExtremeParams.ATR_LENGTH == 14

    def test_rsi_momentum_params_exist(self):
        """RSI Momentumパラメータ定数が存在すること"""
        assert hasattr(SeedStrategyFactory, "RSIMomentumParams")
        assert SeedStrategyFactory.RSIMomentumParams.RSI_LONG_THRESHOLD == 75.0
        assert SeedStrategyFactory.RSIMomentumParams.RSI_SHORT_THRESHOLD == 25.0
        assert SeedStrategyFactory.RSIMomentumParams.RSI_PERIOD == 14

    def test_bollinger_breakout_params_exist(self):
        """Bollinger Breakoutパラメータ定数が存在すること"""
        assert hasattr(SeedStrategyFactory, "BollingerBreakoutParams")
        assert SeedStrategyFactory.BollingerBreakoutParams.BB_LENGTH == 20
        assert SeedStrategyFactory.BollingerBreakoutParams.BB_STD == 2.0

    def test_kama_adx_hybrid_params_exist(self):
        """KAMA-ADX Hybridパラメータ定数が存在すること"""
        assert hasattr(SeedStrategyFactory, "KAMAADXHybridParams")
        assert SeedStrategyFactory.KAMAADXHybridParams.KAMA_LENGTH == 30
        assert SeedStrategyFactory.KAMAADXHybridParams.ADX_LENGTH == 13
        assert SeedStrategyFactory.KAMAADXHybridParams.DI_THRESHOLD == 40.0

    def test_wae_params_exist(self):
        """WAEパラメータ定数が存在すること"""
        assert hasattr(SeedStrategyFactory, "WAEParams")
        assert SeedStrategyFactory.WAEParams.MACD_FAST == 12
        assert SeedStrategyFactory.WAEParams.MACD_SLOW == 26
        assert SeedStrategyFactory.WAEParams.MACD_SIGNAL == 9

    def test_trendilo_params_exist(self):
        """Trendiloパラメータ定数が存在すること"""
        assert hasattr(SeedStrategyFactory, "TrendiloParams")
        assert SeedStrategyFactory.TrendiloParams.T3_LENGTH == 200
        assert SeedStrategyFactory.TrendiloParams.T3_A == 0.7
        assert SeedStrategyFactory.TrendiloParams.ADX_LENGTH == 14
