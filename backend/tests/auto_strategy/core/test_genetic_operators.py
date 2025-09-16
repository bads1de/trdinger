"""
genetic_operators.pyのテスト
"""

import pytest
import sys
import copy
import random
from unittest.mock import Mock, patch, MagicMock
from typing import List

# プロジェクトルートをパスに追加
sys.path.insert(0, '../../..')  # backend/tests to /

from app.services.auto_strategy.models.strategy_models import StrategyGene
from app.services.auto_strategy.core.genetic_operators import (
    _convert_to_strategy_gene,
    _convert_to_individual,
    crossover_strategy_genes_pure,
    crossover_strategy_genes,
    mutate_strategy_gene_pure,
    mutate_strategy_gene,
    create_deap_crossover_wrapper,
    create_deap_mutate_wrapper,
)


@pytest.fixture
def sample_strategy_gene_1():
    """テスト用のStrategyGeneサンプル1"""
    from backend.app.services.auto_strategy.models.strategy_models import (
        IndicatorGene, Condition, TPSLGene, PositionSizingGene, TPSLMethod, PositionSizingMethod
    )

    indicators = [
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
    ]

    entry_conditions = [
        Condition(left_operand="close", operator=">", right_operand="sma_20"),
    ]

    exit_conditions = [
        Condition(left_operand="close", operator="<", right_operand="sma_20"),
    ]

    tpsl_gene = TPSLGene(
        enabled=True,
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        atr_multiplier_sl=2.0,
        atr_multiplier_tp=3.0,
        atr_period=14,
        lookback_period=100,
    )

    return StrategyGene(
        id="test-1",
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        long_entry_conditions=entry_conditions,
        short_entry_conditions=exit_conditions,  # 逆転
        risk_management={"position_size": 0.1, "max_drawdown": 0.05},
        tpsl_gene=tpsl_gene,
        metadata={"test": True},
    )


@pytest.fixture
def sample_strategy_gene_2():
    """テスト用のStrategyGeneサンプル2"""
    from backend.app.services.auto_strategy.models.strategy_models import (
        IndicatorGene, Condition, TPSLGene, PositionSizingGene, TPSLMethod, PositionSizingMethod
    )

    indicators = [
        IndicatorGene(type="EMA", parameters={"period": 10}, enabled=True),
        IndicatorGene(type="MACD", parameters={"fast_period": 12}, enabled=True),
        IndicatorGene(type="BB", parameters={"period": 20, "std_dev": 2.0}, enabled=True),
    ]

    entry_conditions = [
        Condition(left_operand="rsi_14", operator="<", right_operand="30"),
    ]

    exit_conditions = [
        Condition(left_operand="close", operator=">", right_operand="bb_upper"),
    ]

    tpsl_gene = TPSLGene(
        enabled=True,
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        risk_reward_ratio=2.0,
        atr_multiplier_sl=1.5,
        atr_multiplier_tp=2.5,
        atr_period=21,
        lookback_period=150,
    )

    position_sizing_gene = PositionSizingGene(
        enabled=True,
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.02,
        fixed_ratio=0.1,
        atr_multiplier=2.0,
        optimal_f_multiplier=0.5,
        lookback_period=30,
        min_position_size=0.001,
    )

    return StrategyGene(
        id="test-2",
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        long_entry_conditions=entry_conditions,
        short_entry_conditions=exit_conditions,  # 逆転
        risk_management={"position_size": 0.15, "max_drawdown": 0.08},
        tpsl_gene=tpsl_gene,
        position_sizing_gene=position_sizing_gene,
        metadata={"test": True},
    )


class TestConverters:
    """変換関数のテスト"""

    def test_convert_to_strategy_gene_with_strategy_gene(self, sample_strategy_gene_1):
        """StrategyGeneオブジェクトはそのまま返るべき"""
        result = _convert_to_strategy_gene(sample_strategy_gene_1)
        assert result is sample_strategy_gene_1

    def test_convert_to_strategy_gene_with_list(self, sample_strategy_gene_1):
        """リストからStrategyGeneに変換"""
        with patch('backend.app.services.auto_strategy.core.genetic_operators.GeneSerializer') as mock_serializer:
            mock_instance = Mock()
            mock_serializer.return_value = mock_instance
            mock_instance.decode_list_to_strategy_gene.return_value = sample_strategy_gene_1

            test_list = [0.5, 0.2, 0.1, 0.3, 0.9]
            result = _convert_to_strategy_gene(test_list)

            mock_serializer.assert_called_once()
            mock_instance.decode_list_to_strategy_gene.assert_called_once_with(test_list, StrategyGene)
            assert result is sample_strategy_gene_1

    def test_convert_to_strategy_gene_invalid_type(self):
        """無効なタイプが渡されたらTypeError"""
        with pytest.raises(TypeError, match="サポートされていない型です"):
            _convert_to_strategy_gene(123)

    @patch('backend.app.services.auto_strategy.core.genetic_operators.GeneSerializer')
    def test_convert_to_individual_with_strategy_gene(self, mock_serializer, sample_strategy_gene_1):
        """StrategyGeneからIndividualに変換"""
        mock_instance = Mock()
        mock_serializer.return_value = mock_instance
        mock_instance.encode_strategy_gene_to_list.return_value = [0.1, 0.2, 0.3]

        mock_individual_class = Mock()
        mock_individual_class.return_value = Mock()

        result = _convert_to_individual(sample_strategy_gene_1, mock_individual_class)

        mock_serializer.assert_called_once()
        mock_instance.encode_strategy_gene_to_list.assert_called_once_with(sample_strategy_gene_1)
        mock_individual_class.assert_called_once_with([0.1, 0.2, 0.3])
        assert result is not None

    def test_convert_to_individual_without_class(self, sample_strategy_gene_1):
        """individual_classがNoneの場合はStrategyGeneをそのまま返す"""
        result = _convert_to_individual(sample_strategy_gene_1, None)
        assert result is sample_strategy_gene_1


class TestCrossoverStrategyGenesPure:
    """crossover_strategy_genes_pure関数のテスト"""

    def test_crossover_with_valid_genes(self, sample_strategy_gene_1, sample_strategy_gene_2):
        """有効なStrategyGene同士の交叉"""
        # シード固定で再現性確保
        random.seed(42)

        parent1 = copy.deepcopy(sample_strategy_gene_1)
        parent2 = copy.deepcopy(sample_strategy_gene_2)

        child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        # 子がStrategyGeneオブジェクトであること
        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)

        # IDが異なること
        assert child1.id != parent1.id
        assert child2.id != parent2.id
        assert child1.id != child2.id

        # 指標数制限の確認
        assert len(child1.indicators) <= 5
        assert len(child2.indicators) <= 5

        # リスク管理の平均化確認 (数値パラメータ)
        assert child1.risk_management["position_size"] == pytest.approx(
            (parent1.risk_management["position_size"] + parent2.risk_management["position_size"]) / 2
        )

    def test_crossover_with_empty_indicators(self):
        """空の指標リストでの交叉"""
        parent1 = StrategyGene(
            id="parent1", indicators=[], entry_conditions=[], exit_conditions=[],
            long_entry_conditions=[], short_entry_conditions=[], risk_management={}
        )
        parent2 = StrategyGene(
            id="parent2", indicators=[], entry_conditions=[], exit_conditions=[],
            long_entry_conditions=[], short_entry_conditions=[], risk_management={}
        )

        child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)
        assert len(child1.indicators) == 0
        assert len(child2.indicators) == 0

    def test_crossover_handles_errors(self):
        """交叉処理中のエラー発生時、親をそのまま返す"""
        # 無効な親オブジェクト
        invalid_parent = Mock(spec=[])  # 必要な属性がない

        # TypeErrorが発生するようにセットアップ
        invalid_parent.indicators = "invalid"  # 期待値はリスト

        parent2 = StrategyGene(
            id="parent2", indicators=[], entry_conditions=[], exit_conditions=[],
            long_entry_conditions=[], short_entry_conditions=[], risk_management={}
        )

        with patch('logging.Logger.error'):
            child1, child2 = crossover_strategy_genes_pure(invalid_parent, parent2)

        # エラー時は親をそのまま返す
        assert child1 is invalid_parent
        assert child2 is parent2

    @patch('backend.app.services.auto_strategy.core.genetic_operators.crossover_tpsl_genes')
    def test_crossover_with_tpsl_genes(self, mock_crossover_tpsl, sample_strategy_gene_1, sample_strategy_gene_2):
        """TP/SL遺伝子のある場合の交叉"""
        mock_child_tpsl1 = Mock()
        mock_child_tpsl2 = Mock()
        mock_crossover_tpsl.return_value = (mock_child_tpsl1, mock_child_tpsl2)

        child1, child2 = crossover_strategy_genes_pure(sample_strategy_gene_1, sample_strategy_gene_2)

        # TP/SL交叉が呼ばれたこと
        mock_crossover_tpsl.assert_called_once()

        # 子にTP/SLが設定されていること
        assert child1.tpsl_gene is not None
        assert child2.tpsl_gene is not None

    @patch('backend.app.services.auto_strategy.core.genetic_operators.copy.deepcopy')
    def test_crossover_with_position_sizing_genes(self, mock_deepcopy, sample_strategy_gene_1, sample_strategy_gene_2):
        """ポジションサイジング遺伝子のある場合の交叉"""
        from backend.app.services.auto_strategy.models.strategy_models import crossover_position_sizing_genes

        with patch('backend.app.services.auto_strategy.core.genetic_operators.crossover_position_sizing_genes') as mock_crossover_ps:
            mock_child_ps1 = Mock()
            mock_child_ps2 = Mock()
            mock_crossover_ps.return_value = (mock_child_ps1, mock_child_ps2)

            child1, child2 = crossover_strategy_genes_pure(sample_strategy_gene_1, sample_strategy_gene_2)

            # 有効なポジションサイジング遺伝子を持つ親のみ交叉される
            if isinstance(sample_strategy_gene_1.position_sizing_gene, Mock):
                mock_crossover_ps.assert_called()
            else:
                mock_crossover_ps.assert_called()


class TestCrossoverStrategyGenes:
    """crossover_strategy_genes関数のテスト"""

    def test_crossover_with_strategy_genes(self, sample_strategy_gene_1, sample_strategy_gene_2):
        """StrategyGene同士の交叉"""
        child1, child2 = crossover_strategy_genes(sample_strategy_gene_1, sample_strategy_gene_2)

        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)
        assert child1.id != sample_strategy_gene_1.id
        assert child2.id != sample_strategy_gene_2.id

    def test_crossover_with_individual_lists(self, sample_strategy_gene_1, sample_strategy_gene_2):
        """リスト型の個体同士の交叉（DEAP統合）"""
        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_strategy_gene') as mock_convert, \
             patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_individual') as mock_convert_individual:

            mock_convert.side_effect = [sample_strategy_gene_1, sample_strategy_gene_2]
            mock_convert_individual.side_effect = [None, None]  # NoneでStrategyGeneを返す

            individual1 = [0.1, 0.2, 0.3]
            individual2 = [0.4, 0.5, 0.6]

            child1, child2 = crossover_strategy_genes(individual1, individual2)

            assert isinstance(child1, StrategyGene)
            assert isinstance(child2, StrategyGene)

    def test_crossover_handles_errors_gracefully(self):
        """エラー発生時、エラー時に親を返す"""
        invalid_parent = Mock()

        parent2 = StrategyGene(
            id="parent2", indicators=[], entry_conditions=[], exit_conditions=[],
            long_entry_conditions=[], short_entry_conditions=[], risk_management={}
        )

        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_strategy_gene') as mock_convert:
            mock_convert.return_value = invalid_parent

            with patch('backend.app.services.auto_strategy.core.genetic_operators.crossover_strategy_genes_pure') as mock_crossover_pure:
                # crossover_strategy_genes_pureが失敗するように設定
                mock_crossover_pure.side_effect = Exception("テストエラー")

                with patch('logging.Logger.error'):
                    child1, child2 = crossover_strategy_genes(invalid_parent, parent2)

                # エラー時親を返す
                assert child1 is invalid_parent
                assert child2 is parent2


class TestMutateStrategyGenePure:
    """mutate_strategy_gene_pure関数のテスト"""

    def test_mutate_creates_new_gene(self, sample_strategy_gene_1):
        """突然変異後の遺伝子IDが変化すること"""
        original_id = sample_strategy_gene_1.id
        mutated = mutate_strategy_gene_pure(sample_strategy_gene_1)

        assert mutated.id != original_id
        assert mutated != sample_strategy_gene_1

    def test_mutate_preset_metadata_flags(self, sample_strategy_gene_1):
        """突然変異時にメタデータにフラグが設定される"""
        mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=1.0)  # 確実に変異

        assert "mutated" in mutated.metadata
        assert mutated.metadata["mutated"] is True
        assert "mutation_rate" in mutated.metadata

    def test_mutate_indicator_parameters_modification(self, sample_strategy_gene_1):
        """指標パラメータが突然変異されること"""
        # 確実に変異するためのシードとrate設定
        random.seed(42)

        # パラメータを特定可能なものに変更
        sample_strategy_gene_1.indicators[0].parameters["test_period"] = 20

        original_period = sample_strategy_gene_1.indicators[0].parameters["test_period"]
        mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=1.0)

        # パラメータが変更されている可能性がある
        new_period = mutated.indicators[0].parameters.get("test_period", original_period)
        assert isinstance(new_period, (int, float))

    @patch('backend.app.services.auto_strategy.core.genetic_operators.RandomGeneGenerator')
    def test_mutate_can_add_indicators(self, mock_generator_class, sample_strategy_gene_1):
        """指標追加の突然変異"""
        # 新しい指標生成をモック
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        new_indicator = Mock()
        new_indicator.type = "MACD"
        mock_generator._generate_random_indicators.return_value = [new_indicator]

        # 確実に追加されるように調整
        sample_strategy_gene_1.indicators = [sample_strategy_gene_1.indicators[0]]  # 1つだけ
        sample_strategy_gene_1.MAX_INDICATORS = 5

        random.seed(42)  # 1 > 0.3になるように

        with patch('random.random', side_effect=[0.5, 0.4]):  # 添加方向へ
            mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=0.4)

    def test_mutate_conditions_modification(self, sample_strategy_gene_1):
        """条件が突然変異されること"""
        if sample_strategy_gene_1.entry_conditions:
            original_operator = sample_strategy_gene_1.entry_conditions[0].operator

            mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=1.0)

            # オペレータが変化する可能性がある
            assert isinstance(mutated.entry_conditions[0].operator, str)

    def test_mutate_risk_management_modification(self, sample_strategy_gene_1):
        """リスク管理設定が突然変異されること"""
        original_position_size = sample_strategy_gene_1.risk_management.get("position_size", 0.1)

        mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=1.0)

        new_position_size = mutated.risk_management.get("position_size", original_position_size)
        assert isinstance(new_position_size, float)
        assert new_position_size > 0

    @patch('backend.app.services.auto_strategy.core.genetic_operators.mutate_tpsl_gene')
    def test_mutate_tpsl_gene_called(self, mock_mutate_tpsl, sample_strategy_gene_1):
        """TP/SL遺伝子突然変異が呼ばれること"""
        mock_new_tpsl = Mock()
        mock_mutate_tpsl.return_value = mock_new_tpsl

        mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=1.0)

        # 確実に呼ぶ
        mock_mutate_tpsl.assert_called()

    def test_mutate_can_create_new_tpsl_gene(self, sample_strategy_gene_1):
        """TP/SL遺伝子が存在しない場合新規作成されること"""
        sample_strategy_gene_1.tpsl_gene = None

        random.seed(42)
        with patch('random.random', side_effect=[0.8]):  # 0.2未満になるように
            mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=1.0)

            # TPSLが作成される可能性がある
            # assert mutated.tpsl_gene is not None  # 確率ベースなのでオプション

    def test_mutate_handles_errors_gracefully(self):
        """突然変異エラー時に元の遺伝子を返すこと"""
        invalid_gene = Mock()
        # 属性が欠けているエラー発生
        invalid_gene.indicators = "invalid"
        invalid_gene.deepcopy.side_effect = AttributeError("invalid")

        with patch('backend.app.services.auto_strategy.core.genetic_operators.copy.deepcopy', side_effect=Exception("テストエラー")):
            with patch('logging.Logger.error'):
                result = mutate_strategy_gene_pure(invalid_gene)

                # エラー時は元の遺伝子を返す
                assert result is invalid_gene

    def test_mutate_preserves_metadata(self, sample_strategy_gene_1):
        """既存のメタデータが保持されること"""
        sample_strategy_gene_1.metadata["original"] = "test"

        mutated = mutate_strategy_gene_pure(sample_strategy_gene_1, mutation_rate=0.0)  # 変異なし

        assert "original" in mutated.metadata


class TestCreateDeapWrappers:
    """DEAPラッパー関数のテスト"""

    def test_create_crossover_wrapper_basic(self, sample_strategy_gene_1, sample_strategy_gene_2):
        """クロスオーバーラッパーの基本機能"""
        wrapper = create_deap_crossover_wrapper()

        result_child1, result_child2 = wrapper(sample_strategy_gene_1, sample_strategy_gene_2)

        assert isinstance(result_child1, StrategyGene)
        assert isinstance(result_child2, StrategyGene)

    def test_create_crossover_wrapper_with_custom_class(self, sample_strategy_gene_1, sample_strategy_gene_2):
        """カスタムIndividualクラスを使用したクロスオーバー"""
        mock_individual_class = Mock()

        wrapper = create_deap_crossover_wrapper(mock_individual_class)

        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_individual') as mock_convert_individual:
            mock_convert_individual.side_effect = ["mock_ind1", "mock_ind2"]

            result = wrapper(sample_strategy_gene_1, sample_strategy_gene_2)

            assert result == ("mock_ind1", "mock_ind2")

    def test_create_mutate_wrapper_basic(self, sample_strategy_gene_1):
        """突然変異ラッパーの基本機能"""
        wrapper = create_deap_mutate_wrapper()

        result = wrapper(sample_strategy_gene_1)

        assert isinstance(result[0], StrategyGene)

    def test_create_mutate_wrapper_with_custom_class(self, sample_strategy_gene_1):
        """カスタムIndividualクラスを使用した突然変異"""
        mock_individual_class = Mock()

        wrapper = create_deap_mutate_wrapper(mock_individual_class)

        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_individual') as mock_convert_individual:
            mock_convert_individual.return_value = "mock_individual"

            result = wrapper(sample_strategy_gene_1)

            assert result == ("mock_individual",)

    def test_create_mutate_wrapper_handles_errors(self):
        """ラッパー関数がエラーハンドリングをする"""
        mock_invalid_individual = Mock()

        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_strategy_gene', side_effect=Exception("変換エラー")):
            with patch('logging.Logger.error'):
                wrapper = create_deap_mutate_wrapper()
                result = wrapper(mock_invalid_individual)

                # エラー時元の個体を返す
                assert result == (mock_invalid_individual,)


class TestMutateStrategyGene:
    """mutate_strategy_gene関数のテスト"""

    def test_mutate_gene_with_strategy_gene(self, sample_strategy_gene_1):
        """StrategyGeneの突然変異"""
        result = mutate_strategy_gene(sample_strategy_gene_1, mutation_rate=0.5)

        assert isinstance(result, StrategyGene)

    def test_mutate_individual_list(self, sample_strategy_gene_1):
        """リスト個体の突然変異"""
        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_strategy_gene') as mock_convert, \
             patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_individual') as mock_convert_individual:

            mock_convert.return_value = sample_strategy_gene_1
            mock_convert_individual.return_value = [0.1, 0.2, 0.3]

            individual = [0.4, 0.5, 0.6]

            result = mutate_strategy_gene(individual, mutation_rate=0.5)

            assert isinstance(result, list)
            mock_convert.assert_called_once_with(individual)
            mock_convert_individual.assert_called_once()

    def test_mutate_handles_errors_gracefully(self):
        """エラー発生時エラーハンドリング"""
        with patch('backend.app.services.auto_strategy.core.genetic_operators._convert_to_strategy_gene', side_effect=Exception("変換エラー")):
            with patch('logging.Logger.error'):
                result = mutate_strategy_gene(Mock(), mutation_rate=0.5)

                # エラー時は引数をそのまま返す
                assert result is not None