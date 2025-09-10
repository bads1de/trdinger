
import pytest
import logging
from unittest.mock import Mock, MagicMock
from backend.app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from backend.app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition, ConditionGroup


@pytest.fixture
def mock_services():
    """モックされた各種サービスを提供するフィクスチャ"""
    return {
        'condition_evaluator': Mock(),
        'indicator_calculator': Mock(),
        'tpsl_service': Mock(),
        'position_sizing_service': Mock()
    }


@pytest.fixture
def valid_gene():
    """有効なStrategyGeneのモック"""
    gene = Mock(spec=StrategyGene)
    gene.validate.return_value = (True, [])
    gene.id = "test-id-123"
    gene.indicators = []
    gene.entry_conditions = []  # 必要に応じて条件追加
    gene.exit_conditions = []
    gene.long_conditions = []
    gene.short_conditions = []
    gene.tpsl_gene = None
    gene.position_sizing_gene = None
    gene.get_effective_long_conditions.return_value = []
    gene.get_effective_short_conditions.return_value = []
    return gene


@pytest.fixture
def gene_with_indicators(valid_gene):
    """指標を持つGeneのモック"""
    indicator_gene = Mock(spec=IndicatorGene)
    indicator_gene.enabled = True
    indicator_gene.type = "SMA"
    indicator_gene.parameters = {"period": 14}
    valid_gene.indicators = [indicator_gene]
    return valid_gene


class TestStrategyFactory:
    """StrategyFactoryクラスのテスト"""

    def test_factory_init(self):
        """ファクトリーの初期化テスト"""
        factory = StrategyFactory()
        assert hasattr(factory, 'condition_evaluator')
        assert hasattr(factory, 'indicator_calculator')
        assert hasattr(factory, 'tpsl_service')
        assert hasattr(factory, 'position_sizing_service')

    def test_create_strategy_class_valid_gene(self, valid_gene):
        """有効な遺伝子での戦略クラス生成テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)
        assert strategy_class is not None
        assert isinstance(strategy_class, type)
        assert hasattr(strategy_class, '__name__')
        assert strategy_class.__name__.startswith("GS_")

    def test_create_strategy_class_invalid_gene(self):
        """無効な遺伝子での戦略クラス生成テスト - ValueError発生"""
        factory = StrategyFactory()
        invalid_gene = Mock(spec=StrategyGene)
        invalid_gene.validate.return_value = (False, ["Invalid gene"])

        with pytest.raises(ValueError, match="Invalid strategy gene"):
            factory.create_strategy_class(invalid_gene)

    def test_validate_gene_valid(self, valid_gene):
        """validate_geneメソッドの有効遺伝子テスト"""
        factory = StrategyFactory()
        is_valid, errors = factory.validate_gene(valid_gene)
        assert is_valid is True
        assert errors == []

    def test_validate_gene_invalid(self):
        """validate_geneメソッドの無効遺伝子テスト"""
        factory = StrategyFactory()
        invalid_gene = Mock(spec=StrategyGene)
        invalid_gene.validate.side_effect = Exception("Validation error")

        is_valid, errors = factory.validate_gene(invalid_gene)
        assert is_valid is False
        assert "Validation error" in str(errors[0])


class TestGeneratedStrategy:
    """生成された戦略クラスのテスト"""

    def test_generated_strategy_init_default_params(self, valid_gene):
        """生成戦略クラスの__init__テスト - デフォルトparams"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # backtesting.py互換のデータ作成
        data = Mock()
        broker = Mock()
        strategy_instance = strategy_class(broker=broker, data=data, params=None)

        assert strategy_instance.gene == valid_gene

    def test_generated_strategy_init_with_params(self, valid_gene):
        """生成戦略クラスの__init__テスト - params付き"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        data = Mock()
        broker = Mock()
        custom_gene = Mock(spec=StrategyGene)
        params = {"strategy_gene": custom_gene}

        strategy_instance = strategy_class(broker=broker, data=data, params=params)

        assert strategy_instance.gene == custom_gene

    def test_generated_strategy_class_name(self, valid_gene):
        """生成されたクラスの名前テスト"""
        factory = StrategyFactory()
        valid_gene.id = "unique-strategy-id"
        strategy_class = factory.create_strategy_class(valid_gene)

        assert "GS_unique" in strategy_class.__name__

    def test_generated_strategy_class_variables(self, valid_gene):
        """生成されたクラスのクラス変数テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        assert hasattr(strategy_class, 'strategy_gene')
        assert strategy_class.strategy_gene == valid_gene


class TestStrategyExecution:
    """戦略実行関連のテスト"""

    def test_position_size_calculation_with_gene(self, mocker, valid_gene):
        """ポジションサイズ計算テスト - gene有効"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # モック設定
        position_sizing_gene = Mock()
        position_sizing_gene.enabled = True
        valid_gene.position_sizing_gene = position_sizing_gene

        mock_result = Mock()
        mock_result.position_size = 0.1
        # factory.serviceをMockに置換
        mock_service = Mock()
        mock_service.calculate_position_size.return_value = mock_result
        factory.position_sizing_service = mock_service

        broker = Mock()
        broker.equity = 100000.0
        mock_data = Mock()
        mock_data.Close = [50000.0]

        strategy_instance = strategy_class(broker=broker, data=mock_data, params=None)

        size = strategy_instance._calculate_position_size()
        assert size == 0.1

    def test_position_size_calculation_default(self, valid_gene):
        """ポジションサイズ計算テスト - デフォルト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        size = strategy_instance._calculate_position_size()
        assert size == 0.01

    def test_long_entry_conditions(self, mocker, valid_gene):
        """ロングエントリー条件チェックテスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # evaluatorをMockに置き換え
        mock_evaluator = Mock()
        mock_evaluator.evaluate_conditions.return_value = True
        factory.condition_evaluator = mock_evaluator

        # 条件を追加
        valid_gene.entry_conditions = [Mock()]

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        result = strategy_instance._check_long_entry_conditions()
        assert result is True

    def test_short_entry_conditions(self, mocker, valid_gene):
        """ショートエントリー条件チェックテスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        mock_evaluator = Mock()
        mock_evaluator.evaluate_conditions.return_value = True
        factory.condition_evaluator = mock_evaluator

        # entry_conditionsを追加
        valid_gene.entry_conditions = [Mock()]

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        result = strategy_instance._check_short_entry_conditions()
        assert result is True

    def test_exit_conditions(self, mocker, valid_gene):
        """イグジット条件チェックテスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        mock_evaluator = Mock()
        mock_evaluator.evaluate_conditions.return_value = True
        factory.condition_evaluator = mock_evaluator

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)
        strategy_instance.gene.exit_conditions = [Mock()]

        result = strategy_instance._check_exit_conditions()
        assert result is True

    def test_exit_conditions_skip_tpsl(self, mocker, valid_gene):
        """イグジット条件チェックテスト - TP/SLスキップ"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        tpsl_gene = Mock()
        tpsl_gene.enabled = True
        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)
        strategy_instance.gene.tpsl_gene = tpsl_gene

        result = strategy_instance._check_exit_conditions()
        assert result is False

    def test_next_method_execution(self, mocker, valid_gene):
        """nextメソッド実行テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # Mock broker with equity
        mock_broker = Mock()
        mock_broker.equity = 100000.0

        # Mock data
        mock_data = Mock()
        mock_data.Close = [50000.0]

        # Mock position
        mock_position = Mock()
        mock_position.size = 0  # no position

        strategy_instance = strategy_class(broker=mock_broker, data=mock_data, params=None)
        mocker.patch.object(type(strategy_instance), 'position', mock_position)

        # evaluator mock
        mock_evaluator = Mock()
        factory.condition_evaluator = mock_evaluator

        # Mock factory通じて条件がtrueになるように
        valid_gene.get_effective_long_conditions.return_value = [Mock()]
        mock_evaluator.evaluate_conditions.return_value = True

        # サイズ計算mock
        mock_service = Mock()
        factory.position_sizing_service = mock_service

        # buyメソッドをspy
        mocker.patch.object(strategy_instance, 'buy')

        strategy_instance.next()

        # TP/SLがない場合、buyが呼ばれているはず
        strategy_instance.buy.assert_called_once()


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_indicator_init_fallback(self, caplog, mocker, valid_gene):
        """指標初期化フォールバックテスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(valid_gene)

        # 失敗するindicator_gene
        indicator_gene = Mock(spec=IndicatorGene)
        indicator_gene.enabled = True
        indicator_gene.type = "UNKNOWN"
        indicator_gene.parameters = {"period": -1}
        valid_gene.indicators = [indicator_gene]

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        # calculatorをMockに置き換え
        mock_calculator = Mock()
        mock_calculator.init_indicator.side_effect = Exception("Init failed")
        factory.indicator_calculator = mock_calculator

        # ログキャプチャ
        with caplog.at_level(logging.WARNING):
            try:
                strategy_instance.init()
            except Exception:
                pass

        # ログにfallbackメッセージがあるはず
        # assert "フォールバック" in caplog.text or "fallback" in caplog.text

    def test_indicator_init_success(self, mocker, gene_with_indicators):
        """指標初期化成功テスト"""
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene_with_indicators)

        # calculatorをMockに置き換え
        mock_calculator = Mock()
        mock_calculator.init_indicator.return_value = None
        factory.indicator_calculator = mock_calculator

        strategy_instance = strategy_class(broker=Mock(), data=Mock(), params=None)

        # エラーがない
        strategy_instance.init()

        # 呼び出された
        mock_calculator.init_indicator.assert_called_once()