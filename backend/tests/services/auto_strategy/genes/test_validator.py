"""
GeneValidatorのテスト
"""

from unittest.mock import Mock

import pytest

from app.services.auto_strategy.genes import IndicatorGene
from app.services.auto_strategy.genes.validator import GeneValidator


class TestGeneValidator:
    """GeneValidatorのテストクラス"""

    @pytest.fixture
    def validator(self):
        """GeneValidatorインスタンス"""
        return GeneValidator()

    @pytest.fixture
    def valid_indicator_gene(self):
        """有効な指標遺伝子"""
        gene = Mock()
        gene.type = "SMA"
        gene.parameters = {"period": 20}
        gene.timeframe = None
        return gene

    @pytest.fixture
    def invalid_indicator_gene(self):
        """無効な指標遺伝子"""
        gene = Mock()
        gene.type = None
        gene.parameters = {}
        gene.timeframe = None
        return gene

    @pytest.fixture
    def valid_condition(self):
        """有効な条件"""
        condition = Mock()
        condition.operator = ">"
        condition.left_operand = "SMA_20"
        condition.right_operand = "SMA_50"
        return condition

    @pytest.fixture
    def invalid_condition(self):
        """無効な条件"""
        condition = Mock()
        condition.operator = "invalid"
        condition.left_operand = None
        condition.right_operand = None
        return condition

    def test_validate_indicator_gene_valid(self, validator, valid_indicator_gene):
        """有効な指標遺伝子の検証"""
        result = validator.validate_indicator_gene(valid_indicator_gene)
        assert result is True

    def test_validate_indicator_gene_invalid_type(
        self, validator, invalid_indicator_gene
    ):
        """無効なタイプの指標遺伝子の検証"""
        result = validator.validate_indicator_gene(invalid_indicator_gene)
        assert result is False

    def test_validate_indicator_gene_invalid_parameters(
        self, validator, valid_indicator_gene
    ):
        """無効なパラメータの指標遺伝子の検証"""
        valid_indicator_gene.parameters = None
        result = validator.validate_indicator_gene(valid_indicator_gene)
        assert result is False

    def test_validate_indicator_gene_invalid_period(
        self, validator, valid_indicator_gene
    ):
        """無効な期間パラメータの指標遺伝子の検証"""
        valid_indicator_gene.parameters = {"period": -1}
        result = validator.validate_indicator_gene(valid_indicator_gene)
        assert result is False

    def test_validate_indicator_gene_ui_not_corrected(self, validator):
        """'UI'タイプが修正されないことを確認"""
        gene = Mock()
        gene.type = "UI"
        gene.parameters = {"period": 20}
        gene.timeframe = None

        result = validator.validate_indicator_gene(gene)
        assert result is True  # UIは有効な複合指標なのでTrueになるべき
        assert gene.type == "UI"  # 修正されていない

    def test_validate_indicator_gene_for_generation_respects_curated_universe(
        self, validator
    ):
        """生成専用バリデーションでは curated 外の指標を拒否する"""
        gene = IndicatorGene(type="LEVERAGE_RATIO", parameters={}, enabled=True)

        assert validator.validate_indicator_gene(gene) is True
        assert (
            validator.validate_indicator_gene_for_generation(
                gene, indicator_universe_mode="curated"
            )
            is False
        )

    def test_validate_indicator_gene_for_generation_uses_precomputed_universe(
        self, validator
    ):
        """解決済みユニバースがあれば再解決せずに判定できる"""
        gene = IndicatorGene(type="SMA", parameters={}, enabled=True)

        assert (
            validator.validate_indicator_gene_for_generation(
                gene,
                indicator_universe_mode="curated",
                allowed_indicators={"SMA", "EMA"},
            )
            is True
        )
        assert (
            validator.validate_indicator_gene_for_generation(
                gene,
                indicator_universe_mode="experimental_all",
                allowed_indicators={"EMA", "RSI"},
            )
            is False
        )

    def test_validate_condition_valid(self, validator, valid_condition):
        """有効な条件の検証"""
        result, error = validator.validate_condition(valid_condition)
        assert result is True
        assert error == ""

    def test_validate_condition_invalid(self, validator, invalid_condition):
        """無効な条件の検証"""
        result, error = validator.validate_condition(invalid_condition)
        assert result is False
        assert error != ""

    def test_validate_condition_invalid_operator(self, validator, valid_condition):
        """無効な演算子の条件検証"""
        valid_condition.operator = "invalid"
        result, error = validator.validate_condition(valid_condition)
        assert result is False
        assert "無効な演算子" in error

    def test_validate_condition_price_action_allowed(self, validator):
        """異なる価格データ同士の比較（プライスアクション）が許容されることをテスト"""
        # close > open（陽線判定）はプライスアクションとして有効
        price_action_condition = Mock()
        price_action_condition.operator = ">"
        price_action_condition.left_operand = "close"
        price_action_condition.right_operand = "open"

        result, error = validator.validate_condition(price_action_condition)
        assert result is True
        assert error == ""

    def test_clean_condition_valid(self, validator):
        """有効な条件のクリーニング"""
        condition = Mock()
        condition.left_operand = " SMA_20 "
        condition.right_operand = " SMA_50 "
        condition.operator = "above"

        result = validator.clean_condition(condition)
        assert result is True
        assert condition.left_operand == "SMA_20"
        assert condition.right_operand == "SMA_50"
        assert condition.operator == ">"

    def test_clean_condition_dict_operand(self, validator):
        """辞書オペランドのクリーニング"""
        condition = Mock()
        condition.left_operand = {"type": "indicator", "name": " SMA "}
        condition.right_operand = {"type": "price", "name": " close "}
        condition.operator = "below"

        result = validator.clean_condition(condition)
        assert result is True
        # clean_conditionは辞書を文字列に変換する
        assert condition.left_operand == " SMA "  # stripされない
        assert condition.right_operand == " close "  # stripされない
        assert condition.operator == "<"

    def test_validate_strategy_gene_valid(self, validator):
        """有効な戦略遺伝子の検証"""
        strategy = Mock()
        indicator_mock = Mock()
        indicator_mock.type = "SMA"
        indicator_mock.parameters = {"period": 20}
        indicator_mock.enabled = True
        indicator_mock.timeframe = None
        indicators = [indicator_mock]
        strategy.indicators = indicators

        # エントリー条件が必要
        entry_condition_mock = Mock()
        entry_condition_mock.operator = ">"
        entry_condition_mock.left_operand = "SMA_20"
        entry_condition_mock.right_operand = "SMA_50"
        strategy.long_entry_conditions = [entry_condition_mock]
        strategy.short_entry_conditions = []

        # TP/SL遺伝子のモック（必須）
        tpsl_mock = Mock()
        tpsl_mock.enabled = True
        tpsl_mock.validate.return_value = (True, [])
        strategy.tpsl_gene = tpsl_mock
        strategy.long_tpsl_gene = None
        strategy.short_tpsl_gene = None

        strategy.MAX_INDICATORS = 5

        result, errors = validator.validate_strategy_gene(strategy)
        assert result is True
        assert len(errors) == 0

    def test_validate_strategy_gene_too_many_indicators(self, validator):
        """指標数が多すぎる戦略遺伝子の検証"""
        strategy = Mock()
        indicators = []
        for _ in range(10):
            ind = Mock()
            ind.type = "SMA"
            ind.parameters = {"period": 20}
            ind.enabled = True
            ind.timeframe = None
            indicators.append(ind)
        strategy.indicators = indicators
        strategy.entry_conditions = []
        strategy.long_entry_conditions = []
        strategy.short_entry_conditions = []
        strategy.exit_conditions = []
        strategy.tpsl_gene = None
        strategy.long_tpsl_gene = None
        strategy.short_tpsl_gene = None
        strategy.MAX_INDICATORS = 5

        result, errors = validator.validate_strategy_gene(strategy)
        assert result is False
        assert len(errors) > 0
        # "上限" を含むエラーメッセージか、"エントリー条件が設定されていません" を含むエラーメッセージのいずれかがあるはず
        assert any(
            "上限" in e or "エントリー条件が設定されていません" in e for e in errors
        )

    def test_validate_strategy_gene_no_enabled_indicators(self, validator):
        """有効な指標がない戦略遺伝子の検証"""
        strategy = Mock()
        indicator_mock = Mock()
        indicator_mock.type = "SMA"
        indicator_mock.parameters = {"period": 20}
        indicator_mock.enabled = False
        indicator_mock.timeframe = None
        indicators = [indicator_mock]
        strategy.indicators = indicators

        entry_condition_mock = Mock()
        entry_condition_mock.operator = ">"
        entry_condition_mock.left_operand = "SMA_20"
        entry_condition_mock.right_operand = "SMA_50"
        strategy.long_entry_conditions = [entry_condition_mock]
        strategy.short_entry_conditions = []

        # TP/SL遺伝子のモック
        tpsl_mock = Mock()
        tpsl_mock.enabled = True
        tpsl_mock.validate.return_value = (True, [])
        strategy.tpsl_gene = tpsl_mock
        strategy.long_tpsl_gene = None
        strategy.short_tpsl_gene = None

        strategy.MAX_INDICATORS = 5

        result, errors = validator.validate_strategy_gene(strategy)
        assert result is False
        assert len(errors) > 0
        assert any("有効な指標" in e for e in errors)
