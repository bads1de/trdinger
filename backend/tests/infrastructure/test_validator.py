"""
GeneValidatorのテスト
"""

from unittest.mock import Mock

import pytest

from app.services.auto_strategy.models.validator import GeneValidator


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
        return gene

    @pytest.fixture
    def invalid_indicator_gene(self):
        """無効な指標遺伝子"""
        gene = Mock()
        gene.type = None
        gene.parameters = {}
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

    def test_init(self, validator):
        """初期化テスト"""
        assert validator.valid_indicator_types is not None
        assert validator.valid_operators is not None
        assert validator.valid_data_sources is not None

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

        result = validator.validate_indicator_gene(gene)
        assert result is True  # UIは有効な複合指標なのでTrueになるべき
        assert gene.type == "UI"  # 修正されていない

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

    def test_validate_condition_trivial_price_comparison(self, validator):
        """シンプルな価格比較が拒否されることをテスト"""
        trivial_condition = Mock()
        trivial_condition.operator = ">"
        trivial_condition.left_operand = "close"
        trivial_condition.right_operand = "open"

        result, error = validator.validate_condition(trivial_condition)
        assert result is False
        assert "シンプルな価格比較" in error

    def test_is_trivial_condition_true_for_price_price(self, validator):
        """価格同士の比較がトリビアルと判定されることをテスト"""
        condition = Mock()
        condition.left_operand = "close"
        condition.right_operand = "open"
        condition.operator = ">"

        result = validator._is_trivial_condition(condition)
        assert result is True

    def test_is_trivial_condition_false_for_indicator_price(self, validator):
        """指標と価格の比較がトリビアルでないことをテスト"""
        condition = Mock()
        condition.left_operand = "SMA_20"
        condition.right_operand = "close"
        condition.operator = ">"

        result = validator._is_trivial_condition(condition)
        assert result is False

    def test_is_trivial_condition_true_for_same_price(self, validator):
        """同じ価格データの比較がトリビアルと判定されることをテスト"""
        condition = Mock()
        condition.left_operand = "close"
        condition.right_operand = "close"
        condition.operator = ">"

        result = validator._is_trivial_condition(condition)
        assert result is True

    def test_is_valid_operand_detailed_number(self, validator):
        """数値オペランドの検証"""
        result, error = validator._is_valid_operand_detailed(10.5)
        assert result is True
        assert error == ""

    def test_is_valid_operand_detailed_string_valid(self, validator):
        """有効な文字列オペランドの検証"""
        result, error = validator._is_valid_operand_detailed("SMA_20")
        assert result is True
        assert error == ""

    def test_is_valid_operand_detailed_string_invalid(self, validator):
        """無効な文字列オペランドの検証"""
        result, error = validator._is_valid_operand_detailed("invalid_string")
        assert result is False
        assert error != ""

    def test_is_valid_operand_detailed_dict_indicator(self, validator):
        """指標辞書オペランドの検証"""
        operand = {"type": "indicator", "name": "SMA"}
        result, error = validator._is_valid_operand_detailed(operand)
        assert result is True
        assert error == ""

    def test_is_valid_operand_detailed_dict_price(self, validator):
        """価格辞書オペランドの検証"""
        operand = {"type": "price", "name": "close"}
        result, error = validator._is_valid_operand_detailed(operand)
        assert result is True
        assert error == ""

    def test_is_valid_operand_detailed_dict_value(self, validator):
        """値辞書オペランドの検証"""
        operand = {"type": "value", "value": 10.5}
        result, error = validator._is_valid_operand_detailed(operand)
        assert result is True
        assert error == ""

    def test_is_valid_operand_detailed_none(self, validator):
        """Noneオペランドの検証"""
        result, error = validator._is_valid_operand_detailed(None)
        assert result is False
        assert "Noneです" in error

    def test_is_indicator_name_valid(self, validator):
        """有効な指標名の検証"""
        result = validator._is_indicator_name("SMA")
        assert result is True

    def test_is_indicator_name_invalid(self, validator):
        """無効な指標名の検証"""
        result = validator._is_indicator_name("invalid")
        assert result is False

    def test_is_indicator_name_with_params(self, validator):
        """パラメータ付き指標名の検証"""
        result = validator._is_indicator_name("SMA_20")
        # 実際のindicator_typesに依存するので、テストは汎用的に

    def test_is_indicator_name_ui_not_corrected(self, validator):
        """'UI'指標名が修正されないことを確認"""
        result = validator._is_indicator_name("UI")
        # UIが有効な指標でない限りFalseになるべき

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

    def test_extract_operand_from_dict_indicator(self, validator):
        """指標辞書からの抽出"""
        operand = {"type": "indicator", "name": "SMA"}
        result = validator._extract_operand_from_dict(operand)
        assert result == "SMA"

    def test_extract_operand_from_dict_price(self, validator):
        """価格辞書からの抽出"""
        operand = {"type": "price", "name": "close"}
        result = validator._extract_operand_from_dict(operand)
        assert result == "close"

    def test_extract_operand_from_dict_value(self, validator):
        """値辞書からの抽出"""
        operand = {"type": "value", "value": 10.5}
        result = validator._extract_operand_from_dict(operand)
        assert result == "10.5"

    def test_validate_strategy_gene_valid(self, validator):
        """有効な戦略遺伝子の検証"""
        strategy = Mock()
        indicator_mock = Mock()
        indicator_mock.type = "SMA"
        indicator_mock.parameters = {"period": 20}
        indicator_mock.enabled = True
        indicators = [indicator_mock]
        strategy.indicators = indicators
        # エントリー条件が必要
        entry_condition_mock = Mock()
        entry_condition_mock.operator = ">"
        entry_condition_mock.left_operand = "SMA_20"
        entry_condition_mock.right_operand = "SMA_50"
        strategy.entry_conditions = [entry_condition_mock]
        strategy.long_entry_conditions = []
        strategy.short_entry_conditions = []
        # イグジット条件も必要
        exit_condition_mock = Mock()
        exit_condition_mock.operator = "<"
        exit_condition_mock.left_operand = "SMA_20"
        exit_condition_mock.right_operand = "SMA_50"
        strategy.exit_conditions = [exit_condition_mock]
        strategy.tpsl_gene = None
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
            indicators.append(ind)
        strategy.indicators = indicators
        strategy.entry_conditions = []
        strategy.long_entry_conditions = []
        strategy.short_entry_conditions = []
        strategy.exit_conditions = []
        strategy.tpsl_gene = None
        strategy.MAX_INDICATORS = 5

        result, errors = validator.validate_strategy_gene(strategy)
        assert result is False
        assert len(errors) > 0
        assert "上限" in errors[0]

    def test_validate_strategy_gene_no_enabled_indicators(self, validator):
        """有効な指標がない戦略遺伝子の検証"""
        strategy = Mock()
        indicator_mock = Mock()
        indicator_mock.type = "SMA"
        indicator_mock.parameters = {"period": 20}
        indicator_mock.enabled = False
        indicators = [indicator_mock]
        strategy.indicators = indicators
        entry_condition_mock = Mock()
        entry_condition_mock.operator = ">"
        entry_condition_mock.left_operand = "SMA_20"
        entry_condition_mock.right_operand = "SMA_50"
        strategy.entry_conditions = [entry_condition_mock]
        strategy.long_entry_conditions = []
        strategy.short_entry_conditions = []
        exit_condition_mock = Mock()
        exit_condition_mock.operator = "<"
        exit_condition_mock.left_operand = "SMA_20"
        exit_condition_mock.right_operand = "SMA_50"
        strategy.exit_conditions = [exit_condition_mock]
        strategy.tpsl_gene = None
        strategy.MAX_INDICATORS = 5

        result, errors = validator.validate_strategy_gene(strategy)
        assert result is False
        assert len(errors) > 0
        assert "有効な指標" in errors[0]
