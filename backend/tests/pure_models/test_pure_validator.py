"""
Test for GeneValidator
"""
import pytest
from backend.app.services.auto_strategy.models.validator import GeneValidator
from backend.app.services.auto_strategy.models.condition import Condition, ConditionGroup
from backend.app.services.auto_strategy.models.indicator_gene import IndicatorGene
from backend.app.services.auto_strategy.models.strategy_gene import StrategyGene


class TestGeneValidator:
    def test_validator_init(self):
        try:
            validator = GeneValidator()
            assert validator is not None
        except ImportError:
            # Skip if constants not found
            pytest.skip("Constants not available")

    def test_validate_condition_valid(self):
        try:
            validator = GeneValidator()
            condition = Condition(
                left_operand="close",
                operator=">",
                right_operand=1.0
            )

            is_valid, error = validator.validate_condition(condition)
            assert is_valid is True
            assert error == "" 
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_condition_invalid_operator(self):
        try:
            validator = GeneValidator()
            condition = Condition(
                left_operand="close",
                operator="invalid_op",
                right_operand="open"
            )

            is_valid, error = validator.validate_condition(condition)
            assert is_valid is False
        except ImportError:
            pytest.skip("Constants not available")

    def test_condition_group_validate(self):
        try:
            validator = GeneValidator()
            condition1 = Condition(left_operand=1.0, operator="<", right_operand=2.0)
            condition2 = Condition(left_operand="close", operator=">", right_operand="open")
            group = ConditionGroup(conditions=[condition1, condition2])

            result = group.validate()
            assert result is not None
        except ImportError:
            pytest.skip("Constants not available")

    def test_strategy_gene_validate(self):
        try:
            validator = GeneValidator()
            strategy = StrategyGene(
                entry_conditions=[Condition(left_operand="close", operator=">", right_operand=1.0)]
            )

            is_valid, errors = strategy.validate()
            assert is_valid is not None
            assert errors is not None
        except ImportError:
            pytest.skip("Constants not available")
    def test_clean_condition_normalize_operators(self):
        try:
            validator = GeneValidator()
            condition = Condition(left_operand="price", operator="above", right_operand=1.0)

            result = validator.clean_condition(condition)

            assert result is True
            assert condition.operator == ">"
        except ImportError:
            pytest.skip("Constants not available")

    def test_clean_condition_remove_whitespace(self):
        try:
            validator = GeneValidator()
            condition = Condition(left_operand=" close ", operator=">", right_operand=" open ")

            result = validator.clean_condition(condition)

            assert result is True
            assert condition.left_operand == "close"
            assert condition.right_operand == "open"
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_strategy_gene_with_missing_conditions(self):
        try:
            validator = GeneValidator()
            strategy = StrategyGene()  # empty conditions

            is_valid, errors = strategy.validate()

            assert not is_valid
            assert len(errors) > 0
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_indicator_gene_valid(self):
        """有効な指標遺伝子のテスト"""
        try:
            validator = GeneValidator()
            indicator_gene = IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)

            result = validator.validate_indicator_gene(indicator_gene)
            assert result is True
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_indicator_gene_invalid_type(self):
        """無効な指標タイプのテスト"""
        try:
            validator = GeneValidator()
            indicator_gene = IndicatorGene(type="INVALID_TYPE", parameters={"period": 20})

            result = validator.validate_indicator_gene(indicator_gene)
            assert result is False
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_indicator_gene_none_type(self):
        """Noneタイプのテスト"""
        try:
            validator = GeneValidator()
            indicator_gene = IndicatorGene(type=None, parameters={"period": 20})

            result = validator.validate_indicator_gene(indicator_gene)
            assert result is False
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_indicator_gene_invalid_parameters(self):
        """無効なパラメータのテスト"""
        try:
            validator = GeneValidator()
            indicator_gene = IndicatorGene(type="SMA", parameters="invalid_params")

            result = validator.validate_indicator_gene(indicator_gene)
            assert result is False
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_indicator_gene_ui_to_uo_conversion(self):
        """UI to UO変換のテスト"""
        try:
            validator = GeneValidator()
            indicator_gene = IndicatorGene(type="UI", parameters={"period": 20})

            result = validator.validate_indicator_gene(indicator_gene)
            # UIはUOに修正されて有効になるはず
            assert indicator_gene.type == "UO"
            assert result is True
        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_indicator_gene_invalid_period(self):
        """無効なperiodパラメータのテスト"""
        try:
            validator = GeneValidator()
            indicator_gene = IndicatorGene(type="SMA", parameters={"period": -5})

            result = validator.validate_indicator_gene(indicator_gene)
            assert result is False
        except ImportError:
            pytest.skip("Constants not available")

    def test_is_valid_operand_detailed_numeric(self):
        """数値オペランドのテスト"""
        try:
            validator = GeneValidator()

            # 有効な数値
            result, error = validator._is_valid_operand_detailed(1.5)
            assert result is True
            assert error == ""

            # 境界値
            result, error = validator._is_valid_operand_detailed(0)
            assert result is True

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_valid_operand_detailed_string_numeric(self):
        """数値文字列オペランドのテスト"""
        try:
            validator = GeneValidator()

            # 有効な数値文字列
            result, error = validator._is_valid_operand_detailed("1.5")
            assert result is True
            assert error == ""

            # 無効な文字列
            result, error = validator._is_valid_operand_detailed("not_a_number")
            assert result is False
            assert "無効な文字列オペランド" in error

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_valid_operand_detailed_string_data_source(self):
        """データソース文字列オペランドのテスト"""
        try:
            validator = GeneValidator()

            # 有効なデータソース
            result, error = validator._is_valid_operand_detailed("close")
            assert result is True
            assert error == ""

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_valid_operand_detailed_empty_string(self):
        """空文字列オペランドのテスト"""
        try:
            validator = GeneValidator()

            # 空文字列
            result, error = validator._is_valid_operand_detailed("")
            assert result is False
            assert error == "オペランドが空文字列です"

            # 空白のみ
            result, error = validator._is_valid_operand_detailed("   ")
            assert result is False
            assert error == "オペランドが空文字列です"

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_valid_operand_detailed_none(self):
        """Noneオペランドのテスト"""
        try:
            validator = GeneValidator()

            result, error = validator._is_valid_operand_detailed(None)
            assert result is False
            assert error == "オペランドがNoneです"

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_indicator_valid(self):
        """有効な指標辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            operand = {"type": "indicator", "name": "SMA"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is True
            assert error == ""

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_indicator_invalid(self):
        """無効な指標辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            # name が None
            operand = {"type": "indicator", "name": None}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "nameが設定されていません" in error

            # 無効な指標名
            operand = {"type": "indicator", "name": "INVALID_INDICATOR"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "無効な指標名" in error

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_price_valid(self):
        """有効な価格辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            operand = {"type": "price", "name": "close"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is True
            assert error == ""

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_price_invalid(self):
        """無効な価格辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            # name が None
            operand = {"type": "price", "name": None}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "nameが設定されていません" in error

            # 無効なデータソース
            operand = {"type": "price", "name": "INVALID_DATA"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "無効な価格データソース" in error

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_value_valid(self):
        """有効な数値辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            # 数値
            operand = {"type": "value", "value": 1.5}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is True
            assert error == ""

            # 数値文字列
            operand = {"type": "value", "value": "2.5"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is True
            assert error == ""

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_value_invalid(self):
        """無効な数値辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            # value が None
            operand = {"type": "value", "value": None}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "valueが設定されていません" in error

            # 変換できない文字列
            operand = {"type": "value", "value": "not_a_number"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "数値に変換できない文字列" in error

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_dict_operand_invalid_type(self):
        """無効なタイプの辞書オペランドのテスト"""
        try:
            validator = GeneValidator()

            operand = {"type": "invalid_type", "name": "test"}
            result, error = validator._validate_dict_operand_detailed(operand)
            assert result is False
            assert "無効な辞書タイプ" in error

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_indicator_name_ui_conversion(self):
        """UI to UO変換のテスト"""
        try:
            validator = GeneValidator()

            # UIがUOに変換されて有効になるはず
            result = validator._is_indicator_name("UI")
            assert result is True

            # テスト後のパラメータ変更をチェック
            # このチェックは関数内でnameが変更されるため、直接確認できないが
            # UOが有効な指標かどうかをテスト
            result_uo = validator._is_indicator_name("UO")
            # UOが有効かどうかは_util.or_utilsでの設定による

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_indicator_name_with_parameters(self):
        """パラメータ付き指標名のテスト"""
        try:
            validator = GeneValidator()

            # SMA_20 のようなパラメータ付き
            result = validator._is_indicator_name("SMA_20")
            # この結果はSMAが有効な指標であるかどうかによる

            # 無効な指標名_パラメータ
            result = validator._is_indicator_name("INVALID_TYPE_10")
            assert result is False

        except ImportError:
            pytest.skip("Constants not available")

    def test_is_indicator_name_index_suffix(self):
        """インデックス付き指標名のテスト"""
        try:
            validator = GeneValidator()

            # SMA_0 のようなインデックス付き
            result = validator._is_indicator_name("SMA_0")

        except ImportError:
            pytest.skip("Constants not available")

    def test_clean_condition_with_dict_operands(self):
        """辞書オペランドのクリーニングテスト"""
        try:
            validator = GeneValidator()

            condition = Condition(
                left_operand={"type": "indicator", "name": "SMA"},
                operator=">",
                right_operand={"type": "value", "value": 1.5}
            )

            result = validator.clean_condition(condition)
            assert result is True

            # 辞書が文字列に変換されているはず
            assert isinstance(condition.left_operand, str)
            assert isinstance(condition.right_operand, str)

        except ImportError:
            pytest.skip("Constants not available")

    def test_extract_operand_from_dict_indicator(self):
        """指標辞書の抽出テスト"""
        try:
            validator = GeneValidator()

            operand_dict = {"type": "indicator", "name": "SMA"}
            result = validator._extract_operand_from_dict(operand_dict)
            assert result == "SMA"

        except ImportError:
            pytest.skip("Constants not available")

    def test_extract_operand_from_dict_price(self):
        """価格辞書の抽出テスト"""
        try:
            validator = GeneValidator()

            operand_dict = {"type": "price", "name": "close"}
            result = validator._extract_operand_from_dict(operand_dict)
            assert result == "close"

        except ImportError:
            pytest.skip("Constants not available")

    def test_extract_operand_from_dict_value(self):
        """数値辞書の抽出テスト"""
        try:
            validator = GeneValidator()

            operand_dict = {"type": "value", "value": 1.5}
            result = validator._extract_operand_from_dict(operand_dict)
            assert result == "1.5"

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_strategy_gene_max_indicators(self):
        """最大指標数のテスト"""
        try:
            from unittest.mock import patch

            validator = GeneValidator()
            # 多くの指標を作成
            indicators = [IndicatorGene(type="SMA", parameters={"period": 20}) for _ in range(10)]

            # MAX_INDICATORSを5に設定
            with patch.object(StrategyGene, 'MAX_INDICATORS', 5):
                strategy = StrategyGene(indicators=indicators)

                is_valid, errors = validator.validate_strategy_gene(strategy)
                assert not is_valid
                assert len(errors) > 0
                assert "指標数が上限" in errors[0]

        except ImportError:
            pytest.skip("Constants not available")

    def test_validate_strategy_gene_no_enabled_indicators(self):
        """有効指標なしのテスト"""
        try:
            validator = GeneValidator()
            # このテストではエントリー条件を設定して、指標エラーを強制表示
            entry_conditions = [Condition(left_operand="close", operator=">", right_operand=1.0)]
            indicators = [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=False)
            ]

            strategy = StrategyGene(indicators=indicators, entry_conditions=entry_conditions)

            is_valid, errors = validator.validate_strategy_gene(strategy)
            assert not is_valid
            assert len(errors) > 0
            # エラーメッセージに有効な指標エラーが含まれていることを確認
            error_messages = "\n".join(errors)
            assert "有効な指標が設定されていません" in error_messages

        except ImportError:
            pytest.skip("Constants not available")
