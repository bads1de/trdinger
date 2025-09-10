"""
strategies/__init__.py のインポートテスト

バグを発見し、修正を行います。
"""

import pytest
from backend.app.services.auto_strategy.generators.strategies import (
    ConditionStrategy,
    DifferentIndicatorsStrategy,
    ComplexConditionsStrategy,
    IndicatorCharacteristicsStrategy,
)


class TestStrategiesImports:
    """strategiesパッケージのインポートテスト"""

    def test_all_classes_imported(self):
        """全てのクラスが正しくインポートされている"""
        # 各クラスがインポート可能であることを確認
        assert ConditionStrategy is not None
        assert DifferentIndicatorsStrategy is not None
        assert ComplexConditionsStrategy is not None
        assert IndicatorCharacteristicsStrategy is not None

    def test_base_class_is_abstract(self):
        """基底クラスが抽象クラスであることを確認"""
        # ConditionStrategyは抽象クラスなので直接インスタンス化できない
        with pytest.raises(TypeError):
            ConditionStrategy(None)

    def test_subclass_inheritance(self):
        """サブクラスが正しく継承されている"""
        # 各サブクラスがConditionStrategyを継承していることを確認
        assert issubclass(DifferentIndicatorsStrategy, ConditionStrategy)
        assert issubclass(ComplexConditionsStrategy, ConditionStrategy)
        assert issubclass(IndicatorCharacteristicsStrategy, ConditionStrategy)

    def test_class_attributes(self):
        """クラスの基本属性確認"""
        # 各クラスの基本的な属性が存在することを確認
        classes = [
            DifferentIndicatorsStrategy,
            ComplexConditionsStrategy,
            IndicatorCharacteristicsStrategy,
        ]

        for cls in classes:
            assert hasattr(cls, '__init__')
            assert hasattr(cls, 'generate_conditions')

    def test_init_file_coverage(self):
        """__all__に全てのクラスが含まれていることを確認"""
        from backend.app.services.auto_strategy.generators.strategies import __all__

        expected_classes = [
            'ConditionStrategy',
            'DifferentIndicatorsStrategy',
            'ComplexConditionsStrategy',
            'IndicatorCharacteristicsStrategy',
        ]

        for class_name in expected_classes:
            assert class_name in __all__, f"{class_name} is missing from __all__"


class TestStrategiesInitBugs:
    """__init__.pyの潜在的バグテスト"""

    def test_missing_import_bug(self):
        """BUG: インポート漏れのチェック"""
        # 全ての必要なクラスがインポートされていることを確認
        # このテストは特に意味はないが、インポートエラーを検知する

        try:
            from backend.app.services.auto_strategy.generators.strategies import (
                ConditionStrategy,
                DifferentIndicatorsStrategy,
                ComplexConditionsStrategy,
                IndicatorCharacteristicsStrategy,
            )
            # インポート成功
            assert True
        except ImportError as e:
            # インポート失敗はバグ
            pytest.fail(f"Import error in strategies/__init__.py: {e}")

    def test_circular_import_prevention(self):
        """循環インポート防止テスト"""
        # 循環インポートがないことを確認
        # 各クラスを個別にインポートできることを確認
        try:
            from backend.app.services.auto_strategy.generators.strategies.base_strategy import ConditionStrategy
            from backend.app.services.auto_strategy.generators.strategies.different_indicators_strategy import DifferentIndicatorsStrategy
            from backend.app.services.auto_strategy.generators.strategies.complex_conditions_strategy import ComplexConditionsStrategy
            from backend.app.services.auto_strategy.generators.strategies.indicator_characteristics_strategy import IndicatorCharacteristicsStrategy
            assert True
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_all_variable_consistency(self):
        """__all__変数の整合性テスト"""
        from backend.app.services.auto_strategy.generators.strategies import __all__

        # __all__に含まれるクラスが実際にインポート可能であることを確認
        for class_name in __all__:
            if class_name == 'ConditionStrategy':
                assert ConditionStrategy is not None
            elif class_name == 'DifferentIndicatorsStrategy':
                assert DifferentIndicatorsStrategy is not None
            elif class_name == 'ComplexConditionsStrategy':
                assert ComplexConditionsStrategy is not None
            elif class_name == 'IndicatorCharacteristicsStrategy':
                assert IndicatorCharacteristicsStrategy is not None

    def test_module_docstring(self):
        """モジュールドックストリングの確認"""
        import backend.app.services.auto_strategy.generators.strategies as strategies_module

        assert hasattr(strategies_module, '__doc__')
        assert strategies_module.__doc__ is not None
        assert "Strategies package" in strategies_module.__doc__


if __name__ == "__main__":
    pytest.main([__file__])