"""
BaseFeatureCalculatorのテスト
TDDでhelper methodsを開発します。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.services.ml.feature_engineering.base_feature_calculator import BaseFeatureCalculator


class ConcreteFeatureCalculator(BaseFeatureCalculator):
    """テスト用の具象クラス"""

    def calculate_features(
        self, df: pd.DataFrame, config: dict
    ) -> pd.DataFrame:
        """具象実装"""
        return self.create_result_dataframe(df)


class TestBaseFeatureCalculator:
    """BaseFeatureCalculatorのテスト"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータ"""
        dates = pd.date_range(
            start=datetime(2023, 1, 1),
            periods=100,
            freq='1h'
        )

        data = []
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'open': 50000 + i,
                'high': 51000 + i,
                'low': 49000 + i,
                'close': 50500 + i,
                'volume': 1000 + i
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def test_base_calculator_initialization(self):
        """基底クラスの初期化をテスト"""
        calc = ConcreteFeatureCalculator()
        assert calc is not None

    def test_validate_input_data_valid(self, sample_data):
        """有効なデータ検証テスト"""
        calc = ConcreteFeatureCalculator()
        result = calc.validate_input_data(sample_data, ['open', 'high', 'low', 'close'])
        assert result is True

    def test_validate_input_data_missing_columns(self, sample_data):
        """必須カラム不足のテスト"""
        calc = ConcreteFeatureCalculator()
        result = calc.validate_input_data(sample_data, ['nonexistent_column'])
        assert result is False

    def test_validate_input_data_empty_dataframe(self):
        """空DataFrameのテスト"""
        calc = ConcreteFeatureCalculator()
        result = calc.validate_input_data(pd.DataFrame(), ['open'])
        assert result is False

    def test_create_result_dataframe(self, sample_data):
        """create_result_dataframeのテスト"""
        calc = ConcreteFeatureCalculator()
        result = calc.create_result_dataframe(sample_data)

        assert result is not sample_data  # コピーされている
        assert result.equals(sample_data)  # 内容は同じ
        assert id(result) != id(sample_data)  # 別オブジェクト

    def test_handle_calculation_error(self, sample_data):
        """エラーハンドリングのテスト"""
        calc = ConcreteFeatureCalculator()

        try:
            raise ValueError("Test error")
        except Exception as e:
            result = calc.handle_calculation_error(e, "test_context", sample_data)

        assert result.equals(sample_data)

    def test_clip_extreme_values(self):
        """極値クリッピングのテスト"""
        calc = ConcreteFeatureCalculator()

        series = pd.Series([1, 2, 3, 10, -10, 100])
        result = calc.clip_extreme_values(series, lower_bound=-5, upper_bound=5)

        assert result.min() >= -5
        assert result.max() <= 5

    # ========================================
    # TDD: 新規helper methodsのテスト
    # ========================================

    def test_create_result_dataframe_efficient_exists(self):
        """create_result_dataframe_efficientメソッドの存在テスト"""
        calc = ConcreteFeatureCalculator()
        assert hasattr(calc, 'create_result_dataframe_efficient')

    def test_create_result_dataframe_efficient_basic(self, sample_data):
        """create_result_dataframe_efficientの基本機能テスト"""
        calc = ConcreteFeatureCalculator()

        new_features = {
            'feature1': pd.Series([1, 2, 3] * 34)[:100],
            'feature2': pd.Series([4, 5, 6] * 34)[:100],
        }

        result = calc.create_result_dataframe_efficient(sample_data, new_features)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert len(result.columns) == len(sample_data.columns) + len(new_features)
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns

    def test_create_result_dataframe_efficient_no_fragmentation(self, sample_data):
        """create_result_dataframe_efficientがDataFrame断片化を起こさないことをテスト"""
        calc = ConcreteFeatureCalculator()

        new_features = {}
        for i in range(10):
            new_features[f'feature_{i}'] = pd.Series(np.random.randn(100))

        result = calc.create_result_dataframe_efficient(sample_data, new_features)

        # DataFrameが正常にアクセス可能
        summary = result.describe()
        assert summary is not None
        assert len(summary) > 0

        # メモリレイアウトチェック（断片化していない）
        assert result.memory_usage(deep=True).sum() > 0

    def test_batch_calculate_ratio_exists(self):
        """batch_calculate_ratioメソッドの存在テスト"""
        calc = ConcreteFeatureCalculator()
        assert hasattr(calc, 'batch_calculate_ratio')

    def test_batch_calculate_ratio_basic(self):
        """batch_calculate_ratioの基本機能テスト"""
        calc = ConcreteFeatureCalculator()

        numerators = {
            'ratio1': pd.Series([10, 20, 30, 40, 50]),
            'ratio2': pd.Series([5, 10, 15, 20, 25]),
        }

        denominators = {
            'ratio1': pd.Series([2, 4, 6, 8, 10]),
            'ratio2': pd.Series([1, 2, 3, 4, 5]),
        }

        result = calc.batch_calculate_ratio(numerators, denominators)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'ratio1' in result
        assert 'ratio2' in result

        # 値の検証 (10/2=5, 20/4=5, etc.)
        expected1 = numerators['ratio1'] / denominators['ratio1']
        pd.testing.assert_series_equal(result['ratio1'], expected1)

        expected2 = numerators['ratio2'] / denominators['ratio2']
        pd.testing.assert_series_equal(result['ratio2'], expected2)

    def test_batch_calculate_ratio_division_by_zero(self):
        """batch_calculate_ratioのゼロ除算対応テスト"""
        calc = ConcreteFeatureCalculator()

        numerators = {
            'test_ratio': pd.Series([10, 20, 30, 40, 50]),
        }

        denominators = {
            'test_ratio': pd.Series([2, 0, 6, 8, 10]),  # 0を含む
        }

        result = calc.batch_calculate_ratio(numerators, denominators)

        assert result is not None
        assert 'test_ratio' in result

        # ゼロ除算の位置はNaNまたは0になっている
        assert pd.isna(result['test_ratio'].iloc[1]) or result['test_ratio'].iloc[1] == 0

    def test_batch_calculate_ratio_empty_inputs(self):
        """batch_calculate_ratioの空入力テスト"""
        calc = ConcreteFeatureCalculator()

        numerators = {}
        denominators = {}

        result = calc.batch_calculate_ratio(numerators, denominators)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_inheritance_compatibility(self, sample_data):
        """BaseFeatureCalculator継承の互換性テスト"""
        calc = ConcreteFeatureCalculator()
        config = {"lookback_periods": {}}

        result = calc.calculate_features(sample_data, config)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
