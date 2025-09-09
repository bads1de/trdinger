"""
バグ22対応: ラベル生成のエラーハンドリングテスト

pct_change(), shift(), fillna()などのpandas操作でtry exceptがなかった箇所にテスト追加
TDD実装によるバグ22修正検証
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from app.utils.label_generation import (
    SimpleLabelGenerator,
    LabelGenerator,
    PriceChangeTransformer,
    ThresholdMethod,
)


class TestLabelGenerationErrorHandling:
    """ラベル生成のエラーハンドリングテスト"""

    def test_price_change_transformer_pandas_operations_error_handling(self):
        """PriceChangeTransformerのpct_change, dropna操作のエラーハンドリングテスト"""
        transformer = PriceChangeTransformer()

        # 正常データでの処理確認
        normal_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = transformer.transform(normal_data)

        # 結果が2次元配列であることの確認
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 1  # 1つの特徴量

        # NaNを含むデータを扱うことの確認
        nan_data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result_with_nan = transformer.transform(nan_data)

        # NaNデータも適切に処理されること
        assert isinstance(result_with_nan, np.ndarray)
        assert not np.any(np.isnan(result_with_nan))  # NaNが除去されている

    def test_price_change_transformer_with_invalid_data(self):
        """無効なデータでのPriceChangeTransformerエラーハンドリングテスト"""
        transformer = PriceChangeTransformer()

        # 空のデータを渡してエラー処理を確認
        empty_data = pd.Series([], dtype=float)
        with pytest.raises((ValueError, AttributeError)):
            transformer.transform(empty_data)

    def test_simple_label_generator_pct_change_index_error(self):
        """SimpleLabelGeneratorのpct_change().dropna().indexエラーハンドリングテスト"""
        generator = SimpleLabelGenerator()
        generator.fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

        # 正常データでの変換確認
        valid_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        labels = generator.transform(valid_data)

        assert isinstance(labels, pd.Series)
        assert len(labels) == 5  # 最初のNaN除去後の長さ

    def test_simple_label_generator_with_empty_data_exception(self):
        """空データでのエラーハンドリングテスト"""
        generator = SimpleLabelGenerator()

        # fitされていない状態でtransformを呼び出し
        empty_data = pd.Series([], dtype=float)
        with pytest.raises(ValueError) as excinfo:
            generator.transform(empty_data)

        assert "fit()を先に実行してください" in str(excinfo.value)

    def test_label_generator_generate_labels_pct_change_error_handling(self):
        """LabelGenerator.generate_labelsでのpct_change操作エラーハンドリングテスト"""
        label_gen = LabelGenerator()

        # 正常データでのラベル生成確認
        price_data = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        labels, info = label_gen.generate_labels(price_data)

        assert isinstance(labels, pd.Series)
        assert isinstance(info, dict)
        assert len(labels) >= len(price_data) - 1  # shift(-1)による長さ減少

    def test_label_generator_with_nan_data_shift_handling(self):
        """NaNデータでのshift操作エラーハンドリングテスト"""
        label_gen = LabelGenerator()

        # NaNを含むデータを渡す
        price_data = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0])
        labels, info = label_gen.generate_labels(price_data)

        # エラーハンドリングにより正常動作すること
        assert isinstance(labels, pd.Series)
        assert not labels.isnull().any()  # NaNが除去されている

    def test_label_generator_empty_series_error(self):
        """空のSeriesでのエラーハンドリングテスト"""
        label_gen = LabelGenerator()

        empty_price_data = pd.Series([], dtype=float)
        with pytest.raises(ValueError) as excinfo:
            label_gen.generate_labels(empty_price_data)

        assert "有効な価格変化率データがありません" in str(excinfo.value)

    def test_kbins_discretizer_threshold_calculation_nan_handling(self):
        """KBinsDiscretizer名の閾値計算でのNaNハンドリングテスト"""
        label_gen = LabelGenerator()

        # NaNのみのデータを渡す
        nan_only_data = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError) as excinfo:
            label_gen.generate_labels(nan_only_data)

        # NaN除去後のデータが空になるケース
        assert "有効な価格変化率データがありません" in str(excinfo.value)

    @patch('app.utils.label_generation.logger')
    def test_error_logging_in_pandas_operations(self, mock_logger):
        """pandas操作エラー時のログ記録テスト"""
        transformer = PriceChangeTransformer()

        # エラーを発生させるようなデータを渡す
        # Noneを直接渡すとAttributeErrorが発生
        with pytest.raises(AttributeError):
            transformer.transform(None)  # Noneを渡すとエラー

        # エラーログが記録されていることを確認
        mock_logger.error.assert_called()

    def test_dynamic_volatility_rolling_std_error_handling(self):
        """動的ボラティリティ計算でのrolling stdエラーハンドリングテスト"""
        label_gen = LabelGenerator()

        # 少ないデータでのボラティリティ計算
        short_data = pd.Series([100.0, 101.0])  # ボラティリティ計算に不十分なデータ
        labels, info = label_gen.generate_labels(
            short_data,
            method=ThresholdMethod.DYNAMIC_VOLATILITY,
            volatility_window=24  # 必要な長さより長い
        )

        # エラーハンドリングにより何かしらの結果が返されること
        assert isinstance(labels, pd.Series) or labels is None

    def test_quantile_calculation_with_tiny_data(self):
        """小さなデータでの分位数計算エラーハンドリングテスト"""
        label_gen = LabelGenerator()

        # 非常に小さなデータでの分位数計算
        tiny_data = pd.Series([100.0])  # 分位数計算が難しいデータ
        results = label_gen.generate_labels(
            tiny_data,
            method=ThresholdMethod.QUANTILE
        )

        # エラーハンドリングにより何らかの結果が得られること
        assert isinstance(results, tuple)

    def test_rolling_volatility_calculation_division_by_zero_protection(self):
        """ローリングボラティリティ計算でのゼロ除算保護テスト"""
        label_gen = LabelGenerator()

        # 全て同じ値のデータ（ボラティリティ=0）
        constant_data = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
        labels, info = label_gen.generate_labels(
            constant_data,
            method=ThresholdMethod.DYNAMIC_VOLATILITY
        )

        # ゼロボラティリティでもエラーが発生せず処理されること
        assert isinstance(labels, pd.Series)

    def test_pct_change_on_single_value_error(self):
        """単一値データでのpct_changeエラーハンドリングテスト"""
        # 単一値データでのpct_changeはNaNを生成
        single_value_data = pd.Series([100.0])
        pct_change_result = single_value_data.pct_change()

        # pct_changeの結果がNaN（計算不能）であることのテスト
        assert pct_change_result.isnull().all()

        # このようなデータを適切に処理できること
        label_gen = LabelGenerator()
        labels, info = label_gen.generate_labels(single_value_data)

        # データが少なすぎる場合、エラーが発生するか適切に処理されること
        assert isinstance(labels, pd.Series) or labels is None