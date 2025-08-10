"""
エッジケースと検証のテスト

型変換なし実装でのエッジケースと入力検証を確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.math_operators import MathOperatorsIndicators
from app.services.indicators.utils import (
    PandasTAError,
    validate_input,
    validate_multi_input,
    validate_series_data,
    validate_indicator_parameters
)


class TestEdgeCasesValidation:
    """エッジケースと検証テストクラス"""

    def test_empty_data_handling(self):
        """空データの処理テスト"""
        empty_series = pd.Series([])
        empty_array = np.array([])
        
        # 空のSeriesでのエラー
        with pytest.raises(Exception):
            TrendIndicators.sma(empty_series, 5)
        
        # 空のArrayでのエラー
        with pytest.raises(Exception):
            TrendIndicators.sma(empty_array, 5)

    def test_single_value_data(self):
        """単一値データの処理テスト"""
        single_series = pd.Series([100.0])
        single_array = np.array([100.0])
        
        # 期間が1の場合は計算可能
        result_series = TrendIndicators.sma(single_series, 1)
        result_array = TrendIndicators.sma(single_array, 1)
        
        assert len(result_series) == 1
        assert len(result_array) == 1
        assert result_series[0] == 100.0
        assert result_array[0] == 100.0
        
        # 期間が1より大きい場合はエラー
        with pytest.raises(Exception):
            TrendIndicators.sma(single_series, 2)

    def test_nan_values_handling(self):
        """NaN値の処理テスト"""
        # NaN値を含むデータ
        data_with_nan = pd.Series([100, 101, np.nan, 103, 104, 105])
        
        # SMA計算（NaN値があっても計算は実行される）
        result = TrendIndicators.sma(data_with_nan, 3)
        assert len(result) == len(data_with_nan)
        
        # 結果にNaNが含まれることを確認
        assert np.isnan(result).any()

    def test_infinite_values_handling(self):
        """無限大値の処理テスト"""
        # 無限大値を含むデータ
        data_with_inf = pd.Series([100, 101, np.inf, 103, 104])
        
        # 検証関数でエラーが発生することを確認
        with pytest.raises(PandasTAError):
            validate_series_data(data_with_inf, 3)

    def test_negative_values_handling(self):
        """負の値の処理テスト"""
        negative_data = pd.Series([-100, -101, -102, -103, -104])
        
        # SMAは負の値でも計算可能
        result = TrendIndicators.sma(negative_data, 3)
        assert len(result) == len(negative_data)
        assert all(result[~np.isnan(result)] < 0)  # NaN以外は負の値

    def test_zero_values_handling(self):
        """ゼロ値の処理テスト"""
        zero_data = pd.Series([0, 0, 0, 0, 0])
        
        # SMAはゼロ値でも計算可能
        result = TrendIndicators.sma(zero_data, 3)
        assert len(result) == len(zero_data)
        assert all(result[~np.isnan(result)] == 0)  # NaN以外はゼロ

    def test_very_large_values(self):
        """非常に大きな値の処理テスト"""
        large_data = pd.Series([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        
        # 大きな値でも計算可能
        result = TrendIndicators.sma(large_data, 3)
        assert len(result) == len(large_data)
        assert not np.isnan(result[-1])  # 最後の値はNaNでない

    def test_very_small_values(self):
        """非常に小さな値の処理テスト"""
        small_data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        
        # 小さな値でも計算可能
        result = TrendIndicators.sma(small_data, 3)
        assert len(result) == len(small_data)
        assert not np.isnan(result[-1])  # 最後の値はNaNでない

    def test_invalid_period_parameters(self):
        """無効な期間パラメータのテスト"""
        data = pd.Series([100, 101, 102, 103, 104])
        
        # ゼロ期間
        with pytest.raises(Exception):
            TrendIndicators.sma(data, 0)
        
        # 負の期間
        with pytest.raises(Exception):
            TrendIndicators.sma(data, -1)
        
        # データ長より長い期間
        with pytest.raises(Exception):
            TrendIndicators.sma(data, 10)

    def test_non_numeric_data_types(self):
        """非数値データ型のテスト"""
        # 文字列データ
        string_data = pd.Series(['a', 'b', 'c', 'd', 'e'])
        
        with pytest.raises(Exception):
            TrendIndicators.sma(string_data, 3)
        
        # 日付データ
        date_data = pd.Series(pd.date_range('2023-01-01', periods=5))
        
        with pytest.raises(Exception):
            TrendIndicators.sma(date_data, 3)

    def test_mixed_data_types(self):
        """混合データ型のテスト"""
        # 数値と文字列の混合
        mixed_data = pd.Series([100, '101', 102, 103, 104])
        
        # pandas.Seriesは自動的に型変換を試みるが、失敗する可能性がある
        with pytest.raises(Exception):
            TrendIndicators.sma(mixed_data, 3)

    def test_parameter_validation_functions(self):
        """パラメータ検証関数のテスト"""
        # 正常なパラメータ
        validate_indicator_parameters(14, 26, 9)
        
        # 無効なパラメータ
        with pytest.raises(PandasTAError):
            validate_indicator_parameters(0)
        
        with pytest.raises(PandasTAError):
            validate_indicator_parameters(-5)

    def test_input_validation_functions(self):
        """入力検証関数のテスト"""
        data = pd.Series([100, 101, 102, 103, 104])
        
        # 正常な検証
        validate_input(data, 3)
        
        # 期間が長すぎる
        with pytest.raises(PandasTAError):
            validate_input(data, 10)
        
        # 無効な期間
        with pytest.raises(PandasTAError):
            validate_input(data, 0)

    def test_multi_input_validation(self):
        """複数入力検証のテスト"""
        high = pd.Series([105, 106, 107, 108, 109])
        low = pd.Series([95, 96, 97, 98, 99])
        close = pd.Series([100, 101, 102, 103, 104])
        
        # 正常な検証
        validate_multi_input(high, low, close, 3)
        
        # 高値が安値より低い場合
        invalid_high = pd.Series([90, 91, 92, 93, 94])  # 安値より低い
        with pytest.raises(PandasTAError):
            validate_multi_input(invalid_high, low, close, 3)

    def test_data_length_mismatch(self):
        """データ長不一致のテスト"""
        data1 = pd.Series([100, 101, 102, 103, 104])
        data2 = pd.Series([200, 201, 202])  # 長さが異なる
        
        # 数学演算子での長さ不一致
        with pytest.raises(Exception):
            MathOperatorsIndicators.add(data1, data2)

    def test_extreme_period_values(self):
        """極端な期間値のテスト"""
        data = pd.Series(range(1000))  # 1000個のデータ
        
        # 非常に大きな期間（データ長と同じ）
        result = TrendIndicators.sma(data, 1000)
        assert len(result) == 1000
        
        # 期間1（最小値）
        result = TrendIndicators.sma(data, 1)
        assert len(result) == 1000
        assert np.array_equal(result, data.values)  # 期間1のSMAは元データと同じ

    def test_memory_intensive_operations(self):
        """メモリ集約的操作のテスト"""
        # 大きなデータセット
        large_data = pd.Series(np.random.rand(100000))
        
        # メモリ使用量を監視しながら計算
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = TrendIndicators.sma(large_data, 1000)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # メモリ増加が合理的であることを確認（100MB以下）
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.2f} MB"
        assert len(result) == len(large_data)

    def test_concurrent_access_edge_cases(self):
        """並行アクセスのエッジケーステスト"""
        import threading
        import time
        
        data = pd.Series(range(1000))
        errors = []
        results = []
        
        def calculate_with_different_periods():
            try:
                # 異なる期間で計算
                for period in [10, 20, 50, 100]:
                    result = TrendIndicators.sma(data, period)
                    results.append(len(result))
                    time.sleep(0.001)  # 少し待機
            except Exception as e:
                errors.append(e)
        
        # 複数スレッドで同時実行
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=calculate_with_different_periods)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # エラーが発生していないことを確認
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # 全ての結果が正しい長さであることを確認
        for result_length in results:
            assert result_length == 1000

    def test_type_coercion_edge_cases(self):
        """型強制のエッジケーステスト"""
        # int型のSeries
        int_data = pd.Series([100, 101, 102, 103, 104], dtype=int)
        result = TrendIndicators.sma(int_data, 3)
        assert result.dtype == np.float64  # 結果はfloat64
        
        # float32型のSeries
        float32_data = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
        result = TrendIndicators.sma(float32_data, 3)
        assert result.dtype == np.float64  # 結果はfloat64

    def test_index_handling_edge_cases(self):
        """インデックス処理のエッジケーステスト"""
        # 非連続インデックス
        data = pd.Series([100, 101, 102, 103, 104], index=[0, 2, 4, 6, 8])
        result = TrendIndicators.sma(data, 3)
        assert len(result) == 5  # 長さは保持される
        
        # 文字列インデックス
        data = pd.Series([100, 101, 102, 103, 104], index=['a', 'b', 'c', 'd', 'e'])
        result = TrendIndicators.sma(data, 3)
        assert len(result) == 5  # 長さは保持される

    def test_boundary_conditions(self):
        """境界条件のテスト"""
        # 最小データセット
        min_data = pd.Series([100.0, 101.0])
        
        # 期間2で計算
        result = TrendIndicators.sma(min_data, 2)
        assert len(result) == 2
        assert not np.isnan(result[-1])  # 最後の値は計算可能
        
        # 期間1で計算
        result = TrendIndicators.sma(min_data, 1)
        assert len(result) == 2
        assert np.array_equal(result, min_data.values)

    def test_numerical_precision_edge_cases(self):
        """数値精度のエッジケーステスト"""
        # 非常に近い値
        close_values = pd.Series([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005])
        result = TrendIndicators.sma(close_values, 3)
        assert len(result) == 5
        assert not np.isnan(result[-1])
        
        # 精度の確認
        expected_last = (1.0000000003 + 1.0000000004 + 1.0000000005) / 3
        assert abs(result[-1] - expected_last) < 1e-15
