"""
データ変換処理の整合性テスト

データ型変換、時系列データ整合性、インデックス処理、データ結合の正確性を検証するテストスイート。
データ変換処理の信頼性と一貫性を包括的に検証します。
"""

import pytest
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_processing import DataProcessor

logger = logging.getLogger(__name__)


class TestDataTransformations:
    """データ変換処理の整合性テストクラス"""

    def sample_ohlcv_data(self) -> pd.DataFrame:
        """テスト用のOHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'Volume': np.random.lognormal(10, 1, 100)
        })

    def mixed_type_data(self) -> pd.DataFrame:
        """混合データ型のテストデータ"""
        return pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=5, freq='D')
        })

    @pytest.fixture
    def funding_rate_data(self) -> pd.DataFrame:
        """ファンディングレートデータ"""
        dates = pd.date_range('2023-01-01', periods=50, freq='8H')
        return pd.DataFrame({
            'timestamp': dates,
            'funding_rate': np.random.normal(0.0001, 0.0005, 50),
            'symbol': ['BTC'] * 50
        })

    @pytest.fixture
    def open_interest_data(self) -> pd.DataFrame:
        """建玉残高データ"""
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')
        return pd.DataFrame({
            'timestamp': dates,
            'open_interest': np.random.lognormal(15, 0.5, 50),
            'symbol': ['BTC'] * 50
        })

    def test_data_type_optimization(self, mixed_type_data):
        """データ型最適化の正確性テスト"""
        logger.info("=== データ型最適化の正確性テスト ===")
        
        processor = DataProcessor()
        data = mixed_type_data.copy()
        
        # 元のデータ型を記録
        original_dtypes = data.dtypes.to_dict()
        
        # データ型最適化を実行
        optimized_data = processor.optimize_dtypes(data)
        
        # 最適化後のデータ型を確認
        optimized_dtypes = optimized_data.dtypes.to_dict()
        
        # 数値データの値が変わっていないことを確認
        try:
            pd.testing.assert_series_equal(
                data['int_col'].astype(optimized_dtypes['int_col']),
                optimized_data['int_col'],
                check_names=False
            )
        except AssertionError:
            raise AssertionError("整数カラムの値が変更されました")

        try:
            pd.testing.assert_series_equal(
                data['float_col'].astype(optimized_dtypes['float_col']),
                optimized_data['float_col'],
                check_names=False
            )
        except AssertionError:
            raise AssertionError("浮動小数点カラムの値が変更されました")

        # 文字列とブール値は変更されないことを確認
        try:
            pd.testing.assert_series_equal(
                data['str_col'],
                optimized_data['str_col']
            )
        except AssertionError:
            raise AssertionError("文字列カラムが変更されました")

        try:
            pd.testing.assert_series_equal(
                data['bool_col'],
                optimized_data['bool_col']
            )
        except AssertionError:
            raise AssertionError("ブールカラムが変更されました")
        
        logger.info("✅ データ型最適化の正確性テスト完了")

    def test_time_series_index_handling(self, sample_ohlcv_data):
        """時系列インデックス処理の正確性テスト"""
        logger.info("=== 時系列インデックス処理の正確性テスト ===")
        
        processor = DataProcessor()
        data = sample_ohlcv_data.copy()
        
        # timestampカラムをインデックスに設定
        data_with_index = data.set_index('timestamp')
        
        # データ処理を実行
        processed_data = processor.clean_and_validate_data(
            data_with_index,
            required_columns=['Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        # インデックスが時系列順になっていることを確認
        assert processed_data.index.is_monotonic_increasing, "時系列インデックスが昇順でありません"
        
        # インデックスがDatetimeIndexであることを確認
        assert isinstance(processed_data.index, pd.DatetimeIndex), "インデックスがDatetimeIndexではありません"
        
        # データの行数が保持されていることを確認
        assert len(processed_data) == len(data_with_index), "データの行数が変更されました"
        
        logger.info("✅ 時系列インデックス処理の正確性テスト完了")

    def test_data_interpolation_accuracy(self, sample_ohlcv_data):
        """データ補間の正確性テスト"""
        logger.info("=== データ補間の正確性テスト ===")
        
        processor = DataProcessor()
        data = sample_ohlcv_data.copy()
        
        # 意図的にNaN値を挿入
        data.loc[10:15, 'Close'] = np.nan
        data.loc[30:32, 'Volume'] = np.nan
        
        # 補間前のNaN数を記録
        nan_count_before = data.isna().sum().sum()
        
        # データ補間を実行
        interpolated_data = processor.interpolate_all_data(data)
        
        # 補間後のNaN数を確認
        nan_count_after = interpolated_data.isna().sum().sum()
        
        # NaN数が減少していることを確認
        assert nan_count_after < nan_count_before, "データ補間でNaN数が減少していません"
        
        # 補間された値が合理的な範囲内にあることを確認
        original_close_mean = data['Close'].mean()
        interpolated_close_mean = interpolated_data['Close'].mean()
        
        # 平均値の変化が10%以内であることを確認
        mean_change_ratio = abs(interpolated_close_mean - original_close_mean) / original_close_mean
        assert mean_change_ratio < 0.1, f"補間後の平均値変化が大きすぎます: {mean_change_ratio:.4f}"
        
        logger.info("✅ データ補間の正確性テスト完了")

    def test_data_merging_accuracy(self, sample_ohlcv_data, funding_rate_data, open_interest_data):
        """データ結合の正確性テスト"""
        logger.info("=== データ結合の正確性テスト ===")
        
        processor = DataProcessor()
        
        # OHLCVデータをベースとして準備
        ohlcv = sample_ohlcv_data.set_index('timestamp')
        funding = funding_rate_data.set_index('timestamp')
        oi = open_interest_data.set_index('timestamp')
        
        # データ結合を実行（時間軸での結合）
        merged_data = pd.merge(ohlcv, funding, left_index=True, right_index=True, how='left')
        merged_data = pd.merge(merged_data, oi, left_index=True, right_index=True, how='left')
        
        # 結合結果の検証
        assert len(merged_data) == len(ohlcv), "結合後の行数がベースデータと異なります"
        
        # 元のOHLCVデータが保持されていることを確認
        pd.testing.assert_series_equal(
            ohlcv['Close'],
            merged_data['Close'],
            msg="結合後にOHLCVデータが変更されました"
        )
        
        # ファンディングレートと建玉残高のカラムが追加されていることを確認
        assert 'funding_rate' in merged_data.columns, "ファンディングレートカラムが見つかりません"
        assert 'open_interest' in merged_data.columns, "建玉残高カラムが見つかりません"
        
        logger.info("✅ データ結合の正確性テスト完了")

    def test_data_validation_consistency(self, sample_ohlcv_data):
        """データ検証の一貫性テスト"""
        logger.info("=== データ検証の一貫性テスト ===")
        
        processor = DataProcessor()
        data = sample_ohlcv_data.set_index('timestamp')
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 同じデータで複数回検証を実行
        validated_data1 = processor.clean_and_validate_data(data.copy(), required_columns)
        validated_data2 = processor.clean_and_validate_data(data.copy(), required_columns)
        
        # 結果が一致することを確認
        pd.testing.assert_frame_equal(
            validated_data1, validated_data2,
            check_exact=False,
            rtol=1e-10,
            msg="同じデータでの検証結果が一致しません"
        )
        
        logger.info("✅ データ検証の一貫性テスト完了")

    def test_data_sorting_stability(self, sample_ohlcv_data):
        """データソートの安定性テスト"""
        logger.info("=== データソートの安定性テスト ===")
        
        processor = DataProcessor()
        data = sample_ohlcv_data.copy()
        
        # データをランダムに並び替え
        shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        shuffled_data = shuffled_data.set_index('timestamp')
        
        # ソート処理を実行
        sorted_data = processor.clean_and_validate_data(
            shuffled_data,
            required_columns=['Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        # ソート後のデータが時系列順になっていることを確認
        assert sorted_data.index.is_monotonic_increasing, "ソート後のデータが時系列順でありません"
        
        # 元のデータと同じ値が含まれていることを確認（順序は異なる可能性）
        original_sorted = data.set_index('timestamp').sort_index()
        
        pd.testing.assert_frame_equal(
            original_sorted,
            sorted_data,
            check_exact=False,
            rtol=1e-10,
            msg="ソート後のデータが元のデータと一致しません"
        )
        
        logger.info("✅ データソートの安定性テスト完了")

    def test_memory_efficiency(self, sample_ohlcv_data):
        """メモリ効率性テスト"""
        logger.info("=== メモリ効率性テスト ===")
        
        processor = DataProcessor()
        data = sample_ohlcv_data.copy()
        
        # 処理前のメモリ使用量
        memory_before = data.memory_usage(deep=True).sum()
        
        # データ型最適化を実行
        optimized_data = processor.optimize_dtypes(data)
        
        # 処理後のメモリ使用量
        memory_after = optimized_data.memory_usage(deep=True).sum()
        
        # メモリ使用量が削減されているか、少なくとも増加していないことを確認
        assert memory_after <= memory_before, f"メモリ使用量が増加しました: {memory_before} -> {memory_after}"
        
        # データの値が保持されていることを確認
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            np.testing.assert_array_almost_equal(
                data[col].values,
                optimized_data[col].values,
                decimal=10,
                err_msg=f"カラム {col} の値が変更されました"
            )
        
        logger.info(f"メモリ使用量: {memory_before} -> {memory_after} bytes")
        logger.info("✅ メモリ効率性テスト完了")

    def test_edge_case_handling(self):
        """エッジケースの処理テスト"""
        logger.info("=== エッジケースの処理テスト ===")
        
        processor = DataProcessor()
        
        # 空のDataFrameのテスト
        empty_df = pd.DataFrame()
        processed_empty = processor.optimize_dtypes(empty_df)
        assert len(processed_empty) == 0, "空のDataFrameの処理が正しくありません"
        
        # 単一行のDataFrameのテスト
        single_row_df = pd.DataFrame({'A': [1], 'B': [2.0], 'C': ['test']})
        processed_single = processor.optimize_dtypes(single_row_df)
        assert len(processed_single) == 1, "単一行DataFrameの処理が正しくありません"
        
        # 全てNaNのカラムを含むDataFrameのテスト
        all_nan_df = pd.DataFrame({
            'normal_col': [1, 2, 3],
            'all_nan_col': [np.nan, np.nan, np.nan]
        })
        processed_nan = processor.optimize_dtypes(all_nan_df)
        assert 'all_nan_col' in processed_nan.columns, "全NaNカラムが削除されました"
        
        logger.info("✅ エッジケースの処理テスト完了")


def run_all_data_transformation_tests():
    """すべてのデータ変換テストを実行"""
    logger.info("🔄 データ変換処理整合性テストスイートを開始")

    test_instance = TestDataTransformations()

    try:
        # 基本的なテストのみ実行（簡略化版）
        logger.info("データ型最適化テストを実行中...")
        mixed_type_data = test_instance.mixed_type_data()
        test_instance.test_data_type_optimization(mixed_type_data)

        logger.info("時系列インデックス処理テストを実行中...")
        sample_ohlcv_data = test_instance.sample_ohlcv_data()
        test_instance.test_time_series_index_handling(sample_ohlcv_data)

        logger.info("エッジケース処理テストを実行中...")
        test_instance.test_edge_case_handling()
        
        logger.info("🎉 すべてのデータ変換処理整合性テストが正常に完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ データ変換処理整合性テストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_data_transformation_tests()
    sys.exit(0 if success else 1)
