"""
MLトレーニング系の計算正確性テスト

数値計算、統計計算、数学的変換の正確性を検証するテストスイート。
計算の精度、数値安定性、エッジケースでの動作を包括的に検証します。
"""

import numpy as np
import pandas as pd
import logging
from decimal import getcontext
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_processing import DataProcessor

logger = logging.getLogger(__name__)

# 高精度計算のための設定
getcontext().prec = 50


class TestMLCalculations:
    """ML計算の正確性テストクラス"""

    def create_sample_data(self) -> pd.DataFrame:
        """テスト用のサンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')

        # 現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 1000)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'Volume': np.random.lognormal(10, 1, 1000)
        }).set_index('timestamp')

    def create_known_values_data(self) -> pd.DataFrame:
        """既知の正解値を持つテストデータ"""
        return pd.DataFrame({
            'values': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'weights': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        })

    def test_basic_statistics_accuracy(self):
        """基本統計量の計算精度テスト"""
        logger.info("=== 基本統計量の計算精度テスト ===")

        known_values_data = self.create_known_values_data()
        values = known_values_data['values']
        
        # 平均の検証
        calculated_mean = values.mean()
        expected_mean = 5.5
        assert abs(calculated_mean - expected_mean) < 1e-10, f"平均計算エラー: {calculated_mean} != {expected_mean}"
        
        # 分散の検証
        calculated_var = values.var(ddof=1)
        expected_var = 9.166666666666666  # 手計算による正解値
        assert abs(calculated_var - expected_var) < 1e-10, f"分散計算エラー: {calculated_var} != {expected_var}"
        
        # 標準偏差の検証
        calculated_std = values.std(ddof=1)
        expected_std = np.sqrt(expected_var)
        assert abs(calculated_std - expected_std) < 1e-10, f"標準偏差計算エラー: {calculated_std} != {expected_std}"
        
        logger.info("✅ 基本統計量の計算精度テスト完了")

    def test_correlation_accuracy(self):
        """相関係数の計算精度テスト"""
        logger.info("=== 相関係数の計算精度テスト ===")

        # 完全相関のテストケース
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 1  # 完全な正の相関
        
        correlation = np.corrcoef(x, y)[0, 1]
        assert abs(correlation - 1.0) < 1e-10, f"完全正相関の計算エラー: {correlation} != 1.0"
        
        # 完全負相関のテストケース
        y_neg = -2 * x + 10
        correlation_neg = np.corrcoef(x, y_neg)[0, 1]
        assert abs(correlation_neg - (-1.0)) < 1e-10, f"完全負相関の計算エラー: {correlation_neg} != -1.0"
        
        # 無相関のテストケース
        np.random.seed(42)
        x_random = np.random.normal(0, 1, 1000)
        y_random = np.random.normal(0, 1, 1000)
        correlation_random = np.corrcoef(x_random, y_random)[0, 1]
        assert abs(correlation_random) < 0.1, f"無相関の計算エラー: |{correlation_random}| >= 0.1"
        
        logger.info("✅ 相関係数の計算精度テスト完了")

    def test_percentile_accuracy(self):
        """パーセンタイル計算の精度テスト"""
        logger.info("=== パーセンタイル計算の精度テスト ===")

        known_values_data = self.create_known_values_data()
        values = known_values_data['values']
        
        # 既知の値でのパーセンタイル計算
        percentiles = [0, 25, 50, 75, 100]
        expected_values = [1.0, 3.25, 5.5, 7.75, 10.0]
        
        for p, expected in zip(percentiles, expected_values):
            calculated = np.percentile(values, p)
            assert abs(calculated - expected) < 1e-10, f"{p}%パーセンタイル計算エラー: {calculated} != {expected}"
        
        logger.info("✅ パーセンタイル計算の精度テスト完了")

    def test_moving_average_accuracy(self):
        """移動平均の計算精度テスト"""
        logger.info("=== 移動平均の計算精度テスト ===")

        # 簡単なテストケース
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window = 3
        
        # pandas rolling meanと手計算の比較
        rolling_mean = values.rolling(window=window).mean()
        
        # 手計算による検証（最初の3つの値）
        expected_third = (1 + 2 + 3) / 3  # 2.0
        expected_fourth = (2 + 3 + 4) / 3  # 3.0
        
        assert abs(rolling_mean.iloc[2] - expected_third) < 1e-10, f"3番目の移動平均エラー: {rolling_mean.iloc[2]} != {expected_third}"
        assert abs(rolling_mean.iloc[3] - expected_fourth) < 1e-10, f"4番目の移動平均エラー: {rolling_mean.iloc[3]} != {expected_fourth}"
        
        logger.info("✅ 移動平均の計算精度テスト完了")

    def test_zscore_accuracy(self):
        """Z-score計算の精度テスト"""
        logger.info("=== Z-score計算の精度テスト ===")

        known_values_data = self.create_known_values_data()
        values = known_values_data['values']
        
        # Z-scoreの手計算
        mean = values.mean()
        std = values.std(ddof=1)
        
        # 最初と最後の値のZ-scoreを検証
        first_zscore = (values.iloc[0] - mean) / std
        last_zscore = (values.iloc[-1] - mean) / std
        
        # scipyのZ-scoreと比較
        from scipy import stats
        scipy_zscores = stats.zscore(values, ddof=1)
        
        assert abs(first_zscore - scipy_zscores[0]) < 1e-10, f"最初のZ-score計算エラー: {first_zscore} != {scipy_zscores[0]}"
        assert abs(last_zscore - scipy_zscores[-1]) < 1e-10, f"最後のZ-score計算エラー: {last_zscore} != {scipy_zscores[-1]}"
        
        logger.info("✅ Z-score計算の精度テスト完了")

    def test_numerical_stability(self):
        """数値安定性テスト"""
        logger.info("=== 数値安定性テスト ===")
        
        # 非常に大きな値での計算
        large_values = pd.Series([1e15, 1e15 + 1, 1e15 + 2, 1e15 + 3])
        mean_large = large_values.mean()
        std_large = large_values.std()
        
        # 計算結果が有限であることを確認
        assert np.isfinite(mean_large), f"大きな値での平均計算が無限大: {mean_large}"
        assert np.isfinite(std_large), f"大きな値での標準偏差計算が無限大: {std_large}"
        
        # 非常に小さな値での計算
        small_values = pd.Series([1e-15, 2e-15, 3e-15, 4e-15])
        mean_small = small_values.mean()
        std_small = small_values.std()
        
        assert np.isfinite(mean_small), f"小さな値での平均計算が無限大: {mean_small}"
        assert np.isfinite(std_small), f"小さな値での標準偏差計算が無限大: {std_small}"
        
        logger.info("✅ 数値安定性テスト完了")

    def test_edge_cases(self):
        """エッジケースのテスト"""
        logger.info("=== エッジケースのテスト ===")
        
        # 単一値のケース
        single_value = pd.Series([5.0])
        assert single_value.mean() == 5.0, "単一値の平均計算エラー"
        # 単一値の標準偏差はNaNになるのが正常
        assert pd.isna(single_value.std()), "単一値の標準偏差計算エラー"
        
        # 同一値のケース
        identical_values = pd.Series([3.0, 3.0, 3.0, 3.0])
        assert identical_values.mean() == 3.0, "同一値の平均計算エラー"
        assert identical_values.std() == 0.0, "同一値の標準偏差計算エラー"
        
        # NaN値を含むケース
        with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        mean_with_nan = with_nan.mean()  # NaNを除外して計算
        expected_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4
        assert abs(mean_with_nan - expected_mean) < 1e-10, f"NaN含有データの平均計算エラー: {mean_with_nan} != {expected_mean}"
        
        logger.info("✅ エッジケースのテスト完了")

    def test_data_processor_calculations(self):
        """DataProcessorの計算精度テスト"""
        logger.info("=== DataProcessorの計算精度テスト ===")

        processor = DataProcessor()
        sample_data = self.create_sample_data()

        # 基本的な前処理の数値精度を検証
        processed_data = processor.preprocess_features(
            sample_data[['Close']].copy(),
            scale_features=True,
            remove_outliers=False
        )
        
        # スケーリング後の統計量を検証
        scaled_mean = processed_data['Close'].mean()
        scaled_std = processed_data['Close'].std()
        
        # StandardScalerの場合、平均は0、標準偏差は1に近くなるはず
        assert abs(scaled_mean) < 1e-10, f"スケーリング後の平均が0でない: {scaled_mean}"
        assert abs(scaled_std - 1.0) < 1e-3, f"スケーリング後の標準偏差が1でない: {scaled_std}"
        
        logger.info("✅ DataProcessorの計算精度テスト完了")


def run_all_calculation_tests():
    """すべての計算テストを実行"""
    logger.info("🧮 ML計算正確性テストスイートを開始")

    test_instance = TestMLCalculations()

    try:
        # 各テストを実行
        test_instance.test_basic_statistics_accuracy()
        test_instance.test_correlation_accuracy()
        test_instance.test_percentile_accuracy()
        test_instance.test_moving_average_accuracy()
        test_instance.test_zscore_accuracy()
        test_instance.test_numerical_stability()
        test_instance.test_edge_cases()
        test_instance.test_data_processor_calculations()
        
        logger.info("🎉 すべての計算正確性テストが正常に完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 計算正確性テストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_calculation_tests()
    sys.exit(0 if success else 1)
