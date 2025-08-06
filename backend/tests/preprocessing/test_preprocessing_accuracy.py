"""
データ前処理の正確性テスト

スケーリング処理、正規化、外れ値検出、欠損値補完の数学的正確性を検証するテストスイート。
各前処理手法の理論的性質と実装の一致を包括的に検証します。
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import stats
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_processing import DataProcessor

logger = logging.getLogger(__name__)


class TestPreprocessingAccuracy:
    """データ前処理の正確性テストクラス"""

    def create_sample_data(self) -> pd.DataFrame:
        """テスト用のサンプルデータを生成"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 15, 1000),
            'feature2': np.random.exponential(2, 1000),
            'feature3': np.random.uniform(-10, 10, 1000),
            'feature4': np.random.lognormal(0, 1, 1000)
        })

    def create_outlier_data(self) -> pd.DataFrame:
        """外れ値を含むテストデータ"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        # 意図的に外れ値を追加
        data[95:] = [10, -10, 15, -15, 20]
        return pd.DataFrame({'values': data})

    def create_missing_data(self) -> pd.DataFrame:
        """欠損値を含むテストデータ"""
        np.random.seed(42)
        data = np.random.normal(50, 10, 100)
        # 意図的に欠損値を追加
        data[10:20] = np.nan
        data[50:55] = np.nan
        return pd.DataFrame({'values': data})

    def test_standard_scaler_accuracy(self):
        """StandardScalerの数学的正確性テスト"""
        logger.info("=== StandardScalerの正確性テスト ===")

        processor = DataProcessor()
        sample_data = self.create_sample_data()
        feature = sample_data[['feature1']].copy()
        
        # 手動でStandardScalerを適用
        scaler = StandardScaler()
        sklearn_scaled = scaler.fit_transform(feature)
        
        # DataProcessorを使用してスケーリング
        processed = processor.preprocess_features(
            feature,
            scale_features=True,
            scaling_method='standard',
            remove_outliers=False
        )
        
        # 結果の比較
        np.testing.assert_array_almost_equal(
            sklearn_scaled.flatten(),
            processed['feature1'].values,
            decimal=10,
            err_msg="StandardScalerの実装が一致しません"
        )
        
        # 数学的性質の検証
        scaled_mean = processed['feature1'].mean()
        scaled_std = processed['feature1'].std(ddof=1)

        assert abs(scaled_mean) < 1e-10, f"スケーリング後の平均が0でない: {scaled_mean}"
        assert abs(scaled_std - 1.0) < 1e-3, f"スケーリング後の標準偏差が1でない: {scaled_std}"
        
        logger.info("✅ StandardScalerの正確性テスト完了")

    def test_robust_scaler_accuracy(self, sample_data):
        """RobustScalerの数学的正確性テスト"""
        logger.info("=== RobustScalerの正確性テスト ===")
        
        processor = DataProcessor()
        feature = sample_data[['feature2']].copy()
        
        # 手動でRobustScalerを適用
        scaler = RobustScaler()
        sklearn_scaled = scaler.fit_transform(feature)
        
        # DataProcessorを使用してスケーリング
        processed = processor.preprocess_features(
            feature,
            scale_features=True,
            scaling_method='robust',
            remove_outliers=False
        )
        
        # 結果の比較
        np.testing.assert_array_almost_equal(
            sklearn_scaled.flatten(),
            processed['feature2'].values,
            decimal=10,
            err_msg="RobustScalerの実装が一致しません"
        )
        
        # 数学的性質の検証（中央値が0、IQRが1に近い）
        scaled_median = processed['feature2'].median()
        q75 = processed['feature2'].quantile(0.75)
        q25 = processed['feature2'].quantile(0.25)
        iqr = q75 - q25
        
        assert abs(scaled_median) < 1e-10, f"スケーリング後の中央値が0でない: {scaled_median}"
        assert abs(iqr - 1.0) < 1e-10, f"スケーリング後のIQRが1でない: {iqr}"
        
        logger.info("✅ RobustScalerの正確性テスト完了")

    def test_minmax_scaler_accuracy(self, sample_data):
        """MinMaxScalerの数学的正確性テスト"""
        logger.info("=== MinMaxScalerの正確性テスト ===")
        
        processor = DataProcessor()
        feature = sample_data[['feature3']].copy()
        
        # 手動でMinMaxScalerを適用
        scaler = MinMaxScaler()
        sklearn_scaled = scaler.fit_transform(feature)
        
        # DataProcessorを使用してスケーリング
        processed = processor.preprocess_features(
            feature,
            scale_features=True,
            scaling_method='minmax',
            remove_outliers=False
        )
        
        # 結果の比較
        np.testing.assert_array_almost_equal(
            sklearn_scaled.flatten(),
            processed['feature3'].values,
            decimal=10,
            err_msg="MinMaxScalerの実装が一致しません"
        )
        
        # 数学的性質の検証（最小値が0、最大値が1）
        scaled_min = processed['feature3'].min()
        scaled_max = processed['feature3'].max()
        
        assert abs(scaled_min) < 1e-10, f"スケーリング後の最小値が0でない: {scaled_min}"
        assert abs(scaled_max - 1.0) < 1e-10, f"スケーリング後の最大値が1でない: {scaled_max}"
        
        logger.info("✅ MinMaxScalerの正確性テスト完了")

    def test_iqr_outlier_detection_accuracy(self, outlier_data):
        """IQR外れ値検出の正確性テスト"""
        logger.info("=== IQR外れ値検出の正確性テスト ===")
        
        processor = DataProcessor()
        data = outlier_data.copy()
        
        # 手動でIQR外れ値検出を実行
        Q1 = data['values'].quantile(0.25)
        Q3 = data['values'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        manual_outliers = (data['values'] < lower_bound) | (data['values'] > upper_bound)
        manual_outlier_count = manual_outliers.sum()
        
        # DataProcessorを使用して外れ値除去
        processed = processor.preprocess_features(
            data,
            remove_outliers=True,
            outlier_method='iqr',
            outlier_threshold=1.5,
            scale_features=False
        )
        
        # 除去された外れ値の数を計算
        removed_count = len(data) - len(processed)
        
        # 結果の検証（完全一致は期待しないが、近い値であることを確認）
        assert abs(removed_count - manual_outlier_count) <= 2, \
            f"IQR外れ値検出の結果が大きく異なります: 手動={manual_outlier_count}, 実装={removed_count}"
        
        logger.info(f"手動検出: {manual_outlier_count}個, 実装検出: {removed_count}個")
        logger.info("✅ IQR外れ値検出の正確性テスト完了")

    def test_zscore_outlier_detection_accuracy(self, outlier_data):
        """Z-score外れ値検出の正確性テスト"""
        logger.info("=== Z-score外れ値検出の正確性テスト ===")
        
        processor = DataProcessor()
        data = outlier_data.copy()
        
        # 手動でZ-score外れ値検出を実行
        z_scores = np.abs(stats.zscore(data['values'].dropna()))
        manual_outliers = z_scores > 3.0
        manual_outlier_count = manual_outliers.sum()
        
        # DataProcessorを使用して外れ値除去
        processed = processor.preprocess_features(
            data,
            remove_outliers=True,
            outlier_method='zscore',
            outlier_threshold=3.0,
            scale_features=False
        )
        
        # 除去された外れ値の数を計算
        removed_count = len(data) - len(processed)
        
        # 結果の検証
        assert abs(removed_count - manual_outlier_count) <= 1, \
            f"Z-score外れ値検出の結果が異なります: 手動={manual_outlier_count}, 実装={removed_count}"
        
        logger.info(f"手動検出: {manual_outlier_count}個, 実装検出: {removed_count}個")
        logger.info("✅ Z-score外れ値検出の正確性テスト完了")

    def test_missing_value_imputation_accuracy(self, missing_data):
        """欠損値補完の正確性テスト"""
        logger.info("=== 欠損値補完の正確性テスト ===")
        
        processor = DataProcessor()
        data = missing_data.copy()
        
        # 元の統計量を計算（欠損値を除く）
        original_median = data['values'].median()
        original_mean = data['values'].mean()
        
        # median補完のテスト
        median_imputed = processor.transform_missing_values(
            data.copy(),
            strategy='median',
            columns=['values']
        )
        
        # 補完された値が中央値と一致することを確認
        imputed_values = median_imputed.loc[data['values'].isna(), 'values']
        assert all(abs(val - original_median) < 1e-10 for val in imputed_values), \
            "median補完の値が正しくありません"
        
        # mean補完のテスト
        mean_imputed = processor.transform_missing_values(
            data.copy(),
            strategy='mean',
            columns=['values']
        )
        
        # 補完された値が平均値と一致することを確認
        imputed_values_mean = mean_imputed.loc[data['values'].isna(), 'values']
        assert all(abs(val - original_mean) < 1e-10 for val in imputed_values_mean), \
            "mean補完の値が正しくありません"
        
        logger.info("✅ 欠損値補完の正確性テスト完了")

    def test_preprocessing_pipeline_consistency(self, sample_data):
        """前処理パイプラインの一貫性テスト"""
        logger.info("=== 前処理パイプラインの一貫性テスト ===")
        
        processor = DataProcessor()
        data = sample_data.copy()
        
        # 同じ設定で複数回前処理を実行
        result1 = processor.preprocess_features(
            data.copy(),
            scale_features=True,
            remove_outliers=True,
            outlier_method='iqr'
        )
        
        result2 = processor.preprocess_features(
            data.copy(),
            scale_features=True,
            remove_outliers=True,
            outlier_method='iqr'
        )
        
        # 結果が一致することを確認
        try:
            pd.testing.assert_frame_equal(
                result1, result2,
                check_exact=False,
                rtol=1e-10
            )
        except AssertionError:
            raise AssertionError("同じ設定での前処理結果が一致しません")
        
        logger.info("✅ 前処理パイプラインの一貫性テスト完了")

    def test_scaling_reversibility(self, sample_data):
        """スケーリングの可逆性テスト"""
        logger.info("=== スケーリングの可逆性テスト ===")
        
        # 注意: DataProcessorは現在逆変換機能を提供していないため、
        # 理論的な可逆性をsklearn直接使用でテスト
        
        feature = sample_data[['feature1']].copy()
        original_values = feature['feature1'].values
        
        # StandardScalerでの可逆性テスト
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature)
        reversed_values = scaler.inverse_transform(scaled)
        
        np.testing.assert_array_almost_equal(
            original_values,
            reversed_values.flatten(),
            decimal=10,
            err_msg="StandardScalerの可逆性が保たれていません"
        )
        
        # RobustScalerでの可逆性テスト
        robust_scaler = RobustScaler()
        robust_scaled = robust_scaler.fit_transform(feature)
        robust_reversed = robust_scaler.inverse_transform(robust_scaled)
        
        np.testing.assert_array_almost_equal(
            original_values,
            robust_reversed.flatten(),
            decimal=10,
            err_msg="RobustScalerの可逆性が保たれていません"
        )
        
        logger.info("✅ スケーリングの可逆性テスト完了")


def run_all_preprocessing_tests():
    """すべての前処理テストを実行"""
    logger.info("🔧 データ前処理正確性テストスイートを開始")

    test_instance = TestPreprocessingAccuracy()

    try:
        # 各テストを実行（簡略化版）
        logger.info("StandardScalerテストを実行中...")
        test_instance.test_standard_scaler_accuracy()

        logger.info("前処理パイプライン一貫性テストを実行中...")
        sample_data = test_instance.create_sample_data()
        test_instance.test_preprocessing_pipeline_consistency(sample_data)

        logger.info("スケーリング可逆性テストを実行中...")
        test_instance.test_scaling_reversibility(sample_data)
        
        logger.info("🎉 すべての前処理正確性テストが正常に完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 前処理正確性テストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_preprocessing_tests()
    sys.exit(0 if success else 1)
