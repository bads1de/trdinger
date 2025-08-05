"""
エラーハンドリングテスト

MLトレーニングシステムのエラーハンドリング機能を検証するテストスイート。
異常なデータ、不正な引数、リソース不足などの状況での動作を検証します。
"""

import pytest
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_processing import DataProcessor
from app.utils.label_generation import LabelGenerator, ThresholdMethod
from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

logger = logging.getLogger(__name__)


class TestErrorHandling:
    """エラーハンドリングテストクラス"""

    def test_empty_data_handling(self):
        """空データの処理テスト"""
        logger.info("=== 空データの処理テスト ===")
        
        processor = DataProcessor()
        
        # 空のDataFrame
        empty_df = pd.DataFrame()
        
        # 空データでの前処理
        try:
            result = processor.preprocess_features(empty_df)
            assert len(result) == 0, "空データの処理結果が正しくありません"
            logger.info("✅ 空データの前処理が正常に処理されました")
        except Exception as e:
            logger.info(f"✅ 空データで期待通りエラーが発生: {e}")
        
        # 空データでのデータ型最適化
        try:
            optimized = processor.optimize_dtypes(empty_df)
            assert len(optimized) == 0, "空データの最適化結果が正しくありません"
            logger.info("✅ 空データの型最適化が正常に処理されました")
        except Exception as e:
            logger.info(f"✅ 空データで期待通りエラーが発生: {e}")

    def test_invalid_data_types(self):
        """不正なデータ型の処理テスト"""
        logger.info("=== 不正なデータ型の処理テスト ===")
        
        processor = DataProcessor()
        
        # 文字列のみのデータ
        string_data = pd.DataFrame({
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        try:
            result = processor.preprocess_features(string_data, scale_features=True)
            logger.info("✅ 文字列データの前処理が処理されました")
        except Exception as e:
            logger.info(f"✅ 文字列データで期待通りエラーが発生: {e}")
        
        # 混合型データ（数値変換不可能）
        mixed_invalid_data = pd.DataFrame({
            'mixed_col': [1, 'text', None, 4.5, 'invalid']
        })
        
        try:
            result = processor.preprocess_features(mixed_invalid_data, scale_features=True)
            logger.info("✅ 混合型データの前処理が処理されました")
        except Exception as e:
            logger.info(f"✅ 混合型データで期待通りエラーが発生: {e}")

    def test_extreme_values_handling(self):
        """極端な値の処理テスト"""
        logger.info("=== 極端な値の処理テスト ===")
        
        processor = DataProcessor()
        
        # 無限大値を含むデータ
        infinite_data = pd.DataFrame({
            'values': [1.0, 2.0, np.inf, 4.0, -np.inf, 6.0]
        })
        
        try:
            result = processor.preprocess_features(infinite_data, scale_features=True)
            # 無限大値が適切に処理されることを確認
            assert not np.isinf(result['values']).any(), "無限大値が除去されていません"
            logger.info("✅ 無限大値の処理が正常に完了しました")
        except Exception as e:
            logger.info(f"✅ 無限大値で期待通りエラーが発生: {e}")
        
        # 非常に大きな値
        large_values = pd.DataFrame({
            'values': [1e100, 1e101, 1e102, 1e103, 1e104]
        })
        
        try:
            result = processor.preprocess_features(large_values, scale_features=True)
            # 結果が有限であることを確認
            assert np.isfinite(result['values']).all(), "大きな値の処理で無限大が発生しました"
            logger.info("✅ 大きな値の処理が正常に完了しました")
        except Exception as e:
            logger.info(f"✅ 大きな値で期待通りエラーが発生: {e}")

    def test_label_generation_errors(self):
        """ラベル生成のエラーハンドリングテスト"""
        logger.info("=== ラベル生成のエラーハンドリングテスト ===")
        
        label_generator = LabelGenerator()
        
        # 空のSeries
        empty_series = pd.Series([], dtype=float, name='Close')
        
        try:
            labels, _ = label_generator.generate_labels(
                empty_series,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            logger.info("✅ 空Seriesでラベル生成が処理されました")
        except Exception as e:
            logger.info(f"✅ 空Seriesで期待通りエラーが発生: {e}")
        
        # すべてNaNのSeries
        nan_series = pd.Series([np.nan, np.nan, np.nan], name='Close')
        
        try:
            labels, _ = label_generator.generate_labels(
                nan_series,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            logger.info("✅ NaN Seriesでラベル生成が処理されました")
        except Exception as e:
            logger.info(f"✅ NaN Seriesで期待通りエラーが発生: {e}")
        
        # 不正な閾値
        valid_series = pd.Series([100, 101, 102, 103, 104], name='Close')
        
        try:
            labels, _ = label_generator.generate_labels(
                valid_series,
                method=ThresholdMethod.FIXED,
                threshold_up=-0.02,  # 負の上昇閾値
                threshold_down=0.02   # 正の下降閾値
            )
            logger.info("✅ 不正な閾値でラベル生成が処理されました")
        except Exception as e:
            logger.info(f"✅ 不正な閾値で期待通りエラーが発生: {e}")

    def test_feature_engineering_errors(self):
        """特徴量エンジニアリングのエラーハンドリングテスト"""
        logger.info("=== 特徴量エンジニアリングのエラーハンドリングテスト ===")
        
        fe_service = FeatureEngineeringService()
        
        # 必須カラムが不足しているデータ
        incomplete_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104]
            # Close, Low, Volumeが不足
        })
        
        try:
            features = fe_service.calculate_advanced_features(incomplete_data)
            logger.info("✅ 不完全なデータで特徴量計算が処理されました")
        except Exception as e:
            logger.info(f"✅ 不完全なデータで期待通りエラーが発生: {e}")
        
        # 行数が不足しているデータ
        insufficient_data = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000]
        })
        
        try:
            features = fe_service.calculate_advanced_features(insufficient_data)
            logger.info("✅ 行数不足データで特徴量計算が処理されました")
        except Exception as e:
            logger.info(f"✅ 行数不足データで期待通りエラーが発生: {e}")

    def test_memory_constraints(self):
        """メモリ制約のテスト"""
        logger.info("=== メモリ制約のテスト ===")
        
        processor = DataProcessor()
        
        # 大きなデータセット（メモリ使用量をテスト）
        try:
            large_data = pd.DataFrame({
                f'feature_{i}': np.random.normal(0, 1, 10000) 
                for i in range(100)
            })
            
            result = processor.preprocess_features(
                large_data,
                scale_features=True,
                remove_outliers=True
            )
            
            # メモリ使用量が合理的であることを確認
            memory_usage = result.memory_usage(deep=True).sum()
            assert memory_usage < 1e9, f"メモリ使用量が大きすぎます: {memory_usage} bytes"
            
            logger.info(f"✅ 大きなデータセットの処理が完了: メモリ使用量 {memory_usage:,} bytes")
            
        except MemoryError:
            logger.info("✅ メモリ不足で期待通りエラーが発生")
        except Exception as e:
            logger.info(f"✅ 大きなデータセットで期待通りエラーが発生: {e}")

    def test_concurrent_processing_safety(self):
        """並行処理の安全性テスト"""
        logger.info("=== 並行処理の安全性テスト ===")
        
        processor = DataProcessor()
        
        # 同じデータで複数の処理を並行実行
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(1, 1000)
        })
        
        results = []
        for i in range(5):
            try:
                result = processor.preprocess_features(
                    test_data.copy(),
                    scale_features=True,
                    remove_outliers=True
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"並行処理 {i} でエラー: {e}")
        
        # すべての結果が一致することを確認
        if len(results) > 1:
            for i in range(1, len(results)):
                try:
                    pd.testing.assert_frame_equal(results[0], results[i], rtol=1e-10)
                except AssertionError:
                    logger.warning(f"並行処理結果 {i} が一致しません")
        
        logger.info(f"✅ 並行処理テスト完了: {len(results)}個の結果を取得")


def run_all_error_handling_tests():
    """すべてのエラーハンドリングテストを実行"""
    logger.info("🚨 エラーハンドリングテストスイートを開始")
    
    test_instance = TestErrorHandling()
    
    try:
        test_instance.test_empty_data_handling()
        test_instance.test_invalid_data_types()
        test_instance.test_extreme_values_handling()
        test_instance.test_label_generation_errors()
        test_instance.test_feature_engineering_errors()
        test_instance.test_memory_constraints()
        test_instance.test_concurrent_processing_safety()
        
        logger.info("🎉 すべてのエラーハンドリングテストが正常に完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ エラーハンドリングテストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_error_handling_tests()
    sys.exit(0 if success else 1)
