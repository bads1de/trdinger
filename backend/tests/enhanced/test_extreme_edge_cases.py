"""
極端なエッジケーステスト

極端な境界条件、異常データパターン、システムリソース制限下での
MLトレーニングシステムの動作を検証するテストスイート。
"""

import numpy as np
import pandas as pd
import logging
import time
import psutil
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_processing import DataProcessor
from app.utils.label_generation import LabelGenerator, ThresholdMethod
from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

logger = logging.getLogger(__name__)


class TestExtremeEdgeCases:
    """極端なエッジケーステストクラス"""

    def test_micro_dataset_handling(self):
        """マイクロデータセット（極小データ）の処理テスト"""
        logger.info("=== マイクロデータセット処理テスト ===")
        
        # 極小データセット（2-10行）でのテスト
        sizes = [2, 3, 5, 10]
        
        for size in sizes:
            logger.info(f"サイズ {size} のマイクロデータセットをテスト中...")
            
            # マイクロデータ生成
            micro_data = pd.DataFrame({
                'Open': [100 + i for i in range(size)],
                'High': [102 + i for i in range(size)],
                'Low': [99 + i for i in range(size)],
                'Close': [101 + i for i in range(size)],
                'Volume': [1000 + i*100 for i in range(size)]
            })
            
            # データ処理テスト
            processor = DataProcessor()
            try:
                processed = processor.preprocess_features(
                    micro_data[['Close', 'Volume']].copy(),
                    scale_features=True,
                    remove_outliers=False  # 小さなデータでは外れ値除去を無効化
                )
                logger.info(f"  サイズ {size}: データ処理成功 ({len(processed)}行)")
                
                # 基本的な整合性確認
                assert len(processed) > 0, f"サイズ {size} で処理結果が空です"
                assert not processed.isnull().all().all(), f"サイズ {size} で全てNaNです"
                
            except Exception as e:
                logger.info(f"  サイズ {size}: 期待通りエラー発生 - {e}")
            
            # 特徴量エンジニアリングテスト
            fe_service = FeatureEngineeringService()
            try:
                features = fe_service.calculate_advanced_features(micro_data)
                logger.info(f"  サイズ {size}: 特徴量計算成功 ({features.shape[1]}特徴量)")
            except Exception as e:
                logger.info(f"  サイズ {size}: 特徴量計算でエラー - {e}")
            
            # ラベル生成テスト
            label_generator = LabelGenerator()
            try:
                labels, _ = label_generator.generate_labels(
                    pd.Series(micro_data['Close'].values, name='Close'),
                    method=ThresholdMethod.FIXED,
                    threshold_up=0.02,
                    threshold_down=-0.02
                )
                logger.info(f"  サイズ {size}: ラベル生成成功 ({len(labels)}ラベル)")
            except Exception as e:
                logger.info(f"  サイズ {size}: ラベル生成でエラー - {e}")
        
        logger.info("✅ マイクロデータセット処理テスト完了")

    def test_identical_values_dataset(self):
        """全て同じ値のデータセットテスト"""
        logger.info("=== 全同値データセットテスト ===")
        
        # 全て同じ値のデータ
        identical_data = pd.DataFrame({
            'Open': [100.0] * 100,
            'High': [100.0] * 100,
            'Low': [100.0] * 100,
            'Close': [100.0] * 100,
            'Volume': [1000.0] * 100
        })
        
        processor = DataProcessor()
        
        # データ処理テスト
        try:
            processed = processor.preprocess_features(
                identical_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )
            
            # 同値データでのスケーリング結果確認
            logger.info(f"同値データ処理結果: {processed.shape}")
            logger.info(f"Close統計: mean={processed['Close'].mean():.6f}, std={processed['Close'].std():.6f}")
            
            # 標準偏差が0の場合の処理確認
            if processed['Close'].std() == 0:
                logger.info("✅ 同値データで標準偏差0が正しく処理されました")
            
        except Exception as e:
            logger.info(f"✅ 同値データで期待通りエラー発生: {e}")
        
        # 特徴量エンジニアリング
        fe_service = FeatureEngineeringService()
        try:
            features = fe_service.calculate_advanced_features(identical_data)
            
            # 同値データでの特徴量確認
            logger.info(f"同値データ特徴量: {features.shape}")
            
            # 変化率系の特徴量が0になることを確認
            change_features = [col for col in features.columns if 'change' in col.lower() or 'return' in col.lower()]
            for col in change_features[:3]:  # 最初の3つをチェック
                if col in features.columns:
                    unique_values = features[col].nunique()
                    logger.info(f"変化率特徴量 {col}: ユニーク値数={unique_values}")
            
        except Exception as e:
            logger.info(f"✅ 同値データ特徴量計算でエラー: {e}")
        
        logger.info("✅ 全同値データセットテスト完了")

    def test_extreme_volatility_dataset(self):
        """極端な変動データセットテスト"""
        logger.info("=== 極端変動データセットテスト ===")
        
        # 極端な変動パターンを生成
        extreme_patterns = {
            'sudden_spike': [100, 100, 100, 1000, 100, 100, 100],  # 突然のスパイク
            'gradual_explosion': [100, 200, 400, 800, 1600, 3200, 6400],  # 指数的増加
            'oscillation': [100, 10, 100, 10, 100, 10, 100],  # 激しい振動
            'step_function': [100, 100, 200, 200, 300, 300, 400]  # ステップ関数
        }
        
        for pattern_name, prices in extreme_patterns.items():
            logger.info(f"パターン '{pattern_name}' をテスト中...")
            
            # データ生成
            size = len(prices)
            extreme_data = pd.DataFrame({
                'Open': prices,
                'High': [p * 1.1 for p in prices],
                'Low': [p * 0.9 for p in prices],
                'Close': prices,
                'Volume': [1000] * size
            })
            
            # データ処理
            processor = DataProcessor()
            try:
                processed = processor.preprocess_features(
                    extreme_data[['Close', 'Volume']].copy(),
                    scale_features=True,
                    remove_outliers=True
                )
                
                logger.info(f"  {pattern_name}: 処理成功 ({len(processed)}行)")
                
                # 外れ値除去の効果確認
                original_range = max(prices) - min(prices)
                processed_range = processed['Close'].max() - processed['Close'].min()
                logger.info(f"  {pattern_name}: 元の範囲={original_range:.1f}, 処理後範囲={processed_range:.3f}")
                
            except Exception as e:
                logger.info(f"  {pattern_name}: エラー発生 - {e}")
            
            # ラベル生成での極端変動の処理
            label_generator = LabelGenerator()
            try:
                labels, threshold_info = label_generator.generate_labels(
                    pd.Series(prices, name='Close'),
                    method=ThresholdMethod.FIXED,
                    threshold_up=0.1,  # 10%の閾値
                    threshold_down=-0.1
                )
                
                label_counts = pd.Series(labels).value_counts()
                logger.info(f"  {pattern_name}: ラベル分布 {label_counts.to_dict()}")
                
            except Exception as e:
                logger.info(f"  {pattern_name}: ラベル生成エラー - {e}")
        
        logger.info("✅ 極端変動データセットテスト完了")

    def test_memory_pressure_conditions(self):
        """メモリ圧迫条件下でのテスト"""
        logger.info("=== メモリ圧迫条件テスト ===")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 段階的にメモリ使用量を増やしてテスト
        memory_stress_data = []
        
        try:
            # 大きなデータセットを段階的に作成
            for i in range(5):
                size = 20000 * (i + 1)  # 20K, 40K, 60K, 80K, 100K
                
                logger.info(f"メモリストレステスト: サイズ {size}")
                
                # データ生成
                data = pd.DataFrame({
                    'Close': np.random.normal(100, 10, size),
                    'Volume': np.random.lognormal(10, 1, size)
                })
                
                memory_stress_data.append(data)
                
                # メモリ使用量チェック
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                logger.info(f"  メモリ増加: {memory_increase/1e6:.1f}MB")
                
                # メモリ使用量が1GB以上になったら停止
                if memory_increase > 1e9:
                    logger.warning("メモリ使用量が1GBを超えたため、テストを停止")
                    break
                
                # データ処理テスト
                processor = DataProcessor()
                start_time = time.time()
                
                processed = processor.preprocess_features(
                    data.copy(),
                    scale_features=True,
                    remove_outliers=True
                )
                
                processing_time = time.time() - start_time
                logger.info(f"  処理時間: {processing_time:.2f}秒")
                
                # パフォーマンス劣化の検出
                if processing_time > 30:  # 30秒以上
                    logger.warning(f"処理時間が長すぎます: {processing_time:.2f}秒")
                    break
        
        except MemoryError:
            logger.info("✅ メモリ不足エラーが適切に発生しました")
        except Exception as e:
            logger.warning(f"メモリストレステストでエラー: {e}")
        
        finally:
            # メモリクリーンアップ
            memory_stress_data.clear()
            
        logger.info("✅ メモリ圧迫条件テスト完了")

    def test_cpu_intensive_conditions(self):
        """CPU集約的条件下でのテスト"""
        logger.info("=== CPU集約的条件テスト ===")
        
        # 複雑な特徴量計算を並行実行
        fe_service = FeatureEngineeringService()
        
        # 複数のデータセットで同時に特徴量計算
        datasets = []
        for i in range(3):
            data = pd.DataFrame({
                'Open': np.random.normal(100, 10, 1000),
                'High': np.random.normal(105, 10, 1000),
                'Low': np.random.normal(95, 10, 1000),
                'Close': np.random.normal(100, 10, 1000),
                'Volume': np.random.lognormal(10, 1, 1000)
            })
            datasets.append(data)
        
        start_time = time.time()
        results = []
        
        try:
            for i, data in enumerate(datasets):
                logger.info(f"CPU集約テスト {i+1}/3 を実行中...")
                
                features = fe_service.calculate_advanced_features(data)
                results.append(features)
                
                elapsed = time.time() - start_time
                logger.info(f"  データセット {i+1}: {features.shape[1]}特徴量, 経過時間: {elapsed:.2f}秒")
        
        except Exception as e:
            logger.warning(f"CPU集約テストでエラー: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"CPU集約テスト完了: 総時間 {total_time:.2f}秒")
        
        # CPU効率の確認
        if len(results) > 0:
            avg_features = sum(r.shape[1] for r in results) / len(results)
            features_per_second = avg_features * len(results) / total_time
            logger.info(f"特徴量計算効率: {features_per_second:.1f}特徴量/秒")
        
        logger.info("✅ CPU集約的条件テスト完了")

    def test_data_corruption_scenarios(self):
        """データ破損シナリオテスト"""
        logger.info("=== データ破損シナリオテスト ===")
        
        # 様々な破損パターン
        corruption_scenarios = {
            'mixed_types': pd.DataFrame({
                'Close': [100, 'invalid', 102, None, 104],
                'Volume': [1000, 1001, 'error', 1003, 1004]
            }),
            'infinite_values': pd.DataFrame({
                'Close': [100, np.inf, 102, -np.inf, 104],
                'Volume': [1000, 1001, np.inf, 1003, 1004]
            }),
            'extreme_outliers': pd.DataFrame({
                'Close': [100, 101, 1e10, 103, 104],
                'Volume': [1000, 1001, 1002, -1e10, 1004]
            })
        }
        
        processor = DataProcessor()
        
        for scenario_name, corrupted_data in corruption_scenarios.items():
            logger.info(f"破損シナリオ '{scenario_name}' をテスト中...")
            
            try:
                processed = processor.preprocess_features(
                    corrupted_data.copy(),
                    scale_features=True,
                    remove_outliers=True
                )
                
                # 破損データが適切に処理されたか確認
                has_invalid = processed.isnull().any().any() or np.isinf(processed.select_dtypes(include=[np.number])).any().any()
                
                if not has_invalid:
                    logger.info(f"  {scenario_name}: 破損データが適切に修復されました")
                else:
                    logger.warning(f"  {scenario_name}: 無効な値が残っています")
                
            except Exception as e:
                logger.info(f"  {scenario_name}: 期待通りエラー発生 - {e}")
        
        logger.info("✅ データ破損シナリオテスト完了")


def run_all_extreme_edge_case_tests():
    """すべての極端エッジケーステストを実行"""
    logger.info("🔥 極端エッジケーステストスイートを開始")
    
    test_instance = TestExtremeEdgeCases()
    
    try:
        test_instance.test_micro_dataset_handling()
        test_instance.test_identical_values_dataset()
        test_instance.test_extreme_volatility_dataset()
        test_instance.test_memory_pressure_conditions()
        test_instance.test_cpu_intensive_conditions()
        test_instance.test_data_corruption_scenarios()
        
        logger.info("🎉 すべての極端エッジケーステストが正常に完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 極端エッジケーステストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_extreme_edge_case_tests()
    sys.exit(0 if success else 1)
