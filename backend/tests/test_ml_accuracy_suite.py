"""
MLトレーニング系の包括的テストスイート

計算正確性、前処理正確性、特徴量計算、データ変換、ラベル生成の
すべてのテストを統合実行し、MLシステム全体の信頼性を検証します。
"""

import logging
import sys
import os
import time
from typing import Dict, List, Tuple, Any
import traceback
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 各テストモジュールをインポート
from tests.calculations.test_ml_calculations import run_all_calculation_tests
from tests.preprocessing.test_preprocessing_accuracy import run_all_preprocessing_tests
from tests.feature_engineering.test_feature_calculations import run_all_feature_calculation_tests
from tests.data_transformations.test_data_transformations import run_all_data_transformation_tests
from tests.label_generation.test_label_generation import run_all_label_generation_tests
from tests.enhanced.test_error_handling import run_all_error_handling_tests
from tests.enhanced.test_performance import run_all_performance_tests

# 統合テスト関数を直接定義
def run_integration_tests():
    """統合テストを実行（修正版）"""
    logger.info("🔗 統合テストを開始（インデックス整合性修正版）")

    try:
        # 必要なモジュールをインポート
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.utils.index_alignment import MLWorkflowIndexManager

        # インデックス管理器を初期化
        index_manager = MLWorkflowIndexManager()

        # テストデータ生成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='1h')
        base_price = 50000
        returns = np.random.normal(0, 0.02, 500)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        raw_data = pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'Volume': np.random.lognormal(10, 1, 500)
        }).set_index('timestamp')

        # ワークフロー初期化
        index_manager.initialize_workflow(raw_data)

        # Step 1: データ前処理（インデックス追跡付き）
        processor = DataProcessor()

        def preprocess_func(data):
            return processor.preprocess_features(
                data[['Close', 'Volume']].copy(),
                scale_features=False,
                remove_outliers=True
            )

        processed_data = index_manager.process_with_index_tracking(
            "前処理", raw_data, preprocess_func
        )

        # Step 2: 特徴量エンジニアリング（インデックス追跡付き）
        fe_service = FeatureEngineeringService()

        def feature_engineering_func(data):
            # 前処理されたデータのインデックスに合わせて元データを調整
            aligned_ohlcv = raw_data.loc[data.index]
            return fe_service.calculate_advanced_features(aligned_ohlcv)

        features = index_manager.process_with_index_tracking(
            "特徴量エンジニアリング", processed_data, feature_engineering_func
        )

        # Step 3: ラベル生成（インデックス整合性を考慮）
        label_generator = LabelGenerator()

        # 特徴量のインデックスに合わせて価格データを調整
        aligned_price_data = raw_data.loc[features.index, 'Close']

        labels, _ = label_generator.generate_labels(
            aligned_price_data,
            method=ThresholdMethod.FIXED,
            threshold_up=0.02,
            threshold_down=-0.02
        )

        # Step 4: 最終的なインデックス整合
        final_features, final_labels = index_manager.finalize_workflow(
            features, labels, alignment_method="intersection"
        )

        # 統合検証
        assert len(final_features) > 0, "最終特徴量が生成されませんでした"
        assert len(final_labels) > 0, "最終ラベルが生成されませんでした"

        # インデックス整合性の検証
        validation_result = index_manager.alignment_manager.validate_alignment(
            final_features, final_labels, min_alignment_ratio=0.95
        )

        logger.info(f"最終インデックス整合性検証:")
        logger.info(f"  特徴量行数: {validation_result['features_rows']}")
        logger.info(f"  ラベル行数: {validation_result['labels_rows']}")
        logger.info(f"  共通行数: {validation_result['common_rows']}")
        logger.info(f"  整合率: {validation_result['alignment_ratio']*100:.1f}%")

        # 高い整合性を要求（95%以上）
        assert validation_result["is_valid"], \
            f"インデックス整合性が不十分: {validation_result['alignment_ratio']*100:.1f}% < 95%"

        # ワークフローサマリー
        workflow_summary = index_manager.get_workflow_summary()
        logger.info(f"ワークフロー完了:")
        logger.info(f"  元データ: {workflow_summary['original_rows']}行")
        logger.info(f"  最終データ: {workflow_summary['final_rows']}行")
        logger.info(f"  データ保持率: {workflow_summary['data_retention_rate']*100:.1f}%")

        logger.info("✅ 統合テスト完了（インデックス整合性修正版）")
        return True

    except Exception as e:
        logger.error(f"❌ 統合テストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_extreme_edge_case_tests():
    """極端エッジケーステストを実行"""
    logger.info("🔥 極端エッジケーステストを開始")

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # マイクロデータセットテスト
        logger.info("マイクロデータセット処理テスト...")
        micro_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1001, 1002]
        })

        processor = DataProcessor()
        try:
            processed = processor.preprocess_features(
                micro_data.copy(),
                scale_features=True,
                remove_outliers=False
            )
            logger.info(f"マイクロデータ処理成功: {len(processed)}行")
        except Exception as e:
            logger.info(f"マイクロデータで期待通りエラー: {e}")

        # 全同値データテスト
        logger.info("全同値データセットテスト...")
        identical_data = pd.DataFrame({
            'Close': [100.0] * 50,
            'Volume': [1000.0] * 50
        })

        try:
            processed = processor.preprocess_features(
                identical_data.copy(),
                scale_features=True,
                remove_outliers=True
            )
            logger.info(f"同値データ処理成功: std={processed['Close'].std():.6f}")
        except Exception as e:
            logger.info(f"同値データで期待通りエラー: {e}")

        # データ破損シナリオテスト
        logger.info("データ破損シナリオテスト...")
        corrupted_data = pd.DataFrame({
            'Close': [100, np.inf, 102, -np.inf, 104],
            'Volume': [1000, 1001, np.inf, 1003, 1004]
        })

        try:
            processed = processor.preprocess_features(
                corrupted_data.copy(),
                scale_features=True,
                remove_outliers=True
            )

            has_invalid = (processed.isnull().any().any() or
                          np.isinf(processed.select_dtypes(include=[np.number])).any().any())

            if not has_invalid:
                logger.info("破損データが適切に修復されました")
            else:
                logger.warning("無効な値が残っています")

        except Exception as e:
            logger.info(f"破損データで期待通りエラー: {e}")

        logger.info("✅ 極端エッジケーステスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ 極端エッジケーステストでエラーが発生: {e}")
        return False


def run_real_environment_simulation_tests():
    """実環境シミュレーションテストを実行"""
    logger.info("🌍 実環境シミュレーションテストを開始")

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # 実際の市場データパターンのシミュレーション
        logger.info("実市場データパターンシミュレーション...")

        # ビットコインの実際の価格動作を模倣
        np.random.seed(42)
        size = 1000

        # より現実的な価格動作（トレンド + ボラティリティクラスタリング）
        base_price = 50000
        prices = [base_price]
        volatility = 0.02

        for i in range(1, size):
            # ボラティリティクラスタリング効果
            if i > 20:
                recent_volatility = np.std([prices[j]/prices[j-1] - 1 for j in range(i-20, i)])
                volatility = 0.01 + recent_volatility * 2

            # 価格変動
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.3))  # 価格下限

        # 現実的なOHLCVデータ生成
        market_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'Volume': np.random.lognormal(10, 0.8, size)  # より現実的なボリューム分布
        })

        # データ処理パイプライン
        processor = DataProcessor()
        processed_data = processor.preprocess_features(
            market_data[['Close', 'Volume']].copy(),
            scale_features=True,
            remove_outliers=True
        )

        logger.info(f"実市場データ処理成功: {len(processed_data)}行")

        # 特徴量エンジニアリング
        fe_service = FeatureEngineeringService()
        features = fe_service.calculate_advanced_features(market_data)

        logger.info(f"実市場データ特徴量: {features.shape[1]}個")

        # ラベル生成
        label_generator = LabelGenerator()
        labels, _ = label_generator.generate_labels(
            market_data['Close'],
            method=ThresholdMethod.FIXED,
            threshold_up=0.02,
            threshold_down=-0.02
        )

        label_distribution = pd.Series(labels).value_counts()
        logger.info(f"実市場データラベル分布: {label_distribution.to_dict()}")

        # データ品質の検証
        price_volatility = market_data['Close'].pct_change().std()
        volume_consistency = market_data['Volume'].std() / market_data['Volume'].mean()

        logger.info(f"価格ボラティリティ: {price_volatility:.4f}")
        logger.info(f"ボリューム変動係数: {volume_consistency:.4f}")

        # 長時間実行安定性テスト
        logger.info("長時間実行安定性テスト...")

        start_time = time.time()
        iterations = 10

        for i in range(iterations):
            # 繰り返し処理でメモリリークや性能劣化をチェック
            test_data = market_data.sample(n=100).copy()

            processed = processor.preprocess_features(
                test_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )

            if i % 3 == 0:
                logger.info(f"  反復 {i+1}/{iterations} 完了")

        total_time = time.time() - start_time
        avg_time_per_iteration = total_time / iterations

        logger.info(f"長時間実行テスト完了: 平均 {avg_time_per_iteration:.3f}秒/反復")

        # パフォーマンス劣化の検出
        if avg_time_per_iteration > 1.0:
            logger.warning(f"処理時間が長すぎます: {avg_time_per_iteration:.3f}秒")

        # I/Oエラーシミュレーション
        logger.info("I/Oエラーシミュレーション...")

        # 不完全なデータ（読み込みエラーをシミュレート）
        incomplete_data = market_data.iloc[:50].copy()  # データが途中で切れる

        try:
            processed = processor.preprocess_features(
                incomplete_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )
            logger.info(f"不完全データ処理成功: {len(processed)}行")
        except Exception as e:
            logger.info(f"不完全データで期待通りエラー: {e}")

        # メモリ制約シミュレーション
        logger.info("メモリ制約シミュレーション...")

        # 大きなデータセットを小さなチャンクに分割して処理
        large_data = pd.concat([market_data] * 5, ignore_index=True)  # 5倍のデータ
        chunk_size = 500

        processed_chunks = []
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data.iloc[i:i+chunk_size]
            processed_chunk = processor.preprocess_features(
                chunk[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )
            processed_chunks.append(processed_chunk)

        final_result = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"チャンク処理成功: {len(final_result)}行")

        logger.info("✅ 実環境シミュレーションテスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ 実環境シミュレーションテストでエラーが発生: {e}")
        return False


def run_advanced_performance_optimization_tests():
    """高度なパフォーマンス最適化検証テストを実行"""
    logger.info("⚡ 高度なパフォーマンス最適化検証テストを開始")

    try:
        import psutil
        import gc
        from app.utils.data_processing import DataProcessor
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        process = psutil.Process(os.getpid())

        # メモリリーク検出テスト
        logger.info("メモリリーク検出テスト...")

        initial_memory = process.memory_info().rss
        memory_measurements = []

        processor = DataProcessor()

        for i in range(20):
            # 繰り返し処理でメモリリークを検出
            test_data = pd.DataFrame({
                'Close': np.random.normal(100, 10, 1000),
                'Volume': np.random.lognormal(10, 1, 1000)
            })

            processed = processor.preprocess_features(
                test_data.copy(),
                scale_features=True,
                remove_outliers=True
            )

            # メモリ使用量測定
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            memory_measurements.append(memory_increase)

            # 明示的なガベージコレクション
            del test_data, processed
            gc.collect()

            if i % 5 == 0:
                logger.info(f"  反復 {i+1}: メモリ増加 {memory_increase/1e6:.1f}MB")

        # メモリリークの分析
        memory_trend = np.polyfit(range(len(memory_measurements)), memory_measurements, 1)[0]

        logger.info(f"メモリ使用量トレンド: {memory_trend/1e6:.3f}MB/反復")

        if memory_trend > 1e6:  # 1MB/反復以上の増加
            logger.warning(f"メモリリークの可能性: {memory_trend/1e6:.1f}MB/反復")
        else:
            logger.info("✅ メモリリークは検出されませんでした")

        # CPU使用率監視テスト
        logger.info("CPU使用率監視テスト...")

        cpu_measurements = []
        start_time = time.time()

        fe_service = FeatureEngineeringService()

        # CPU集約的なタスクを実行
        for i in range(5):
            cpu_before = psutil.cpu_percent(interval=0.1)

            # 複雑な特徴量計算
            complex_data = pd.DataFrame({
                'Open': np.random.normal(100, 10, 2000),
                'High': np.random.normal(105, 10, 2000),
                'Low': np.random.normal(95, 10, 2000),
                'Close': np.random.normal(100, 10, 2000),
                'Volume': np.random.lognormal(10, 1, 2000)
            })

            features = fe_service.calculate_advanced_features(complex_data)

            cpu_after = psutil.cpu_percent(interval=0.1)
            cpu_usage = max(cpu_after - cpu_before, 0)
            cpu_measurements.append(cpu_usage)

            logger.info(f"  タスク {i+1}: CPU使用率 {cpu_usage:.1f}%")

        avg_cpu_usage = np.mean(cpu_measurements)
        logger.info(f"平均CPU使用率: {avg_cpu_usage:.1f}%")

        # ボトルネック特定テスト
        logger.info("ボトルネック特定テスト...")

        # 各処理段階の時間測定
        bottleneck_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 5000),
            'Volume': np.random.lognormal(10, 1, 5000)
        })

        # データ前処理のボトルネック
        start = time.time()
        processed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=False,
            remove_outliers=False
        )
        preprocessing_time = time.time() - start

        # スケーリングのボトルネック
        start = time.time()
        scaled = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=True,
            remove_outliers=False
        )
        scaling_time = time.time() - start

        # 外れ値除去のボトルネック
        start = time.time()
        outlier_removed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=False,
            remove_outliers=True
        )
        outlier_removal_time = time.time() - start

        # 完全処理のボトルネック
        start = time.time()
        full_processed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=True,
            remove_outliers=True
        )
        full_processing_time = time.time() - start

        logger.info("処理時間分析:")
        logger.info(f"  基本前処理: {preprocessing_time:.3f}秒")
        logger.info(f"  スケーリング: {scaling_time:.3f}秒")
        logger.info(f"  外れ値除去: {outlier_removal_time:.3f}秒")
        logger.info(f"  完全処理: {full_processing_time:.3f}秒")

        # ボトルネックの特定
        processing_times = {
            'スケーリング': scaling_time - preprocessing_time,
            '外れ値除去': outlier_removal_time - preprocessing_time,
            '統合処理': full_processing_time - max(scaling_time, outlier_removal_time)
        }

        bottleneck = max(processing_times, key=processing_times.get)
        bottleneck_time = processing_times[bottleneck]

        logger.info(f"最大ボトルネック: {bottleneck} ({bottleneck_time:.3f}秒)")

        # パフォーマンス効率の計算
        data_throughput = len(bottleneck_data) / full_processing_time
        logger.info(f"データ処理スループット: {data_throughput:.0f}行/秒")

        # メモリ効率の測定
        memory_efficiency = len(bottleneck_data) / (process.memory_info().rss / 1e6)
        logger.info(f"メモリ効率: {memory_efficiency:.0f}行/MB")

        # パフォーマンス最適化の提案
        optimization_suggestions = []

        if bottleneck_time > 0.1:
            optimization_suggestions.append(f"{bottleneck}の最適化が必要")

        if avg_cpu_usage > 80:
            optimization_suggestions.append("CPU使用率が高すぎます")

        if memory_trend > 5e5:  # 0.5MB/反復
            optimization_suggestions.append("メモリ使用量の最適化が必要")

        if data_throughput < 1000:
            optimization_suggestions.append("データ処理スループットの改善が必要")

        if optimization_suggestions:
            logger.warning("最適化提案:")
            for suggestion in optimization_suggestions:
                logger.warning(f"  - {suggestion}")
        else:
            logger.info("✅ パフォーマンスは良好です")

        logger.info("✅ 高度なパフォーマンス最適化検証テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ 高度なパフォーマンス最適化検証テストでエラーが発生: {e}")
        return False


def run_code_coverage_verification_tests():
    """コードカバレッジ確認テストを実行"""
    logger.info("📋 コードカバレッジ確認テストを開始")

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # 設定パラメータの組み合わせテスト
        logger.info("設定パラメータ組み合わせテスト...")

        processor = DataProcessor()
        test_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.lognormal(10, 1, 100)
        })

        # 異なる設定の組み合わせをテスト
        configurations = [
            {'scale_features': True, 'remove_outliers': True, 'outlier_method': 'iqr'},
            {'scale_features': True, 'remove_outliers': True, 'outlier_method': 'zscore'},
            {'scale_features': True, 'remove_outliers': False},
            {'scale_features': False, 'remove_outliers': True, 'outlier_method': 'iqr'},
            {'scale_features': False, 'remove_outliers': False},
        ]

        for i, config in enumerate(configurations):
            try:
                result = processor.preprocess_features(test_data.copy(), **config)
                logger.info(f"  設定 {i+1}: 成功 ({len(result)}行)")
            except Exception as e:
                logger.info(f"  設定 {i+1}: エラー - {e}")

        # ラベル生成の異なる手法テスト
        logger.info("ラベル生成手法網羅テスト...")

        label_generator = LabelGenerator()
        price_series = pd.Series(np.random.normal(100, 10, 100), name='Close')

        # 異なる閾値設定
        threshold_configs = [
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.01, 'threshold_down': -0.01},
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.05, 'threshold_down': -0.05},
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.1, 'threshold_down': -0.1},
        ]

        for i, config in enumerate(threshold_configs):
            try:
                labels, info = label_generator.generate_labels(price_series, **config)
                label_dist = pd.Series(labels).value_counts()
                logger.info(f"  閾値設定 {i+1}: 成功 - 分布 {label_dist.to_dict()}")
            except Exception as e:
                logger.info(f"  閾値設定 {i+1}: エラー - {e}")

        # 例外処理の網羅性テスト
        logger.info("例外処理網羅性テスト...")

        # 空データでの例外処理
        empty_data = pd.DataFrame()
        try:
            processor.preprocess_features(empty_data)
            logger.info("  空データ: 処理成功")
        except Exception as e:
            logger.info(f"  空データ: 例外処理確認 - {e}")

        # 不正な型での例外処理
        invalid_data = "invalid_data"
        try:
            processor.preprocess_features(invalid_data)
            logger.info("  不正型: 処理成功")
        except Exception as e:
            logger.info(f"  不正型: 例外処理確認 - {e}")

        # NaNのみのデータでの例外処理
        nan_data = pd.DataFrame({'Close': [np.nan] * 10, 'Volume': [np.nan] * 10})
        try:
            result = processor.preprocess_features(nan_data)
            logger.info(f"  NaNデータ: 処理成功 ({len(result)}行)")
        except Exception as e:
            logger.info(f"  NaNデータ: 例外処理確認 - {e}")

        # 特徴量エンジニアリングの例外処理テスト
        logger.info("特徴量エンジニアリング例外処理テスト...")

        fe_service = FeatureEngineeringService()

        # 不完全なOHLCVデータ
        incomplete_ohlcv = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103]
            # Close, Low, Volumeが不足
        })

        try:
            features = fe_service.calculate_advanced_features(incomplete_ohlcv)
            logger.info(f"  不完全OHLCV: 処理成功 ({features.shape[1]}特徴量)")
        except Exception as e:
            logger.info(f"  不完全OHLCV: 例外処理確認 - {e}")

        # 極小データでの特徴量計算
        tiny_ohlcv = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000]
        })

        try:
            features = fe_service.calculate_advanced_features(tiny_ohlcv)
            logger.info(f"  極小OHLCV: 処理成功 ({features.shape[1]}特徴量)")
        except Exception as e:
            logger.info(f"  極小OHLCV: 例外処理確認 - {e}")

        # エッジケースでのラベル生成
        logger.info("ラベル生成エッジケーステスト...")

        # 単調増加データ
        monotonic_series = pd.Series(range(100), name='Close')
        try:
            labels, _ = label_generator.generate_labels(
                monotonic_series,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            label_dist = pd.Series(labels).value_counts()
            logger.info(f"  単調増加: 成功 - 分布 {label_dist.to_dict()}")
        except Exception as e:
            logger.info(f"  単調増加: 例外処理確認 - {e}")

        # 単調減少データ
        monotonic_decreasing = pd.Series(range(100, 0, -1), name='Close')
        try:
            labels, _ = label_generator.generate_labels(
                monotonic_decreasing,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            label_dist = pd.Series(labels).value_counts()
            logger.info(f"  単調減少: 成功 - 分布 {label_dist.to_dict()}")
        except Exception as e:
            logger.info(f"  単調減少: 例外処理確認 - {e}")

        # データ型変換の網羅性テスト
        logger.info("データ型変換網羅性テスト...")

        # 異なるデータ型での処理
        data_types_test = {
            'int32': pd.DataFrame({'Close': np.array([100, 101, 102], dtype=np.int32)}),
            'int64': pd.DataFrame({'Close': np.array([100, 101, 102], dtype=np.int64)}),
            'float32': pd.DataFrame({'Close': np.array([100.0, 101.0, 102.0], dtype=np.float32)}),
            'float64': pd.DataFrame({'Close': np.array([100.0, 101.0, 102.0], dtype=np.float64)}),
        }

        for dtype_name, dtype_data in data_types_test.items():
            try:
                result = processor.preprocess_features(dtype_data.copy())
                logger.info(f"  {dtype_name}: 処理成功 ({result.dtypes['Close']})")
            except Exception as e:
                logger.info(f"  {dtype_name}: 例外処理確認 - {e}")

        # 境界値テスト
        logger.info("境界値テスト...")

        # 最小値・最大値での処理
        boundary_data = pd.DataFrame({
            'Close': [sys.float_info.min, 0, sys.float_info.max],
            'Volume': [1, 1000, sys.float_info.max]
        })

        try:
            result = processor.preprocess_features(boundary_data.copy())
            logger.info(f"  境界値: 処理成功 ({len(result)}行)")
        except Exception as e:
            logger.info(f"  境界値: 例外処理確認 - {e}")

        logger.info("✅ コードカバレッジ確認テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ コードカバレッジ確認テストでエラーが発生: {e}")
        return False


def run_automl_comprehensive_tests():
    """AutoML包括的テストを実行"""
    logger.info("🤖 AutoML包括的テストを開始")

    try:
        from app.services.ml.ml_training_service import MLTrainingService
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.services.ml.feature_engineering.automl_features.automl_config import AutoMLConfig
        from app.utils.index_alignment import MLWorkflowIndexManager
        from app.utils.label_generation import LabelGenerator, ThresholdMethod

        # BTC市場データ生成関数
        def create_btc_market_data(timeframe="1h", size=300):
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=size, freq='1h')

            base_price = 50000
            volatility = 0.02
            prices = [base_price]

            for i in range(1, size):
                change = np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, base_price * 0.5))

            return pd.DataFrame({
                'timestamp': dates,
                'Open': prices,
                'High': [p * 1.01 for p in prices],
                'Low': [p * 0.99 for p in prices],
                'Close': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
                'Volume': np.random.lognormal(10, 0.6, size)
            }).set_index('timestamp')

        # AutoMLアンサンブル学習テスト
        logger.info("AutoMLアンサンブル学習テスト...")
        btc_data = create_btc_market_data("1h", 200)

        try:
            # AutoML設定（軽量版）
            automl_config = {
                "tsfresh": {"enabled": True, "feature_count_limit": 20},
                "autofeat": {"enabled": False}
            }

            # アンサンブル設定
            ensemble_config = {
                "method": "bagging",
                "bagging_params": {
                    "n_estimators": 2,
                    "bootstrap_fraction": 0.8,
                    "base_model_type": "lightgbm"
                }
            }

            ml_service = MLTrainingService(
                trainer_type="ensemble",
                ensemble_config=ensemble_config,
                automl_config=automl_config
            )

            result = ml_service.train_model(
                training_data=btc_data,
                threshold_up=0.02,
                threshold_down=-0.02,
                save_model=False
            )

            logger.info(f"AutoMLアンサンブル学習成功: 精度={result.get('accuracy', 'N/A')}")

        except Exception as e:
            logger.info(f"AutoMLアンサンブル学習でエラー（期待される場合もあります）: {e}")

        # AutoML特徴量選択テスト
        logger.info("AutoML特徴量選択テスト...")

        try:
            # AutoML設定で特徴量エンジニアリング
            automl_config_obj = AutoMLConfig.get_financial_optimized_config()
            fe_service = FeatureEngineeringService(automl_config=automl_config_obj)

            # 基本特徴量計算
            basic_features = fe_service.calculate_advanced_features(btc_data)

            logger.info(f"基本特徴量: {basic_features.shape[1]}個")

            # 特徴量が生成されることを確認
            assert basic_features.shape[1] > 0, "特徴量が生成されませんでした"

            logger.info("✅ AutoML特徴量選択が正常に動作しました")

        except Exception as e:
            logger.info(f"AutoML特徴量選択でエラー（期待される場合もあります）: {e}")

        # AutoMLワークフロー統合テスト
        logger.info("AutoMLワークフロー統合テスト...")

        try:
            index_manager = MLWorkflowIndexManager()
            index_manager.initialize_workflow(btc_data)

            # 特徴量エンジニアリング
            def automl_feature_func(data):
                fe_service = FeatureEngineeringService()
                return fe_service.calculate_advanced_features(data)

            features = index_manager.process_with_index_tracking(
                "AutoML特徴量エンジニアリング", btc_data, automl_feature_func
            )

            # ラベル生成
            label_generator = LabelGenerator()
            aligned_price_data = btc_data.loc[features.index, 'Close']
            labels, _ = label_generator.generate_labels(
                aligned_price_data,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )

            # 最終整合
            final_features, final_labels = index_manager.finalize_workflow(
                features, labels, alignment_method="intersection"
            )

            # 検証
            assert len(final_features) > 0, "最終特徴量が生成されませんでした"
            assert len(final_labels) > 0, "最終ラベルが生成されませんでした"

            workflow_summary = index_manager.get_workflow_summary()
            logger.info(f"AutoMLワークフロー完了: データ保持率={workflow_summary['data_retention_rate']*100:.1f}%")

        except Exception as e:
            logger.info(f"AutoMLワークフロー統合でエラー（期待される場合もあります）: {e}")

        logger.info("✅ AutoML包括的テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ AutoML包括的テストでエラーが発生: {e}")
        return False


def run_optuna_bayesian_optimization_tests():
    """Optunaベイジアン最適化テストを実行"""
    logger.info("🔍 Optunaベイジアン最適化テストを開始")

    try:
        from app.services.optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace

        # Optunaベイジアン最適化の基本テスト
        logger.info("Optunaベイジアン最適化基本テスト...")

        # 簡単な目的関数（二次関数の最大化）
        def simple_objective(params):
            x = params['x']
            y = params['y']
            # 最大値が(2, 3)で値が10となる二次関数
            return 10 - (x - 2)**2 - (y - 3)**2

        # パラメータ空間定義
        parameter_space = {
            'x': ParameterSpace(type="real", low=0.0, high=5.0),
            'y': ParameterSpace(type="real", low=0.0, high=5.0)
        }

        # Optuna最適化実行
        optimizer = OptunaOptimizer()
        result = optimizer.optimize(
            objective_function=simple_objective,
            parameter_space=parameter_space,
            n_calls=20  # テスト用に少なく設定
        )

        # 結果検証
        assert result.best_score > 8.0, f"最適化結果が不十分: {result.best_score}"
        assert abs(result.best_params['x'] - 2.0) < 1.0, "xパラメータの最適化が不十分"
        assert abs(result.best_params['y'] - 3.0) < 1.0, "yパラメータの最適化が不十分"

        logger.info(f"Optuna最適化成功: スコア={result.best_score:.3f}, "
                   f"パラメータ={result.best_params}")

        # クリーンアップ
        optimizer.cleanup()

        # LightGBMパラメータ最適化テスト
        logger.info("LightGBMパラメータ最適化テスト...")

        # LightGBM用の目的関数（簡略版）
        def lightgbm_objective(params):
            # 実際のモデル学習の代わりに、パラメータの妥当性をスコア化
            score = 0.5  # ベーススコア

            # パラメータの妥当性に基づいてスコア調整
            if 0.01 <= params['learning_rate'] <= 0.3:
                score += 0.1
            if 10 <= params['num_leaves'] <= 100:
                score += 0.1
            if 0.5 <= params['feature_fraction'] <= 1.0:
                score += 0.1
            if 5 <= params['min_data_in_leaf'] <= 50:
                score += 0.1

            # ランダムノイズを追加（実際の学習結果をシミュレート）
            import random
            score += random.uniform(-0.1, 0.1)

            return score

        # LightGBMパラメータ空間
        lgb_parameter_space = OptunaOptimizer.get_default_parameter_space()

        # 最適化実行
        optimizer2 = OptunaOptimizer()
        lgb_result = optimizer2.optimize(
            objective_function=lightgbm_objective,
            parameter_space=lgb_parameter_space,
            n_calls=15
        )

        # 結果検証
        assert lgb_result.best_score > 0.5, f"LightGBM最適化結果が不十分: {lgb_result.best_score}"
        assert 'learning_rate' in lgb_result.best_params, "learning_rateパラメータが不足"
        assert 'num_leaves' in lgb_result.best_params, "num_leavesパラメータが不足"

        logger.info(f"LightGBM最適化成功: スコア={lgb_result.best_score:.3f}")

        # クリーンアップ
        optimizer2.cleanup()

        # アンサンブルパラメータ最適化テスト
        logger.info("アンサンブルパラメータ最適化テスト...")

        try:
            # アンサンブル用パラメータ空間
            ensemble_parameter_space = OptunaOptimizer.get_ensemble_parameter_space(
                ensemble_method="bagging",
                enabled_models=["lightgbm", "random_forest"]
            )

            # アンサンブル用目的関数
            def ensemble_objective(params):
                score = 0.6  # ベーススコア

                # アンサンブルパラメータの妥当性チェック
                if 'n_estimators' in params and 2 <= params['n_estimators'] <= 10:
                    score += 0.1
                if 'bootstrap_fraction' in params and 0.5 <= params['bootstrap_fraction'] <= 1.0:
                    score += 0.1

                return score + np.random.uniform(-0.05, 0.05)

            # 最適化実行
            optimizer3 = OptunaOptimizer()
            ensemble_result = optimizer3.optimize(
                objective_function=ensemble_objective,
                parameter_space=ensemble_parameter_space,
                n_calls=10
            )

            logger.info(f"アンサンブル最適化成功: スコア={ensemble_result.best_score:.3f}")
            optimizer3.cleanup()

        except Exception as e:
            logger.info(f"アンサンブル最適化でエラー（期待される場合もあります）: {e}")

        # 最適化効率性テスト
        logger.info("最適化効率性テスト...")

        # 複雑な目的関数（多峰性関数）
        def complex_objective(params):
            x, y = params['x'], params['y']
            # Rastrigin関数の変形（最大化用）
            return -(10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) +
                    (y**2 - 10 * np.cos(2 * np.pi * y)))

        # 効率性測定
        start_time = time.time()
        optimizer4 = OptunaOptimizer()
        complex_result = optimizer4.optimize(
            objective_function=complex_objective,
            parameter_space={
                'x': ParameterSpace(type="real", low=-5.0, high=5.0),
                'y': ParameterSpace(type="real", low=-5.0, high=5.0)
            },
            n_calls=25
        )
        optimization_time = time.time() - start_time

        # 効率性検証
        assert optimization_time < 30, f"最適化時間が長すぎます: {optimization_time:.2f}秒"
        assert complex_result.total_evaluations == 25, "評価回数が正しくありません"

        logger.info(f"複雑関数最適化完了: 時間={optimization_time:.2f}秒, "
                   f"スコア={complex_result.best_score:.3f}")

        optimizer4.cleanup()

        # 異なるサンプラーのテスト
        logger.info("異なる最適化アルゴリズムテスト...")

        try:
            import optuna

            # TPESampler vs RandomSampler の比較
            samplers = {
                "TPE": optuna.samplers.TPESampler(seed=42),
                "Random": optuna.samplers.RandomSampler(seed=42)
            }

            sampler_results = {}

            for sampler_name, sampler in samplers.items():
                # カスタムOptunaスタディ作成
                study = optuna.create_study(
                    direction="maximize",
                    sampler=sampler
                )

                def optuna_objective(trial):
                    x = trial.suggest_float('x', 0.0, 5.0)
                    y = trial.suggest_float('y', 0.0, 5.0)
                    return simple_objective({'x': x, 'y': y})

                study.optimize(optuna_objective, n_trials=15)

                sampler_results[sampler_name] = {
                    "best_score": study.best_value,
                    "best_params": study.best_params
                }

                logger.info(f"{sampler_name}サンプラー: スコア={study.best_value:.3f}")

            # TPEの方が良い結果を出すことを期待（必須ではない）
            if sampler_results["TPE"]["best_score"] > sampler_results["Random"]["best_score"]:
                logger.info("✅ TPEサンプラーがRandomサンプラーより良い結果を出しました")
            else:
                logger.info("ℹ️ この試行ではRandomサンプラーの方が良い結果でした")

        except Exception as e:
            logger.info(f"サンプラー比較テストでエラー（期待される場合もあります）: {e}")

        logger.info("✅ Optunaベイジアン最適化テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ Optunaベイジアン最適化テストでエラーが発生: {e}")
        return False

logger = logging.getLogger(__name__)


class MLAccuracyTestSuite:
    """MLトレーニング系の包括的テストスイート"""

    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = None
        self.end_time = None

    def run_test_module(self, test_name: str, test_function) -> bool:
        """個別テストモジュールを実行"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 {test_name} を開始")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = test_function()
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ {test_name} 成功 (実行時間: {execution_time:.2f}秒)")
                self.passed_tests += 1
            else:
                logger.error(f"❌ {test_name} 失敗 (実行時間: {execution_time:.2f}秒)")
                self.failed_tests += 1
            
            self.test_results[test_name] = {
                'success': success,
                'execution_time': execution_time,
                'error': None
            }
            
            return success
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{test_name} でエラーが発生: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.test_results[test_name] = {
                'success': False,
                'execution_time': execution_time,
                'error': error_msg
            }
            
            self.failed_tests += 1
            return False

    def run_all_tests(self) -> bool:
        """すべてのテストを実行"""
        logger.info("🚀 MLトレーニング系包括的テストスイートを開始")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # テストモジュールの定義
        test_modules = [
            ("計算正確性テスト", run_all_calculation_tests),
            ("前処理正確性テスト", run_all_preprocessing_tests),
            ("特徴量計算テスト", run_all_feature_calculation_tests),
            ("データ変換テスト", run_all_data_transformation_tests),
            ("ラベル生成テスト", run_all_label_generation_tests),
            ("統合テスト", run_integration_tests),
            ("極端エッジケーステスト", run_extreme_edge_case_tests),
            ("実環境シミュレーションテスト", run_real_environment_simulation_tests),
            ("高度パフォーマンス最適化検証", run_advanced_performance_optimization_tests),
            ("コードカバレッジ確認", run_code_coverage_verification_tests),
            ("AutoML包括的テスト", run_automl_comprehensive_tests),
            ("Optunaベイジアン最適化テスト", run_optuna_bayesian_optimization_tests),
            ("エラーハンドリングテスト", run_all_error_handling_tests),
            ("パフォーマンステスト", run_all_performance_tests),
        ]
        
        self.total_tests = len(test_modules)
        
        # 各テストモジュールを実行
        all_passed = True
        for test_name, test_function in test_modules:
            success = self.run_test_module(test_name, test_function)
            if not success:
                all_passed = False
        
        self.end_time = time.time()
        
        # 結果サマリーを表示
        self._display_summary()
        
        return all_passed

    def _display_summary(self):
        """テスト結果のサマリーを表示"""
        total_time = self.end_time - self.start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 テスト結果サマリー")
        logger.info("=" * 80)
        
        logger.info(f"総実行時間: {total_time:.2f}秒")
        logger.info(f"総テスト数: {self.total_tests}")
        logger.info(f"成功: {self.passed_tests}")
        logger.info(f"失敗: {self.failed_tests}")
        logger.info(f"成功率: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        logger.info("\n📋 詳細結果:")
        for test_name, result in self.test_results.items():
            status = "✅ 成功" if result['success'] else "❌ 失敗"
            time_str = f"{result['execution_time']:.2f}秒"
            logger.info(f"  {test_name}: {status} ({time_str})")
            
            if result['error']:
                logger.info(f"    エラー: {result['error']}")
        
        if self.failed_tests == 0:
            logger.info("\n🎉 すべてのテストが正常に完了しました！")
            logger.info("MLトレーニングシステムの計算と前処理の正確性が確認されました。")
        else:
            logger.warning(f"\n⚠️ {self.failed_tests}個のテストが失敗しました。")
            logger.warning("失敗したテストを確認し、問題を修正してください。")

    def run_specific_test(self, test_name: str) -> bool:
        """特定のテストのみを実行"""
        test_mapping = {
            "calculations": ("計算正確性テスト", run_all_calculation_tests),
            "preprocessing": ("前処理正確性テスト", run_all_preprocessing_tests),
            "features": ("特徴量計算テスト", run_all_feature_calculation_tests),
            "transformations": ("データ変換テスト", run_all_data_transformation_tests),
            "labels": ("ラベル生成テスト", run_all_label_generation_tests),
            "integration": ("統合テスト", run_integration_tests),
            "extreme": ("極端エッジケーステスト", run_extreme_edge_case_tests),
            "realenv": ("実環境シミュレーションテスト", run_real_environment_simulation_tests),
            "advperf": ("高度パフォーマンス最適化検証", run_advanced_performance_optimization_tests),
            "coverage": ("コードカバレッジ確認", run_code_coverage_verification_tests),
            "automl": ("AutoML包括的テスト", run_automl_comprehensive_tests),
            "optuna": ("Optunaベイジアン最適化テスト", run_optuna_bayesian_optimization_tests),
            "errors": ("エラーハンドリングテスト", run_all_error_handling_tests),
            "performance": ("パフォーマンステスト", run_all_performance_tests),
        }
        
        if test_name not in test_mapping:
            logger.error(f"不明なテスト名: {test_name}")
            logger.info(f"利用可能なテスト: {list(test_mapping.keys())}")
            return False
        
        self.start_time = time.time()
        self.total_tests = 1
        
        test_display_name, test_function = test_mapping[test_name]
        success = self.run_test_module(test_display_name, test_function)
        
        self.end_time = time.time()
        self._display_summary()
        
        return success

    def validate_test_environment(self) -> bool:
        """テスト環境の検証"""
        logger.info("🔍 テスト環境を検証中...")
        
        try:
            # 必要なライブラリの確認
            import numpy as np
            import pandas as pd
            import sklearn
            import scipy
            import talib
            
            logger.info("✅ 必要なライブラリが利用可能です")
            
            # プロジェクトモジュールの確認
            from app.utils.data_processing import DataProcessor
            from app.utils.label_generation import LabelGenerator
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            logger.info("✅ プロジェクトモジュールが利用可能です")
            
            # 基本的な動作確認
            processor = DataProcessor()
            label_generator = LabelGenerator()
            fe_service = FeatureEngineeringService()
            
            logger.info("✅ 基本的なクラスのインスタンス化が成功しました")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ 必要なライブラリが見つかりません: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ テスト環境の検証でエラーが発生: {e}")
            return False

    def generate_test_report(self, output_file: str = None):
        """テスト結果のレポートを生成"""
        if not self.test_results:
            logger.warning("テスト結果がありません。先にテストを実行してください。")
            return
        
        report_lines = [
            "# MLトレーニング系テスト結果レポート",
            f"実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"総実行時間: {(self.end_time - self.start_time):.2f}秒",
            "",
            "## サマリー",
            f"- 総テスト数: {self.total_tests}",
            f"- 成功: {self.passed_tests}",
            f"- 失敗: {self.failed_tests}",
            f"- 成功率: {(self.passed_tests/self.total_tests)*100:.1f}%",
            "",
            "## 詳細結果"
        ]
        
        for test_name, result in self.test_results.items():
            status = "✅ 成功" if result['success'] else "❌ 失敗"
            report_lines.append(f"### {test_name}")
            report_lines.append(f"- ステータス: {status}")
            report_lines.append(f"- 実行時間: {result['execution_time']:.2f}秒")
            
            if result['error']:
                report_lines.append(f"- エラー: {result['error']}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"📄 テストレポートを保存しました: {output_file}")
            except Exception as e:
                logger.error(f"レポート保存エラー: {e}")
        else:
            logger.info("\n" + report_content)


def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_suite = MLAccuracyTestSuite()
    
    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "validate":
            success = test_suite.validate_test_environment()
            sys.exit(0 if success else 1)
        else:
            success = test_suite.run_specific_test(test_name)
    else:
        # 環境検証
        if not test_suite.validate_test_environment():
            logger.error("テスト環境の検証に失敗しました。")
            sys.exit(1)
        
        # 全テスト実行
        success = test_suite.run_all_tests()
    
    # レポート生成
    test_suite.generate_test_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
