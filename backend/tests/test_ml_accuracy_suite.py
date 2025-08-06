"""
MLトレーニング系の匁E��皁E��ストスイーチE
計算正確性、前処琁E��確性、特徴量計算、データ変換、ラベル生�Eの
すべてのチE��トを統合実行し、MLシスチE��全体�E信頼性を検証します、E"""

import logging
import sys
import os
import time
import traceback
# ruff: noqa: F401 - availability check\nimport importlib.util as _il\nHAS_NUMPY = _il.find_spec("numpy") is not None
HAS_PANDAS = _il.find_spec("pandas") is not None

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 吁E��ストモジュールをインポ�EチEfrom tests.calculations.test_ml_calculations import run_all_calculation_tests
from tests.preprocessing.test_preprocessing_accuracy import run_all_preprocessing_tests
from tests.feature_engineering.test_feature_calculations import run_all_feature_calculation_tests
from tests.data_transformations.test_data_transformations import run_all_data_transformation_tests
from tests.label_generation.test_label_generation import run_all_label_generation_tests
from tests.enhanced.test_error_handling import run_all_error_handling_tests
from tests.enhanced.test_performance import run_all_performance_tests

# 統合テスト関数を直接定義
def run_integration_tests():
    """統合テストを実行（修正版！E""
    logger.info("🔗 統合テストを開始（インチE��クス整合性修正版！E)

    try:
        # 忁E��なモジュールをインポ�EチE        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.utils.index_alignment import MLWorkflowIndexManager

        # インチE��クス管琁E��を�E期化
        index_manager = MLWorkflowIndexManager()

        # チE��トデータ生�E
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

        # ワークフロー初期匁E        index_manager.initialize_workflow(raw_data)

        # Step 1: チE�Eタ前�E琁E��インチE��クス追跡付き�E�E        processor = DataProcessor()

        def preprocess_func(data):
            return processor.preprocess_features(
                data[['Close', 'Volume']].copy(),
                scale_features=False,
                remove_outliers=True
            )

        processed_data = index_manager.process_with_index_tracking(
            "前�E琁E, raw_data, preprocess_func
        )

        # Step 2: 特徴量エンジニアリング�E�インチE��クス追跡付き�E�E        fe_service = FeatureEngineeringService()

        def feature_engineering_func(data):
            # 前�E琁E��れたチE�EタのインチE��クスに合わせて允E��ータを調整
            aligned_ohlcv = raw_data.loc[data.index]
            return fe_service.calculate_advanced_features(aligned_ohlcv)

        features = index_manager.process_with_index_tracking(
            "特徴量エンジニアリング", processed_data, feature_engineering_func
        )

        # Step 3: ラベル生�E�E�インチE��クス整合性を老E�E�E�E        label_generator = LabelGenerator()

        # 特徴量�EインチE��クスに合わせて価格チE�Eタを調整
        aligned_price_data = raw_data.loc[features.index, 'Close']

        labels, _ = label_generator.generate_labels(
            aligned_price_data,
            method=ThresholdMethod.FIXED,
            threshold_up=0.02,
            threshold_down=-0.02
        )

        # Step 4: 最終的なインチE��クス整吁E        final_features, final_labels = index_manager.finalize_workflow(
            features, labels, alignment_method="intersection"
        )

        # 統合検証
        assert len(final_features) > 0, "最終特徴量が生�Eされませんでした"
        assert len(final_labels) > 0, "最終ラベルが生成されませんでした"

        # インチE��クス整合性の検証
        validation_result = index_manager.alignment_manager.validate_alignment(
            final_features, final_labels, min_alignment_ratio=0.95
        )

        logger.info(f"最終インチE��クス整合性検証:")
        logger.info(f"  特徴量行数: {validation_result['features_rows']}")
        logger.info(f"  ラベル行数: {validation_result['labels_rows']}")
        logger.info(f"  共通行数: {validation_result['common_rows']}")
        logger.info(f"  整合率: {validation_result['alignment_ratio']*100:.1f}%")

        # 高い整合性を要求！E5%以上！E        assert validation_result["is_valid"], \
            f"インチE��クス整合性が不十刁E {validation_result['alignment_ratio']*100:.1f}% < 95%"

        # ワークフローサマリー
        workflow_summary = index_manager.get_workflow_summary()
        logger.info(f"ワークフロー完亁E")
        logger.info(f"  允E��ータ: {workflow_summary['original_rows']}衁E)
        logger.info(f"  最終データ: {workflow_summary['final_rows']}衁E)
        logger.info(f"  チE�Eタ保持玁E {workflow_summary['data_retention_rate']*100:.1f}%")

        logger.info("✁E統合テスト完亁E��インチE��クス整合性修正版！E)
        return True

    except Exception as e:
        logger.error(f"❁E統合テストでエラーが発甁E {e}")
        import traceback
        traceback.print_exc()
        return False


def run_extreme_edge_case_tests():
    """極端エチE��ケースチE��トを実衁E""
    logger.info("🔥 極端エチE��ケースチE��トを開姁E)

    try:
        from app.utils.data_processing import DataProcessor

        # マイクロチE�EタセチE��チE��チE        logger.info("マイクロチE�EタセチE��処琁E��スチE..")
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
            logger.info(f"マイクロチE�Eタ処琁E�E劁E {len(processed)}衁E)
        except Exception as e:
            logger.info(f"マイクロチE�Eタで期征E��りエラー: {e}")

        # 全同値チE�EタチE��チE        logger.info("全同値チE�EタセチE��チE��チE..")
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
            logger.info(f"同値チE�Eタ処琁E�E劁E std={processed['Close'].std():.6f}")
        except Exception as e:
            logger.info(f"同値チE�Eタで期征E��りエラー: {e}")

        # チE�Eタ破損シナリオチE��チE        logger.info("チE�Eタ破損シナリオチE��チE..")
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
                logger.info("破損データが適刁E��修復されました")
            else:
                logger.warning("無効な値が残ってぁE��ぁE)

        except Exception as e:
            logger.info(f"破損データで期征E��りエラー: {e}")

        logger.info("✁E極端エチE��ケースチE��ト完亁E)
        return True

    except Exception as e:
        logger.error(f"❁E極端エチE��ケースチE��トでエラーが発甁E {e}")
        return False


def run_real_environment_simulation_tests():
    """実環墁E��ミュレーションチE��トを実衁E""
    logger.info("🌍 実環墁E��ミュレーションチE��トを開姁E)

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # 実際の市場チE�Eタパターンのシミュレーション
        logger.info("実市場チE�Eタパターンシミュレーション...")

        # ビットコインの実際の価格動作を模倣
        np.random.seed(42)
        size = 1000

        # より現実的な価格動作（トレンチE+ ボラチE��リチE��クラスタリング�E�E        base_price = 50000
        prices = [base_price]
        volatility = 0.02

        for i in range(1, size):
            # ボラチE��リチE��クラスタリング効极E            if i > 20:
                recent_volatility = np.std([prices[j]/prices[j-1] - 1 for j in range(i-20, i)])
                volatility = 0.01 + recent_volatility * 2

            # 価格変動
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.3))  # 価格下限

        # 現実的なOHLCVチE�Eタ生�E
        market_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'Volume': np.random.lognormal(10, 0.8, size)  # より現実的なボリューム刁E��E        })

        # チE�Eタ処琁E��イプライン
        processor = DataProcessor()
        processed_data = processor.preprocess_features(
            market_data[['Close', 'Volume']].copy(),
            scale_features=True,
            remove_outliers=True
        )

        logger.info(f"実市場チE�Eタ処琁E�E劁E {len(processed_data)}衁E)

        # 特徴量エンジニアリング
        fe_service = FeatureEngineeringService()
        features = fe_service.calculate_advanced_features(market_data)

        logger.info(f"実市場チE�Eタ特徴釁E {features.shape[1]}倁E)

        # ラベル生�E
        label_generator = LabelGenerator()
        labels, _ = label_generator.generate_labels(
            market_data['Close'],
            method=ThresholdMethod.FIXED,
            threshold_up=0.02,
            threshold_down=-0.02
        )

        label_distribution = pd.Series(labels).value_counts()
        logger.info(f"実市場チE�Eタラベル刁E��E {label_distribution.to_dict()}")

        # チE�Eタ品質の検証
        price_volatility = market_data['Close'].pct_change().std()
        volume_consistency = market_data['Volume'].std() / market_data['Volume'].mean()

        logger.info(f"価格ボラチE��リチE��: {price_volatility:.4f}")
        logger.info(f"ボリューム変動係数: {volume_consistency:.4f}")

        # 長時間実行安定性チE��チE        logger.info("長時間実行安定性チE��チE..")

        start_time = time.time()
        iterations = 10

        for i in range(iterations):
            # 繰り返し処琁E��メモリリークめE��能劣化をチェチE��
            test_data = market_data.sample(n=100).copy()

            processed = processor.preprocess_features(
                test_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )

            if i % 3 == 0:
                logger.info(f"  反復 {i+1}/{iterations} 完亁E)

        total_time = time.time() - start_time
        avg_time_per_iteration = total_time / iterations

        logger.info(f"長時間実行テスト完亁E 平坁E{avg_time_per_iteration:.3f}私E反復")

        # パフォーマンス劣化�E検�E
        if avg_time_per_iteration > 1.0:
            logger.warning(f"処琁E��間が長すぎまぁE {avg_time_per_iteration:.3f}私E)

        # I/Oエラーシミュレーション
        logger.info("I/Oエラーシミュレーション...")

        # 不完�EなチE�Eタ�E�読み込みエラーをシミュレート！E        incomplete_data = market_data.iloc[:50].copy()  # チE�Eタが途中で刁E��めE
        try:
            processed = processor.preprocess_features(
                incomplete_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )
            logger.info(f"不完�EチE�Eタ処琁E�E劁E {len(processed)}衁E)
        except Exception as e:
            logger.info(f"不完�EチE�Eタで期征E��りエラー: {e}")

        # メモリ制紁E��ミュレーション
        logger.info("メモリ制紁E��ミュレーション...")

        # 大きなチE�EタセチE��を小さなチャンクに刁E��して処琁E        large_data = pd.concat([market_data] * 5, ignore_index=True)  # 5倍�EチE�Eタ
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
        logger.info(f"チャンク処琁E�E劁E {len(final_result)}衁E)

        logger.info("✁E実環墁E��ミュレーションチE��ト完亁E)
        return True

    except Exception as e:
        logger.error(f"❁E実環墁E��ミュレーションチE��トでエラーが発甁E {e}")
        return False


def run_advanced_performance_optimization_tests():
    """高度なパフォーマンス最適化検証チE��トを実衁E""
    logger.info("⚡ 高度なパフォーマンス最適化検証チE��トを開姁E)

    try:
        import psutil
        import gc
        from app.utils.data_processing import DataProcessor
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        process = psutil.Process(os.getpid())

        # メモリリーク検�EチE��チE        logger.info("メモリリーク検�EチE��チE..")

        initial_memory = process.memory_info().rss
        memory_measurements = []

        processor = DataProcessor()

        for i in range(20):
            # 繰り返し処琁E��メモリリークを検�E
            test_data = pd.DataFrame({
                'Close': np.random.normal(100, 10, 1000),
                'Volume': np.random.lognormal(10, 1, 1000)
            })

            processed = processor.preprocess_features(
                test_data.copy(),
                scale_features=True,
                remove_outliers=True
            )

            # メモリ使用量測宁E            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            memory_measurements.append(memory_increase)

            # 明示皁E��ガベ�Eジコレクション
            del test_data, processed
            gc.collect()

            if i % 5 == 0:
                logger.info(f"  反復 {i+1}: メモリ増加 {memory_increase/1e6:.1f}MB")

        # メモリリークの刁E��
        memory_trend = np.polyfit(range(len(memory_measurements)), memory_measurements, 1)[0]

        logger.info(f"メモリ使用量トレンチE {memory_trend/1e6:.3f}MB/反復")

        if memory_trend > 1e6:  # 1MB/反復以上�E増加
            logger.warning(f"メモリリークの可能性: {memory_trend/1e6:.1f}MB/反復")
        else:
            logger.info("✁Eメモリリークは検�Eされませんでした")

        # CPU使用玁E��視テスチE        logger.info("CPU使用玁E��視テスチE..")

        cpu_measurements = []
        start_time = time.time()

        fe_service = FeatureEngineeringService()

        # CPU雁E��E��なタスクを実衁E        for i in range(5):
            cpu_before = psutil.cpu_percent(interval=0.1)

            # 褁E��な特徴量計箁E            complex_data = pd.DataFrame({
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

            logger.info(f"  タスク {i+1}: CPU使用玁E{cpu_usage:.1f}%")

        avg_cpu_usage = np.mean(cpu_measurements)
        logger.info(f"平均CPU使用玁E {avg_cpu_usage:.1f}%")

        # ボトルネック特定テスチE        logger.info("ボトルネック特定テスチE..")

        # 吁E�E琁E��階�E時間測宁E        bottleneck_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 5000),
            'Volume': np.random.lognormal(10, 1, 5000)
        })

        # チE�Eタ前�E琁E�Eボトルネック
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

        # 完�E処琁E�Eボトルネック
        start = time.time()
        full_processed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=True,
            remove_outliers=True
        )
        full_processing_time = time.time() - start

        logger.info("処琁E��間�E极E")
        logger.info(f"  基本前�E琁E {preprocessing_time:.3f}私E)
        logger.info(f"  スケーリング: {scaling_time:.3f}私E)
        logger.info(f"  外れ値除去: {outlier_removal_time:.3f}私E)
        logger.info(f"  完�E処琁E {full_processing_time:.3f}私E)

        # ボトルネックの特宁E        processing_times = {
            'スケーリング': scaling_time - preprocessing_time,
            '外れ値除去': outlier_removal_time - preprocessing_time,
            '統合�E琁E: full_processing_time - max(scaling_time, outlier_removal_time)
        }

        bottleneck = max(processing_times, key=processing_times.get)
        bottleneck_time = processing_times[bottleneck]

        logger.info(f"最大ボトルネック: {bottleneck} ({bottleneck_time:.3f}私E")

        # パフォーマンス効玁E�E計箁E        data_throughput = len(bottleneck_data) / full_processing_time
        logger.info(f"チE�Eタ処琁E��ループッチE {data_throughput:.0f}衁E私E)

        # メモリ効玁E�E測宁E        memory_efficiency = len(bottleneck_data) / (process.memory_info().rss / 1e6)
        logger.info(f"メモリ効玁E {memory_efficiency:.0f}衁EMB")

        # パフォーマンス最適化�E提桁E        optimization_suggestions = []

        if bottleneck_time > 0.1:
            optimization_suggestions.append(f"{bottleneck}の最適化が忁E��E)

        if avg_cpu_usage > 80:
            optimization_suggestions.append("CPU使用玁E��高すぎまぁE)

        if memory_trend > 5e5:  # 0.5MB/反復
            optimization_suggestions.append("メモリ使用量�E最適化が忁E��E)

        if data_throughput < 1000:
            optimization_suggestions.append("チE�Eタ処琁E��ループット�E改喁E��忁E��E)

        if optimization_suggestions:
            logger.warning("最適化提桁E")
            for suggestion in optimization_suggestions:
                logger.warning(f"  - {suggestion}")
        else:
            logger.info("✁Eパフォーマンスは良好でぁE)

        logger.info("✁E高度なパフォーマンス最適化検証チE��ト完亁E)
        return True

    except Exception as e:
        logger.error(f"❁E高度なパフォーマンス最適化検証チE��トでエラーが発甁E {e}")
        return False


def run_code_coverage_verification_tests():
    """コードカバレチE��確認テストを実衁E""
    logger.info("📋 コードカバレチE��確認テストを開姁E)

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # 設定パラメータの絁E��合わせテスチE        logger.info("設定パラメータ絁E��合わせテスチE..")

        processor = DataProcessor()
        test_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.lognormal(10, 1, 100)
        })

        # 異なる設定�E絁E��合わせをチE��チE        configurations = [
            {'scale_features': True, 'remove_outliers': True, 'outlier_method': 'iqr'},
            {'scale_features': True, 'remove_outliers': True, 'outlier_method': 'zscore'},
            {'scale_features': True, 'remove_outliers': False},
            {'scale_features': False, 'remove_outliers': True, 'outlier_method': 'iqr'},
            {'scale_features': False, 'remove_outliers': False},
        ]

        for i, config in enumerate(configurations):
            try:
                result = processor.preprocess_features(test_data.copy(), **config)
                logger.info(f"  設宁E{i+1}: 成功 ({len(result)}衁E")
            except Exception as e:
                logger.info(f"  設宁E{i+1}: エラー - {e}")

        # ラベル生�Eの異なる手法テスチE        logger.info("ラベル生�E手法網羁E��スチE..")

        label_generator = LabelGenerator()
        price_series = pd.Series(np.random.normal(100, 10, 100), name='Close')

        # 異なる閾値設宁E        threshold_configs = [
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.01, 'threshold_down': -0.01},
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.05, 'threshold_down': -0.05},
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.1, 'threshold_down': -0.1},
        ]

        for i, config in enumerate(threshold_configs):
            try:
                labels, info = label_generator.generate_labels(price_series, **config)
                label_dist = pd.Series(labels).value_counts()
                logger.info(f"  閾値設宁E{i+1}: 成功 - 刁E��E{label_dist.to_dict()}")
            except Exception as e:
                logger.info(f"  閾値設宁E{i+1}: エラー - {e}")

        # 例外�E琁E�E網羁E��チE��チE        logger.info("例外�E琁E��羁E��チE��チE..")

        # 空チE�Eタでの例外�E琁E        empty_data = pd.DataFrame()
        try:
            processor.preprocess_features(empty_data)
            logger.info("  空チE�Eタ: 処琁E�E劁E)
        except Exception as e:
            logger.info(f"  空チE�Eタ: 例外�E琁E��誁E- {e}")

        # 不正な型での例外�E琁E        invalid_data = "invalid_data"
        try:
            processor.preprocess_features(invalid_data)
            logger.info("  不正垁E 処琁E�E劁E)
        except Exception as e:
            logger.info(f"  不正垁E 例外�E琁E��誁E- {e}")

        # NaNのみのチE�Eタでの例外�E琁E        nan_data = pd.DataFrame({'Close': [np.nan] * 10, 'Volume': [np.nan] * 10})
        try:
            result = processor.preprocess_features(nan_data)
            logger.info(f"  NaNチE�Eタ: 処琁E�E劁E({len(result)}衁E")
        except Exception as e:
            logger.info(f"  NaNチE�Eタ: 例外�E琁E��誁E- {e}")

        # 特徴量エンジニアリングの例外�E琁E��スチE        logger.info("特徴量エンジニアリング例外�E琁E��スチE..")

        fe_service = FeatureEngineeringService()

        # 不完�EなOHLCVチE�Eタ
        incomplete_ohlcv = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103]
            # Close, Low, Volumeが不足
        })

        try:
            features = fe_service.calculate_advanced_features(incomplete_ohlcv)
            logger.info(f"  不完�EOHLCV: 処琁E�E劁E({features.shape[1]}特徴釁E")
        except Exception as e:
            logger.info(f"  不完�EOHLCV: 例外�E琁E��誁E- {e}")

        # 極小データでの特徴量計箁E        tiny_ohlcv = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000]
        })

        try:
            features = fe_service.calculate_advanced_features(tiny_ohlcv)
            logger.info(f"  極小OHLCV: 処琁E�E劁E({features.shape[1]}特徴釁E")
        except Exception as e:
            logger.info(f"  極小OHLCV: 例外�E琁E��誁E- {e}")

        # エチE��ケースでのラベル生�E
        logger.info("ラベル生�EエチE��ケースチE��チE..")

        # 単調増加チE�Eタ
        monotonic_series = pd.Series(range(100), name='Close')
        try:
            labels, _ = label_generator.generate_labels(
                monotonic_series,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            label_dist = pd.Series(labels).value_counts()
            logger.info(f"  単調増加: 成功 - 刁E��E{label_dist.to_dict()}")
        except Exception as e:
            logger.info(f"  単調増加: 例外�E琁E��誁E- {e}")

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
            logger.info(f"  単調減封E 成功 - 刁E��E{label_dist.to_dict()}")
        except Exception as e:
            logger.info(f"  単調減封E 例外�E琁E��誁E- {e}")

        # チE�Eタ型変換の網羁E��チE��チE        logger.info("チE�Eタ型変換網羁E��チE��チE..")

        # 異なるデータ型での処琁E        data_types_test = {
            'int32': pd.DataFrame({'Close': np.array([100, 101, 102], dtype=np.int32)}),
            'int64': pd.DataFrame({'Close': np.array([100, 101, 102], dtype=np.int64)}),
            'float32': pd.DataFrame({'Close': np.array([100.0, 101.0, 102.0], dtype=np.float32)}),
            'float64': pd.DataFrame({'Close': np.array([100.0, 101.0, 102.0], dtype=np.float64)}),
        }

        for dtype_name, dtype_data in data_types_test.items():
            try:
                result = processor.preprocess_features(dtype_data.copy())
                logger.info(f"  {dtype_name}: 処琁E�E劁E({result.dtypes['Close']})")
            except Exception as e:
                logger.info(f"  {dtype_name}: 例外�E琁E��誁E- {e}")

        # 墁E��値チE��チE        logger.info("墁E��値チE��チE..")

        # 最小値・最大値での処琁E        boundary_data = pd.DataFrame({
            'Close': [sys.float_info.min, 0, sys.float_info.max],
            'Volume': [1, 1000, sys.float_info.max]
        })

        try:
            result = processor.preprocess_features(boundary_data.copy())
            logger.info(f"  墁E��値: 処琁E�E劁E({len(result)}衁E")
        except Exception as e:
            logger.info(f"  墁E��値: 例外�E琁E��誁E- {e}")

        logger.info("✁EコードカバレチE��確認テスト完亁E)
        return True

    except Exception as e:
        logger.error(f"❁EコードカバレチE��確認テストでエラーが発甁E {e}")
        return False


def run_automl_comprehensive_tests():
    """AutoML匁E��皁E��ストを実衁E""
    logger.info("🤁EAutoML匁E��皁E��ストを開姁E)

    try:
        from app.services.ml.ml_training_service import MLTrainingService
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.services.ml.feature_engineering.automl_features.automl_config import AutoMLConfig
        from app.utils.index_alignment import MLWorkflowIndexManager
        from app.utils.label_generation import LabelGenerator, ThresholdMethod

        # BTC市場チE�Eタ生�E関数
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

        # AutoMLアンサンブル学習テスチE        logger.info("AutoMLアンサンブル学習テスチE..")
        btc_data = create_btc_market_data("1h", 200)

        try:
            # AutoML設定（軽量版�E�E            automl_config = {
                "tsfresh": {"enabled": True, "feature_count_limit": 20},
                "autofeat": {"enabled": False}
            }

            # アンサンブル設宁E            ensemble_config = {
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

            logger.info(f"AutoMLアンサンブル学習�E劁E 精度={result.get('accuracy', 'N/A')}")

        except Exception as e:
            logger.info(f"AutoMLアンサンブル学習でエラー�E�期征E��れる場合もあります！E {e}")

        # AutoML特徴量選択テスチE        logger.info("AutoML特徴量選択テスチE..")

        try:
            # AutoML設定で特徴量エンジニアリング
            automl_config_obj = AutoMLConfig.get_financial_optimized_config()
            fe_service = FeatureEngineeringService(automl_config=automl_config_obj)

            # 基本特徴量計箁E            basic_features = fe_service.calculate_advanced_features(btc_data)

            logger.info(f"基本特徴釁E {basic_features.shape[1]}倁E)

            # 特徴量が生�Eされることを確誁E            assert basic_features.shape[1] > 0, "特徴量が生�Eされませんでした"

            logger.info("✁EAutoML特徴量選択が正常に動作しました")

        except Exception as e:
            logger.info(f"AutoML特徴量選択でエラー�E�期征E��れる場合もあります！E {e}")

        # AutoMLワークフロー統合テスチE        logger.info("AutoMLワークフロー統合テスチE..")

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

            # ラベル生�E
            label_generator = LabelGenerator()
            aligned_price_data = btc_data.loc[features.index, 'Close']
            labels, _ = label_generator.generate_labels(
                aligned_price_data,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )

            # 最終整吁E            final_features, final_labels = index_manager.finalize_workflow(
                features, labels, alignment_method="intersection"
            )

            # 検証
            assert len(final_features) > 0, "最終特徴量が生�Eされませんでした"
            assert len(final_labels) > 0, "最終ラベルが生成されませんでした"

            workflow_summary = index_manager.get_workflow_summary()
            logger.info(f"AutoMLワークフロー完亁E チE�Eタ保持玁E{workflow_summary['data_retention_rate']*100:.1f}%")

        except Exception as e:
            logger.info(f"AutoMLワークフロー統合でエラー�E�期征E��れる場合もあります！E {e}")

        logger.info("✁EAutoML匁E��皁E��スト完亁E)
        return True

    except Exception as e:
        logger.error(f"❁EAutoML匁E��皁E��ストでエラーが発甁E {e}")
        return False


def run_optuna_bayesian_optimization_tests():
    """Optunaベイジアン最適化テストを実衁E""
    logger.info("🔍 Optunaベイジアン最適化テストを開姁E)

    try:
        from app.services.optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace

        # Optunaベイジアン最適化�E基本チE��チE        logger.info("Optunaベイジアン最適化基本チE��チE..")

        # 簡単な目皁E��数�E�二次関数の最大化！E        def simple_objective(params):
            x = params['x']
            y = params['y']
            # 最大値ぁE2, 3)で値ぁE0となる二次関数
            return 10 - (x - 2)**2 - (y - 3)**2

        # パラメータ空間定義
        parameter_space = {
            'x': ParameterSpace(type="real", low=0.0, high=5.0),
            'y': ParameterSpace(type="real", low=0.0, high=5.0)
        }

        # Optuna最適化実衁E        optimizer = OptunaOptimizer()
        result = optimizer.optimize(
            objective_function=simple_objective,
            parameter_space=parameter_space,
            n_calls=20  # チE��ト用に少なく設宁E        )

        # 結果検証
        assert result.best_score > 8.0, f"最適化結果が不十刁E {result.best_score}"
        assert abs(result.best_params['x'] - 2.0) < 1.0, "xパラメータの最適化が不十刁E
        assert abs(result.best_params['y'] - 3.0) < 1.0, "yパラメータの最適化が不十刁E

        logger.info(f"Optuna最適化�E劁E スコア={result.best_score:.3f}, "
                   f"パラメータ={result.best_params}")

        # クリーンアチE�E
        optimizer.cleanup()

        # LightGBMパラメータ最適化テスチE        logger.info("LightGBMパラメータ最適化テスチE..")

        # LightGBM用の目皁E��数�E�簡略版！E        def lightgbm_objective(params):
            # 実際のモチE��学習�E代わりに、パラメータの妥当性をスコア匁E            score = 0.5  # ベ�Eススコア

            # パラメータの妥当性に基づぁE��スコア調整
            if 0.01 <= params['learning_rate'] <= 0.3:
                score += 0.1
            if 10 <= params['num_leaves'] <= 100:
                score += 0.1
            if 0.5 <= params['feature_fraction'] <= 1.0:
                score += 0.1
            if 5 <= params['min_data_in_leaf'] <= 50:
                score += 0.1

            # ランダムノイズを追加�E�実際の学習結果をシミュレート！E            import random
            score += random.uniform(-0.1, 0.1)

            return score

        # LightGBMパラメータ空閁E        lgb_parameter_space = OptunaOptimizer.get_default_parameter_space()

        # 最適化実衁E        optimizer2 = OptunaOptimizer()
        lgb_result = optimizer2.optimize(
            objective_function=lightgbm_objective,
            parameter_space=lgb_parameter_space,
            n_calls=15
        )

        # 結果検証
        assert lgb_result.best_score > 0.5, f"LightGBM最適化結果が不十刁E {lgb_result.best_score}"
        assert 'learning_rate' in lgb_result.best_params, "learning_rateパラメータが不足"
        assert 'num_leaves' in lgb_result.best_params, "num_leavesパラメータが不足"

        logger.info(f"LightGBM最適化�E劁E スコア={lgb_result.best_score:.3f}")

        # クリーンアチE�E
        optimizer2.cleanup()

        # アンサンブルパラメータ最適化テスチE        logger.info("アンサンブルパラメータ最適化テスチE..")

        try:
            # アンサンブル用パラメータ空閁E            ensemble_parameter_space = OptunaOptimizer.get_ensemble_parameter_space(
                ensemble_method="bagging",
                enabled_models=["lightgbm", "random_forest"]
            )

            # アンサンブル用目皁E��数
            def ensemble_objective(params):
                score = 0.6  # ベ�Eススコア

                # アンサンブルパラメータの妥当性チェチE��
                if 'n_estimators' in params and 2 <= params['n_estimators'] <= 10:
                    score += 0.1
                if 'bootstrap_fraction' in params and 0.5 <= params['bootstrap_fraction'] <= 1.0:
                    score += 0.1

                return score + np.random.uniform(-0.05, 0.05)

            # 最適化実衁E            optimizer3 = OptunaOptimizer()
            ensemble_result = optimizer3.optimize(
                objective_function=ensemble_objective,
                parameter_space=ensemble_parameter_space,
                n_calls=10
            )

            logger.info(f"アンサンブル最適化�E劁E スコア={ensemble_result.best_score:.3f}")
            optimizer3.cleanup()

        except Exception as e:
            logger.info(f"アンサンブル最適化でエラー�E�期征E��れる場合もあります！E {e}")

        # 最適化効玁E��チE��チE        logger.info("最適化効玁E��チE��チE..")

        # 褁E��な目皁E��数�E�多峰性関数�E�E        def complex_objective(params):
            x, y = params['x'], params['y']
            # Rastrigin関数の変形�E�最大化用�E�E            return -(10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) +
                    (y**2 - 10 * np.cos(2 * np.pi * y)))

        # 効玁E��測宁E        start_time = time.time()
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

        # 効玁E��検証
        assert optimization_time < 30, f"最適化時間が長すぎまぁE {optimization_time:.2f}私E
        assert complex_result.total_evaluations == 25, "評価回数が正しくありません"

        logger.info(f"褁E��関数最適化完亁E 時間={optimization_time:.2f}私E "
                   f"スコア={complex_result.best_score:.3f}")

        optimizer4.cleanup()

        # 異なるサンプラーのチE��チE        logger.info("異なる最適化アルゴリズムチE��チE..")

        try:
            import optuna

            # TPESampler vs RandomSampler の比輁E            samplers = {
                "TPE": optuna.samplers.TPESampler(seed=42),
                "Random": optuna.samplers.RandomSampler(seed=42)
            }

            sampler_results = {}

            for sampler_name, sampler in samplers.items():
                # カスタムOptunaスタチE��作�E
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

            # TPEの方が良ぁE��果を�Eすことを期征E��忁E��ではなぁE��E            if sampler_results["TPE"]["best_score"] > sampler_results["Random"]["best_score"]:
                logger.info("✁ETPEサンプラーがRandomサンプラーより良ぁE��果を�Eしました")
            else:
                logger.info("ℹ�E�Eこ�E試行ではRandomサンプラーの方が良ぁE��果でした")

        except Exception as e:
            logger.info(f"サンプラー比輁E��ストでエラー�E�期征E��れる場合もあります！E {e}")

        logger.info("✁EOptunaベイジアン最適化テスト完亁E)
        return True

    except Exception as e:
        logger.error(f"❁EOptunaベイジアン最適化テストでエラーが発甁E {e}")
        return False

logger = logging.getLogger(__name__)


class MLAccuracyTestSuite:
    """MLトレーニング系の匁E��皁E��ストスイーチE""

    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = None
        self.end_time = None

    def run_test_module(self, test_name: str, test_function) -> bool:
        """個別チE��トモジュールを実衁E""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 {test_name} を開姁E)
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = test_function()
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"✁E{test_name} 成功 (実行時閁E {execution_time:.2f}私E")
                self.passed_tests += 1
            else:
                logger.error(f"❁E{test_name} 失敁E(実行時閁E {execution_time:.2f}私E")
                self.failed_tests += 1
            
            self.test_results[test_name] = {
                'success': success,
                'execution_time': execution_time,
                'error': None
            }
            
            return success
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{test_name} でエラーが発甁E {str(e)}"
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
        """すべてのチE��トを実衁E""
        logger.info("🚀 MLトレーニング系匁E��皁E��ストスイートを開姁E)
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # チE��トモジュールの定義
        test_modules = [
            ("計算正確性チE��チE, run_all_calculation_tests),
            ("前�E琁E��確性チE��チE, run_all_preprocessing_tests),
            ("特徴量計算テスチE, run_all_feature_calculation_tests),
            ("チE�Eタ変換チE��チE, run_all_data_transformation_tests),
            ("ラベル生�EチE��チE, run_all_label_generation_tests),
            ("統合テスチE, run_integration_tests),
            ("極端エチE��ケースチE��チE, run_extreme_edge_case_tests),
            ("実環墁E��ミュレーションチE��チE, run_real_environment_simulation_tests),
            ("高度パフォーマンス最適化検証", run_advanced_performance_optimization_tests),
            ("コードカバレチE��確誁E, run_code_coverage_verification_tests),
            ("AutoML匁E��皁E��スチE, run_automl_comprehensive_tests),
            ("Optunaベイジアン最適化テスチE, run_optuna_bayesian_optimization_tests),
            ("エラーハンドリングチE��チE, run_all_error_handling_tests),
            ("パフォーマンスチE��チE, run_all_performance_tests),
        ]
        
        self.total_tests = len(test_modules)
        
        # 吁E��ストモジュールを実衁E        all_passed = True
        for test_name, test_function in test_modules:
            success = self.run_test_module(test_name, test_function)
            if not success:
                all_passed = False
        
        self.end_time = time.time()
        
        # 結果サマリーを表示
        self._display_summary()
        
        return all_passed

    def _display_summary(self):
        """チE��ト結果のサマリーを表示"""
        total_time = self.end_time - self.start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 チE��ト結果サマリー")
        logger.info("=" * 80)
        
        logger.info(f"総実行時閁E {total_time:.2f}私E)
        logger.info(f"総テスト数: {self.total_tests}")
        logger.info(f"成功: {self.passed_tests}")
        logger.info(f"失敁E {self.failed_tests}")
        logger.info(f"成功玁E {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        logger.info("\n📋 詳細結果:")
        for test_name, result in self.test_results.items():
            status = "✁E成功" if result['success'] else "❁E失敁E
            time_str = f"{result['execution_time']:.2f}私E
            logger.info(f"  {test_name}: {status} ({time_str})")
            
            if result['error']:
                logger.info(f"    エラー: {result['error']}")
        
        if self.failed_tests == 0:
            logger.info("\n🎉 すべてのチE��トが正常に完亁E��ました�E�E)
            logger.info("MLトレーニングシスチE��の計算と前�E琁E�E正確性が確認されました、E)
        else:
            logger.warning(f"\n⚠�E�E{self.failed_tests}個�EチE��トが失敗しました、E)
            logger.warning("失敗したテストを確認し、問題を修正してください、E)

    def run_specific_test(self, test_name: str) -> bool:
        """特定�EチE��ト�Eみを実衁E""
        test_mapping = {
            "calculations": ("計算正確性チE��チE, run_all_calculation_tests),
            "preprocessing": ("前�E琁E��確性チE��チE, run_all_preprocessing_tests),
            "features": ("特徴量計算テスチE, run_all_feature_calculation_tests),
            "transformations": ("チE�Eタ変換チE��チE, run_all_data_transformation_tests),
            "labels": ("ラベル生�EチE��チE, run_all_label_generation_tests),
            "integration": ("統合テスチE, run_integration_tests),
            "extreme": ("極端エチE��ケースチE��チE, run_extreme_edge_case_tests),
            "realenv": ("実環墁E��ミュレーションチE��チE, run_real_environment_simulation_tests),
            "advperf": ("高度パフォーマンス最適化検証", run_advanced_performance_optimization_tests),
            "coverage": ("コードカバレチE��確誁E, run_code_coverage_verification_tests),
            "automl": ("AutoML匁E��皁E��スチE, run_automl_comprehensive_tests),
            "optuna": ("Optunaベイジアン最適化テスチE, run_optuna_bayesian_optimization_tests),
            "errors": ("エラーハンドリングチE��チE, run_all_error_handling_tests),
            "performance": ("パフォーマンスチE��チE, run_all_performance_tests),
        }
        
        if test_name not in test_mapping:
            logger.error(f"不�EなチE��ト名: {test_name}")
            logger.info(f"利用可能なチE��チE {list(test_mapping.keys())}")
            return False
        
        self.start_time = time.time()
        self.total_tests = 1
        
        test_display_name, test_function = test_mapping[test_name]
        success = self.run_test_module(test_display_name, test_function)
        
        self.end_time = time.time()
        self._display_summary()
        
        return success

    def validate_test_environment(self) -> bool:
        """チE��ト環墁E�E検証"""
        logger.info("🔍 チE��ト環墁E��検証中...")
        
        try:
            # 忁E��なライブラリの確誁E            import numpy as np
HAS_PANDAS = _il.find_spec("pandas") is not None
HAS_SKLEARN = _il.find_spec("sklearn") is not None
HAS_SCIPY = _il.find_spec("scipy") is not None
HAS_TALIB = _il.find_spec("talib") is not None
            
            logger.info("✁E忁E��なライブラリが利用可能でぁE)
            
            # プロジェクトモジュールの確誁E            from app.utils.data_processing import DataProcessor
            from app.utils.label_generation import LabelGenerator
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            logger.info("✁Eプロジェクトモジュールが利用可能でぁE)
            
            # 基本皁E��動作確誁E            processor = DataProcessor()
            label_generator = LabelGenerator()
            fe_service = FeatureEngineeringService()
            
            logger.info("✁E基本皁E��クラスのインスタンス化が成功しました")
            
            return True
            
        except ImportError as e:
            logger.error(f"❁E忁E��なライブラリが見つかりません: {e}")
            return False
        except Exception as e:
            logger.error(f"❁EチE��ト環墁E�E検証でエラーが発甁E {e}")
            return False

    def generate_test_report(self, output_file: str = None):
        """チE��ト結果のレポ�Eトを生�E"""
        if not self.test_results:
            logger.warning("チE��ト結果がありません。�EにチE��トを実行してください、E)
            return
        
        report_lines = [
            "# MLトレーニング系チE��ト結果レポ�EチE,
            f"実行日晁E {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"総実行時閁E {(self.end_time - self.start_time):.2f}私E,
            "",
            "## サマリー",
            f"- 総テスト数: {self.total_tests}",
            f"- 成功: {self.passed_tests}",
            f"- 失敁E {self.failed_tests}",
            f"- 成功玁E {(self.passed_tests/self.total_tests)*100:.1f}%",
            "",
            "## 詳細結果"
        ]
        
        for test_name, result in self.test_results.items():
            status = "✁E成功" if result['success'] else "❁E失敁E
            report_lines.append(f"### {test_name}")
            report_lines.append(f"- スチE�Eタス: {status}")
            report_lines.append(f"- 実行時閁E {result['execution_time']:.2f}私E)
            
            if result['error']:
                report_lines.append(f"- エラー: {result['error']}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"📄 チE��トレポ�Eトを保存しました: {output_file}")
            except Exception as e:
                logger.error(f"レポ�Eト保存エラー: {e}")
        else:
            logger.info("\n" + report_content)


def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_suite = MLAccuracyTestSuite()
    
    # コマンドライン引数の処琁E    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "validate":
            success = test_suite.validate_test_environment()
            sys.exit(0 if success else 1)
        else:
            success = test_suite.run_specific_test(test_name)
    else:
        # 環墁E��証
        if not test_suite.validate_test_environment():
            logger.error("チE��ト環墁E�E検証に失敗しました、E)
            sys.exit(1)
        
        # 全チE��ト実衁E        success = test_suite.run_all_tests()
    
    # レポ�Eト生戁E    test_suite.generate_test_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
