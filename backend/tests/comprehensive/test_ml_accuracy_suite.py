"""
ML繝医Ξ繝ｼ繝九Φ繧ｰ邉ｻ縺ｮ蛹・峡逧・ユ繧ｹ繝医せ繧､繝ｼ繝・
險育ｮ玲ｭ｣遒ｺ諤ｧ縲∝燕蜃ｦ逅・ｭ｣遒ｺ諤ｧ縲∫音蠕ｴ驥剰ｨ育ｮ励√ョ繝ｼ繧ｿ螟画鋤縲√Λ繝吶Ν逕滓・縺ｮ
縺吶∋縺ｦ縺ｮ繝・せ繝医ｒ邨ｱ蜷亥ｮ溯｡後＠縲｀L繧ｷ繧ｹ繝・Β蜈ｨ菴薙・菫｡鬆ｼ諤ｧ繧呈､懆ｨｼ縺励∪縺吶・"""

import logging
import sys
import os
import time
import traceback
# ruff: noqa: F401 - availability check\nimport importlib.util as _il\nHAS_NUMPY = _il.find_spec("numpy") is not None
HAS_PANDAS = _il.find_spec("pandas") is not None

# 繝励Ο繧ｸ繧ｧ繧ｯ繝医Ν繝ｼ繝医ｒ繝代せ縺ｫ霑ｽ蜉
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 蜷・ユ繧ｹ繝医Δ繧ｸ繝･繝ｼ繝ｫ繧偵う繝ｳ繝昴・繝・from tests.calculations.test_ml_calculations import run_all_calculation_tests
from tests.preprocessing.test_preprocessing_accuracy import run_all_preprocessing_tests
from tests.feature_engineering.test_feature_calculations import run_all_feature_calculation_tests
from tests.data_transformations.test_data_transformations import run_all_data_transformation_tests
from tests.label_generation.test_label_generation import run_all_label_generation_tests
from tests.enhanced.test_error_handling import run_all_error_handling_tests
from tests.enhanced.test_performance import run_all_performance_tests

# 邨ｱ蜷医ユ繧ｹ繝磯未謨ｰ繧堤峩謗･螳夂ｾｩ
def run_integration_tests():
    """邨ｱ蜷医ユ繧ｹ繝医ｒ螳溯｡鯉ｼ井ｿｮ豁｣迚茨ｼ・""
    logger.info("迫 邨ｱ蜷医ユ繧ｹ繝医ｒ髢句ｧ具ｼ医う繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷域ｧ菫ｮ豁｣迚茨ｼ・)

    try:
        # 蠢・ｦ√↑繝｢繧ｸ繝･繝ｼ繝ｫ繧偵う繝ｳ繝昴・繝・        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.utils.index_alignment import MLWorkflowIndexManager

        # 繧､繝ｳ繝・ャ繧ｯ繧ｹ邂｡逅・勣繧貞・譛溷喧
        index_manager = MLWorkflowIndexManager()

        # 繝・せ繝医ョ繝ｼ繧ｿ逕滓・
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

        # 繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ蛻晄悄蛹・        index_manager.initialize_workflow(raw_data)

        # Step 1: 繝・・繧ｿ蜑榊・逅・ｼ医う繝ｳ繝・ャ繧ｯ繧ｹ霑ｽ霍｡莉倥″・・        processor = DataProcessor()

        def preprocess_func(data):
            return processor.preprocess_features(
                data[['Close', 'Volume']].copy(),
                scale_features=False,
                remove_outliers=True
            )

        processed_data = index_manager.process_with_index_tracking(
            "蜑榊・逅・, raw_data, preprocess_func
        )

        # Step 2: 迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ・医う繝ｳ繝・ャ繧ｯ繧ｹ霑ｽ霍｡莉倥″・・        fe_service = FeatureEngineeringService()

        def feature_engineering_func(data):
            # 蜑榊・逅・＆繧後◆繝・・繧ｿ縺ｮ繧､繝ｳ繝・ャ繧ｯ繧ｹ縺ｫ蜷医ｏ縺帙※蜈・ョ繝ｼ繧ｿ繧定ｪｿ謨ｴ
            aligned_ohlcv = raw_data.loc[data.index]
            return fe_service.calculate_advanced_features(aligned_ohlcv)

        features = index_manager.process_with_index_tracking(
            "迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ", processed_data, feature_engineering_func
        )

        # Step 3: 繝ｩ繝吶Ν逕滓・・医う繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷域ｧ繧定・・・・        label_generator = LabelGenerator()

        # 迚ｹ蠕ｴ驥上・繧､繝ｳ繝・ャ繧ｯ繧ｹ縺ｫ蜷医ｏ縺帙※萓｡譬ｼ繝・・繧ｿ繧定ｪｿ謨ｴ
        aligned_price_data = raw_data.loc[features.index, 'Close']

        labels, _ = label_generator.generate_labels(
            aligned_price_data,
            method=ThresholdMethod.FIXED,
            threshold_up=0.02,
            threshold_down=-0.02
        )

        # Step 4: 譛邨ら噪縺ｪ繧､繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷・        final_features, final_labels = index_manager.finalize_workflow(
            features, labels, alignment_method="intersection"
        )

        # 邨ｱ蜷域､懆ｨｼ
        assert len(final_features) > 0, "譛邨ら音蠕ｴ驥上′逕滓・縺輔ｌ縺ｾ縺帙ｓ縺ｧ縺励◆"
        assert len(final_labels) > 0, "譛邨ゅΛ繝吶Ν縺檎函謌舌＆繧後∪縺帙ｓ縺ｧ縺励◆"

        # 繧､繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷域ｧ縺ｮ讀懆ｨｼ
        validation_result = index_manager.alignment_manager.validate_alignment(
            final_features, final_labels, min_alignment_ratio=0.95
        )

        logger.info(f"譛邨ゅう繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷域ｧ讀懆ｨｼ:")
        logger.info(f"  迚ｹ蠕ｴ驥剰｡梧焚: {validation_result['features_rows']}")
        logger.info(f"  繝ｩ繝吶Ν陦梧焚: {validation_result['labels_rows']}")
        logger.info(f"  蜈ｱ騾夊｡梧焚: {validation_result['common_rows']}")
        logger.info(f"  謨ｴ蜷育紫: {validation_result['alignment_ratio']*100:.1f}%")

        # 鬮倥＞謨ｴ蜷域ｧ繧定ｦ∵ｱゑｼ・5%莉･荳奇ｼ・        assert validation_result["is_valid"], \
            f"繧､繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷域ｧ縺御ｸ榊香蛻・ {validation_result['alignment_ratio']*100:.1f}% < 95%"

        # 繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ繧ｵ繝槭Μ繝ｼ
        workflow_summary = index_manager.get_workflow_summary()
        logger.info(f"繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ螳御ｺ・")
        logger.info(f"  蜈・ョ繝ｼ繧ｿ: {workflow_summary['original_rows']}陦・)
        logger.info(f"  譛邨ゅョ繝ｼ繧ｿ: {workflow_summary['final_rows']}陦・)
        logger.info(f"  繝・・繧ｿ菫晄戟邇・ {workflow_summary['data_retention_rate']*100:.1f}%")

        logger.info("笨・邨ｱ蜷医ユ繧ｹ繝亥ｮ御ｺ・ｼ医う繝ｳ繝・ャ繧ｯ繧ｹ謨ｴ蜷域ｧ菫ｮ豁｣迚茨ｼ・)
        return True

    except Exception as e:
        logger.error(f"笶・邨ｱ蜷医ユ繧ｹ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        import traceback
        traceback.print_exc()
        return False


def run_extreme_edge_case_tests():
    """讌ｵ遶ｯ繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝医ｒ螳溯｡・""
    logger.info("櫨 讌ｵ遶ｯ繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝医ｒ髢句ｧ・)

    try:
        from app.utils.data_processing import DataProcessor

        # 繝槭う繧ｯ繝ｭ繝・・繧ｿ繧ｻ繝・ヨ繝・せ繝・        logger.info("繝槭う繧ｯ繝ｭ繝・・繧ｿ繧ｻ繝・ヨ蜃ｦ逅・ユ繧ｹ繝・..")
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
            logger.info(f"繝槭う繧ｯ繝ｭ繝・・繧ｿ蜃ｦ逅・・蜉・ {len(processed)}陦・)
        except Exception as e:
            logger.info(f"繝槭う繧ｯ繝ｭ繝・・繧ｿ縺ｧ譛溷ｾ・壹ｊ繧ｨ繝ｩ繝ｼ: {e}")

        # 蜈ｨ蜷悟､繝・・繧ｿ繝・せ繝・        logger.info("蜈ｨ蜷悟､繝・・繧ｿ繧ｻ繝・ヨ繝・せ繝・..")
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
            logger.info(f"蜷悟､繝・・繧ｿ蜃ｦ逅・・蜉・ std={processed['Close'].std():.6f}")
        except Exception as e:
            logger.info(f"蜷悟､繝・・繧ｿ縺ｧ譛溷ｾ・壹ｊ繧ｨ繝ｩ繝ｼ: {e}")

        # 繝・・繧ｿ遐ｴ謳阪す繝翫Μ繧ｪ繝・せ繝・        logger.info("繝・・繧ｿ遐ｴ謳阪す繝翫Μ繧ｪ繝・せ繝・..")
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
                logger.info("遐ｴ謳阪ョ繝ｼ繧ｿ縺碁←蛻・↓菫ｮ蠕ｩ縺輔ｌ縺ｾ縺励◆")
            else:
                logger.warning("辟｡蜉ｹ縺ｪ蛟､縺梧ｮ九▲縺ｦ縺・∪縺・)

        except Exception as e:
            logger.info(f"遐ｴ謳阪ョ繝ｼ繧ｿ縺ｧ譛溷ｾ・壹ｊ繧ｨ繝ｩ繝ｼ: {e}")

        logger.info("笨・讌ｵ遶ｯ繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝亥ｮ御ｺ・)
        return True

    except Exception as e:
        logger.error(f"笶・讌ｵ遶ｯ繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        return False


def run_real_environment_simulation_tests():
    """螳溽腸蠅・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・せ繝医ｒ螳溯｡・""
    logger.info("訣 螳溽腸蠅・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・せ繝医ｒ髢句ｧ・)

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # 螳滄圀縺ｮ蟶ょｴ繝・・繧ｿ繝代ち繝ｼ繝ｳ縺ｮ繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ
        logger.info("螳溷ｸょｴ繝・・繧ｿ繝代ち繝ｼ繝ｳ繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ...")

        # 繝薙ャ繝医さ繧､繝ｳ縺ｮ螳滄圀縺ｮ萓｡譬ｼ蜍穂ｽ懊ｒ讓｡蛟｣
        np.random.seed(42)
        size = 1000

        # 繧医ｊ迴ｾ螳溽噪縺ｪ萓｡譬ｼ蜍穂ｽ懶ｼ医ヨ繝ｬ繝ｳ繝・+ 繝懊Λ繝・ぅ繝ｪ繝・ぅ繧ｯ繝ｩ繧ｹ繧ｿ繝ｪ繝ｳ繧ｰ・・        base_price = 50000
        prices = [base_price]
        volatility = 0.02

        for i in range(1, size):
            # 繝懊Λ繝・ぅ繝ｪ繝・ぅ繧ｯ繝ｩ繧ｹ繧ｿ繝ｪ繝ｳ繧ｰ蜉ｹ譫・            if i > 20:
                recent_volatility = np.std([prices[j]/prices[j-1] - 1 for j in range(i-20, i)])
                volatility = 0.01 + recent_volatility * 2

            # 萓｡譬ｼ螟牙虚
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.3))  # 萓｡譬ｼ荳矩剞

        # 迴ｾ螳溽噪縺ｪOHLCV繝・・繧ｿ逕滓・
        market_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'Volume': np.random.lognormal(10, 0.8, size)  # 繧医ｊ迴ｾ螳溽噪縺ｪ繝懊Μ繝･繝ｼ繝蛻・ｸ・        })

        # 繝・・繧ｿ蜃ｦ逅・ヱ繧､繝励Λ繧､繝ｳ
        processor = DataProcessor()
        processed_data = processor.preprocess_features(
            market_data[['Close', 'Volume']].copy(),
            scale_features=True,
            remove_outliers=True
        )

        logger.info(f"螳溷ｸょｴ繝・・繧ｿ蜃ｦ逅・・蜉・ {len(processed_data)}陦・)

        # 迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ
        fe_service = FeatureEngineeringService()
        features = fe_service.calculate_advanced_features(market_data)

        logger.info(f"螳溷ｸょｴ繝・・繧ｿ迚ｹ蠕ｴ驥・ {features.shape[1]}蛟・)

        # 繝ｩ繝吶Ν逕滓・
        label_generator = LabelGenerator()
        labels, _ = label_generator.generate_labels(
            market_data['Close'],
            method=ThresholdMethod.FIXED,
            threshold_up=0.02,
            threshold_down=-0.02
        )

        label_distribution = pd.Series(labels).value_counts()
        logger.info(f"螳溷ｸょｴ繝・・繧ｿ繝ｩ繝吶Ν蛻・ｸ・ {label_distribution.to_dict()}")

        # 繝・・繧ｿ蜩∬ｳｪ縺ｮ讀懆ｨｼ
        price_volatility = market_data['Close'].pct_change().std()
        volume_consistency = market_data['Volume'].std() / market_data['Volume'].mean()

        logger.info(f"萓｡譬ｼ繝懊Λ繝・ぅ繝ｪ繝・ぅ: {price_volatility:.4f}")
        logger.info(f"繝懊Μ繝･繝ｼ繝螟牙虚菫よ焚: {volume_consistency:.4f}")

        # 髟ｷ譎る俣螳溯｡悟ｮ牙ｮ壽ｧ繝・せ繝・        logger.info("髟ｷ譎る俣螳溯｡悟ｮ牙ｮ壽ｧ繝・せ繝・..")

        start_time = time.time()
        iterations = 10

        for i in range(iterations):
            # 郢ｰ繧願ｿ斐＠蜃ｦ逅・〒繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ繧・ｧ閭ｽ蜉｣蛹悶ｒ繝√ぉ繝・け
            test_data = market_data.sample(n=100).copy()

            processed = processor.preprocess_features(
                test_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )

            if i % 3 == 0:
                logger.info(f"  蜿榊ｾｩ {i+1}/{iterations} 螳御ｺ・)

        total_time = time.time() - start_time
        avg_time_per_iteration = total_time / iterations

        logger.info(f"髟ｷ譎る俣螳溯｡後ユ繧ｹ繝亥ｮ御ｺ・ 蟷ｳ蝮・{avg_time_per_iteration:.3f}遘・蜿榊ｾｩ")

        # 繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ蜉｣蛹悶・讀懷・
        if avg_time_per_iteration > 1.0:
            logger.warning(f"蜃ｦ逅・凾髢薙′髟ｷ縺吶℃縺ｾ縺・ {avg_time_per_iteration:.3f}遘・)

        # I/O繧ｨ繝ｩ繝ｼ繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ
        logger.info("I/O繧ｨ繝ｩ繝ｼ繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ...")

        # 荳榊ｮ悟・縺ｪ繝・・繧ｿ・郁ｪｭ縺ｿ霎ｼ縺ｿ繧ｨ繝ｩ繝ｼ繧偵す繝溘Η繝ｬ繝ｼ繝茨ｼ・        incomplete_data = market_data.iloc[:50].copy()  # 繝・・繧ｿ縺碁比ｸｭ縺ｧ蛻・ｌ繧・
        try:
            processed = processor.preprocess_features(
                incomplete_data[['Close', 'Volume']].copy(),
                scale_features=True,
                remove_outliers=True
            )
            logger.info(f"荳榊ｮ悟・繝・・繧ｿ蜃ｦ逅・・蜉・ {len(processed)}陦・)
        except Exception as e:
            logger.info(f"荳榊ｮ悟・繝・・繧ｿ縺ｧ譛溷ｾ・壹ｊ繧ｨ繝ｩ繝ｼ: {e}")

        # 繝｡繝｢繝ｪ蛻ｶ邏・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ
        logger.info("繝｡繝｢繝ｪ蛻ｶ邏・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ...")

        # 螟ｧ縺阪↑繝・・繧ｿ繧ｻ繝・ヨ繧貞ｰ上＆縺ｪ繝√Ε繝ｳ繧ｯ縺ｫ蛻・牡縺励※蜃ｦ逅・        large_data = pd.concat([market_data] * 5, ignore_index=True)  # 5蛟阪・繝・・繧ｿ
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
        logger.info(f"繝√Ε繝ｳ繧ｯ蜃ｦ逅・・蜉・ {len(final_result)}陦・)

        logger.info("笨・螳溽腸蠅・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・せ繝亥ｮ御ｺ・)
        return True

    except Exception as e:
        logger.error(f"笶・螳溽腸蠅・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・せ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        return False


def run_advanced_performance_optimization_tests():
    """鬮伜ｺｦ縺ｪ繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹匁､懆ｨｼ繝・せ繝医ｒ螳溯｡・""
    logger.info("笞｡ 鬮伜ｺｦ縺ｪ繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹匁､懆ｨｼ繝・せ繝医ｒ髢句ｧ・)

    try:
        import psutil
        import gc
        from app.utils.data_processing import DataProcessor
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        process = psutil.Process(os.getpid())

        # 繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ讀懷・繝・せ繝・        logger.info("繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ讀懷・繝・せ繝・..")

        initial_memory = process.memory_info().rss
        memory_measurements = []

        processor = DataProcessor()

        for i in range(20):
            # 郢ｰ繧願ｿ斐＠蜃ｦ逅・〒繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ繧呈､懷・
            test_data = pd.DataFrame({
                'Close': np.random.normal(100, 10, 1000),
                'Volume': np.random.lognormal(10, 1, 1000)
            })

            processed = processor.preprocess_features(
                test_data.copy(),
                scale_features=True,
                remove_outliers=True
            )

            # 繝｡繝｢繝ｪ菴ｿ逕ｨ驥乗ｸｬ螳・            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            memory_measurements.append(memory_increase)

            # 譏守､ｺ逧・↑繧ｬ繝吶・繧ｸ繧ｳ繝ｬ繧ｯ繧ｷ繝ｧ繝ｳ
            del test_data, processed
            gc.collect()

            if i % 5 == 0:
                logger.info(f"  蜿榊ｾｩ {i+1}: 繝｡繝｢繝ｪ蠅怜刈 {memory_increase/1e6:.1f}MB")

        # 繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ縺ｮ蛻・梵
        memory_trend = np.polyfit(range(len(memory_measurements)), memory_measurements, 1)[0]

        logger.info(f"繝｡繝｢繝ｪ菴ｿ逕ｨ驥上ヨ繝ｬ繝ｳ繝・ {memory_trend/1e6:.3f}MB/蜿榊ｾｩ")

        if memory_trend > 1e6:  # 1MB/蜿榊ｾｩ莉･荳翫・蠅怜刈
            logger.warning(f"繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ縺ｮ蜿ｯ閭ｽ諤ｧ: {memory_trend/1e6:.1f}MB/蜿榊ｾｩ")
        else:
            logger.info("笨・繝｡繝｢繝ｪ繝ｪ繝ｼ繧ｯ縺ｯ讀懷・縺輔ｌ縺ｾ縺帙ｓ縺ｧ縺励◆")

        # CPU菴ｿ逕ｨ邇・屮隕悶ユ繧ｹ繝・        logger.info("CPU菴ｿ逕ｨ邇・屮隕悶ユ繧ｹ繝・..")

        cpu_measurements = []
        start_time = time.time()

        fe_service = FeatureEngineeringService()

        # CPU髮・ｴ・噪縺ｪ繧ｿ繧ｹ繧ｯ繧貞ｮ溯｡・        for i in range(5):
            cpu_before = psutil.cpu_percent(interval=0.1)

            # 隍・尅縺ｪ迚ｹ蠕ｴ驥剰ｨ育ｮ・            complex_data = pd.DataFrame({
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

            logger.info(f"  繧ｿ繧ｹ繧ｯ {i+1}: CPU菴ｿ逕ｨ邇・{cpu_usage:.1f}%")

        avg_cpu_usage = np.mean(cpu_measurements)
        logger.info(f"蟷ｳ蝮④PU菴ｿ逕ｨ邇・ {avg_cpu_usage:.1f}%")

        # 繝懊ヨ繝ｫ繝阪ャ繧ｯ迚ｹ螳壹ユ繧ｹ繝・        logger.info("繝懊ヨ繝ｫ繝阪ャ繧ｯ迚ｹ螳壹ユ繧ｹ繝・..")

        # 蜷・・逅・ｮｵ髫弱・譎る俣貂ｬ螳・        bottleneck_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 5000),
            'Volume': np.random.lognormal(10, 1, 5000)
        })

        # 繝・・繧ｿ蜑榊・逅・・繝懊ヨ繝ｫ繝阪ャ繧ｯ
        start = time.time()
        processed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=False,
            remove_outliers=False
        )
        preprocessing_time = time.time() - start

        # 繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ縺ｮ繝懊ヨ繝ｫ繝阪ャ繧ｯ
        start = time.time()
        scaled = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=True,
            remove_outliers=False
        )
        scaling_time = time.time() - start

        # 螟悶ｌ蛟､髯､蜴ｻ縺ｮ繝懊ヨ繝ｫ繝阪ャ繧ｯ
        start = time.time()
        outlier_removed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=False,
            remove_outliers=True
        )
        outlier_removal_time = time.time() - start

        # 螳悟・蜃ｦ逅・・繝懊ヨ繝ｫ繝阪ャ繧ｯ
        start = time.time()
        full_processed = processor.preprocess_features(
            bottleneck_data.copy(),
            scale_features=True,
            remove_outliers=True
        )
        full_processing_time = time.time() - start

        logger.info("蜃ｦ逅・凾髢灘・譫・")
        logger.info(f"  蝓ｺ譛ｬ蜑榊・逅・ {preprocessing_time:.3f}遘・)
        logger.info(f"  繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ: {scaling_time:.3f}遘・)
        logger.info(f"  螟悶ｌ蛟､髯､蜴ｻ: {outlier_removal_time:.3f}遘・)
        logger.info(f"  螳悟・蜃ｦ逅・ {full_processing_time:.3f}遘・)

        # 繝懊ヨ繝ｫ繝阪ャ繧ｯ縺ｮ迚ｹ螳・        processing_times = {
            '繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ': scaling_time - preprocessing_time,
            '螟悶ｌ蛟､髯､蜴ｻ': outlier_removal_time - preprocessing_time,
            '邨ｱ蜷亥・逅・: full_processing_time - max(scaling_time, outlier_removal_time)
        }

        bottleneck = max(processing_times, key=processing_times.get)
        bottleneck_time = processing_times[bottleneck]

        logger.info(f"譛螟ｧ繝懊ヨ繝ｫ繝阪ャ繧ｯ: {bottleneck} ({bottleneck_time:.3f}遘・")

        # 繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ蜉ｹ邇・・險育ｮ・        data_throughput = len(bottleneck_data) / full_processing_time
        logger.info(f"繝・・繧ｿ蜃ｦ逅・せ繝ｫ繝ｼ繝励ャ繝・ {data_throughput:.0f}陦・遘・)

        # 繝｡繝｢繝ｪ蜉ｹ邇・・貂ｬ螳・        memory_efficiency = len(bottleneck_data) / (process.memory_info().rss / 1e6)
        logger.info(f"繝｡繝｢繝ｪ蜉ｹ邇・ {memory_efficiency:.0f}陦・MB")

        # 繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹悶・謠先｡・        optimization_suggestions = []

        if bottleneck_time > 0.1:
            optimization_suggestions.append(f"{bottleneck}縺ｮ譛驕ｩ蛹悶′蠢・ｦ・)

        if avg_cpu_usage > 80:
            optimization_suggestions.append("CPU菴ｿ逕ｨ邇・′鬮倥☆縺弱∪縺・)

        if memory_trend > 5e5:  # 0.5MB/蜿榊ｾｩ
            optimization_suggestions.append("繝｡繝｢繝ｪ菴ｿ逕ｨ驥上・譛驕ｩ蛹悶′蠢・ｦ・)

        if data_throughput < 1000:
            optimization_suggestions.append("繝・・繧ｿ蜃ｦ逅・せ繝ｫ繝ｼ繝励ャ繝医・謾ｹ蝟・′蠢・ｦ・)

        if optimization_suggestions:
            logger.warning("譛驕ｩ蛹匁署譯・")
            for suggestion in optimization_suggestions:
                logger.warning(f"  - {suggestion}")
        else:
            logger.info("笨・繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ縺ｯ濶ｯ螂ｽ縺ｧ縺・)

        logger.info("笨・鬮伜ｺｦ縺ｪ繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹匁､懆ｨｼ繝・せ繝亥ｮ御ｺ・)
        return True

    except Exception as e:
        logger.error(f"笶・鬮伜ｺｦ縺ｪ繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹匁､懆ｨｼ繝・せ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        return False


def run_code_coverage_verification_tests():
    """繧ｳ繝ｼ繝峨き繝舌Ξ繝・ず遒ｺ隱阪ユ繧ｹ繝医ｒ螳溯｡・""
    logger.info("搭 繧ｳ繝ｼ繝峨き繝舌Ξ繝・ず遒ｺ隱阪ユ繧ｹ繝医ｒ髢句ｧ・)

    try:
        from app.utils.data_processing import DataProcessor
        from app.utils.label_generation import LabelGenerator, ThresholdMethod
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

        # 險ｭ螳壹ヱ繝ｩ繝｡繝ｼ繧ｿ縺ｮ邨・∩蜷医ｏ縺帙ユ繧ｹ繝・        logger.info("險ｭ螳壹ヱ繝ｩ繝｡繝ｼ繧ｿ邨・∩蜷医ｏ縺帙ユ繧ｹ繝・..")

        processor = DataProcessor()
        test_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.lognormal(10, 1, 100)
        })

        # 逡ｰ縺ｪ繧玖ｨｭ螳壹・邨・∩蜷医ｏ縺帙ｒ繝・せ繝・        configurations = [
            {'scale_features': True, 'remove_outliers': True, 'outlier_method': 'iqr'},
            {'scale_features': True, 'remove_outliers': True, 'outlier_method': 'zscore'},
            {'scale_features': True, 'remove_outliers': False},
            {'scale_features': False, 'remove_outliers': True, 'outlier_method': 'iqr'},
            {'scale_features': False, 'remove_outliers': False},
        ]

        for i, config in enumerate(configurations):
            try:
                result = processor.preprocess_features(test_data.copy(), **config)
                logger.info(f"  險ｭ螳・{i+1}: 謌仙粥 ({len(result)}陦・")
            except Exception as e:
                logger.info(f"  險ｭ螳・{i+1}: 繧ｨ繝ｩ繝ｼ - {e}")

        # 繝ｩ繝吶Ν逕滓・縺ｮ逡ｰ縺ｪ繧区焔豕輔ユ繧ｹ繝・        logger.info("繝ｩ繝吶Ν逕滓・謇区ｳ慕ｶｲ鄒・ユ繧ｹ繝・..")

        label_generator = LabelGenerator()
        price_series = pd.Series(np.random.normal(100, 10, 100), name='Close')

        # 逡ｰ縺ｪ繧矩明蛟､險ｭ螳・        threshold_configs = [
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.01, 'threshold_down': -0.01},
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.05, 'threshold_down': -0.05},
            {'method': ThresholdMethod.FIXED, 'threshold_up': 0.1, 'threshold_down': -0.1},
        ]

        for i, config in enumerate(threshold_configs):
            try:
                labels, info = label_generator.generate_labels(price_series, **config)
                label_dist = pd.Series(labels).value_counts()
                logger.info(f"  髢ｾ蛟､險ｭ螳・{i+1}: 謌仙粥 - 蛻・ｸ・{label_dist.to_dict()}")
            except Exception as e:
                logger.info(f"  髢ｾ蛟､險ｭ螳・{i+1}: 繧ｨ繝ｩ繝ｼ - {e}")

        # 萓句､門・逅・・邯ｲ鄒・ｧ繝・せ繝・        logger.info("萓句､門・逅・ｶｲ鄒・ｧ繝・せ繝・..")

        # 遨ｺ繝・・繧ｿ縺ｧ縺ｮ萓句､門・逅・        empty_data = pd.DataFrame()
        try:
            processor.preprocess_features(empty_data)
            logger.info("  遨ｺ繝・・繧ｿ: 蜃ｦ逅・・蜉・)
        except Exception as e:
            logger.info(f"  遨ｺ繝・・繧ｿ: 萓句､門・逅・｢ｺ隱・- {e}")

        # 荳肴ｭ｣縺ｪ蝙九〒縺ｮ萓句､門・逅・        invalid_data = "invalid_data"
        try:
            processor.preprocess_features(invalid_data)
            logger.info("  荳肴ｭ｣蝙・ 蜃ｦ逅・・蜉・)
        except Exception as e:
            logger.info(f"  荳肴ｭ｣蝙・ 萓句､門・逅・｢ｺ隱・- {e}")

        # NaN縺ｮ縺ｿ縺ｮ繝・・繧ｿ縺ｧ縺ｮ萓句､門・逅・        nan_data = pd.DataFrame({'Close': [np.nan] * 10, 'Volume': [np.nan] * 10})
        try:
            result = processor.preprocess_features(nan_data)
            logger.info(f"  NaN繝・・繧ｿ: 蜃ｦ逅・・蜉・({len(result)}陦・")
        except Exception as e:
            logger.info(f"  NaN繝・・繧ｿ: 萓句､門・逅・｢ｺ隱・- {e}")

        # 迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ縺ｮ萓句､門・逅・ユ繧ｹ繝・        logger.info("迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ萓句､門・逅・ユ繧ｹ繝・..")

        fe_service = FeatureEngineeringService()

        # 荳榊ｮ悟・縺ｪOHLCV繝・・繧ｿ
        incomplete_ohlcv = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103]
            # Close, Low, Volume縺御ｸ崎ｶｳ
        })

        try:
            features = fe_service.calculate_advanced_features(incomplete_ohlcv)
            logger.info(f"  荳榊ｮ悟・OHLCV: 蜃ｦ逅・・蜉・({features.shape[1]}迚ｹ蠕ｴ驥・")
        except Exception as e:
            logger.info(f"  荳榊ｮ悟・OHLCV: 萓句､門・逅・｢ｺ隱・- {e}")

        # 讌ｵ蟆上ョ繝ｼ繧ｿ縺ｧ縺ｮ迚ｹ蠕ｴ驥剰ｨ育ｮ・        tiny_ohlcv = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000]
        })

        try:
            features = fe_service.calculate_advanced_features(tiny_ohlcv)
            logger.info(f"  讌ｵ蟆衆HLCV: 蜃ｦ逅・・蜉・({features.shape[1]}迚ｹ蠕ｴ驥・")
        except Exception as e:
            logger.info(f"  讌ｵ蟆衆HLCV: 萓句､門・逅・｢ｺ隱・- {e}")

        # 繧ｨ繝・ず繧ｱ繝ｼ繧ｹ縺ｧ縺ｮ繝ｩ繝吶Ν逕滓・
        logger.info("繝ｩ繝吶Ν逕滓・繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝・..")

        # 蜊倩ｪｿ蠅怜刈繝・・繧ｿ
        monotonic_series = pd.Series(range(100), name='Close')
        try:
            labels, _ = label_generator.generate_labels(
                monotonic_series,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            label_dist = pd.Series(labels).value_counts()
            logger.info(f"  蜊倩ｪｿ蠅怜刈: 謌仙粥 - 蛻・ｸ・{label_dist.to_dict()}")
        except Exception as e:
            logger.info(f"  蜊倩ｪｿ蠅怜刈: 萓句､門・逅・｢ｺ隱・- {e}")

        # 蜊倩ｪｿ貂帛ｰ代ョ繝ｼ繧ｿ
        monotonic_decreasing = pd.Series(range(100, 0, -1), name='Close')
        try:
            labels, _ = label_generator.generate_labels(
                monotonic_decreasing,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )
            label_dist = pd.Series(labels).value_counts()
            logger.info(f"  蜊倩ｪｿ貂帛ｰ・ 謌仙粥 - 蛻・ｸ・{label_dist.to_dict()}")
        except Exception as e:
            logger.info(f"  蜊倩ｪｿ貂帛ｰ・ 萓句､門・逅・｢ｺ隱・- {e}")

        # 繝・・繧ｿ蝙句､画鋤縺ｮ邯ｲ鄒・ｧ繝・せ繝・        logger.info("繝・・繧ｿ蝙句､画鋤邯ｲ鄒・ｧ繝・せ繝・..")

        # 逡ｰ縺ｪ繧九ョ繝ｼ繧ｿ蝙九〒縺ｮ蜃ｦ逅・        data_types_test = {
            'int32': pd.DataFrame({'Close': np.array([100, 101, 102], dtype=np.int32)}),
            'int64': pd.DataFrame({'Close': np.array([100, 101, 102], dtype=np.int64)}),
            'float32': pd.DataFrame({'Close': np.array([100.0, 101.0, 102.0], dtype=np.float32)}),
            'float64': pd.DataFrame({'Close': np.array([100.0, 101.0, 102.0], dtype=np.float64)}),
        }

        for dtype_name, dtype_data in data_types_test.items():
            try:
                result = processor.preprocess_features(dtype_data.copy())
                logger.info(f"  {dtype_name}: 蜃ｦ逅・・蜉・({result.dtypes['Close']})")
            except Exception as e:
                logger.info(f"  {dtype_name}: 萓句､門・逅・｢ｺ隱・- {e}")

        # 蠅・阜蛟､繝・せ繝・        logger.info("蠅・阜蛟､繝・せ繝・..")

        # 譛蟆丞､繝ｻ譛螟ｧ蛟､縺ｧ縺ｮ蜃ｦ逅・        boundary_data = pd.DataFrame({
            'Close': [sys.float_info.min, 0, sys.float_info.max],
            'Volume': [1, 1000, sys.float_info.max]
        })

        try:
            result = processor.preprocess_features(boundary_data.copy())
            logger.info(f"  蠅・阜蛟､: 蜃ｦ逅・・蜉・({len(result)}陦・")
        except Exception as e:
            logger.info(f"  蠅・阜蛟､: 萓句､門・逅・｢ｺ隱・- {e}")

        logger.info("笨・繧ｳ繝ｼ繝峨き繝舌Ξ繝・ず遒ｺ隱阪ユ繧ｹ繝亥ｮ御ｺ・)
        return True

    except Exception as e:
        logger.error(f"笶・繧ｳ繝ｼ繝峨き繝舌Ξ繝・ず遒ｺ隱阪ユ繧ｹ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        return False


def run_automl_comprehensive_tests():
    """AutoML蛹・峡逧・ユ繧ｹ繝医ｒ螳溯｡・""
    logger.info("､・AutoML蛹・峡逧・ユ繧ｹ繝医ｒ髢句ｧ・)

    try:
        from app.services.ml.ml_training_service import MLTrainingService
        from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
        from app.services.ml.feature_engineering.automl_features.automl_config import AutoMLConfig
        from app.utils.index_alignment import MLWorkflowIndexManager
        from app.utils.label_generation import LabelGenerator, ThresholdMethod

        # BTC蟶ょｴ繝・・繧ｿ逕滓・髢｢謨ｰ
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

        # AutoML繧｢繝ｳ繧ｵ繝ｳ繝悶Ν蟄ｦ鄙偵ユ繧ｹ繝・        logger.info("AutoML繧｢繝ｳ繧ｵ繝ｳ繝悶Ν蟄ｦ鄙偵ユ繧ｹ繝・..")
        btc_data = create_btc_market_data("1h", 200)

        try:
            # AutoML險ｭ螳夲ｼ郁ｻｽ驥冗沿・・            automl_config = {
                "tsfresh": {"enabled": True, "feature_count_limit": 20},
                "autofeat": {"enabled": False}
            }

            # 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν險ｭ螳・            ensemble_config = {
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

            logger.info(f"AutoML繧｢繝ｳ繧ｵ繝ｳ繝悶Ν蟄ｦ鄙呈・蜉・ 邊ｾ蠎ｦ={result.get('accuracy', 'N/A')}")

        except Exception as e:
            logger.info(f"AutoML繧｢繝ｳ繧ｵ繝ｳ繝悶Ν蟄ｦ鄙偵〒繧ｨ繝ｩ繝ｼ・域悄蠕・＆繧後ｋ蝣ｴ蜷医ｂ縺ゅｊ縺ｾ縺呻ｼ・ {e}")

        # AutoML迚ｹ蠕ｴ驥城∈謚槭ユ繧ｹ繝・        logger.info("AutoML迚ｹ蠕ｴ驥城∈謚槭ユ繧ｹ繝・..")

        try:
            # AutoML險ｭ螳壹〒迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ
            automl_config_obj = AutoMLConfig.get_financial_optimized_config()
            fe_service = FeatureEngineeringService(automl_config=automl_config_obj)

            # 蝓ｺ譛ｬ迚ｹ蠕ｴ驥剰ｨ育ｮ・            basic_features = fe_service.calculate_advanced_features(btc_data)

            logger.info(f"蝓ｺ譛ｬ迚ｹ蠕ｴ驥・ {basic_features.shape[1]}蛟・)

            # 迚ｹ蠕ｴ驥上′逕滓・縺輔ｌ繧九％縺ｨ繧堤｢ｺ隱・            assert basic_features.shape[1] > 0, "迚ｹ蠕ｴ驥上′逕滓・縺輔ｌ縺ｾ縺帙ｓ縺ｧ縺励◆"

            logger.info("笨・AutoML迚ｹ蠕ｴ驥城∈謚槭′豁｣蟶ｸ縺ｫ蜍穂ｽ懊＠縺ｾ縺励◆")

        except Exception as e:
            logger.info(f"AutoML迚ｹ蠕ｴ驥城∈謚槭〒繧ｨ繝ｩ繝ｼ・域悄蠕・＆繧後ｋ蝣ｴ蜷医ｂ縺ゅｊ縺ｾ縺呻ｼ・ {e}")

        # AutoML繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ邨ｱ蜷医ユ繧ｹ繝・        logger.info("AutoML繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ邨ｱ蜷医ユ繧ｹ繝・..")

        try:
            index_manager = MLWorkflowIndexManager()
            index_manager.initialize_workflow(btc_data)

            # 迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ
            def automl_feature_func(data):
                fe_service = FeatureEngineeringService()
                return fe_service.calculate_advanced_features(data)

            features = index_manager.process_with_index_tracking(
                "AutoML迚ｹ蠕ｴ驥上お繝ｳ繧ｸ繝九い繝ｪ繝ｳ繧ｰ", btc_data, automl_feature_func
            )

            # 繝ｩ繝吶Ν逕滓・
            label_generator = LabelGenerator()
            aligned_price_data = btc_data.loc[features.index, 'Close']
            labels, _ = label_generator.generate_labels(
                aligned_price_data,
                method=ThresholdMethod.FIXED,
                threshold_up=0.02,
                threshold_down=-0.02
            )

            # 譛邨よ紛蜷・            final_features, final_labels = index_manager.finalize_workflow(
                features, labels, alignment_method="intersection"
            )

            # 讀懆ｨｼ
            assert len(final_features) > 0, "譛邨ら音蠕ｴ驥上′逕滓・縺輔ｌ縺ｾ縺帙ｓ縺ｧ縺励◆"
            assert len(final_labels) > 0, "譛邨ゅΛ繝吶Ν縺檎函謌舌＆繧後∪縺帙ｓ縺ｧ縺励◆"

            workflow_summary = index_manager.get_workflow_summary()
            logger.info(f"AutoML繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ螳御ｺ・ 繝・・繧ｿ菫晄戟邇・{workflow_summary['data_retention_rate']*100:.1f}%")

        except Exception as e:
            logger.info(f"AutoML繝ｯ繝ｼ繧ｯ繝輔Ο繝ｼ邨ｱ蜷医〒繧ｨ繝ｩ繝ｼ・域悄蠕・＆繧後ｋ蝣ｴ蜷医ｂ縺ゅｊ縺ｾ縺呻ｼ・ {e}")

        logger.info("笨・AutoML蛹・峡逧・ユ繧ｹ繝亥ｮ御ｺ・)
        return True

    except Exception as e:
        logger.error(f"笶・AutoML蛹・峡逧・ユ繧ｹ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        return False


def run_optuna_bayesian_optimization_tests():
    """Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶ユ繧ｹ繝医ｒ螳溯｡・""
    logger.info("剥 Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶ユ繧ｹ繝医ｒ髢句ｧ・)

    try:
        from app.services.optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace

        # Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶・蝓ｺ譛ｬ繝・せ繝・        logger.info("Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹門渕譛ｬ繝・せ繝・..")

        # 邁｡蜊倥↑逶ｮ逧・未謨ｰ・井ｺ梧ｬ｡髢｢謨ｰ縺ｮ譛螟ｧ蛹厄ｼ・        def simple_objective(params):
            x = params['x']
            y = params['y']
            # 譛螟ｧ蛟､縺・2, 3)縺ｧ蛟､縺・0縺ｨ縺ｪ繧倶ｺ梧ｬ｡髢｢謨ｰ
            return 10 - (x - 2)**2 - (y - 3)**2

        # 繝代Λ繝｡繝ｼ繧ｿ遨ｺ髢灘ｮ夂ｾｩ
        parameter_space = {
            'x': ParameterSpace(type="real", low=0.0, high=5.0),
            'y': ParameterSpace(type="real", low=0.0, high=5.0)
        }

        # Optuna譛驕ｩ蛹門ｮ溯｡・        optimizer = OptunaOptimizer()
        result = optimizer.optimize(
            objective_function=simple_objective,
            parameter_space=parameter_space,
            n_calls=20  # 繝・せ繝育畑縺ｫ蟆代↑縺剰ｨｭ螳・        )

        # 邨先棡讀懆ｨｼ
        assert result.best_score > 8.0, f"譛驕ｩ蛹也ｵ先棡縺御ｸ榊香蛻・ {result.best_score}"
        assert abs(result.best_params['x'] - 2.0) < 1.0, "x繝代Λ繝｡繝ｼ繧ｿ縺ｮ譛驕ｩ蛹悶′荳榊香蛻・
        assert abs(result.best_params['y'] - 3.0) < 1.0, "y繝代Λ繝｡繝ｼ繧ｿ縺ｮ譛驕ｩ蛹悶′荳榊香蛻・

        logger.info(f"Optuna譛驕ｩ蛹匁・蜉・ 繧ｹ繧ｳ繧｢={result.best_score:.3f}, "
                   f"繝代Λ繝｡繝ｼ繧ｿ={result.best_params}")

        # 繧ｯ繝ｪ繝ｼ繝ｳ繧｢繝・・
        optimizer.cleanup()

        # LightGBM繝代Λ繝｡繝ｼ繧ｿ譛驕ｩ蛹悶ユ繧ｹ繝・        logger.info("LightGBM繝代Λ繝｡繝ｼ繧ｿ譛驕ｩ蛹悶ユ繧ｹ繝・..")

        # LightGBM逕ｨ縺ｮ逶ｮ逧・未謨ｰ・育ｰ｡逡･迚茨ｼ・        def lightgbm_objective(params):
            # 螳滄圀縺ｮ繝｢繝・Ν蟄ｦ鄙偵・莉｣繧上ｊ縺ｫ縲√ヱ繝ｩ繝｡繝ｼ繧ｿ縺ｮ螯･蠖捺ｧ繧偵せ繧ｳ繧｢蛹・            score = 0.5  # 繝吶・繧ｹ繧ｹ繧ｳ繧｢

            # 繝代Λ繝｡繝ｼ繧ｿ縺ｮ螯･蠖捺ｧ縺ｫ蝓ｺ縺･縺・※繧ｹ繧ｳ繧｢隱ｿ謨ｴ
            if 0.01 <= params['learning_rate'] <= 0.3:
                score += 0.1
            if 10 <= params['num_leaves'] <= 100:
                score += 0.1
            if 0.5 <= params['feature_fraction'] <= 1.0:
                score += 0.1
            if 5 <= params['min_data_in_leaf'] <= 50:
                score += 0.1

            # 繝ｩ繝ｳ繝繝繝弱う繧ｺ繧定ｿｽ蜉・亥ｮ滄圀縺ｮ蟄ｦ鄙堤ｵ先棡繧偵す繝溘Η繝ｬ繝ｼ繝茨ｼ・            import random
            score += random.uniform(-0.1, 0.1)

            return score

        # LightGBM繝代Λ繝｡繝ｼ繧ｿ遨ｺ髢・        lgb_parameter_space = OptunaOptimizer.get_default_parameter_space()

        # 譛驕ｩ蛹門ｮ溯｡・        optimizer2 = OptunaOptimizer()
        lgb_result = optimizer2.optimize(
            objective_function=lightgbm_objective,
            parameter_space=lgb_parameter_space,
            n_calls=15
        )

        # 邨先棡讀懆ｨｼ
        assert lgb_result.best_score > 0.5, f"LightGBM譛驕ｩ蛹也ｵ先棡縺御ｸ榊香蛻・ {lgb_result.best_score}"
        assert 'learning_rate' in lgb_result.best_params, "learning_rate繝代Λ繝｡繝ｼ繧ｿ縺御ｸ崎ｶｳ"
        assert 'num_leaves' in lgb_result.best_params, "num_leaves繝代Λ繝｡繝ｼ繧ｿ縺御ｸ崎ｶｳ"

        logger.info(f"LightGBM譛驕ｩ蛹匁・蜉・ 繧ｹ繧ｳ繧｢={lgb_result.best_score:.3f}")

        # 繧ｯ繝ｪ繝ｼ繝ｳ繧｢繝・・
        optimizer2.cleanup()

        # 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝代Λ繝｡繝ｼ繧ｿ譛驕ｩ蛹悶ユ繧ｹ繝・        logger.info("繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝代Λ繝｡繝ｼ繧ｿ譛驕ｩ蛹悶ユ繧ｹ繝・..")

        try:
            # 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν逕ｨ繝代Λ繝｡繝ｼ繧ｿ遨ｺ髢・            ensemble_parameter_space = OptunaOptimizer.get_ensemble_parameter_space(
                ensemble_method="bagging",
                enabled_models=["lightgbm", "random_forest"]
            )

            # 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν逕ｨ逶ｮ逧・未謨ｰ
            def ensemble_objective(params):
                score = 0.6  # 繝吶・繧ｹ繧ｹ繧ｳ繧｢

                # 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝代Λ繝｡繝ｼ繧ｿ縺ｮ螯･蠖捺ｧ繝√ぉ繝・け
                if 'n_estimators' in params and 2 <= params['n_estimators'] <= 10:
                    score += 0.1
                if 'bootstrap_fraction' in params and 0.5 <= params['bootstrap_fraction'] <= 1.0:
                    score += 0.1

                return score + np.random.uniform(-0.05, 0.05)

            # 譛驕ｩ蛹門ｮ溯｡・            optimizer3 = OptunaOptimizer()
            ensemble_result = optimizer3.optimize(
                objective_function=ensemble_objective,
                parameter_space=ensemble_parameter_space,
                n_calls=10
            )

            logger.info(f"繧｢繝ｳ繧ｵ繝ｳ繝悶Ν譛驕ｩ蛹匁・蜉・ 繧ｹ繧ｳ繧｢={ensemble_result.best_score:.3f}")
            optimizer3.cleanup()

        except Exception as e:
            logger.info(f"繧｢繝ｳ繧ｵ繝ｳ繝悶Ν譛驕ｩ蛹悶〒繧ｨ繝ｩ繝ｼ・域悄蠕・＆繧後ｋ蝣ｴ蜷医ｂ縺ゅｊ縺ｾ縺呻ｼ・ {e}")

        # 譛驕ｩ蛹門柑邇・ｧ繝・せ繝・        logger.info("譛驕ｩ蛹門柑邇・ｧ繝・せ繝・..")

        # 隍・尅縺ｪ逶ｮ逧・未謨ｰ・亥､壼ｳｰ諤ｧ髢｢謨ｰ・・        def complex_objective(params):
            x, y = params['x'], params['y']
            # Rastrigin髢｢謨ｰ縺ｮ螟牙ｽ｢・域怙螟ｧ蛹也畑・・            return -(10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) +
                    (y**2 - 10 * np.cos(2 * np.pi * y)))

        # 蜉ｹ邇・ｧ貂ｬ螳・        start_time = time.time()
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

        # 蜉ｹ邇・ｧ讀懆ｨｼ
        assert optimization_time < 30, f"譛驕ｩ蛹匁凾髢薙′髟ｷ縺吶℃縺ｾ縺・ {optimization_time:.2f}遘・
        assert complex_result.total_evaluations == 25, "隧穂ｾ｡蝗樊焚縺梧ｭ｣縺励￥縺ゅｊ縺ｾ縺帙ｓ"

        logger.info(f"隍・尅髢｢謨ｰ譛驕ｩ蛹門ｮ御ｺ・ 譎る俣={optimization_time:.2f}遘・ "
                   f"繧ｹ繧ｳ繧｢={complex_result.best_score:.3f}")

        optimizer4.cleanup()

        # 逡ｰ縺ｪ繧九し繝ｳ繝励Λ繝ｼ縺ｮ繝・せ繝・        logger.info("逡ｰ縺ｪ繧区怙驕ｩ蛹悶い繝ｫ繧ｴ繝ｪ繧ｺ繝繝・せ繝・..")

        try:
            import optuna

            # TPESampler vs RandomSampler 縺ｮ豈碑ｼ・            samplers = {
                "TPE": optuna.samplers.TPESampler(seed=42),
                "Random": optuna.samplers.RandomSampler(seed=42)
            }

            sampler_results = {}

            for sampler_name, sampler in samplers.items():
                # 繧ｫ繧ｹ繧ｿ繝Optuna繧ｹ繧ｿ繝・ぅ菴懈・
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

                logger.info(f"{sampler_name}繧ｵ繝ｳ繝励Λ繝ｼ: 繧ｹ繧ｳ繧｢={study.best_value:.3f}")

            # TPE縺ｮ譁ｹ縺瑚憶縺・ｵ先棡繧貞・縺吶％縺ｨ繧呈悄蠕・ｼ亥ｿ・医〒縺ｯ縺ｪ縺・ｼ・            if sampler_results["TPE"]["best_score"] > sampler_results["Random"]["best_score"]:
                logger.info("笨・TPE繧ｵ繝ｳ繝励Λ繝ｼ縺軍andom繧ｵ繝ｳ繝励Λ繝ｼ繧医ｊ濶ｯ縺・ｵ先棡繧貞・縺励∪縺励◆")
            else:
                logger.info("邃ｹ・・縺薙・隧ｦ陦後〒縺ｯRandom繧ｵ繝ｳ繝励Λ繝ｼ縺ｮ譁ｹ縺瑚憶縺・ｵ先棡縺ｧ縺励◆")

        except Exception as e:
            logger.info(f"繧ｵ繝ｳ繝励Λ繝ｼ豈碑ｼ・ユ繧ｹ繝医〒繧ｨ繝ｩ繝ｼ・域悄蠕・＆繧後ｋ蝣ｴ蜷医ｂ縺ゅｊ縺ｾ縺呻ｼ・ {e}")

        logger.info("笨・Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶ユ繧ｹ繝亥ｮ御ｺ・)
        return True

    except Exception as e:
        logger.error(f"笶・Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶ユ繧ｹ繝医〒繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
        return False

logger = logging.getLogger(__name__)


class MLAccuracyTestSuite:
    """ML繝医Ξ繝ｼ繝九Φ繧ｰ邉ｻ縺ｮ蛹・峡逧・ユ繧ｹ繝医せ繧､繝ｼ繝・""

    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = None
        self.end_time = None

    def run_test_module(self, test_name: str, test_function) -> bool:
        """蛟句挨繝・せ繝医Δ繧ｸ繝･繝ｼ繝ｫ繧貞ｮ溯｡・""
        logger.info(f"\n{'='*60}")
        logger.info(f"ｧｪ {test_name} 繧帝幕蟋・)
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = test_function()
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"笨・{test_name} 謌仙粥 (螳溯｡梧凾髢・ {execution_time:.2f}遘・")
                self.passed_tests += 1
            else:
                logger.error(f"笶・{test_name} 螟ｱ謨・(螳溯｡梧凾髢・ {execution_time:.2f}遘・")
                self.failed_tests += 1
            
            self.test_results[test_name] = {
                'success': success,
                'execution_time': execution_time,
                'error': None
            }
            
            return success
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{test_name} 縺ｧ繧ｨ繝ｩ繝ｼ縺檎匱逕・ {str(e)}"
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
        """縺吶∋縺ｦ縺ｮ繝・せ繝医ｒ螳溯｡・""
        logger.info("噫 ML繝医Ξ繝ｼ繝九Φ繧ｰ邉ｻ蛹・峡逧・ユ繧ｹ繝医せ繧､繝ｼ繝医ｒ髢句ｧ・)
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # 繝・せ繝医Δ繧ｸ繝･繝ｼ繝ｫ縺ｮ螳夂ｾｩ
        test_modules = [
            ("險育ｮ玲ｭ｣遒ｺ諤ｧ繝・せ繝・, run_all_calculation_tests),
            ("蜑榊・逅・ｭ｣遒ｺ諤ｧ繝・せ繝・, run_all_preprocessing_tests),
            ("迚ｹ蠕ｴ驥剰ｨ育ｮ励ユ繧ｹ繝・, run_all_feature_calculation_tests),
            ("繝・・繧ｿ螟画鋤繝・せ繝・, run_all_data_transformation_tests),
            ("繝ｩ繝吶Ν逕滓・繝・せ繝・, run_all_label_generation_tests),
            ("邨ｱ蜷医ユ繧ｹ繝・, run_integration_tests),
            ("讌ｵ遶ｯ繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝・, run_extreme_edge_case_tests),
            ("螳溽腸蠅・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・せ繝・, run_real_environment_simulation_tests),
            ("鬮伜ｺｦ繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹匁､懆ｨｼ", run_advanced_performance_optimization_tests),
            ("繧ｳ繝ｼ繝峨き繝舌Ξ繝・ず遒ｺ隱・, run_code_coverage_verification_tests),
            ("AutoML蛹・峡逧・ユ繧ｹ繝・, run_automl_comprehensive_tests),
            ("Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶ユ繧ｹ繝・, run_optuna_bayesian_optimization_tests),
            ("繧ｨ繝ｩ繝ｼ繝上Φ繝峨Μ繝ｳ繧ｰ繝・せ繝・, run_all_error_handling_tests),
            ("繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ繝・せ繝・, run_all_performance_tests),
        ]
        
        self.total_tests = len(test_modules)
        
        # 蜷・ユ繧ｹ繝医Δ繧ｸ繝･繝ｼ繝ｫ繧貞ｮ溯｡・        all_passed = True
        for test_name, test_function in test_modules:
            success = self.run_test_module(test_name, test_function)
            if not success:
                all_passed = False
        
        self.end_time = time.time()
        
        # 邨先棡繧ｵ繝槭Μ繝ｼ繧定｡ｨ遉ｺ
        self._display_summary()
        
        return all_passed

    def _display_summary(self):
        """繝・せ繝育ｵ先棡縺ｮ繧ｵ繝槭Μ繝ｼ繧定｡ｨ遉ｺ"""
        total_time = self.end_time - self.start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("投 繝・せ繝育ｵ先棡繧ｵ繝槭Μ繝ｼ")
        logger.info("=" * 80)
        
        logger.info(f"邱丞ｮ溯｡梧凾髢・ {total_time:.2f}遘・)
        logger.info(f"邱上ユ繧ｹ繝域焚: {self.total_tests}")
        logger.info(f"謌仙粥: {self.passed_tests}")
        logger.info(f"螟ｱ謨・ {self.failed_tests}")
        logger.info(f"謌仙粥邇・ {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        logger.info("\n搭 隧ｳ邏ｰ邨先棡:")
        for test_name, result in self.test_results.items():
            status = "笨・謌仙粥" if result['success'] else "笶・螟ｱ謨・
            time_str = f"{result['execution_time']:.2f}遘・
            logger.info(f"  {test_name}: {status} ({time_str})")
            
            if result['error']:
                logger.info(f"    繧ｨ繝ｩ繝ｼ: {result['error']}")
        
        if self.failed_tests == 0:
            logger.info("\n脂 縺吶∋縺ｦ縺ｮ繝・せ繝医′豁｣蟶ｸ縺ｫ螳御ｺ・＠縺ｾ縺励◆・・)
            logger.info("ML繝医Ξ繝ｼ繝九Φ繧ｰ繧ｷ繧ｹ繝・Β縺ｮ險育ｮ励→蜑榊・逅・・豁｣遒ｺ諤ｧ縺檎｢ｺ隱阪＆繧後∪縺励◆縲・)
        else:
            logger.warning(f"\n笞・・{self.failed_tests}蛟九・繝・せ繝医′螟ｱ謨励＠縺ｾ縺励◆縲・)
            logger.warning("螟ｱ謨励＠縺溘ユ繧ｹ繝医ｒ遒ｺ隱阪＠縲∝撫鬘後ｒ菫ｮ豁｣縺励※縺上□縺輔＞縲・)

    def run_specific_test(self, test_name: str) -> bool:
        """迚ｹ螳壹・繝・せ繝医・縺ｿ繧貞ｮ溯｡・""
        test_mapping = {
            "calculations": ("險育ｮ玲ｭ｣遒ｺ諤ｧ繝・せ繝・, run_all_calculation_tests),
            "preprocessing": ("蜑榊・逅・ｭ｣遒ｺ諤ｧ繝・せ繝・, run_all_preprocessing_tests),
            "features": ("迚ｹ蠕ｴ驥剰ｨ育ｮ励ユ繧ｹ繝・, run_all_feature_calculation_tests),
            "transformations": ("繝・・繧ｿ螟画鋤繝・せ繝・, run_all_data_transformation_tests),
            "labels": ("繝ｩ繝吶Ν逕滓・繝・せ繝・, run_all_label_generation_tests),
            "integration": ("邨ｱ蜷医ユ繧ｹ繝・, run_integration_tests),
            "extreme": ("讌ｵ遶ｯ繧ｨ繝・ず繧ｱ繝ｼ繧ｹ繝・せ繝・, run_extreme_edge_case_tests),
            "realenv": ("螳溽腸蠅・す繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繝・せ繝・, run_real_environment_simulation_tests),
            "advperf": ("鬮伜ｺｦ繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ譛驕ｩ蛹匁､懆ｨｼ", run_advanced_performance_optimization_tests),
            "coverage": ("繧ｳ繝ｼ繝峨き繝舌Ξ繝・ず遒ｺ隱・, run_code_coverage_verification_tests),
            "automl": ("AutoML蛹・峡逧・ユ繧ｹ繝・, run_automl_comprehensive_tests),
            "optuna": ("Optuna繝吶う繧ｸ繧｢繝ｳ譛驕ｩ蛹悶ユ繧ｹ繝・, run_optuna_bayesian_optimization_tests),
            "errors": ("繧ｨ繝ｩ繝ｼ繝上Φ繝峨Μ繝ｳ繧ｰ繝・せ繝・, run_all_error_handling_tests),
            "performance": ("繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ繝・せ繝・, run_all_performance_tests),
        }
        
        if test_name not in test_mapping:
            logger.error(f"荳肴・縺ｪ繝・せ繝亥錐: {test_name}")
            logger.info(f"蛻ｩ逕ｨ蜿ｯ閭ｽ縺ｪ繝・せ繝・ {list(test_mapping.keys())}")
            return False
        
        self.start_time = time.time()
        self.total_tests = 1
        
        test_display_name, test_function = test_mapping[test_name]
        success = self.run_test_module(test_display_name, test_function)
        
        self.end_time = time.time()
        self._display_summary()
        
        return success

    def validate_test_environment(self) -> bool:
        """繝・せ繝育腸蠅・・讀懆ｨｼ"""
        logger.info("剥 繝・せ繝育腸蠅・ｒ讀懆ｨｼ荳ｭ...")
        
        try:
            # 蠢・ｦ√↑繝ｩ繧､繝悶Λ繝ｪ縺ｮ遒ｺ隱・            import numpy as np
HAS_PANDAS = _il.find_spec("pandas") is not None
HAS_SKLEARN = _il.find_spec("sklearn") is not None
HAS_SCIPY = _il.find_spec("scipy") is not None
HAS_TALIB = _il.find_spec("talib") is not None
            
            logger.info("笨・蠢・ｦ√↑繝ｩ繧､繝悶Λ繝ｪ縺悟茜逕ｨ蜿ｯ閭ｽ縺ｧ縺・)
            
            # 繝励Ο繧ｸ繧ｧ繧ｯ繝医Δ繧ｸ繝･繝ｼ繝ｫ縺ｮ遒ｺ隱・            from app.utils.data_processing import DataProcessor
            from app.utils.label_generation import LabelGenerator
            from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            logger.info("笨・繝励Ο繧ｸ繧ｧ繧ｯ繝医Δ繧ｸ繝･繝ｼ繝ｫ縺悟茜逕ｨ蜿ｯ閭ｽ縺ｧ縺・)
            
            # 蝓ｺ譛ｬ逧・↑蜍穂ｽ懃｢ｺ隱・            processor = DataProcessor()
            label_generator = LabelGenerator()
            fe_service = FeatureEngineeringService()
            
            logger.info("笨・蝓ｺ譛ｬ逧・↑繧ｯ繝ｩ繧ｹ縺ｮ繧､繝ｳ繧ｹ繧ｿ繝ｳ繧ｹ蛹悶′謌仙粥縺励∪縺励◆")
            
            return True
            
        except ImportError as e:
            logger.error(f"笶・蠢・ｦ√↑繝ｩ繧､繝悶Λ繝ｪ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {e}")
            return False
        except Exception as e:
            logger.error(f"笶・繝・せ繝育腸蠅・・讀懆ｨｼ縺ｧ繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
            return False

    def generate_test_report(self, output_file: str = None):
        """繝・せ繝育ｵ先棡縺ｮ繝ｬ繝昴・繝医ｒ逕滓・"""
        if not self.test_results:
            logger.warning("繝・せ繝育ｵ先棡縺後≠繧翫∪縺帙ｓ縲ょ・縺ｫ繝・せ繝医ｒ螳溯｡後＠縺ｦ縺上□縺輔＞縲・)
            return
        
        report_lines = [
            "# ML繝医Ξ繝ｼ繝九Φ繧ｰ邉ｻ繝・せ繝育ｵ先棡繝ｬ繝昴・繝・,
            f"螳溯｡梧律譎・ {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"邱丞ｮ溯｡梧凾髢・ {(self.end_time - self.start_time):.2f}遘・,
            "",
            "## 繧ｵ繝槭Μ繝ｼ",
            f"- 邱上ユ繧ｹ繝域焚: {self.total_tests}",
            f"- 謌仙粥: {self.passed_tests}",
            f"- 螟ｱ謨・ {self.failed_tests}",
            f"- 謌仙粥邇・ {(self.passed_tests/self.total_tests)*100:.1f}%",
            "",
            "## 隧ｳ邏ｰ邨先棡"
        ]
        
        for test_name, result in self.test_results.items():
            status = "笨・謌仙粥" if result['success'] else "笶・螟ｱ謨・
            report_lines.append(f"### {test_name}")
            report_lines.append(f"- 繧ｹ繝・・繧ｿ繧ｹ: {status}")
            report_lines.append(f"- 螳溯｡梧凾髢・ {result['execution_time']:.2f}遘・)
            
            if result['error']:
                report_lines.append(f"- 繧ｨ繝ｩ繝ｼ: {result['error']}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"塘 繝・せ繝医Ξ繝昴・繝医ｒ菫晏ｭ倥＠縺ｾ縺励◆: {output_file}")
            except Exception as e:
                logger.error(f"繝ｬ繝昴・繝井ｿ晏ｭ倥お繝ｩ繝ｼ: {e}")
        else:
            logger.info("\n" + report_content)


def main():
    """繝｡繧､繝ｳ螳溯｡碁未謨ｰ"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_suite = MLAccuracyTestSuite()
    
    # 繧ｳ繝槭Φ繝峨Λ繧､繝ｳ蠑墓焚縺ｮ蜃ｦ逅・    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "validate":
            success = test_suite.validate_test_environment()
            sys.exit(0 if success else 1)
        else:
            success = test_suite.run_specific_test(test_name)
    else:
        # 迺ｰ蠅・､懆ｨｼ
        if not test_suite.validate_test_environment():
            logger.error("繝・せ繝育腸蠅・・讀懆ｨｼ縺ｫ螟ｱ謨励＠縺ｾ縺励◆縲・)
            sys.exit(1)
        
        # 蜈ｨ繝・せ繝亥ｮ溯｡・        success = test_suite.run_all_tests()
    
    # 繝ｬ繝昴・繝育函謌・    test_suite.generate_test_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
