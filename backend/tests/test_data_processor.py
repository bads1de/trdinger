"""
データプロセッサーの統合インターフェースに対するテスト
TDD原則に基づき、統合インターフェースの各メソッドをテスト
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock


class TestDataProcessor:
    """DataProcessor統合インターフェースのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        np.random.seed(42)

        # OHLCの関係を満たすようにデータを生成
        base_prices = np.random.uniform(100, 110, 100)
        volatility = np.random.uniform(0.01, 0.05, 100)  # 1% to 5% volatility

        opens = base_prices
        highs = base_prices * (1 + volatility)
        lows = base_prices * (1 - volatility)
        closes = base_prices + np.random.uniform(-volatility, volatility, 100) * base_prices

        # 確実にlow <= open/close <= highを満たす
        lows = np.minimum(lows, np.minimum(opens, closes))
        highs = np.maximum(highs, np.maximum(opens, closes))

        data = {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, 100),
            'fear_greed_value': np.random.uniform(0, 100, 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def data_processor(self):
        """DataProcessorインスタンス"""
        # 統合インターフェースが実装されるまでMockを使用
        from backend.app.utils.data_processing import data_processor
        return data_processor

    def test_clean_and_validate_data_basic(self, data_processor, sample_data):
        """clean_and_validate_dataの基本機能テスト"""
        required_columns = ['open', 'high', 'low', 'close']

        result = data_processor.clean_and_validate_data(
            sample_data, required_columns
        )

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in required_columns)

    def test_prepare_training_data_basic(self, data_processor, sample_data):
        """prepare_training_dataの基本機能テスト"""
        from unittest.mock import Mock

        # Mockラベルジェネレータ
        label_generator = Mock()
        # インデックスをsample_dataと同じものにする
        mock_labels = pd.Series([0, 1, 0] * (len(sample_data) // 3 + 1))[:len(sample_data)]
        mock_labels.index = sample_data.index
        label_generator.generate_labels.return_value = (
            mock_labels,
            {"threshold": 0.02}
        )

        features, labels, threshold_info = data_processor.prepare_training_data(
            sample_data, label_generator
        )

        # 結果の検証
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert isinstance(threshold_info, dict)
        assert len(features) == len(labels)

    def test_preprocess_with_pipeline_basic(self, data_processor, sample_data):
        """preprocess_with_pipelineの基本機能テスト"""
        result = data_processor.preprocess_with_pipeline(sample_data)

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_process_data_efficiently_basic(self, data_processor, sample_data):
        """process_data_efficientlyの基本機能テスト"""
        result = data_processor.process_data_efficiently(sample_data)

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_create_optimized_pipeline_basic(self, data_processor):
        """create_optimized_pipelineの基本機能テスト"""
        pipeline = data_processor.create_optimized_pipeline()

        # 結果の検証
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')

    def test_get_pipeline_info_basic(self, data_processor):
        """get_pipeline_infoの基本機能テスト"""
        # 存在しないパイプラインの場合
        info = data_processor.get_pipeline_info("nonexistent")
        assert info["exists"] is False

        # パイプラインを作成してからテスト
        pipeline = data_processor.create_optimized_pipeline()
        data_processor.fitted_pipelines["test_pipeline"] = pipeline

        info = data_processor.get_pipeline_info("test_pipeline")
        assert info["exists"] is True
        assert "n_steps" in info

    def test_clear_cache_basic(self, data_processor):
        """clear_cacheの基本機能テスト"""
        # キャッシュにデータを追加
        data_processor.fitted_pipelines["test"] = "dummy"

        # キャッシュクリア
        data_processor.clear_cache()

        # キャッシュが空であることを確認
        assert len(data_processor.fitted_pipelines) == 0
        assert len(data_processor.imputation_stats) == 0

    def test_clean_and_validate_data_missing_required_columns(self, data_processor):
        """必須カラムが欠けている場合のテスト"""
        dates = pd.date_range('2023-01-01', periods=10, freq='h')
        data = {
            'timestamp': dates,
            'some_other_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        df = pd.DataFrame(data)

        # 必須カラムが欠けている場合、ValueErrorが発生することを期待
        with pytest.raises(ValueError, match="Missing required columns"):
            data_processor.clean_and_validate_data(
                df,
                required_columns=['open', 'high', 'low', 'close', 'volume']
            )


class TestDataProcessorIntegration:
    """統合テスト"""

    def test_backward_compatibility(self):
        """後方互換性のテスト"""
        # 既存のグローバルインスタンスが利用可能であることを確認
        from backend.app.utils.data_processing import data_processor

        assert hasattr(data_processor, 'clean_and_validate_data')
        assert callable(data_processor.clean_and_validate_data)

    def test_module_imports(self):
        """モジュールインポートテスト"""
        # transformers
        from backend.app.utils.data_processing.transformers.outlier_remover import OutlierRemover
        from backend.app.utils.data_processing.transformers.categorical_encoder import CategoricalEncoder
        from backend.app.utils.data_processing.transformers.data_imputer import DataImputer
        from backend.app.utils.data_processing.transformers.dtype_optimizer import DtypeOptimizer

        # pipelines (関数ベース)
        from backend.app.utils.data_processing.pipelines.preprocessing_pipeline import create_preprocessing_pipeline
        from backend.app.utils.data_processing.pipelines.ml_pipeline import create_ml_pipeline
        from backend.app.utils.data_processing.pipelines.comprehensive_pipeline import create_comprehensive_pipeline

        # validators (関数ベース)
        from backend.app.utils.data_processing.validators.data_validator import validate_ohlcv_data

        # data_processor
        from backend.app.utils.data_processing import data_processor as dp

        # 全てのモジュールがインポート可能であることを確認
        assert True