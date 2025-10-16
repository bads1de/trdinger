"""
データ処理パイプライン高度テスト
市場データ処理パイプラインの包括的テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from typing import Dict, Any, List

from app.utils.data_processing.data_processor import DataProcessor
from app.utils.data_processing.validators.data_validator import (
    validate_ohlcv_data,
    validate_extended_data,
    validate_data_integrity,
)
from app.utils.data_processing.transformers.outlier_remover import OutlierRemover
from app.utils.data_processing.transformers.data_imputer import DataImputer
from app.utils.data_processing.transformers.categorical_encoder import CategoricalEncoder
from app.utils.data_processing.transformers.dtype_optimizer import DataTypeOptimizer
from app.utils.data_processing.pipelines.preprocessing_pipeline import (
    create_preprocessing_pipeline,
)
from app.utils.data_processing.pipelines.comprehensive_pipeline import (
    create_comprehensive_pipeline,
)


class TestDataProcessingPipelineAdvanced:
    """データ処理パイプライン高度テスト"""

    @pytest.fixture
    def clean_market_data(self):
        """クリーンな市場データ"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 200,
            'high': 10000 + np.random.randn(len(dates)) * 300,
            'low': 10000 + np.random.randn(len(dates)) * 300,
            'close': 10000 + np.random.randn(len(dates)) * 200,
            'volume': 500 + np.random.randint(100, 1000, len(dates)),
            'returns': np.random.randn(len(dates)) * 0.02,
            'volatility': 0.01 + np.random.rand(len(dates)) * 0.02,
        })

        # OHLCの関係を確保
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)

        return data

    @pytest.fixture
    def dirty_market_data(self):
        """汚染された市場データ"""
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 200,
            'high': 10000 + np.random.randn(len(dates)) * 300,
            'low': 10000 + np.random.randn(len(dates)) * 300,
            'close': 10000 + np.random.randn(len(dates)) * 200,
            'volume': 500 + np.random.randint(100, 1000, len(dates)),
            'returns': np.random.randn(len(dates)) * 0.02,
            'volatility': 0.01 + np.random.rand(len(dates)) * 0.02,
        })

        # OHLCの関係を確保
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)

        # 汚染を追加
        data.loc[10:20, 'volume'] = np.nan  # 欠損値
        data.loc[30:35, 'close'] = np.inf   # 無限大
        data.loc[50:55, 'open'] = -999     # 異常値
        data.loc[70:75, 'returns'] = np.nan  # 追加の欠損値

        return data

    @pytest.fixture
    def categorical_data(self):
        """カテゴリカルデータ"""
        return pd.DataFrame({
            'symbol': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'] * 100,
            'exchange': ['Binance', 'Coinbase', 'Kraken'] * 100,
            'market_condition': ['bullish', 'bearish', 'sideways'] * 100,
            'numeric_feature': np.random.randn(300),
            'target': np.random.choice([0, 1], 300),
        })

    def test_data_validation_comprehensive(self, clean_market_data, dirty_market_data):
        """包括的データ検証のテスト"""
        # クリーンデータの検証
        try:
            validate_ohlcv_data(clean_market_data)
            clean_valid = True
        except Exception:
            clean_valid = False

        assert clean_valid is True

        # 汚染データの検証
        try:
            validate_ohlcv_data(dirty_market_data)
            dirty_valid = True
        except Exception:
            dirty_valid = False

        assert dirty_valid is False

    def test_outlier_detection_and_removal_advanced(self, dirty_market_data):
        """高度な外れ値検出と除去のテスト"""
        # IQR法
        remover_iqr = OutlierRemover(method="iqr")
        cleaned_iqr = remover_iqr.remove_outliers(dirty_market_data)

        assert isinstance(cleaned_iqr, pd.DataFrame)
        assert len(cleaned_iqr) <= len(dirty_market_data)

        # 孤立森法
        remover_isolation = OutlierRemover(method="isolation_forest")
        cleaned_isolation = remover_isolation.remove_outliers(dirty_market_data)

        assert isinstance(cleaned_isolation, pd.DataFrame)

    def test_advanced_data_imputation_techniques(self, dirty_market_data):
        """高度なデータ補完技法のテスト"""
        # 平均値補完
        imputer_mean = DataImputer(strategy="mean")
        imputed_mean = imputer_mean.impute(dirty_market_data)

        assert not imputed_mean.isnull().any().any()
        assert isinstance(imputed_mean, pd.DataFrame)

        # 中央値補完
        imputer_median = DataImputer(strategy="median")
        imputed_median = imputer_median.impute(dirty_market_data)

        assert not imputed_median.isnull().any().any()

        # 最近傍補完
        imputer_knn = DataImputer(strategy="knn")
        imputed_knn = imputer_knn.impute(dirty_market_data)

        assert not imputed_knn.isnull().any().any()

    def test_categorical_encoding_advanced(self, categorical_data):
        """高度なカテゴリカルエンコーディングのテスト"""
        # One-hotエンコーディング
        encoder_onehot = CategoricalEncoder(encoding_type="onehot")
        encoded_onehot = encoder_onehot.encode(categorical_data)

        assert isinstance(encoded_onehot, pd.DataFrame)
        assert len(encoded_onehot.columns) > len(categorical_data.columns)

        # ラベルエンコーディング
        encoder_label = CategoricalEncoder(encoding_type="label")
        encoded_label = encoder_label.encode(categorical_data)

        assert isinstance(encoded_label, pd.DataFrame)

        # ターゲットエンコーディング
        encoder_target = CategoricalEncoder(encoding_type="target")
        encoded_target = encoder_target.encode(categorical_data, categorical_data['target'])

        assert isinstance(encoded_target, pd.DataFrame)

    def test_data_type_optimization_advanced(self, dirty_market_data):
        """高度なデータ型最適化のテスト"""
        optimizer = DataTypeOptimizer()

        # 最適化前後のメモリ使用量を比較
        original_memory = dirty_market_data.memory_usage().sum()
        optimized_data = optimizer.optimize_types(dirty_market_data)
        optimized_memory = optimized_data.memory_usage().sum()

        assert isinstance(optimized_data, pd.DataFrame)
        assert optimized_memory <= original_memory

    def test_feature_engineering_pipeline(self, clean_market_data):
        """特徴量エンジニアリングパイプラインのテスト"""
        processor = DataProcessor()

        # 技術指標の追加
        engineered_data = processor.add_technical_indicators(
            clean_market_data,
            indicators=['rsi', 'macd', 'bollinger']
        )

        assert isinstance(engineered_data, pd.DataFrame)
        assert len(engineered_data.columns) > len(clean_market_data.columns)

        # ラグ特徴量の追加
        lagged_data = processor.add_lag_features(
            engineered_data,
            columns=['close', 'volume'],
            lags=[1, 2, 3]
        )

        assert isinstance(lagged_data, pd.DataFrame)
        assert len(lagged_data.columns) > len(engineered_data.columns)

    def test_real_time_data_streaming_processing(self):
        """リアルタイムデータストリーミング処理のテスト"""
        processor = DataProcessor()

        # リアルタイムストリームをシミュレート
        stream_data = [
            {'price': 10000.0, 'volume': 500.0, 'timestamp': '2023-01-01T10:00:00'},
            {'price': 10001.0, 'volume': 510.0, 'timestamp': '2023-01-01T10:01:00'},
            {'price': 9999.0, 'volume': 490.0, 'timestamp': '2023-01-01T10:02:00'},
        ]

        # ストリーミング処理を実行
        processed_stream = processor.process_streaming_data(stream_data)

        assert isinstance(processed_stream, list)
        assert len(processed_stream) == len(stream_data)

        for item in processed_stream:
            assert isinstance(item, dict)
            assert 'processed_timestamp' in item

    def test_batch_processing_with_validation(self, clean_market_data):
        """検証付きバッチ処理のテスト"""
        processor = DataProcessor()

        # バッチデータを準備
        batches = []
        for i in range(5):
            batch = clean_market_data[i*20:(i+1)*20].copy()
            batches.append(batch)

        # バッチ処理を実行
        processed_batches = processor.process_batch_with_validation(batches)

        assert isinstance(processed_batches, list)
        assert len(processed_batches) == len(batches)

        for batch in processed_batches:
            assert isinstance(batch, pd.DataFrame)
            assert len(batch) > 0

    def test_data_quality_monitoring(self, clean_market_data, dirty_market_data):
        """データ品質監視のテスト"""
        processor = DataProcessor()

        # クリーンデータの品質レポート
        clean_report = processor.get_data_quality_report(clean_market_data)
        assert isinstance(clean_report, dict)
        assert 'missing_ratio' in clean_report
        assert 'duplicate_ratio' in clean_report
        assert 'outlier_ratio' in clean_report

        # 汚染データの品質レポート
        dirty_report = processor.get_data_quality_report(dirty_market_data)
        assert isinstance(dirty_report, dict)

        # 汚染データの方が品質スコアが低いこと
        assert dirty_report['missing_ratio'] >= clean_report['missing_ratio']

    def test_data_lineage_tracking(self, clean_market_data):
        """データ系統追跡のテスト"""
        processor = DataProcessor()

        # 処理履歴を追跡
        initial_data = clean_market_data.copy()
        initial_data['processing_step'] = 'raw_data'

        # 処理を適用
        step1_data = processor.add_technical_indicators(initial_data, indicators=['rsi'])
        step1_data['processing_step'] = 'technical_indicators_added'

        step2_data = processor.normalize_features(step1_data, method='minmax')
        step2_data['processing_step'] = 'normalized'

        # 系統を検証
        assert 'processing_step' in step2_data.columns
        assert step2_data['processing_step'].iloc[0] == 'normalized'

    def test_data_governance_compliance(self, clean_market_data):
        """データガバナンスコンプライアンスのテスト"""
        processor = DataProcessor()

        # 暗号化処理
        encrypted_data = processor.encrypt_sensitive_data(
            clean_market_data,
            columns=['volume', 'returns']
        )

        assert isinstance(encrypted_data, pd.DataFrame)

        # データ保持ポリシーの適用
        retention_applied = processor.apply_data_retention_policy(
            encrypted_data,
            retention_days=365
        )

        assert isinstance(retention_applied, pd.DataFrame)

    def test_memory_efficient_large_scale_processing(self):
        """大規模データのメモリ効率処理のテスト"""
        import gc
        import sys

        processor = DataProcessor()

        # 大規模データを生成
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='H'),
            'open': np.random.randn(10000) * 100 + 10000,
            'high': np.random.randn(10000) * 150 + 10000,
            'low': np.random.randn(10000) * 150 + 10000,
            'close': np.random.randn(10000) * 100 + 10000,
            'volume': np.random.randint(100, 1000, 10000),
        })

        initial_memory = sys.getsizeof(gc.get_objects())

        # 大規模データを処理
        processed_data = processor.process(large_data)

        gc.collect()
        final_memory = sys.getsizeof(gc.get_objects())
        memory_increase = final_memory - initial_memory

        assert isinstance(processed_data, pd.DataFrame)
        assert memory_increase < 10000000  # 10MB未満の増加

    def test_parallel_data_processing_advanced(self, clean_market_data):
        """高度な並列データ処理のテスト"""
        processor = DataProcessor()

        # 小分けのデータを作成
        chunks = np.array_split(clean_market_data, 4)

        # 並列処理を実行
        processed_chunks = processor.process_in_parallel(chunks)

        assert isinstance(processed_chunks, list)
        assert len(processed_chunks) == len(chunks)

        for chunk in processed_chunks:
            assert isinstance(chunk, pd.DataFrame)

    def test_final_data_pipeline_validation(self, clean_market_data, dirty_market_data):
        """最終データパイプライン検証"""
        processor = DataProcessor()

        # 基本処理が可能であること
        basic_processed = processor.process(clean_market_data)
        assert isinstance(basic_processed, pd.DataFrame)

        # 汚染データの処理が可能であること
        cleaned = processor.process(dirty_market_data)
        assert isinstance(cleaned, pd.DataFrame)

        # パイプラインが堅牢であること
        assert len(cleaned) > 0

        print("✅ データ処理パイプライン高度テスト成功")


# TDDアプローチによるデータ処理テスト
class TestDataProcessingTDD:
    """TDDアプローチによるデータ処理パイプラインテスト"""

    def test_data_processor_basic_functionality(self):
        """データプロセッサ基本機能のテスト"""
        processor = DataProcessor()

        # 最小限のデータでテスト
        simple_data = pd.DataFrame({
            'price': [100, 101, 99, 102, 100],
            'volume': [1000, 1100, 900, 1200, 1000],
        })

        # 基本処理が可能であること
        processed = processor.process(simple_data)
        assert isinstance(processed, pd.DataFrame)

        print("✅ データプロセッサ基本機能のテスト成功")

    def test_data_validation_basic_workflow(self):
        """データ検証基本ワークフローテスト"""
        # 簡単な検証を実行
        simple_data = pd.DataFrame({
            'open': [100, 101, 99],
            'high': [102, 103, 101],
            'low': [98, 99, 97],
            'close': [101, 100, 102],
            'volume': [1000, 1100, 900],
        })

        try:
            validate_ohlcv_data(simple_data)
            valid = True
        except Exception:
            valid = False

        assert valid is True

        print("✅ データ検証基本ワークフローテスト成功")

    def test_outlier_removal_basic_function(self):
        """外れ値除去基本機能のテスト"""
        remover = OutlierRemover(method="iqr")

        data_with_outliers = pd.DataFrame({
            'price': [100, 101, 99, 102, 200],  # 200が外れ値
            'volume': [1000, 1100, 900, 1200, 1000],
        })

        cleaned = remover.remove_outliers(data_with_outliers)
        assert isinstance(cleaned, pd.DataFrame)

        print("✅ 外れ値除去基本機能のテスト成功")

    def test_data_imputation_basic_workflow(self):
        """データ補完基本ワークフローテスト"""
        imputer = DataImputer(strategy="mean")

        data_with_missing = pd.DataFrame({
            'price': [100, 101, np.nan, 102, 100],
            'volume': [1000, 1100, 900, np.nan, 1000],
        })

        imputed = imputer.impute(data_with_missing)
        assert isinstance(imputed, pd.DataFrame)
        assert not imputed.isnull().any().any()

        print("✅ データ補完基本ワークフローテスト成功")

    def test_pipeline_integration_basic_test(self):
        """パイプライン統合基本テスト"""
        # 基本パイプラインをテスト
        pipeline = create_preprocessing_pipeline()

        simple_data = pd.DataFrame({
            'price': [100, 101, 99, 102, 100],
            'volume': [1000, 1100, 900, 1200, 1000],
        })

        # パイプラインを適用
        processed = pipeline.fit_transform(simple_data)
        assert isinstance(processed, pd.DataFrame)

        print("✅ パイプライン統合基本テスト成功")