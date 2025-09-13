"""
統合データフロー

データ収集から処理、分析までの完全なフローをテスト
TDD原則に基づき、エンドツーエンドの機能をテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# データ関連
from app.utils.data_conversion import OHLCVDataConverter
from backend.app.utils.data_processing import data_processor
from app.utils.data_validation import DataValidator

# バックテスト関連
from backend.app.services.backtest.execution.backtest_executor import BacktestExecutor
from backend.app.services.backtest.backtest_data_service import BacktestDataService
from backend.app.services.auto_strategy.generators.strategy_factory import StrategyFactory

# 指標関連
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestDataFlowIntegration:
    """データフローの統合テスト"""

    @pytest.fixture
    def comprehensive_ohlcv_data(self):
        """包括的なOHLCVデータ"""
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        np.random.seed(42)

        # 現実的な価格変動をシミュレート
        base_prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        volatility = np.random.uniform(0.005, 0.02, 200)

        opens = base_prices
        highs = base_prices * (1 + volatility)
        lows = base_prices * (1 - volatility)
        closes = base_prices + np.random.uniform(-volatility, volatility, 200) * base_prices

        # OHLCの整合性を確保
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        data = {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, 200),
            'open_interest': np.random.uniform(500000, 2000000, 200),
            'funding_rate': np.random.uniform(-0.001, 0.001, 200),
            'fear_greed_value': np.random.uniform(20, 80, 200)
        }
        return pd.DataFrame(data)

    def test_ccxt_to_processed_dataframe_flow(self, comprehensive_ohlcv_data):
        """CCXT形式から処理済みDataFrameへの完全フロー"""
        # 1. CCXT形式のデータを作成
        ccxt_data = []
        for idx, row in comprehensive_ohlcv_data.iterrows():
            ccxt_data.append([
                int(row['timestamp'].timestamp() * 1000),  # ミリ秒
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ])

        # 2. CCXT → DB形式変換
        db_records = OHLCVDataConverter.ccxt_to_db_format(
            ccxt_data, "BTC/USDT", "1h"
        )
        assert len(db_records) == len(ccxt_data)

        # 3. DB形式 → API形式変換
        db_mocks = []
        for record in db_records:
            mock = Mock()
            mock.timestamp = record["timestamp"]
            mock.open = record["open"]
            mock.high = record["high"]
            mock.low = record["low"]
            mock.close = record["close"]
            mock.volume = record["volume"]
            db_mocks.append(mock)

        api_data = OHLCVDataConverter.db_to_api_format(db_mocks)
        assert len(api_data) == len(ccxt_data)

    def test_data_processing_pipeline_flow(self, comprehensive_ohlcv_data):
        """データ処理パイプラインの完全フロー"""
        # 1. データ検証
        validation_result = DataValidator.validate_ohlcv_data(comprehensive_ohlcv_data)
        assert validation_result["is_valid"] is True

        # 2. データクリーニングと検証
        cleaned_data = data_processor.clean_and_validate_data(
            comprehensive_ohlcv_data,
            required_columns=['open', 'high', 'low', 'close']
        )
        assert len(cleaned_data) > 0

        # 3. 前処理パイプライン適用
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)
        assert isinstance(processed_data, pd.DataFrame)

        # 4. 効率的な処理
        efficient_data = data_processor.process_data_efficiently(cleaned_data)
        assert isinstance(efficient_data, pd.DataFrame)

    def test_indicator_calculation_flow(self, comprehensive_ohlcv_data):
        """指標計算の完全フロー"""
        service = TechnicalIndicatorService()

        # 複数の指標を計算
        indicators_to_test = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS']

        for indicator in indicators_to_test:
            result = service.calculate_indicator(comprehensive_ohlcv_data, indicator, {})
            if result is not None:
                assert len(result) == len(comprehensive_ohlcv_data)

    def test_backtest_data_preparation_flow(self, comprehensive_ohlcv_data):
        """バックテストデータ準備の完全フロー"""
        # 1. データ準備
        processed_data = data_processor.clean_and_validate_data(
            comprehensive_ohlcv_data,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        # 2. ラベル生成のモック
        from unittest.mock import Mock
        label_generator = Mock()
        mock_labels = pd.Series([0, 1, 0] * (len(processed_data) // 3 + 1))[:len(processed_data)]
        mock_labels.index = processed_data.index
        label_generator.generate_labels.return_value = (
            mock_labels,
            {"threshold": 0.02}
        )

        # 3. 学習データ準備
        features, labels, threshold_info = data_processor.prepare_training_data(
            processed_data, label_generator
        )

        assert len(features) == len(labels)
        assert isinstance(threshold_info, dict)

    def test_strategy_generation_and_execution_flow(self):
        """戦略生成と実行の完全フロー"""
        # 1. 戦略遺伝子の作成
        from backend.app.services.auto_strategy.models.strategy_models import StrategyGene
        gene = Mock(spec=StrategyGene)
        gene.validate.return_value = (True, [])
        gene.id = "test-integration-gene"

        # 2. 戦略ファクトリーでの戦略生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        # 3. 戦略インスタンスの作成
        broker = Mock()
        data = Mock()
        data.Close = [50000.0]
        strategy_instance = strategy_class(broker=broker, data=data, params=None)

        assert strategy_instance.gene == gene


class TestEndToEndDataProcessing:
    """エンドツーエンドデータ処理テスト"""

    def test_full_data_processing_workflow(self):
        """完全なデータ処理ワークフロー"""
        # 1. 生データの作成
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        raw_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(49000, 51000, 100),
            'high': np.random.uniform(50000, 52000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(49000, 51000, 100),
            'volume': np.random.randint(1000, 5000, 100)
        })

        # 2. データ検証
        validation_result = DataValidator.validate_ohlcv_data(raw_data)
        assert validation_result["is_valid"] is True

        # 3. データクリーニング
        cleaned_data = data_processor.clean_and_validate_data(
            raw_data, ['open', 'high', 'low', 'close', 'volume']
        )
        assert len(cleaned_data) > 0

        # 4. パイプライン処理
        pipeline = data_processor.create_optimized_pipeline()
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)
        assert isinstance(processed_data, pd.DataFrame)

        # 5. 指標計算
        service = TechnicalIndicatorService()
        sma_result = service.calculate_indicator(cleaned_data, 'SMA', {'length': 20})
        if sma_result is not None:
            assert len(sma_result) == len(cleaned_data)

    def test_error_handling_in_data_flow(self):
        """データフロー内のエラーハンドリング"""
        # 1. 空のDataFrame
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            data_processor.clean_and_validate_data(empty_df, ['open'])

        # 2. 必須カラムのないDataFrame
        invalid_df = pd.DataFrame({'timestamp': [datetime.now()], 'value': [100]})

        with pytest.raises(ValueError):
            data_processor.clean_and_validate_data(
                invalid_df, ['open', 'high', 'low', 'close']
            )

    def test_data_quality_through_pipeline(self):
        """パイプラインを通じたデータ品質テスト"""
        # 高品質のテストデータ
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        quality_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(50000, 51000, 50),
            'high': np.linspace(50500, 51500, 50),
            'low': np.linspace(49500, 50500, 50),
            'close': np.linspace(50000, 51000, 50),
            'volume': np.linspace(1000, 5000, 50)
        })

        # 各処理ステップでの品質チェック
        # 1. 検証
        validation = DataValidator.validate_ohlcv_data(quality_data)
        assert validation["is_valid"] is True
        assert validation["data_quality_score"] > 90

        # 2. クリーニング
        cleaned = data_processor.clean_and_validate_data(
            quality_data, ['open', 'high', 'low', 'close']
        )
        assert len(cleaned) == len(quality_data)

        # 3. 変換
        processed = data_processor.preprocess_with_pipeline(cleaned)
        assert isinstance(processed, pd.DataFrame)


class TestPerformanceAndScalability:
    """パフォーマンスとスケーラビリティテスト"""

    def test_large_dataset_processing(self):
        """大規模データセットの処理"""
        # 大規模データセットの作成
        dates = pd.date_range('2020-01-01', periods=10000, freq='h')
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(30000, 70000, 10000),
            'high': np.random.uniform(35000, 75000, 10000),
            'low': np.random.uniform(25000, 65000, 10000),
            'close': np.random.uniform(30000, 70000, 10000),
            'volume': np.random.randint(100, 10000, 10000)
        })

        # 処理時間を計測しながら実行
        import time
        start_time = time.time()

        # データ処理
        cleaned = data_processor.clean_and_validate_data(
            large_data, ['open', 'high', 'low', 'close']
        )

        processing_time = time.time() - start_time
        assert processing_time < 10.0  # 10秒以内に完了すべき
        assert len(cleaned) == len(large_data)

    def test_memory_efficiency(self):
        """メモリ効率テスト"""
        # メモリ使用量を監視しながら処理
        dates = pd.date_range('2023-01-01', periods=5000, freq='h')
        memory_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 60000, 5000),
            'high': np.random.uniform(45000, 65000, 5000),
            'low': np.random.uniform(35000, 55000, 5000),
            'close': np.random.uniform(40000, 60000, 5000),
            'volume': np.random.randint(500, 5000, 5000)
        })

        # 処理前のメモリ使用量
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # データ処理
        result = data_processor.process_data_efficiently(memory_data)

        # 処理後のメモリ使用量
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        # メモリ使用量の増加が合理的であることを確認
        memory_increase = memory_after - memory_before
        assert memory_increase < 100  # 100MB以内の増加


class TestDataConsistency:
    """データ整合性テスト"""

    def test_data_consistency_across_formats(self):
        """異なるデータ形式間での整合性"""
        # テストデータ
        original_data = pd.DataFrame({
            'timestamp': [datetime(2023, 1, 1, 12, 0, 0)],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [1000.0]
        })

        # CCXT形式への変換
        ccxt_format = OHLCVDataConverter.db_to_api_format([
            Mock(
                timestamp=original_data.iloc[0]['timestamp'],
                open=original_data.iloc[0]['open'],
                high=original_data.iloc[0]['high'],
                low=original_data.iloc[0]['low'],
                close=original_data.iloc[0]['close'],
                volume=original_data.iloc[0]['volume']
            )
        ])

        # 逆変換
        db_format = OHLCVDataConverter.ccxt_to_db_format(
            ccxt_format, "BTC/USDT", "1h"
        )

        # 整合性チェック
        assert abs(db_format[0]['open'] - original_data.iloc[0]['open']) < 1e-6
        assert abs(db_format[0]['close'] - original_data.iloc[0]['close']) < 1e-6

    def test_pipeline_idempotency(self):
        """パイプラインのべき等性テスト"""
        # テストデータ
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=20, freq='h'),
            'open': np.random.uniform(49000, 51000, 20),
            'high': np.random.uniform(50000, 52000, 20),
            'low': np.random.uniform(48000, 50000, 20),
            'close': np.random.uniform(49000, 51000, 20),
            'volume': np.random.randint(1000, 5000, 20)
        })

        # パイプライン適用
        result1 = data_processor.process_data_efficiently(test_data)
        result2 = data_processor.process_data_efficiently(result1)

        # 結果が同じであることを確認（べき等性）
        pd.testing.assert_frame_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__])