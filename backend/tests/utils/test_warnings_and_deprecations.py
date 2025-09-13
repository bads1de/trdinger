"""
統合ユーティリティテスト

警告、非推奨機能、ログ関連のテストを統合
TDD原則に基づき、各機能を包括的にテスト
"""

import pytest
import pandas as pd
import numpy as np
import logging
import warnings
from unittest.mock import patch, MagicMock
from datetime import datetime

# 非推奨機能テスト
import pandas as pd

# 警告関連テスト
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

# ログ削除テスト
from database.repositories.ohlcv_repository import OHLCVRepository
from app.services.backtest.data.data_retrieval_service import DataRetrievalService
from app.utils.data_processing.data_processor import DataProcessor
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.backtest.execution.backtest_executor import BacktestExecutor


class TestDeprecationWarnings:
    """非推奨機能のテスト"""

    def test_pandas_fillna_method_deprecation(self):
        """pandas fillna methodパラメータの非推奨テスト"""
        # 非推奨方法を使用
        series = pd.Series([1, np.nan, 3, np.nan, 5])

        # 非推奨のmethodパラメータを使用した場合、FutureWarningが発生することを確認
        with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
            result_bfill = series.fillna(method="bfill")

        with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
            result_ffill = series.fillna(method="ffill")

        # 結果は正しいことを確認
        assert not result_bfill.isna().any()
        assert not result_ffill.isna().any()

    def test_pandas_fillna_new_api_recommendation(self):
        """新しいAPIの推奨テスト"""
        series = pd.Series([1, np.nan, 3, np.nan, 5])

        # 新しいAPIを使用
        result_bfill = series.bfill()
        result_ffill = series.ffill()

        expected_bfill = pd.Series([1.0, 3.0, 3.0, 5.0, 5.0], index=series.index)
        expected_ffill = pd.Series([1.0, 1.0, 3.0, 3.0, 5.0], index=series.index)

        pd.testing.assert_series_equal(result_bfill, expected_bfill)
        pd.testing.assert_series_equal(result_ffill, expected_ffill)

    def test_dataframe_fillna_method_deprecation(self):
        """DataFrame fillna methodパラメータの非推奨テスト"""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})

        # 非推奨のmethodパラメータを使用した場合、FutureWarningが発生することを確認
        with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
            result_bfill = df.fillna(method="bfill")

        with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
            result_ffill = df.fillna(method="ffill")

        # 結果は正しいことを確認
        assert not result_bfill.isna().any().any()
        assert not result_ffill.isna().any().any()

    def test_dataframe_fillna_new_api_recommendation(self):
        """DataFrame新しいAPIの推奨テスト"""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})

        # 新しいAPIを使用
        result_bfill = df.bfill()
        result_ffill = df.ffill()

        expected_bfill = pd.DataFrame({"A": [1.0, 3.0, 3.0], "B": [4.0, 6.0, 6.0]})
        expected_ffill = pd.DataFrame({"A": [1.0, 1.0, 3.0], "B": [4.0, 4.0, 6.0]})

        pd.testing.assert_frame_equal(result_bfill, expected_bfill)
        pd.testing.assert_frame_equal(result_ffill, expected_ffill)


class TestIndicatorWarnings:
    """指標関連の警告テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        df = pd.DataFrame({
            'Close': np.random.rand(100) * 100
        })
        return df

    def test_indicator_no_futurewarning(self, sample_data):
        """指標計算でFutureWarningが発生しないことをテスト"""
        service = TechnicalIndicatorService()

        # VIDYA計算時のFutureWarningなしを確認
        with pytest.warns(None) as record:
            result = service.calculate_indicator(sample_data, 'VIDYA', {'period': 14, 'adjust': True})

        # FutureWarning about dtype incompatibilityがないことを確認
        future_warnings = [w for w in record.list
                          if "FutureWarning" in str(w.message) and "dtype incompatible" in str(w.message)]
        assert len(future_warnings) == 0
        assert result is not None

    def test_indicator_no_unexpected_keyword_error(self, sample_data):
        """指標計算でunexpected keyword argumentエラーが発生しないことをテスト"""
        service = TechnicalIndicatorService()

        # PVR計算時のエラーなしを確認
        try:
            result = service.calculate_indicator(sample_data, 'PVR', {})
            assert result is not None
            assert len(result) > 0
        except TypeError as e:
            assert "unexpected keyword argument" not in str(e)

    def test_indicator_no_missing_argument_error(self, sample_data):
        """指標計算でmissing argumentエラーが発生しないことをテスト"""
        service = TechnicalIndicatorService()

        # LINREG計算時のエラーなしを確認
        try:
            result = service.calculate_indicator(sample_data, 'LINREG', {'period': 14})
            assert result is not None
            assert isinstance(result, (np.ndarray, pd.Series))
            assert len(result) > 0
        except TypeError as e:
            assert "unexpected keyword argument" not in str(e)

    def test_indicator_stc_no_missing_arg(self, sample_data):
        """STC指標計算でmissing argumentエラーが発生しないことをテスト"""
        service = TechnicalIndicatorService()

        try:
            result = service.calculate_indicator(sample_data, 'STC', {'length': 10, 'fast_length': 23, 'slow_length': 50})
            assert result is not None
        except TypeError as e:
            assert "missing 1 required positional argument" not in str(e)

    def test_indicator_cv_calculation(self, sample_data):
        """CV指標計算テスト"""
        service = TechnicalIndicatorService()
        result = service.calculate_indicator(sample_data, 'CV', {'length': 14})
        assert result is not None

    def test_indicator_irm_calculation(self):
        """IRM指標計算テスト"""
        df = pd.DataFrame({
            'High': np.random.rand(100) * 110,
            'Low': np.random.rand(100) * 90,
            'Close': np.random.rand(100) * 100 + 10
        })
        service = TechnicalIndicatorService()
        result = service.calculate_indicator(df, 'IRM', {'length': 14})
        assert result is not None
        assert len(result) > 0


class TestLogRemoval:
    """ログ削除テスト"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """サンプルOHLCVデータ"""
        return [
            {
                'timestamp': datetime(2025, 6, 3, 0, 0, 0),
                'symbol': 'BTC/USDT:USDT',
                'timeframe': '1h',
                'open': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'close': 50500.0,
                'volume': 100.0
            }
        ]

    @pytest.fixture
    def sample_dataframe(self):
        """サンプルDataFrame"""
        data = {
            'timestamp': [datetime(2025, 6, 3, 0, 0, 0)],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [100.0],
            'open_interest': [1000000.0],
            'funding_rate': [0.0001]
        }
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # timestampをdatetime型に
        return df

    def test_ohlcv_repository_no_query_log(self, sample_ohlcv_data, caplog):
        """OHLCVRepositoryのクエリログが出力されないことを確認"""
        with patch('database.repositories.ohlcv_repository.OHLCVRepository.get_filtered_data') as mock_get:
            mock_get.return_value = sample_ohlcv_data

            # Mock session
            mock_session = MagicMock()
            repo = OHLCVRepository(mock_session)

            with caplog.at_level(logging.INFO):
                repo.get_ohlcv_data(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_time=datetime(2025, 6, 3, 0, 0, 0),
                    end_time=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "OHLCVRepository - Querying data:" not in "\n".join(log_messages)
            assert "OHLCVRepository - Found" not in "\n".join(log_messages)

    def test_data_retrieval_service_no_retrieved_log(self, sample_ohlcv_data, caplog):
        """DataRetrievalServiceの取得ログが出力されないことを確認"""
        with patch('database.repositories.ohlcv_repository.OHLCVRepository') as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_ohlcv_data.return_value = sample_ohlcv_data

            service = DataRetrievalService(ohlcv_repo=mock_repo)

            with caplog.at_level(logging.INFO):
                result = service.get_ohlcv_data(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_date=datetime(2025, 6, 3, 0, 0, 0),
                    end_date=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "DataRetrievalService - Retrieved" not in "\n".join(log_messages)

    def test_data_integration_service_no_logs(self, sample_ohlcv_data, sample_dataframe, caplog):
        """DataIntegrationServiceのログが出力されないことを確認"""
        from app.services.backtest.data.data_integration_service import DataIntegrationService

        with patch('app.services.backtest.data.data_retrieval_service.DataRetrievalService') as mock_retrieval:
            mock_retrieval_instance = mock_retrieval.return_value
            mock_retrieval_instance.get_ohlcv_data.return_value = sample_ohlcv_data

            with patch('app.services.backtest.data.data_conversion_service.DataConversionService') as mock_conversion:
                mock_conversion_instance = mock_conversion.return_value
                mock_conversion_instance.convert_ohlcv_to_dataframe.return_value = sample_dataframe

                service = DataIntegrationService(
                    retrieval_service=mock_retrieval_instance,
                    conversion_service=mock_conversion_instance
                )

                with caplog.at_level(logging.INFO):
                    result = service._get_base_ohlcv_dataframe(
                        symbol='BTC/USDT:USDT',
                        timeframe='1h',
                        start_date=datetime(2025, 6, 3, 0, 0, 0),
                        end_date=datetime(2025, 9, 3, 0, 0, 0)
                    )

                # 削除されたログメッセージが含まれないことを確認
                log_messages = [record.message for record in caplog.records]
                assert "DataIntegrationService - Retrieved" not in "\n".join(log_messages)
                assert "DataIntegrationService - Converted to DataFrame" not in "\n".join(log_messages)

    def test_data_processor_no_cleaning_logs(self, sample_dataframe, caplog):
        """DataProcessorのクリーニングログが出力されないことを確認"""
        processor = DataProcessor()

        with caplog.at_level(logging.INFO):
            result = processor.clean_and_validate_data(
                df=sample_dataframe,
                required_columns=['open', 'high', 'low', 'close', 'volume', 'open_interest', 'funding_rate']
            )

        # 削除されたログメッセージが含まれないことを確認
        log_messages = [record.message for record in caplog.records]
        assert "データクリーニングと検証を開始" not in "\n".join(log_messages)
        assert "カラム名を小文字に統一しました" not in "\n".join(log_messages)
        assert "拡張データの範囲クリップを開始" not in "\n".join(log_messages)

    def test_backtest_data_service_no_logs(self, sample_dataframe, caplog):
        """BacktestDataServiceのログが出力されないことを確認"""
        with patch('app.services.backtest.data.data_integration_service.DataIntegrationService') as mock_integration:
            mock_integration_instance = mock_integration.return_value
            mock_integration_instance.create_backtest_dataframe.return_value = sample_dataframe

            service = BacktestDataService()
            service._integration_service = mock_integration_instance

            with caplog.at_level(logging.INFO):
                result = service.get_data_for_backtest(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_date=datetime(2025, 6, 3, 0, 0, 0),
                    end_date=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "BacktestDataService - DataFrame created with shape:" not in "\n".join(log_messages)
            assert "BacktestDataService - funding_rate stats:" not in "\n".join(log_messages)

    def test_backtest_executor_no_logs(self, sample_dataframe, caplog):
        """BacktestExecutorのログが出力されないことを確認"""
        with patch('app.services.backtest.backtest_data_service.BacktestDataService') as mock_data_service:
            mock_data_service_instance = mock_data_service.return_value
            mock_data_service_instance.get_data_for_backtest.return_value = sample_dataframe

            executor = BacktestExecutor(mock_data_service_instance)

            with caplog.at_level(logging.INFO):
                result = executor._get_backtest_data(
                    symbol='BTC/USDT:USDT',
                    timeframe='1h',
                    start_date=datetime(2025, 6, 3, 0, 0, 0),
                    end_date=datetime(2025, 9, 3, 0, 0, 0)
                )

            # 削除されたログメッセージが含まれないことを確認
            log_messages = [record.message for record in caplog.records]
            assert "BacktestExecutor - Retrieved data shape:" not in "\n".join(log_messages)
            assert "BacktestExecutor - Data columns:" not in "\n".join(log_messages)
            assert "BacktestExecutor - Data head:" not in "\n".join(log_messages)

    def test_feature_calculator_no_completion_logs(self, sample_dataframe, caplog):
        """特徴量計算の完了ログが出力されないことを確認"""
        from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator

        calculator = PriceFeatureCalculator()

        # メソッドを直接呼び出してログが出力されないことを確認
        with caplog.at_level(logging.DEBUG):
            calculator.log_feature_calculation_complete("test_feature")

        # 削除されたログメッセージが含まれないことを確認
        log_messages = [record.message for record in caplog.records]
        assert "test_feature特徴量計算が完了しました" not in "\n".join(log_messages)


class TestPandasDeprecationComprehensive:
    """pandas非推奨機能の包括テスト"""

    def test_fillna_method_warnings_comprehensive(self):
        """fillna methodパラメータの警告を包括的にテスト"""
        # Seriesでのテスト
        series = pd.Series([1, np.nan, 3, np.nan, 5])

        with pytest.warns(FutureWarning) as warning_info:
            series.fillna(method="bfill")

        assert len(warning_info) > 0
        assert "method" in str(warning_info[0].message)

        with pytest.warns(FutureWarning) as warning_info:
            series.fillna(method="ffill")

        assert len(warning_info) > 0
        assert "method" in str(warning_info[0].message)

        # DataFrameでのテスト
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})

        with pytest.warns(FutureWarning) as warning_info:
            df.fillna(method="bfill")

        assert len(warning_info) > 0
        assert "method" in str(warning_info[0].message)

    def test_fillna_new_methods_correctness(self):
        """新しいfillnaメソッドの正確性をテスト"""
        # Seriesテスト
        series = pd.Series([1, np.nan, 3, np.nan, 5])

        # bfill
        result_bfill = series.bfill()
        expected_bfill = pd.Series([1.0, 3.0, 3.0, 5.0, 5.0])
        pd.testing.assert_series_equal(result_bfill, expected_bfill)

        # ffill
        result_ffill = series.ffill()
        expected_ffill = pd.Series([1.0, 1.0, 3.0, 3.0, 5.0])
        pd.testing.assert_series_equal(result_ffill, expected_ffill)

        # DataFrameテスト
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})

        # bfill
        result_bfill = df.bfill()
        expected_bfill = pd.DataFrame({"A": [1.0, 3.0, 3.0], "B": [4.0, 6.0, 6.0]})
        pd.testing.assert_frame_equal(result_bfill, expected_bfill)

        # ffill
        result_ffill = df.ffill()
        expected_ffill = pd.DataFrame({"A": [1.0, 1.0, 3.0], "B": [4.0, 4.0, 6.0]})
        pd.testing.assert_frame_equal(result_ffill, expected_ffill)


if __name__ == "__main__":
    pytest.main([__file__])