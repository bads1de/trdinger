"""
データ統合サービステスト

DataIntegrationServiceの機能をテストします。
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.backtest.data.data_conversion_service import DataConversionService
from app.services.backtest.data.data_integration_service import (
    DataIntegrationService,
)
from app.services.backtest.data.data_retrieval_service import DataRetrievalService

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_retrieval_service():
    """モックデータ取得サービス"""
    return MagicMock(spec=DataRetrievalService)


@pytest.fixture
def mock_conversion_service():
    """モックデータ変換サービス"""
    return MagicMock(spec=DataConversionService)


@pytest.fixture
def integration_service(mock_retrieval_service, mock_conversion_service):
    """DataIntegrationServiceインスタンス"""
    return DataIntegrationService(
        retrieval_service=mock_retrieval_service,
        conversion_service=mock_conversion_service,
    )


@pytest.fixture
def sample_ohlcv_data():
    """テスト用OHLCVデータ"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [105.0 + i * 0.1 for i in range(100)],
            "low": [95.0 + i * 0.1 for i in range(100)],
            "close": [102.0 + i * 0.1 for i in range(100)],
            "volume": [1000.0 + i * 10 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_oi_data():
    """テスト用OIデータ"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "open_interest": [5000.0 + i * 50 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_fr_data():
    """テスト用FRデータ"""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "funding_rate": [0.0001 + i * 0.00001 for i in range(100)],
        },
        index=dates,
    )


class TestServiceInitialization:
    """サービス初期化テスト"""

    def test_initialize_with_services(
        self, mock_retrieval_service, mock_conversion_service
    ):
        """サービス付きで初期化できること"""
        service = DataIntegrationService(
            retrieval_service=mock_retrieval_service,
            conversion_service=mock_conversion_service,
        )

        assert service.retrieval_service == mock_retrieval_service
        assert service.conversion_service == mock_conversion_service

    def test_initialize_without_conversion_service(self, mock_retrieval_service):
        """変換サービスなしで初期化できること"""
        service = DataIntegrationService(
            retrieval_service=mock_retrieval_service,
            conversion_service=None,
        )

        assert service.retrieval_service == mock_retrieval_service
        assert service.conversion_service is not None  # 自動作成される

    def test_initialize_oi_merger(self, mock_retrieval_service):
        """OIマージャーが初期化されること"""
        mock_retrieval_service.oi_repo = MagicMock()

        service = DataIntegrationService(
            retrieval_service=mock_retrieval_service,
        )

        assert service.oi_merger is not None

    def test_initialize_fr_merger(self, mock_retrieval_service):
        """FRマージャーが初期化されること"""
        mock_retrieval_service.fr_repo = MagicMock()

        service = DataIntegrationService(
            retrieval_service=mock_retrieval_service,
        )

        assert service.fr_merger is not None


class TestBacktestDataframeCreation:
    """バックテスト用DataFrame作成テスト"""

    def test_create_backtest_dataframe_success(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
        sample_ohlcv_data,
    ):
        """バックテスト用DataFrameを正常に作成できること"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            sample_ohlcv_data
        )

        with (
            patch.object(
                integration_service,
                "_integrate_open_interest_data",
                return_value=sample_ohlcv_data,
            ),
            patch.object(
                integration_service,
                "_integrate_funding_rate_data",
                return_value=sample_ohlcv_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            mock_processor.clean_and_validate_data.return_value = sample_ohlcv_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                include_oi=True,
                include_fr=True,
            )

            assert not result.empty
            assert len(result) == 100

    def test_create_backtest_dataframe_with_oi(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
        sample_ohlcv_data,
    ):
        """OIデータを含むDataFrameを作成できること"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            sample_ohlcv_data
        )

        integrated_data = sample_ohlcv_data.copy()
        integrated_data["open_interest"] = 5000.0

        with (
            patch.object(
                integration_service,
                "_integrate_open_interest_data",
                return_value=integrated_data,
            ),
            patch.object(
                integration_service,
                "_integrate_funding_rate_data",
                return_value=integrated_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            mock_processor.clean_and_validate_data.return_value = integrated_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                include_oi=True,
                include_fr=True,
            )

            assert "open_interest" in result.columns

    def test_create_backtest_dataframe_without_oi(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
        sample_ohlcv_data,
    ):
        """OIデータなしでDataFrameを作成できること"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            sample_ohlcv_data
        )

        with (
            patch.object(
                integration_service,
                "_integrate_funding_rate_data",
                return_value=sample_ohlcv_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            result_data = sample_ohlcv_data.copy()
            result_data["open_interest"] = 0.0
            mock_processor.clean_and_validate_data.return_value = result_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                include_oi=False,
                include_fr=True,
            )

            assert "open_interest" in result.columns
            assert (result["open_interest"] == 0.0).all()

    def test_create_backtest_dataframe_without_fr(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
        sample_ohlcv_data,
    ):
        """FRデータなしでDataFrameを作成できること"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            sample_ohlcv_data
        )

        with (
            patch.object(
                integration_service,
                "_integrate_open_interest_data",
                return_value=sample_ohlcv_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            result_data = sample_ohlcv_data.copy()
            result_data["funding_rate"] = 0.0
            mock_processor.clean_and_validate_data.return_value = result_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                include_oi=True,
                include_fr=False,
            )

            assert "funding_rate" in result.columns
            assert (result["funding_rate"] == 0.0).all()


class TestMLTrainingDataframeCreation:
    """MLトレーニング用DataFrame作成テスト"""

    def test_create_ml_training_dataframe(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
        sample_ohlcv_data,
    ):
        """MLトレーニング用DataFrameを作成できること"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            sample_ohlcv_data
        )

        with patch.object(
            integration_service,
            "create_backtest_dataframe",
            return_value=sample_ohlcv_data,
        ):
            result = integration_service.create_ml_training_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert not result.empty
            assert len(result) == 100

    def test_create_ml_training_dataframe_includes_all_data(self, integration_service):
        """MLトレーニング用DataFrameに全データが含まれること"""
        with patch.object(
            integration_service, "create_backtest_dataframe"
        ) as mock_create:
            integration_service.create_ml_training_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            # OIとFRの両方がTrueで呼ばれることを確認
            mock_create.assert_called_once_with(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
                include_oi=True,
                include_fr=True,
            )


class TestOHLCVDataRetrieval:
    """OHLCVデータ取得テスト"""

    def test_get_base_ohlcv_dataframe(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
        sample_ohlcv_data,
    ):
        """ベースOHLCVデータを取得できること"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            sample_ohlcv_data
        )

        result = integration_service._get_base_ohlcv_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert not result.empty
        mock_retrieval_service.get_ohlcv_data.assert_called_once()
        mock_conversion_service.convert_ohlcv_to_dataframe.assert_called_once()


class TestOpenInterestIntegration:
    """Open Interest統合テスト"""

    def test_integrate_open_interest_data_with_merger(
        self, integration_service, sample_ohlcv_data, sample_oi_data
    ):
        """OIマージャー付きでOIデータを統合できること"""
        mock_merger = MagicMock()
        integrated_data = sample_ohlcv_data.copy()
        integrated_data["open_interest"] = sample_oi_data["open_interest"]
        mock_merger.merge_oi_data.return_value = integrated_data

        integration_service.oi_merger = mock_merger

        result = integration_service._integrate_open_interest_data(
            df=sample_ohlcv_data,
            symbol="BTC/USDT:USDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert "open_interest" in result.columns
        mock_merger.merge_oi_data.assert_called_once()

    def test_integrate_open_interest_data_without_merger(
        self, integration_service, sample_ohlcv_data
    ):
        """OIマージャーなしでの処理"""
        integration_service.oi_merger = None

        result = integration_service._integrate_open_interest_data(
            df=sample_ohlcv_data,
            symbol="BTC/USDT:USDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert "open_interest" in result.columns
        assert result["open_interest"].isna().all()


class TestFundingRateIntegration:
    """Funding Rate統合テスト"""

    def test_integrate_funding_rate_data_with_merger(
        self, integration_service, sample_ohlcv_data, sample_fr_data
    ):
        """FRマージャー付きでFRデータを統合できること"""
        mock_merger = MagicMock()
        integrated_data = sample_ohlcv_data.copy()
        integrated_data["funding_rate"] = sample_fr_data["funding_rate"]
        mock_merger.merge_fr_data.return_value = integrated_data

        integration_service.fr_merger = mock_merger

        result = integration_service._integrate_funding_rate_data(
            df=sample_ohlcv_data,
            symbol="BTC/USDT:USDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert "funding_rate" in result.columns
        mock_merger.merge_fr_data.assert_called_once()

    def test_integrate_funding_rate_data_without_merger(
        self, integration_service, sample_ohlcv_data
    ):
        """FRマージャーなしでの処理"""
        integration_service.fr_merger = None

        result = integration_service._integrate_funding_rate_data(
            df=sample_ohlcv_data,
            symbol="BTC/USDT:USDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert "funding_rate" in result.columns
        assert (result["funding_rate"] == 0.0).all()

    def test_integrate_funding_rate_data_error_handling(
        self, integration_service, sample_ohlcv_data
    ):
        """FRデータ統合エラーの処理"""
        mock_merger = MagicMock()
        mock_merger.merge_fr_data.side_effect = Exception("Merge error")

        integration_service.fr_merger = mock_merger

        result = integration_service._integrate_funding_rate_data(
            df=sample_ohlcv_data,
            symbol="BTC/USDT:USDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        # エラー時は0.0で埋められる
        assert "funding_rate" in result.columns
        assert (result["funding_rate"] == 0.0).all()


class TestDataCleaningAndOptimization:
    """データクリーニングと最適化テスト"""

    def test_clean_and_optimize_dataframe(self, integration_service, sample_ohlcv_data):
        """DataFrameをクリーニングと最適化できること"""
        test_data = sample_ohlcv_data.copy()
        test_data["open_interest"] = 5000.0
        test_data["funding_rate"] = 0.0001

        with patch(
            "app.services.backtest.data.data_integration_service.data_processor"
        ) as mock_processor:
            mock_processor.clean_and_validate_data.return_value = test_data

            integration_service._clean_and_optimize_dataframe(test_data)

            mock_processor.clean_and_validate_data.assert_called_once()
            call_kwargs = mock_processor.clean_and_validate_data.call_args[1]
            assert call_kwargs["interpolate"] is True
            assert call_kwargs["optimize"] is True

    def test_clean_and_optimize_required_columns(self, integration_service):
        """必須カラムが指定されること"""
        test_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
                "open_interest": [5000.0],
                "funding_rate": [0.0001],
            }
        )

        with patch(
            "app.services.backtest.data.data_integration_service.data_processor"
        ) as mock_processor:
            mock_processor.clean_and_validate_data.return_value = test_data

            integration_service._clean_and_optimize_dataframe(test_data)

            call_kwargs = mock_processor.clean_and_validate_data.call_args[1]
            required_columns = call_kwargs["required_columns"]
            assert "open" in required_columns
            assert "high" in required_columns
            assert "low" in required_columns
            assert "close" in required_columns
            assert "volume" in required_columns
            assert "open_interest" in required_columns
            assert "funding_rate" in required_columns


class TestDataSummary:
    """データ概要テスト"""

    def test_get_data_summary_success(self, integration_service, sample_ohlcv_data):
        """データ概要を正常に取得できること"""
        summary = integration_service.get_data_summary(sample_ohlcv_data)

        assert "total_records" in summary
        assert "start_date" in summary
        assert "end_date" in summary
        assert "columns" in summary
        assert "price_range" in summary
        assert "volume_stats" in summary

    def test_get_data_summary_with_additional_data(
        self, integration_service, sample_ohlcv_data
    ):
        """追加データを含む概要を取得できること"""
        test_data = sample_ohlcv_data.copy()
        test_data["open_interest"] = 5000.0
        test_data["funding_rate"] = 0.0001

        summary = integration_service.get_data_summary(test_data)

        assert "open_interest_stats" in summary
        assert "funding_rate_stats" in summary

    def test_get_data_summary_empty_dataframe(self, integration_service):
        """空のDataFrameの概要取得"""
        summary = integration_service.get_data_summary(pd.DataFrame())

        assert "error" in summary
        assert summary["error"] == "データがありません"

    def test_get_data_summary_structure(self, integration_service, sample_ohlcv_data):
        """データ概要の構造が正しいこと"""
        summary = integration_service.get_data_summary(sample_ohlcv_data)

        assert summary["total_records"] == 100
        assert isinstance(summary["columns"], list)

        price_range = summary["price_range"]
        assert "min" in price_range
        assert "max" in price_range
        assert "first_close" in price_range
        assert "last_close" in price_range

        volume_stats = summary["volume_stats"]
        assert "total" in volume_stats
        assert "average" in volume_stats
        assert "max" in volume_stats


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_handle_retrieval_error(self, integration_service, mock_retrieval_service):
        """データ取得エラーの処理"""
        mock_retrieval_service.get_ohlcv_data.side_effect = Exception("Database error")

        # safe_operationデコレーターにより空のDataFrameが返される
        result = integration_service.create_backtest_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert result.empty

    def test_handle_conversion_error(
        self, integration_service, mock_retrieval_service, mock_conversion_service
    ):
        """データ変換エラーの処理"""
        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.side_effect = Exception(
            "Conversion error"
        )

        result = integration_service.create_backtest_dataframe(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
        )

        assert result.empty


class TestEdgeCases:
    """エッジケーステスト"""

    def test_create_dataframe_with_minimal_data(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
    ):
        """最小限のデータでDataFrameを作成"""
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            },
            index=dates,
        )

        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = minimal_data

        with (
            patch.object(
                integration_service,
                "_integrate_open_interest_data",
                return_value=minimal_data,
            ),
            patch.object(
                integration_service,
                "_integrate_funding_rate_data",
                return_value=minimal_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            mock_processor.clean_and_validate_data.return_value = minimal_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
            )

            assert len(result) == 1

    def test_create_dataframe_with_large_data(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
    ):
        """大量データでDataFrameを作成"""
        dates = pd.date_range("2024-01-01", periods=10000, freq="h")
        large_data = pd.DataFrame(
            {
                "open": [100.0] * 10000,
                "high": [105.0] * 10000,
                "low": [95.0] * 10000,
                "close": [102.0] * 10000,
                "volume": [1000.0] * 10000,
            },
            index=dates,
        )

        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = large_data

        with (
            patch.object(
                integration_service,
                "_integrate_open_interest_data",
                return_value=large_data,
            ),
            patch.object(
                integration_service,
                "_integrate_funding_rate_data",
                return_value=large_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            mock_processor.clean_and_validate_data.return_value = large_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2025, 2, 1),
            )

            assert len(result) == 10000

    def test_create_dataframe_with_missing_columns(
        self,
        integration_service,
        mock_retrieval_service,
        mock_conversion_service,
    ):
        """カラムが欠けているDataFrameの作成"""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        incomplete_data = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "close": [102.0] * 10,
                # high, low, volumeが欠けている
            },
            index=dates,
        )

        mock_retrieval_service.get_ohlcv_data.return_value = []
        mock_conversion_service.convert_ohlcv_to_dataframe.return_value = (
            incomplete_data
        )

        with (
            patch.object(
                integration_service,
                "_integrate_open_interest_data",
                return_value=incomplete_data,
            ),
            patch.object(
                integration_service,
                "_integrate_funding_rate_data",
                return_value=incomplete_data,
            ),
            patch(
                "app.services.backtest.data.data_integration_service.data_processor"
            ) as mock_processor,
        ):
            # clean_and_validate_dataが欠けているカラムを補完すると仮定
            complete_data = incomplete_data.copy()
            complete_data["high"] = 105.0
            complete_data["low"] = 95.0
            complete_data["volume"] = 1000.0
            mock_processor.clean_and_validate_data.return_value = complete_data

            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

            # クリーニング後は全カラムが存在する
            assert not result.empty


class TestIntegrationWithRealServices:
    """実際のサービスとの統合テスト"""

    def test_integration_with_real_conversion_service(self, mock_retrieval_service):
        """実際の変換サービスとの統合"""
        service = DataIntegrationService(
            retrieval_service=mock_retrieval_service,
            conversion_service=DataConversionService(),
        )

        assert service.conversion_service is not None
        assert isinstance(service.conversion_service, DataConversionService)

    def test_safe_operation_decorator_applied(self, integration_service):
        """safe_operationデコレーターが適用されていること"""
        # デコレーターが適用されているメソッドは、エラー時に空のDataFrameを返す
        with patch.object(
            integration_service,
            "_get_base_ohlcv_dataframe",
            side_effect=Exception("Error"),
        ):
            result = integration_service.create_backtest_dataframe(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5),
            )

            assert isinstance(result, pd.DataFrame)
            assert result.empty


