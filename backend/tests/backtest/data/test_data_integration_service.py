import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from app.services.backtest.data.data_integration_service import DataIntegrationService

class TestDataIntegrationService:
    @pytest.fixture
    def retrieval_service(self):
        service = MagicMock()
        service.oi_repo = MagicMock()
        service.fr_repo = MagicMock()
        return service

    @pytest.fixture
    def conversion_service(self):
        return MagicMock()

    @pytest.fixture
    def integration_service(self, retrieval_service, conversion_service):
        return DataIntegrationService(retrieval_service, conversion_service)

    @pytest.fixture
    def sample_ohlcv_df(self):
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 20]
        }, index=pd.to_datetime(["2023-01-01 10:00", "2023-01-01 11:00"]))
        return df

    def test_init_with_repos(self, retrieval_service):
        service = DataIntegrationService(retrieval_service)
        assert service.oi_merger is not None
        assert service.fr_merger is not None

    def test_init_without_repos(self):
        retrieval = MagicMock()
        retrieval.oi_repo = None
        retrieval.fr_repo = None
        service = DataIntegrationService(retrieval)
        assert service.oi_merger is None
        assert service.fr_merger is None

    def test_create_backtest_dataframe_full(self, integration_service, sample_ohlcv_df):
        # Arrange
        integration_service.retrieval_service.get_ohlcv_data.return_value = []
        integration_service.conversion_service.convert_ohlcv_to_dataframe.return_value = sample_ohlcv_df
        
        integration_service.oi_merger = MagicMock()
        integration_service.oi_merger.merge_oi_data.side_effect = lambda df, *args: df.assign(open_interest=[1000, 1100])
        
        integration_service.fr_merger = MagicMock()
        integration_service.fr_merger.merge_fr_data.side_effect = lambda df, *args: df.assign(funding_rate=[0.0001, 0.0002])

        # Act
        df = integration_service.create_backtest_dataframe(
            "BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2)
        )

        # Assert
        assert "open_interest" in df.columns
        assert "funding_rate" in df.columns
        assert df["open_interest"].iloc[0] == 1000
        integration_service.oi_merger.merge_oi_data.assert_called_once()
        integration_service.fr_merger.merge_fr_data.assert_called_once()

    def test_create_backtest_dataframe_exclude_optional(self, integration_service, sample_ohlcv_df):
        integration_service.conversion_service.convert_ohlcv_to_dataframe.return_value = sample_ohlcv_df
        
        df = integration_service.create_backtest_dataframe(
            "BTC/USDT", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2),
            include_oi=False, include_fr=False
        )
        
        assert df["open_interest"].iloc[0] == 0.0
        assert df["funding_rate"].iloc[0] == 0.0

    def test_clean_and_optimize_dataframe(self, integration_service, sample_ohlcv_df):
        df_in = sample_ohlcv_df.assign(open_interest=0, funding_rate=0)
        
        with patch("app.services.backtest.data.data_integration_service.data_processor") as mock_processor:
            mock_processor.clean_and_validate_data.return_value = df_in
            
            result = integration_service._clean_and_optimize_dataframe(df_in)
            
            mock_processor.clean_and_validate_data.assert_called_once()
            assert result is df_in

    def test_get_data_summary(self, integration_service, sample_ohlcv_df):
        df = sample_ohlcv_df.assign(open_interest=[1000, 1100], funding_rate=[0.01, 0.02])
        
        summary = integration_service.get_data_summary(df)
        
        assert summary["total_records"] == 2
        assert "price_range" in summary
        assert summary["open_interest_stats"]["average"] == 1050.0
        assert summary["funding_rate_stats"]["min"] == 0.01

    def test_get_data_summary_empty(self, integration_service):
        summary = integration_service.get_data_summary(pd.DataFrame())
        assert "error" in summary
