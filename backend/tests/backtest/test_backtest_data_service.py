import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from app.services.backtest.backtest_data_service import BacktestDataService

class TestBacktestDataService:
    @pytest.fixture
    def mock_repos(self):
        return {
            "ohlcv": MagicMock(),
            "oi": MagicMock(),
            "fr": MagicMock()
        }

    @pytest.fixture
    def service(self, mock_repos):
        return BacktestDataService(
            ohlcv_repo=mock_repos["ohlcv"],
            oi_repo=mock_repos["oi"],
            fr_repo=mock_repos["fr"]
        )

    def test_get_data_for_backtest(self, service):
        mock_df = pd.DataFrame({"close": [100]})
        # 内部のintegration_serviceをモック
        with patch.object(service._integration_service, 'create_backtest_dataframe', return_value=mock_df) as mock_method:
            result = service.get_data_for_backtest("BTC", "1h", datetime.now(), datetime.now())
            
            assert result is mock_df
            mock_method.assert_called_once()
            # include_oi/fr が False で呼ばれていることを確認
            args, kwargs = mock_method.call_args
            assert kwargs["include_oi"] is False
            assert kwargs["include_fr"] is False

    def test_get_ohlcv_data(self, service):
        mock_raw = [MagicMock(timestamp=datetime(2023, 1, 1), open=100, high=101, low=99, close=100, volume=10)]
        mock_df = pd.DataFrame({
            "timestamp": [datetime(2023, 1, 1)],
            "open": ["100"], # 文字列で入ってきても変換されるかテスト
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [10]
        })
        
        service._retrieval_service.get_ohlcv_data = MagicMock(return_value=mock_raw)
        service._conversion_service.convert_ohlcv_to_dataframe = MagicMock(return_value=mock_df)
        
        result = service.get_ohlcv_data("BTC", "1h", datetime.now(), datetime.now())
        
        assert len(result) == 1
        assert pd.api.types.is_numeric_dtype(result["open"])
        assert "timestamp" not in result.columns # dropされていること

    def test_get_ml_training_data(self, service):
        mock_df = pd.DataFrame({"close": [100]})
        with patch.object(service._integration_service, 'create_ml_training_dataframe', return_value=mock_df) as mock_method:
            result = service.get_ml_training_data("BTC", "1h", datetime.now(), datetime.now())
            assert result is mock_df

    def test_get_event_labeled_training_data(self, service):
        market_df = pd.DataFrame({"close": [100, 101]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        labels_df = pd.DataFrame({"label": [1, 0]}, index=market_df.index)
        profile = {"info": "test"}
        
        service._integration_service.create_ml_training_dataframe = MagicMock(return_value=market_df)
        service._event_label_generator.generate_hrhp_lrlp_labels = MagicMock(return_value=(labels_df, profile))
        
        df, info = service.get_event_labeled_training_data("BTC", "1h", datetime.now(), datetime.now())
        
        assert "label" in df.columns
        assert info == profile
        service._event_label_generator.generate_hrhp_lrlp_labels.assert_called_once()

    def test_get_data_summary(self, service):
        mock_df = pd.DataFrame()
        service._integration_service.get_data_summary = MagicMock(return_value={"total": 0})
        
        summary = service.get_data_summary(mock_df)
        assert summary["total"] == 0
        service._integration_service.get_data_summary.assert_called_with(mock_df)
