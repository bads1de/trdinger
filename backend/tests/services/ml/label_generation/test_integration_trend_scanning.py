import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from app.services.ml.label_generation.label_generation_service import (
    LabelGenerationService,
)


class TestTrendScanningIntegration:
    @pytest.fixture
    def sample_ohlcv(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)
        # Create a trend
        close_prices = np.linspace(100, 110, 100) + np.random.normal(0, 0.1, 100)

        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.1,
                "low": close_prices - 0.1,
                "close": close_prices,
                "volume": 1000,
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def sample_features(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["feature_1"] = df["close"].pct_change().fillna(0)
        return df[["feature_1"]]

    def test_trend_scanning_integration(self, sample_features, sample_ohlcv):
        service = LabelGenerationService()

        # Patch the unified_config where it is defined, so that when LabelGenerationService imports it, it gets the mock.
        # Since LabelGenerationService imports it inside the method, we must patch 'app.config.unified_config.unified_config'.

        with patch("app.config.unified_config.unified_config") as mock_config:
            # Setup mock config
            mock_config.ml.training.label_generation.threshold_method = "TREND_SCANNING"
            mock_config.ml.training.label_generation.horizon_n = 20  # max_window
            mock_config.ml.training.label_generation.threshold = 2.0  # min_t_value
            mock_config.ml.training.label_generation.timeframe = "1h"
            mock_config.ml.training.label_generation.price_column = "close"

            features, labels = service.prepare_labels(
                features_df=sample_features,
                ohlcv_df=sample_ohlcv,
                min_window=5,
                window_step=1,
                use_signal_generator=False,
            )

            assert len(features) > 0
            assert len(labels) == len(features)

            # Since we have a strong uptrend, we expect mostly 1s
            assert labels.sum() > 0
            assert labels.isin([0, 1]).all()


