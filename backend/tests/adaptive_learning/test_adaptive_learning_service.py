import pandas as pd
import numpy as np

from app.services.ml.adaptive_learning.adaptive_learning_service import (
    AdaptiveLearningService,
    AdaptiveLearningConfig,
    MarketRegime,
)


def _make_dummy_ohlcv(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 30000 + np.cumsum(rng.normal(0, 50, size=n))
    highs = prices + rng.normal(20, 10, size=n)
    lows = prices - rng.normal(20, 10, size=n)
    opens = prices + rng.normal(0, 5, size=n)
    volumes = np.abs(rng.normal(1000, 100, size=n))

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(highs, opens),
            "Low": np.minimum(lows, opens),
            "Close": prices,
            "Volume": volumes,
        }
    )
    return df


def test_adaptive_learning_service_with_basic_detector():
    df = _make_dummy_ohlcv(150)
    svc = AdaptiveLearningService()

    result = svc.adapt_to_market_changes(df)
    assert result is not None
    assert isinstance(result.regime_detected, MarketRegime)
    assert 0.0 <= result.confidence <= 1.0


def test_adaptive_learning_service_with_enhanced_detector_kmeans():
    df = _make_dummy_ohlcv(200)
    cfg = AdaptiveLearningConfig(
        use_enhanced_regime_detector=True,
        detection_method="kmeans",
        regime_detection_window=100,
    )
    svc = AdaptiveLearningService(config=cfg)

    result = svc.adapt_to_market_changes(df)
    assert result is not None
    assert isinstance(result.regime_detected, MarketRegime)
    assert 0.0 <= result.confidence <= 1.0

