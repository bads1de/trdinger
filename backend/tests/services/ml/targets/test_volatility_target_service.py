import numpy as np
import pandas as pd

from app.services.ml.targets.volatility_target_service import VolatilityTargetService


def test_prepare_targets_generates_future_log_realized_vol():
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    features = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [6, 5, 4, 3, 2, 1],
        },
        index=index,
    )
    ohlcv = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
        },
        index=index,
    )

    service = VolatilityTargetService()
    X, y = service.prepare_targets(features, ohlcv, prediction_horizon=2)

    assert not X.empty
    assert not y.empty
    assert y.name == "future_log_realized_vol"
    assert X.index.equals(y.index)


def test_prepare_targets_excludes_tail_without_forward_window():
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    features = pd.DataFrame({"f1": np.arange(5, dtype=float)}, index=index)
    ohlcv = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=index)

    service = VolatilityTargetService()
    X, y = service.prepare_targets(features, ohlcv, prediction_horizon=2)

    assert X.index.max() < index[-1]
    assert len(X) == len(y)
