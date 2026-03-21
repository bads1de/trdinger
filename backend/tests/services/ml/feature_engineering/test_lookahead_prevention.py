import numpy as np
import pandas as pd

from app.services.ml.feature_engineering.microstructure_features import (
    MicrostructureFeatureCalculator,
)
from app.services.ml.feature_engineering.time_anomaly_features import (
    TimeAnomalyFeatures,
)


def _build_ohlcv_frame(periods: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1h")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 120, periods),
            "high": np.linspace(101, 121, periods),
            "low": np.linspace(99, 119, periods),
            "close": np.linspace(100, 130, periods),
            "volume": np.linspace(1000, 2000, periods),
        },
        index=index,
    )


def test_microstructure_features_do_not_backfill_future_auxiliary_data():
    ohlcv = _build_ohlcv_frame(100)
    partial_ohlcv = ohlcv.iloc[:50]

    future_index = ohlcv.index[80:100]
    partial_index = ohlcv.index[:50]

    future_fr = pd.DataFrame(
        {"funding_rate": np.linspace(0.001, 0.02, len(future_index))},
        index=future_index,
    )
    future_ls = pd.DataFrame(
        {"long_short_ratio": np.linspace(0.5, 1.5, len(future_index))},
        index=future_index,
    )

    partial_fr = pd.DataFrame(
        {"funding_rate": np.full(len(partial_index), np.nan)},
        index=partial_index,
    )
    partial_ls = pd.DataFrame(
        {"long_short_ratio": np.full(len(partial_index), np.nan)},
        index=partial_index,
    )

    calculator = MicrostructureFeatureCalculator()
    full_features = calculator.calculate_features(
        ohlcv, fr_df=future_fr, ls_df=future_ls
    )
    partial_features = calculator.calculate_features(
        partial_ohlcv, fr_df=partial_fr, ls_df=partial_ls
    )

    common_index = partial_features.index.intersection(full_features.index)
    common_columns = partial_features.columns.intersection(full_features.columns)

    for col in common_columns:
        partial_values = partial_features.loc[common_index, col].dropna()
        if len(partial_values) == 0:
            continue

        full_values = full_features.loc[partial_values.index, col]
        np.testing.assert_allclose(
            partial_values.values,
            full_values.values,
            rtol=1e-5,
            err_msg=f"microstructure feature '{col}' differs when future aux data is removed",
        )


def test_time_anomaly_features_do_not_use_future_values():
    ohlcv = _build_ohlcv_frame(100)
    partial_ohlcv = ohlcv.iloc[:50]

    calculator = TimeAnomalyFeatures()
    full_features = calculator.calculate_features(ohlcv)
    partial_features = calculator.calculate_features(partial_ohlcv)

    common_index = partial_features.index.intersection(full_features.index)

    np.testing.assert_allclose(
        partial_features.loc[common_index, "time_adaptive_vol_ratio"].values,
        full_features.loc[common_index, "time_adaptive_vol_ratio"].values,
        rtol=1e-5,
        err_msg="time_adaptive_vol_ratio differs when future rows are removed",
    )
