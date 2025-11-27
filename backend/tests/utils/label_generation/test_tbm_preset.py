import pytest
import pandas as pd
import numpy as np
from app.services.ml.label_generation.presets import triple_barrier_method_preset, apply_preset_by_name

def test_triple_barrier_method_preset_basic():
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="4h")
    close = pd.Series(100 + np.random.randn(100).cumsum(), index=dates, name="close")
    df = pd.DataFrame({"close": close})
    
    # Run preset
    labels = triple_barrier_method_preset(
        df=df,
        timeframe="4h",
        horizon_n=4,
        pt=1.0,
        sl=1.0,
        min_ret=0.001,
        volatility_window=10
    )
    
    assert isinstance(labels, pd.Series)
    assert len(labels) == 100
    # Check if labels are in expected set (NaN is allowed for last horizon_n)
    unique_labels = labels.dropna().unique()
    for label in unique_labels:
        assert label in ["UP", "DOWN", "RANGE"]

def test_apply_preset_by_name_tbm():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="4h")
    close = pd.Series(100 + np.random.randn(100).cumsum(), index=dates, name="close")
    df = pd.DataFrame({"close": close})
    
    # Use one of the registered presets
    labels, info = apply_preset_by_name(df, "tbm_4h_1.0_1.0")
    
    assert isinstance(labels, pd.Series)
    assert info["preset_name"] == "tbm_4h_1.0_1.0"
    assert info["pt"] == 1.0
