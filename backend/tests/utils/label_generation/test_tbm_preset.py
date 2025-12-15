import pytest
import pandas as pd
import numpy as np
from app.services.ml.label_generation.presets import triple_barrier_method_preset, apply_preset_by_name

def test_triple_barrier_method_preset_basic():
    """トリプルバリアメソッドプリセットの基本テスト"""
    # ダミーデータを作成
    dates = pd.date_range(start="2023-01-01", periods=100, freq="4h")
    close = pd.Series(100 + np.random.randn(100).cumsum(), index=dates, name="close")
    df = pd.DataFrame({"close": close})
    
    # プリセットを実行
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
    # ラベルが期待されるセット内にあることを確認 (最後のhorizon_n期間はNaNを許容)
    unique_labels = labels.dropna().unique()
    for label in unique_labels:
        assert label in ["UP", "DOWN", "RANGE"]

def test_apply_preset_by_name_tbm():
    """名前指定によるプリセット適用テスト (TBM)"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="4h")
    close = pd.Series(100 + np.random.randn(100).cumsum(), index=dates, name="close")
    df = pd.DataFrame({"close": close})
    
    # 登録されているプリセットのいずれかを使用
    labels, info = apply_preset_by_name(df, "tbm_4h_1.0_1.0")
    
    assert isinstance(labels, pd.Series)
    assert info["preset_name"] == "tbm_4h_1.0_1.0"
    assert info["pt"] == 1.0



