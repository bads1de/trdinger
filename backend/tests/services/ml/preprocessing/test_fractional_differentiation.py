import pytest
import pandas as pd
import numpy as np
from app.services.ml.preprocessing.fractional_differentiation import FractionalDifferentiation

class TestFractionalDifferentiation:
    def test_get_weights(self):
        """重み係数の計算テスト"""
        fd = FractionalDifferentiation(d=0.5)
        # d=0.5 の場合: w0=1, w1=-0.5, w2=-0.125, ...
        weights = fd._get_weights(d=0.5, size=3)
        assert weights[0] == 1.0
        assert weights[1] == -0.5
        assert weights[2] == -0.125

    def test_transform_series(self):
        """Seriesに対する変換テスト"""
        # d=1.0 ならば通常の1次差分と同じになる（w0=1, w1=-1, w2=0...）
        fd = FractionalDifferentiation(d=1.0, window_size=2)
        s = pd.Series([10, 12, 15, 14])
        
        res = fd.transform(s)
        # rolling dot [10, 12] * [-1, 1] = 2
        # rolling dot [12, 15] * [-1, 1] = 3
        # rolling dot [15, 14] * [-1, 1] = -1
        assert res.iloc[0] is np.nan or np.isnan(res.iloc[0])
        assert res.iloc[1] == 2.0
        assert res.iloc[2] == 3.0
        assert res.iloc[3] == -1.0

    def test_transform_dataframe(self):
        """DataFrameに対する変換テスト"""
        fd = FractionalDifferentiation(d=0.5, window_size=5)
        df = pd.DataFrame({
            "A": np.random.randn(20),
            "B": np.random.randn(20)
        })
        
        res = fd.transform(df)
        assert isinstance(res, pd.DataFrame)
        assert res.shape == df.shape
        assert res["A"].isnull().sum() == 4 # window_size-1

    def test_insufficient_data(self):
        """データ不足時のテスト"""
        fd = FractionalDifferentiation(d=0.5, window_size=10)
        s = pd.Series([1, 2, 3])
        res = fd.transform(s)
        assert res.isnull().all()

    def test_invalid_input_type(self):
        """不正な入力型"""
        fd = FractionalDifferentiation()
        with pytest.raises(TypeError):
            fd.transform([1, 2, 3])
