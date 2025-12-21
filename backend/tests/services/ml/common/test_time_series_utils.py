import pytest
import pandas as pd
from app.services.ml.common.utils import infer_timeframe, get_t1_series

class TestTimeSeriesUtils:
    def test_infer_timeframe_standard(self):
        """標準的な時間足の推定"""
        # 1h
        idx_1h = pd.date_range("2023-01-01", periods=10, freq="h")
        assert infer_timeframe(idx_1h) == "1h"
        
        # 15m
        idx_15m = pd.date_range("2023-01-01", periods=10, freq="15min")
        assert infer_timeframe(idx_15m) == "15m"
        
        # 1d
        idx_1d = pd.date_range("2023-01-01", periods=10, freq="D")
        assert infer_timeframe(idx_1d) == "1d"

    def test_infer_timeframe_custom(self):
        """カスタム時間足（Nh, Nm）"""
        # 2h
        idx_2h = pd.date_range("2023-01-01", periods=10, freq="2h")
        assert infer_timeframe(idx_2h) == "2h"
        
        # 5m
        idx_5m = pd.date_range("2023-01-01", periods=10, freq="5min")
        assert infer_timeframe(idx_5m) == "5m"

    def test_infer_timeframe_insufficient_data(self):
        """データ不足時のデフォルト"""
        assert infer_timeframe(pd.to_datetime(["2023-01-01"])) == "1h"

    def test_get_t1_series(self):
        """t1系列（パージング用）の計算"""
        idx = pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 01:00:00"])
        # horizon=2, tf=1h なら delta=2h
        res = get_t1_series(idx, horizon_n=2, timeframe="1h")
        
        assert res.iloc[0] == pd.Timestamp("2023-01-01 02:00:00")
        assert res.iloc[1] == pd.Timestamp("2023-01-01 03:00:00")

    def test_get_t1_series_fallback(self):
        """不明なフォーマット時のフォールバック"""
        idx = pd.to_datetime(["2023-01-01 00:00:00"])
        # 不明な形式 "xyz" 指定時
        res = get_t1_series(idx, horizon_n=1, timeframe="xyz")
        # 1h とみなされる
        assert res.iloc[0] == pd.Timestamp("2023-01-01 01:00:00")
