import pytest
import pandas as pd
import numpy as np
from app.services.ml.label_generation.presets import (
    triple_barrier_method_preset,
    trend_scanning_preset,
    apply_preset_by_name,
    get_common_presets
)

class TestLabelPresets:
    @pytest.fixture
    def sample_df(self):
        n = 200
        # 緩やかなトレンドにノイズを加えてボラティリティを発生させる
        prices = np.linspace(100, 110, n) + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": [100]*n
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))
        return df

    def test_triple_barrier_method_preset(self, sample_df):
        """TBMプリセットの動作確認"""
        res = triple_barrier_method_preset(sample_df, timeframe="1h", horizon_n=10)
        assert isinstance(res, pd.Series)
        assert not res.empty

    def test_trend_scanning_preset(self, sample_df):
        """TSプリセットの動作確認"""
        res = trend_scanning_preset(sample_df, timeframe="1h", horizon_n=50)
        assert isinstance(res, pd.Series)
        assert not res.empty

    def test_apply_preset_by_name_tbm(self, sample_df):
        """名前指定によるTBM適用"""
        labels, info = apply_preset_by_name(sample_df, "tbm_4h_1.0_1.0")
        assert "preset_name" in info
        assert info["preset_name"] == "tbm_4h_1.0_1.0"
        assert not labels.empty

    def test_apply_preset_by_name_ts(self, sample_df):
        """名前指定によるTS適用"""
        labels, info = apply_preset_by_name(sample_df, "trend_scanning_medium")
        assert "trend_scanning" in info["preset_name"]
        assert not labels.empty

    def test_apply_preset_invalid_name(self, sample_df):
        """存在しないプリセット名"""
        with pytest.raises(ValueError, match="見つかりません"):
            apply_preset_by_name(sample_df, "invalid_preset")

    def test_unsupported_timeframe(self, sample_df):
        """サポート外の時間足"""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            triple_barrier_method_preset(sample_df, timeframe="2h")
