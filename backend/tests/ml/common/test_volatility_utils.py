"""
ボラティリティ計算ユーティリティのテスト
"""

import pytest
import numpy as np
import pandas as pd
from backend.app.services.ml.common.volatility_utils import (
    calculate_volatility_std,
    calculate_volatility_atr,
)


class TestVolatilityUtils:
    """ボラティリティ計算ユーティリティのテストケース"""

    @pytest.fixture
    def sample_price_data(self):
        """テスト用価格データ"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="H")

        # 現実的な価格データを生成
        base_price = 50000
        returns = np.random.randn(100) * 0.01  # 1%の標準偏差
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices * (1 + np.abs(np.random.randn(100) * 0.005)),
                "low": prices * (1 - np.abs(np.random.randn(100) * 0.005)),
            },
            index=dates,
        )

        return df

    def test_calculate_volatility_std_basic(self, sample_price_data):
        """基本的なSTDベースのボラティリティ計算"""
        returns = sample_price_data["close"].pct_change()

        result = calculate_volatility_std(returns, window=24)

        # 結果の基本チェック
        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)
        assert not result.iloc[24:].isna().all()  # 24以降はNaNでない

    def test_calculate_volatility_std_window_sizes(self, sample_price_data):
        """異なるウィンドウサイズでの計算"""
        returns = sample_price_data["close"].pct_change()

        vol_24 = calculate_volatility_std(returns, window=24)
        vol_48 = calculate_volatility_std(returns, window=48)

        # ウィンドウが大きいほど滑らかになるはず
        assert vol_24.std() >= vol_48.std()

    def test_calculate_volatility_atr_basic(self, sample_price_data):
        """基本的なATRベースのボラティリティ計算"""
        result = calculate_volatility_atr(
            high=sample_price_data["high"],
            low=sample_price_data["low"],
            close=sample_price_data["close"],
            window=14,
        )

        # 結果の基本チェック
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert (result.dropna() >= 0).all()  # ATRは常に正

    def test_calculate_volatility_atr_as_percentage(self, sample_price_data):
        """ATRをパーセンテージで返す"""
        result = calculate_volatility_atr(
            high=sample_price_data["high"],
            low=sample_price_data["low"],
            close=sample_price_data["close"],
            window=14,
            as_percentage=True,
        )

        # パーセンテージなので0以上のはず
        valid_result = result.dropna()
        assert (valid_result >= 0).all()
        # 通常のボラティリティは数%程度
        assert valid_result.median() < 0.5  # 50%以下であれば正常
