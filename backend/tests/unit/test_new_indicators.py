"""
新しく追加されたインジケーターのテスト

拡張されたトレンド系、モメンタム系、ボラティリティ系、
および新しいカテゴリ（Volume、PriceTransform、Cycle）のテスト
"""

import pytest
import numpy as np
import pandas as pd
import talib
from unittest.mock import patch

# 新しい指標クラスをインポート
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.technical_indicators.price_transform import PriceTransformIndicators
from app.services.indicators.technical_indicators.cycle import CycleIndicators
from app.services.indicators.utils import TALibError


class TestNewTrendIndicators:
    """新しいトレンド系インジケーターのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        return {
            "close": np.random.uniform(100, 200, size),
            "high": np.random.uniform(150, 250, size),
            "low": np.random.uniform(50, 150, size),
            "open": np.random.uniform(75, 175, size),
            "volume": np.random.uniform(1000, 10000, size),
        }

    def test_wma(self, sample_data):
        """WMA（加重移動平均）のテスト"""
        result = TrendIndicators.wma(sample_data["close"], period=14)
        
        # Ta-libの結果と比較
        expected = talib.WMA(sample_data["close"], timeperiod=14)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_trima(self, sample_data):
        """TRIMA（三角移動平均）のテスト"""
        result = TrendIndicators.trima(sample_data["close"], period=14)
        
        # Ta-libの結果と比較
        expected = talib.TRIMA(sample_data["close"], timeperiod=14)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_kama(self, sample_data):
        """KAMA（カウフマン適応移動平均）のテスト"""
        result = TrendIndicators.kama(sample_data["close"], period=30)
        
        # Ta-libの結果と比較
        expected = talib.KAMA(sample_data["close"], timeperiod=30)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_sar(self, sample_data):
        """SAR（パラボリックSAR）のテスト"""
        result = TrendIndicators.sar(sample_data["high"], sample_data["low"])
        
        # Ta-libの結果と比較
        expected = talib.SAR(sample_data["high"], sample_data["low"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["high"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestNewMomentumIndicators:
    """新しいモメンタム系インジケーターのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        return {
            "close": np.random.uniform(100, 200, size),
            "high": np.random.uniform(150, 250, size),
            "low": np.random.uniform(50, 150, size),
            "open": np.random.uniform(75, 175, size),
            "volume": np.random.uniform(1000, 10000, size),
        }

    def test_adx(self, sample_data):
        """ADX（平均方向性指数）のテスト"""
        result = MomentumIndicators.adx(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.ADX(sample_data["high"], sample_data["low"], sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_cci(self, sample_data):
        """CCI（商品チャネル指数）のテスト"""
        result = MomentumIndicators.cci(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.CCI(sample_data["high"], sample_data["low"], sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_mfi(self, sample_data):
        """MFI（マネーフローインデックス）のテスト"""
        result = MomentumIndicators.mfi(
            sample_data["high"], sample_data["low"], sample_data["close"], sample_data["volume"]
        )
        
        # Ta-libの結果と比較
        expected = talib.MFI(
            sample_data["high"], sample_data["low"], sample_data["close"], sample_data["volume"]
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_willr(self, sample_data):
        """WILLR（ウィリアムズ%R）のテスト"""
        result = MomentumIndicators.willr(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.WILLR(sample_data["high"], sample_data["low"], sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestNewVolatilityIndicators:
    """新しいボラティリティ系インジケーターのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        return {
            "close": np.random.uniform(100, 200, size),
            "high": np.random.uniform(150, 250, size),
            "low": np.random.uniform(50, 150, size),
        }

    def test_natr(self, sample_data):
        """NATR（正規化平均真の値幅）のテスト"""
        result = VolatilityIndicators.natr(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.NATR(sample_data["high"], sample_data["low"], sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_trange(self, sample_data):
        """TRANGE（真の値幅）のテスト"""
        result = VolatilityIndicators.trange(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.TRANGE(sample_data["high"], sample_data["low"], sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestVolumeIndicators:
    """出来高系インジケーターのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        return {
            "close": np.random.uniform(100, 200, size),
            "high": np.random.uniform(150, 250, size),
            "low": np.random.uniform(50, 150, size),
            "volume": np.random.uniform(1000, 10000, size),
        }

    def test_obv(self, sample_data):
        """OBV（オンバランスボリューム）のテスト"""
        result = VolumeIndicators.obv(sample_data["close"], sample_data["volume"])
        
        # Ta-libの結果と比較
        expected = talib.OBV(sample_data["close"], sample_data["volume"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_ad(self, sample_data):
        """AD（チャイキンA/Dライン）のテスト"""
        result = VolumeIndicators.ad(
            sample_data["high"], sample_data["low"], sample_data["close"], sample_data["volume"]
        )
        
        # Ta-libの結果と比較
        expected = talib.AD(
            sample_data["high"], sample_data["low"], sample_data["close"], sample_data["volume"]
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestPriceTransformIndicators:
    """価格変換系インジケーターのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        return {
            "close": np.random.uniform(100, 200, size),
            "high": np.random.uniform(150, 250, size),
            "low": np.random.uniform(50, 150, size),
            "open": np.random.uniform(75, 175, size),
        }

    def test_avgprice(self, sample_data):
        """AVGPRICE（平均価格）のテスト"""
        result = PriceTransformIndicators.avgprice(
            sample_data["open"], sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.AVGPRICE(
            sample_data["open"], sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_typprice(self, sample_data):
        """TYPPRICE（典型価格）のテスト"""
        result = PriceTransformIndicators.typprice(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        
        # Ta-libの結果と比較
        expected = talib.TYPPRICE(sample_data["high"], sample_data["low"], sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestCycleIndicators:
    """サイクル系インジケーターのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        size = 100
        return {
            "close": np.random.uniform(100, 200, size),
        }

    def test_ht_dcperiod(self, sample_data):
        """HT_DCPERIOD（ヒルベルト変換支配的サイクル期間）のテスト"""
        result = CycleIndicators.ht_dcperiod(sample_data["close"])
        
        # Ta-libの結果と比較
        expected = talib.HT_DCPERIOD(sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_ht_trendmode(self, sample_data):
        """HT_TRENDMODE（ヒルベルト変換トレンドモード）のテスト"""
        result = CycleIndicators.ht_trendmode(sample_data["close"])
        
        # Ta-libの結果と比較
        expected = talib.HT_TRENDMODE(sample_data["close"])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data["close"])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestIndicatorInitialization:
    """インジケーターの初期化テスト"""

    def test_all_indicators_can_be_imported(self):
        """全てのインジケーターがインポート可能であることを確認"""
        from app.services.indicators import (
            TrendIndicators,
            MomentumIndicators,
            VolatilityIndicators,
            VolumeIndicators,
            PriceTransformIndicators,
            CycleIndicators,
        )
        
        # 各クラスが正しくインポートされていることを確認
        assert TrendIndicators is not None
        assert MomentumIndicators is not None
        assert VolatilityIndicators is not None
        assert VolumeIndicators is not None
        assert PriceTransformIndicators is not None
        assert CycleIndicators is not None

    def test_indicator_registry_initialization(self):
        """インジケーターレジストリの初期化テスト"""
        from app.services.indicators.config.indicator_definitions import initialize_all_indicators
        from app.services.indicators.config.indicator_config import indicator_registry

        # 初期化前のレジストリをクリア
        indicator_registry._configs.clear()

        # 初期化実行
        initialize_all_indicators()

        # 新しいインジケーターが登録されていることを確認
        registered_indicators = list(indicator_registry._configs.keys())
        
        # 新しく追加されたインジケーターが含まれていることを確認
        new_indicators = [
            "WMA", "TRIMA", "KAMA", "SAR",  # トレンド系
            "ADX", "CCI", "MFI", "WILLR",  # モメンタム系
            "NATR", "TRANGE",  # ボラティリティ系
            "AD", "ADOSC", "OBV",  # 出来高系
            "AVGPRICE", "TYPPRICE",  # 価格変換系
            "HT_DCPERIOD", "HT_TRENDMODE",  # サイクル系
        ]
        
        for indicator in new_indicators:
            assert indicator in registered_indicators, f"{indicator} が登録されていません"
        
        # 合計で25個以上のインジケーターが登録されていることを確認（段階的拡張）
        assert len(registered_indicators) >= 25, f"登録されたインジケーター数: {len(registered_indicators)}"

        # 登録されたインジケーターの一覧を出力（デバッグ用）
        print(f"登録されたインジケーター ({len(registered_indicators)}個): {sorted(registered_indicators)}")
