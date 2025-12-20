"""
TrendIndicatorsのテスト

pandas-taのtrendカテゴリに対応する指標のテスト。
SAR, ADX, AROON, VORTEXなどのトレンド方向系指標を含む。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.trend import TrendIndicators


class TestTrendIndicators:
    """TrendIndicatorsのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        pass

    def test_init(self):
        """初期化のテスト"""
        # クラス自体を直接テスト
        assert TrendIndicators is not None

    def test_calculate_sar_valid_data(self):
        """有効データでのSAR計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = TrendIndicators.sar(data["high"], data["low"], af=0.02, max_af=0.2)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # SARは高値と安値の範囲内
        assert result.dropna().min() >= 98
        assert result.dropna().max() <= 111

    def test_calculate_sar_insufficient_data(self):
        """不十分なデータでのSAR計算テスト"""
        data = pd.DataFrame({"high": [102], "low": [98]})

        result = TrendIndicators.sar(data["high"], data["low"])

        assert isinstance(result, pd.Series)
        # 1つのデータではNaN
        assert result.isna().all()

    def test_calculate_amat_valid_data(self):
        """有効データでのAMAT計算テスト"""
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                ]
            }
        )

        # AMATは十分なデータがあれば計算可能
        result = TrendIndicators.amat(data["close"], fast=3, slow=10, signal=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_amat_insufficient_data(self):
        """不十分なデータでのAMAT計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})  # 不十分

        result = TrendIndicators.amat(data["close"], fast=3, slow=30, signal=10)
        assert result.isna().all()

    def test_calculate_dpo_valid_data(self):
        """有効データでのDPO計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.dpo(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_vortex_valid_data(self):
        """有効データでのVORTEX計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = TrendIndicators.vortex(
            data["high"], data["low"], data["close"], length=5
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        # 両方のシリーズを確認
        assert len(result[0]) == len(data)
        assert len(result[1]) == len(data)

    def test_calculate_vortex_invalid_drift(self):
        """無効なdriftパラメータのVORTEXテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106],
                "low": [98, 99, 100, 101, 102],
                "close": [100, 101, 102, 103, 104],
            }
        )

        with pytest.raises(ValueError, match="drift must be positive"):
            TrendIndicators.vortex(
                data["high"], data["low"], data["close"], length=3, drift=0
            )

    def test_calculate_adx_valid_data(self):
        """有効データでのADX計算テスト"""
        np.random.seed(42)
        length = 50
        close = 100 + np.cumsum(np.random.randn(length))
        high = close + abs(np.random.randn(length))
        low = close - abs(np.random.randn(length))

        data = pd.DataFrame({"high": high, "low": low, "close": close})

        result = TrendIndicators.adx(
            data["high"], data["low"], data["close"], length=14
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        # 全てのシリーズの長さをチェック
        for series in result:
            assert len(series) == len(data)

    def test_calculate_adx_insufficient_data(self):
        """不十分なデータでのADX計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "close": [100, 101, 102],
            }
        )

        result = TrendIndicators.adx(
            data["high"], data["low"], data["close"], length=14
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        # 不十分なデータではNaN
        for series in result:
            assert series.isna().all()

    def test_calculate_aroon_valid_data(self):
        """有効データでのAROON計算テスト"""
        np.random.seed(42)
        length = 50
        close = 100 + np.cumsum(np.random.randn(length))
        high = close + abs(np.random.randn(length))
        low = close - abs(np.random.randn(length))

        data = pd.DataFrame({"high": high, "low": low})

        result = TrendIndicators.aroon(data["high"], data["low"], length=14)

        assert isinstance(result, tuple)
        assert len(result) == 3  # up, down, osc
        # 全てのシリーズの長さをチェック
        for series in result:
            assert len(series) == len(data)

    def test_calculate_aroon_insufficient_data(self):
        """不十分なデータでのAROON計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104],
                "low": [98, 99, 100],
            }
        )

        result = TrendIndicators.aroon(data["high"], data["low"], length=14)

        assert isinstance(result, tuple)
        assert len(result) == 3
        # 不十分なデータではNaN
        for series in result:
            assert series.isna().all()

    def test_calculate_chop_valid_data(self):
        """有効データでのCHOP計算テスト"""
        np.random.seed(42)
        length = 50
        close = 100 + np.cumsum(np.random.randn(length))
        high = close + abs(np.random.randn(length))
        low = close - abs(np.random.randn(length))

        data = pd.DataFrame({"high": high, "low": low, "close": close})

        result = TrendIndicators.chop(
            data["high"], data["low"], data["close"], length=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_chop_insufficient_data(self):
        """不十分なデータでのCHOP計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "close": [100, 101, 102],
            }
        )

        result = TrendIndicators.chop(
            data["high"], data["low"], data["close"], length=14
        )

        # 不十分なデータでは計算できない
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_calculate_vhf_valid_data(self):
        """有効データでのVHF計算テスト"""
        np.random.seed(42)
        length = 100
        close = 100 + np.cumsum(np.random.randn(length))

        data = pd.DataFrame({"close": close})

        result = TrendIndicators.vhf(data["close"], length=28)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_vhf_insufficient_data(self):
        """不十分なデータでのVHF計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})

        result = TrendIndicators.vhf(data["close"], length=28)

        # 不十分なデータではNaN
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_handle_invalid_data_types(self):
        """無効なデータ型のテスト"""
        # 数値以外のデータ
        data = pd.DataFrame({"close": ["invalid", "data"]})

        # DPOは無効なデータでもエラーではなくNaNを返す場合がある
        result = TrendIndicators.dpo(data["close"], length=5)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_handle_mismatched_lengths(self):
        """長さ不一致のテスト"""
        data_high = pd.Series([102, 103, 104])
        data_low = pd.Series([98, 99])  # 長さが異なる
        data_close = pd.Series([100, 101, 102])

        with pytest.raises(ValueError, match="All series must have the same length"):
            TrendIndicators.vortex(data_high, data_low, data_close, length=3)

    def test_edge_case_empty_data(self):
        """空データのテスト"""
        data = pd.DataFrame({"high": [], "low": []})

        result = TrendIndicators.sar(data["high"], data["low"])

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_edge_case_single_value(self):
        """単一値のテスト"""
        data = pd.DataFrame({"close": [100]})

        result = TrendIndicators.dpo(data["close"], length=5)

        # 単一値では計算できない
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_all_trend_indicators_with_sample_data(self):
        """代表的なトレンド指標の統合テスト"""
        # 十分な長さのテストデータ
        np.random.seed(42)
        base = 1000
        trend = np.linspace(0, 100, 100)
        noise = np.random.normal(0, 10, 100)
        close_prices = base + trend + noise

        data = pd.DataFrame(
            {
                "close": close_prices,
                "high": close_prices + np.random.normal(0, 5, 100),
                "low": close_prices - np.random.normal(0, 5, 100),
            }
        )

        # SAR
        sar_result = TrendIndicators.sar(data["high"], data["low"])
        assert isinstance(sar_result, pd.Series)
        assert len(sar_result) == len(data)

        # AMAT
        amat_result = TrendIndicators.amat(data["close"], fast=3, slow=10, signal=5)
        assert isinstance(amat_result, pd.Series)
        assert len(amat_result) == len(data)

        # DPO
        dpo_result = TrendIndicators.dpo(data["close"], length=20)
        assert isinstance(dpo_result, pd.Series)
        assert len(dpo_result) == len(data)

        # VORTEX
        vortex_result = TrendIndicators.vortex(
            data["high"], data["low"], data["close"], length=14
        )
        assert isinstance(vortex_result, tuple)
        assert len(vortex_result) == 2

        # ADX
        adx_result = TrendIndicators.adx(
            data["high"], data["low"], data["close"], length=14
        )
        assert isinstance(adx_result, tuple)
        assert len(adx_result) == 3

        # AROON
        aroon_result = TrendIndicators.aroon(data["high"], data["low"], length=14)
        assert isinstance(aroon_result, tuple)
        assert len(aroon_result) == 3

        # CHOP
        chop_result = TrendIndicators.chop(
            data["high"], data["low"], data["close"], length=14
        )
        assert isinstance(chop_result, pd.Series)
        assert len(chop_result) == len(data)

        # VHF
        vhf_result = TrendIndicators.vhf(data["close"], length=28)
        assert isinstance(vhf_result, pd.Series)
        assert len(vhf_result) == len(data)

    @staticmethod
    def subTest(indicator):
        """サブテスト用ダミーメソッド"""
        return indicator


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])
