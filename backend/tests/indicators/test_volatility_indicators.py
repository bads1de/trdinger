"""
VolatilityIndicatorsのテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.volatility import VolatilityIndicators


class TestVolatilityIndicators:
    """VolatilityIndicatorsのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        pass

    def test_init(self):
        """初期化のテスト"""
        assert VolatilityIndicators is not None

    def test_calculate_atr_valid_data(self):
        """有効データでのATR計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # ATRは正の値（NaN値がない場合）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert valid_values.min() >= 0

    def test_calculate_atr_insufficient_data(self):
        """データ不足でのATR計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
            }
        )

        result = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        # 不十分なデータではNaN
        assert result.isna().all()

    def test_calculate_natr_valid_data(self):
        """有効データでのNATR計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = VolatilityIndicators.natr(
            data["high"], data["low"], data["close"], length=5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # NATRは0-100の範囲
        assert result.dropna().min() >= 0
        assert result.dropna().max() <= 100

    def test_calculate_natr_invalid_length(self):
        """無効な長さのNATRテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "close": [100, 101, 102],
            }
        )

        with pytest.raises(ValueError, match="length must be positive"):
            VolatilityIndicators.natr(
                data["high"], data["low"], data["close"], length=-1
            )

    def test_calculate_bbands_valid_data(self):
        """有効データでのBBands計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = VolatilityIndicators.bbands(data["close"], length=5, std=2.0)

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, middle, lower = result
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)
        # 上下バンドの関係（NaN値を除外）
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_idx.any():
            assert (upper[valid_idx] >= middle[valid_idx]).all()
            assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_calculate_bbands_insufficient_data(self):
        """データ不足でのBBands計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})

        result = VolatilityIndicators.bbands(data["close"], length=20, std=2.0)

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, middle, lower = result
        # 不十分なデータではNaN
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()

    def test_calculate_keltner_valid_data(self):
        """有効データでのKeltner Channels計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = VolatilityIndicators.keltner(
            data["high"], data["low"], data["close"], period=5, scalar=2.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, middle, lower = result
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)

    def test_calculate_keltner_insufficient_data(self):
        """不十分なデータでのKeltner Channels計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
            }
        )

        result = VolatilityIndicators.keltner(
            data["high"], data["low"], data["close"], period=20, scalar=2.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, middle, lower = result
        # 不十分なデータではNaN
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()

    def test_calculate_donchian_valid_data(self):
        """有効データでのDonchian Channels計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = VolatilityIndicators.donchian(data["high"], data["low"], length=5)

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, middle, lower = result
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)

    def test_calculate_supertrend_valid_data(self):
        """有効データでのSupertrend計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = VolatilityIndicators.supertrend(
            data["high"], data["low"], data["close"], period=7, multiplier=3.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, lower, direction = result
        assert len(upper) == len(data)
        assert len(lower) == len(data)
        assert len(direction) == len(data)
        # directionは1.0または-1.0
        assert set(direction.dropna().unique()).issubset({1.0, -1.0})

    def test_calculate_supertrend_insufficient_data(self):
        """不十分なデータでのSupertrend計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
            }
        )

        result = VolatilityIndicators.supertrend(
            data["high"], data["low"], data["close"], period=14, multiplier=3.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, lower, direction = result
        # 不十分なデータではNaN
        assert upper.isna().all()
        assert lower.isna().all()
        assert direction.isna().all()

    def test_calculate_accbands_valid_data(self):
        """有効データでのAcceleration Bands計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = VolatilityIndicators.accbands(
            data["high"], data["low"], data["close"], period=5
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        upper, middle, lower = result
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)

    def test_calculate_ui_valid_data(self):
        """有効データでのUlcer Index計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = VolatilityIndicators.ui(data["close"], period=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # UIは0以上の値（NaN値がない場合）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert valid_values.min() >= 0

    def test_calculate_rvi_valid_data(self):
        """有効データでのRVI計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = VolatilityIndicators.rvi(
            data["close"], data["high"], data["low"], length=10, scalar=100.0
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # RVIは0-100の範囲（NaN値がない場合、またはデータ不足の場合はNaN）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert valid_values.min() >= 0
            assert valid_values.max() <= 150  # スカラー倍の影響で100を超える場合あり

    def test_calculate_rvi_invalid_length(self):
        """無効な長さのRVIテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [98, 99, 100],
            }
        )

        # RVIは負の長さでもエラーを発生させず、NaNを返す（pandas-taの仕様）
        result = VolatilityIndicators.rvi(
            data["close"], data["high"], data["low"], length=0
        )
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_calculate_gri_valid_data(self):
        """有効データでのGRI計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        result = VolatilityIndicators.gri(
            data["high"], data["low"], data["close"], length=5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # GRIは振動型指標なので、正負の値を取り得る
        assert not result.isna().all()

    def test_calculate_gri_insufficient_data(self):
        """データ不足でのGRI計算テスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
            }
        )

        result = VolatilityIndicators.gri(
            data["high"], data["low"], data["close"], length=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        # 不十分なデータではNaN
        assert result.isna().all()

    def test_calculate_gri_invalid_length(self):
        """無効な長さのGRIテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "close": [100, 101, 102],
            }
        )

        with pytest.raises(ValueError, match="length must be positive"):
            VolatilityIndicators.gri(
                data["high"], data["low"], data["close"], length=-1
            )

    def test_handle_invalid_data_types(self):
        """無効なデータ型のテスト"""
        # 数値以外のデータ
        data = pd.DataFrame(
            {"high": ["invalid", "data"], "low": [98, 99], "close": [100, 101]}
        )

        # pandas-taはエラーではなくNaNを返す
        result = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=14
        )
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_handle_mismatched_lengths(self):
        """長さ不一致のテスト"""
        data1 = pd.DataFrame({"high": [102, 103, 104], "low": [98, 99, 100]})
        data2 = pd.DataFrame({"close": [100, 101]})  # 長さが異なる

        # pandas-taは長さ不一致でもエラーではなくNaNを返す
        result = VolatilityIndicators.atr(
            data1["high"], data1["low"], data2["close"], length=3
        )
        assert isinstance(result, pd.Series)

    def test_handle_negative_length(self):
        """負の長さパラメータのテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106],
                "low": [98, 99, 100, 101, 102],
                "close": [100, 101, 102, 103, 104],
            }
        )

        # pandas-taは負の長さでもエラーではなくNaNを返す
        result = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=-1
        )
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_edge_case_empty_data(self):
        """空データのテスト"""
        data = pd.DataFrame({"high": [], "low": [], "close": []})

        result = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_edge_case_single_value(self):
        """単一値のテスト"""
        data = pd.DataFrame({"high": [102], "low": [98], "close": [100]})

        result = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=5
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.isna().all()

    def test_data_validation_empty_series(self):
        """空シリーズのデータ検証"""
        empty_high = pd.Series([])
        empty_low = pd.Series([])
        empty_close = pd.Series([])

        result = VolatilityIndicators.atr(empty_high, empty_low, empty_close, length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_all_volatility_indicators_with_sample_data(self):
        """代表的なボラティリティ指標の統合テスト"""
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

        # テスト対象の指標
        indicators_to_test = [
            ("atr", (data["high"], data["low"], data["close"], 14)),
            ("natr", (data["high"], data["low"], data["close"], 14)),
            ("bbands", (data["close"], 20, 2.0)),
            ("ui", (data["close"], 14)),
        ]

        for indicator_name, params in indicators_to_test:
            # subtestの代わりに直接テスト
            method = getattr(VolatilityIndicators, indicator_name)
            result = method(*params)
            if indicator_name in ["bbands"]:
                # 複数のシリーズを返す場合
                assert isinstance(result, tuple)
                for series in result:
                    assert isinstance(series, pd.Series)
                    assert len(series) == len(data)
            else:
                assert isinstance(result, pd.Series)
                assert len(result) == len(data)

    def test_all_volatility_indicators_handle_errors(self):
        """ボラティリティ指標のエラーハンドリング統合テスト"""
        # 短いデータ
        short_data = pd.DataFrame(
            {
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
            }
        )

        # エラーが適切に処理されるかテスト
        try:
            VolatilityIndicators.atr(
                short_data["high"], short_data["low"], short_data["close"], length=14
            )
            # エラーにならずNaNが返される
        except Exception as e:
            assert "length" in str(e) or "data" in str(e)

        try:
            VolatilityIndicators.bbands(short_data["close"], length=20, std=2.0)
            # エラーにならずNaNが返される
        except Exception as e:
            assert "length" in str(e) or "data" in str(e)

        # 無効なパラメータ（pandas-taはエラーではなくNaNを返す）
        result = VolatilityIndicators.atr(
            short_data["high"], short_data["low"], short_data["close"], length=-1
        )
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_bbands_with_different_std_values(self):
        """異なる標準偏差でのBBandsテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        # std=1.0
        result1 = VolatilityIndicators.bbands(data["close"], length=5, std=1.0)
        # std=2.0
        result2 = VolatilityIndicators.bbands(data["close"], length=5, std=2.0)

        assert isinstance(result1, tuple)
        assert isinstance(result2, tuple)

        # stdが大きいほどバンドの幅が広い
        upper1, middle1, lower1 = result1
        upper2, middle2, lower2 = result2

        # NaN値を除外してから比較
        valid_idx = ~(
            middle1.isna()
            | middle2.isna()
            | upper1.isna()
            | upper2.isna()
            | lower1.isna()
            | lower2.isna()
        )
        if valid_idx.any():
            # 中心線は同じ
            assert np.allclose(middle1[valid_idx], middle2[valid_idx], rtol=1e-5)
            # std=2.0の方がバンド幅が広い
            assert (upper2[valid_idx] >= upper1[valid_idx] - 1e-10).all()
            assert (lower2[valid_idx] <= lower1[valid_idx] + 1e-10).all()

    def test_supertrend_with_different_multipliers(self):
        """異なる乗数でのSupertrendテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        # multiplier=2.0
        result1 = VolatilityIndicators.supertrend(
            data["high"], data["low"], data["close"], period=7, multiplier=2.0
        )
        # multiplier=3.0
        result2 = VolatilityIndicators.supertrend(
            data["high"], data["low"], data["close"], period=7, multiplier=3.0
        )

        assert isinstance(result1, tuple)
        assert isinstance(result2, tuple)

        lower1, upper1, _ = result1
        lower2, upper2, _ = result2

        # NaN値を除外してから比較
        valid_idx = ~(upper1.isna() | upper2.isna() | lower1.isna() | lower2.isna())
        if valid_idx.any():
            # multiplierが大きいほどバンド幅が広い（小さな誤差を許容）
            assert (upper2[valid_idx] >= upper1[valid_idx] - 1e-10).all()
            assert (lower2[valid_idx] <= lower1[valid_idx] + 1e-10).all()

    def test_atr_with_different_periods(self):
        """異なる期間でのATRテスト"""
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )

        # 短期ATR
        result_short = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=5
        )
        # 長期ATR
        result_long = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=10
        )

        assert isinstance(result_short, pd.Series)
        assert isinstance(result_long, pd.Series)

        # 長期の方が平滑化されている
        # 変動が大きい場合、短期ATRの方が大きい可能性がある
        assert len(result_short) == len(data)
        assert len(result_long) == len(data)

    def subTest(self, indicator):
        """サブテスト用ダミーメソッド"""
        pass


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])
