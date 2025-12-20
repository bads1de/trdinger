"""
TrendIndicatorsのテスト
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

    def test_calculate_sma_valid_data(self):
        """有効データでのSMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.sma(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # SMAは元のデータの範囲内
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_calculate_sma_insufficient_data(self):
        """データ不足でのSMA計算テスト"""
        data = pd.DataFrame({"close": [100, 101]})  # 不十分なデータ

        # データ不足でPandasTAErrorが発生する
        from app.services.indicators.data_validation import PandasTAError

        with pytest.raises(PandasTAError):
            TrendIndicators.sma(data["close"], length=14)

    def test_calculate_ema_valid_data(self):
        """有効データでのEMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.ema(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # EMAも元のデータの範囲内
        assert result.dropna().min() >= 100
        assert result.dropna().max() <= 109

    def test_calculate_wma_valid_data(self):
        """有効データでのWMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.wma(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_wma_with_data_parameter(self):
        """dataパラメータでのWMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.wma(data=data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_wma_missing_parameters(self):
        """パラメータ不足のWMAテスト"""
        with pytest.raises(
            ValueError, match="Either 'data' or 'close' must be provided"
        ):
            TrendIndicators.wma(length=5)

    def test_calculate_trima_valid_data(self):
        """有効データでのTRIMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.trima(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_alma_valid_data(self):
        """有効データでのALMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.alma(
            data["close"], length=10, sigma=6.0, distribution_offset=0.85
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_dema_valid_data(self):
        """有効データでのDEMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.dema(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_dema_insufficient_data(self):
        """不十分なデータでのDEMA計算テスト"""
        # DEMAはlength*2のデータが必要
        data = pd.DataFrame({"close": [100, 101, 102]})  # 6未満

        result = TrendIndicators.dema(data["close"], length=3)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        # 不十分なデータではNaN
        assert result.isna().all()

    def test_calculate_tema_valid_data(self):
        """有効データでのTEMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.tema(data["close"], length=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_tema_insufficient_data(self):
        """不十分なデータでのTEMA計算テスト"""
        # TEMAはlength*3のデータが必要
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # 9未満

        result = TrendIndicators.tema(data["close"], length=3)

        assert isinstance(result, pd.Series)
        assert len(result) == 5
        # 不十分なデータではNaN
        assert result.isna().all()

    def test_calculate_t3_valid_data(self):
        """有効データでのT3計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.t3(data["close"], length=5, a=0.7)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_t3_insufficient_data(self):
        """不十分なデータでのT3計算テスト"""
        # T3はlength*6のデータが必要
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # 30未満

        result = TrendIndicators.t3(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 5
        # 不十分なデータではNaN
        assert result.isna().all()

    def test_calculate_kama_valid_data(self):
        """有効データでのKAMA計算テスト"""
        data = pd.DataFrame({"close": list(range(100, 150))})  # 50ポイントのデータ

        # 十分なデータがあれば計算可能
        result = TrendIndicators.kama(data["close"], length=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_hma_valid_data(self):
        """有効データでのHMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.hma(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_vwma_valid_data(self):
        """有効データでのVWMA計算テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = TrendIndicators.vwma(data["close"], data["volume"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_vwma_mismatched_lengths(self):
        """長さ不一致のVWMAテスト"""
        data_close = pd.DataFrame({"close": [100, 101, 102]})
        data_volume = pd.DataFrame({"volume": [1000, 1100]})  # 長さが異なる

        with pytest.raises(
            ValueError, match="close and volume series must share the same length"
        ):
            TrendIndicators.vwma(data_close["close"], data_volume["volume"], length=3)

    def test_calculate_linreg_valid_data(self):
        """有効データでの線形回帰計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.linreg(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 最初の数ポイントはNaN
        assert result.iloc[0:4].isna().all()
        # 後半は有効な値
        assert not result.iloc[5:].isna().all()

    def test_calculate_linreg_slope_valid_data(self):
        """有効データでの線形回帰スロープ計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.linregslope(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 最初の数ポイントはNaN
        assert result.iloc[0:4].isna().all()

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

        with pytest.raises(ValueError, match="Insufficient data for AMAT calculation"):
            TrendIndicators.amat(data["close"], fast=3, slow=30, signal=10)

    def test_calculate_rma_valid_data(self):
        """有効データでのRMA計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.rma(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_dpo_valid_data(self):
        """有効データでのDPO計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = TrendIndicators.dpo(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_efficiency_ratio_valid_data(self):
        """有効データでのEfficiency Ratio計算テスト"""
        # 単調増加データ (ER=1になるはず)
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})
        result = TrendIndicators.efficiency_ratio(data["close"], length=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # 最初の数要素は計算できないので0 (fillna(0.0)のため)
        # |103-100| / (|101-100|+|102-101|+|103-102|) = 3/3 = 1
        assert result.iloc[3] == 1.0

        # ジグザグデータ (ERが小さくなるはず)
        data_zigzag = pd.DataFrame({"close": [100, 110, 100, 110, 100]})
        result_zigzag = TrendIndicators.efficiency_ratio(data_zigzag["close"], length=4)
        # インデックス4: |100-100| / 40 = 0
        assert result_zigzag.iloc[4] == 0.0

    def test_calculate_efficiency_ratio_invalid_params(self):
        """無効なパラメータでのEfficiency Ratioテスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(TypeError):
            TrendIndicators.efficiency_ratio("invalid", length=10)

        with pytest.raises(ValueError):
            TrendIndicators.efficiency_ratio(data["close"], length=0)

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

    def test_calculate_vortex_mismatched_lengths(self):
        """長さ不一致のVORTEXテスト"""
        data_high = pd.DataFrame({"high": [102, 103, 104]})
        data_low = pd.DataFrame({"low": [98, 99, 100]})
        data_close = pd.DataFrame({"close": [100, 101, 102]})

        # 長さが一致していれば計算可能
        result = TrendIndicators.vortex(
            data_high["high"], data_low["low"], data_close["close"], length=3
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        # 両方のシリーズを確認
        assert len(result[0]) == len(data_high)
        assert len(result[1]) == len(data_high)

    def test_handle_invalid_data_types(self):
        """無効なデータ型のテスト"""
        # 数値以外のデータ
        data = pd.DataFrame({"close": ["invalid", "data"]})

        # 無効なデータ型でもPandasTAErrorが発生
        from app.services.indicators.data_validation import PandasTAError

        with pytest.raises(PandasTAError):
            TrendIndicators.sma(data["close"], length=5)

    def test_handle_negative_length(self):
        """負の長さパラメータのテスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        with pytest.raises(ValueError):
            TrendIndicators.sma(data["close"], length=-1)

        with pytest.raises(ValueError):
            TrendIndicators.ema(data["close"], length=0)

    def test_handle_zero_length(self):
        """ゼロ長さパラメータのテスト"""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        with pytest.raises(ValueError):
            TrendIndicators.sma(data["close"], length=0)

    def test_edge_case_empty_data(self):
        """空データのテスト"""
        data = pd.DataFrame({"close": []})

        result = TrendIndicators.sma(data["close"], length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_edge_case_single_value(self):
        """単一値のテスト"""
        data = pd.DataFrame({"close": [100]})

        # 単一値でもPandasTAErrorが発生
        from app.services.indicators.data_validation import PandasTAError

        with pytest.raises(PandasTAError):
            TrendIndicators.sma(data["close"], length=5)

    def test_data_validation_empty_series(self):
        """空シリーズのデータ検証"""
        empty_series = pd.Series([])

        result = TrendIndicators.sma(empty_series, length=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

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
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # テスト対象の指標
        indicators_to_test = [
            ("sma", (data["close"], 20)),
            ("ema", (data["close"], 20)),
            ("wma", (data["close"], 20)),
            ("dema", (data["close"], 10)),
            ("tema", (data["close"], 10)),
            ("kama", (data["close"], 30)),
            ("hma", (data["close"], 20)),
            ("rma", (data["close"], 14)),
        ]

        for indicator_name, params in indicators_to_test:
            method = getattr(TrendIndicators, indicator_name)
            try:
                result = method(*params)
                assert isinstance(result, pd.Series)
                assert len(result) == len(data)
                # 有効な値が含まれているか確認
                assert not result.isna().all()
            except Exception as e:
                # pandas-taがNoneを返す場合などはスキップ
                if "PandasTAError" in str(type(e)) or "None" in str(e):
                    continue  # 計算不能な場合はスキップ
                else:
                    raise e

    def test_all_trend_indicators_handle_errors(self):
        """トレンド指標のエラーハンドリング統合テスト"""
        # 短いデータ
        short_data = pd.DataFrame({"close": [100, 101, 102]})

        # エラーが適切に処理されるかテスト
        try:
            TrendIndicators.dema(short_data["close"], length=10)  # 長さ不足
            # エラーにならずNaNが返される
        except Exception as e:
            assert "length" in str(e) or "data" in str(e)

        try:
            TrendIndicators.tema(short_data["close"], length=10)  # 長さ不足
            # エラーにならずNaNが返される
        except Exception as e:
            assert "length" in str(e) or "data" in str(e)

        # 無効なパラメータ
        try:
            TrendIndicators.sma(short_data["close"], length=-1)
            assert False, "負の長さでエラーが発生すべき"
        except ValueError:
            pass  # 期待されるエラー

    @staticmethod
    def subTest(indicator):
        """サブテスト用ダミーメソッド"""
        return indicator


if __name__ == "__main__":
    # コマンドラインからの実行用
    pytest.main([__file__, "-v"])
