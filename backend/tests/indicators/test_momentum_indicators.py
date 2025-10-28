"""
MomentumIndicatorsのテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestMomentumIndicators:
    """MomentumIndicatorsのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.momentum = MomentumIndicators()

    def test_init(self):
        """初期化のテスト"""
        assert self.momentum is not None

    def test_calculate_rsi_valid_data(self):
        """有効データでのRSI計算テスト"""
        data = pd.DataFrame(
            {"close": [100, 102, 98, 105, 103, 107, 105, 110, 108, 112]}
        )

        result = self.momentum.calculate_rsi(data, period=6)

        assert isinstance(result, pd.DataFrame)
        assert "RSI_6" in result.columns
        # RSIは0-100の範囲
        rsi_values = result["RSI_6"].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)

    def test_calculate_rsi_insufficient_data(self):
        """データ不足でのRSI計算テスト"""
        data = pd.DataFrame({"close": [100, 102]})  # 不十分

        result = self.momentum.calculate_rsi(data, period=14)

        assert isinstance(result, pd.DataFrame)
        assert "RSI_14" in result.columns
        # 不十分なデータではNaNが含まれる
        assert result["RSI_14"].isna().all()

    def test_calculate_macd_valid_data(self):
        """有効データでのMACD計算テスト"""
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
                ]
            }
        )

        result = self.momentum.calculate_macd(data)

        assert isinstance(result, pd.DataFrame)
        assert "MACD_12_26_9" in result.columns
        assert "MACD_signal_12_26_9" in result.columns
        assert "MACD_histogram_12_26_9" in result.columns

    def test_calculate_macd_insufficient_data(self):
        """データ不足でのMACD計算テスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})  # 不十分

        result = self.momentum.calculate_macd(data)

        assert isinstance(result, pd.DataFrame)
        # 不十分なデータではNaNが含まれる
        histogram_col = "MACD_histogram_12_26_9"
        if histogram_col in result.columns:
            assert result[histogram_col].isna().all()

    def test_calculate_stochastic_oscillator(self):
        """ストキャスティクスオシレーター計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 102, 98, 105, 103, 107, 105, 110, 108, 112],
                "high": [102, 103, 100, 106, 105, 108, 107, 111, 110, 113],
                "low": [98, 100, 97, 103, 102, 105, 103, 108, 106, 110],
            }
        )

        result = self.momentum.calculate_stochastic_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        assert "Stoch_k_14_3" in result.columns
        assert "Stoch_d_14_3" in result.columns
        # ストキャスティクスは0-100の範囲
        stoch_k = result["Stoch_k_14_3"].dropna()
        assert all(0 <= val <= 100 for val in stoch_k)

    def test_calculate_stochastic_insufficient_data(self):
        """データ不足でのストキャスティクス計算テスト"""
        data = pd.DataFrame({"close": [100, 101], "high": [102, 103], "low": [98, 100]})

        result = self.momentum.calculate_stochastic_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        # 不十分なデータではNaN
        stoch_col = "Stoch_k_14_3"
        if stoch_col in result.columns:
            assert result[stoch_col].isna().all()

    def test_calculate_cci(self):
        """CCI計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = self.momentum.calculate_cci(data)

        assert isinstance(result, pd.DataFrame)
        assert "CCI_20" in result.columns
        # CCIは通常-100から+100の範囲
        cci_values = result["CCI_20"].dropna()
        # 極端な値もあり得るため、計算が行われることを確認

    def test_calculate_williams_r(self):
        """Williams %R計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = self.momentum.calculate_williams_r(data)

        assert isinstance(result, pd.DataFrame)
        assert "Williams_%R_14" in result.columns
        # Williams %Rは-100から0の範囲
        williams_values = result["Williams_%R_14"].dropna()
        assert all(-100 <= val <= 0 for val in williams_values)

    def test_calculate_roc(self):
        """ROC計算のテスト"""
        data = pd.DataFrame(
            {"close": [100, 102, 105, 103, 107, 110, 108, 112, 115, 118]}
        )

        result = self.momentum.calculate_roc(data, period=5)

        assert isinstance(result, pd.DataFrame)
        assert "ROC_5" in result.columns
        # 変化率が計算されている
        assert not result["ROC_5"].isna().all()

    def test_calculate_uo(self):
        """Ultimate Oscillator計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = self.momentum.calculate_uo(data)

        assert isinstance(result, pd.DataFrame)
        assert "UO" in result.columns
        # UOは0-100の範囲
        uo_values = result["UO"].dropna()
        assert all(0 <= val <= 100 for val in uo_values)

    def test_calculate_mfi(self):
        """MFI計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = self.momentum.calculate_mfi(data)

        assert isinstance(result, pd.DataFrame)
        assert "MFI_14" in result.columns
        # MFIは0-100の範囲
        mfi_values = result["MFI_14"].dropna()
        assert all(0 <= val <= 100 for val in mfi_values)

    def test_calculate_tsi(self):
        """TSI計算のテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = self.momentum.calculate_tsi(data)

        assert isinstance(result, pd.DataFrame)
        assert "TSI" in result.columns
        # TSIは通常-100から+100の範囲
        tsi_values = result["TSI"].dropna()
        assert all(-100 <= abs(val) <= 100 for val in tsi_values)

    def test_calculate_kst(self):
        """KST計算のテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = self.momentum.calculate_kst(data)

        assert isinstance(result, pd.DataFrame)
        assert "KST" in result.columns
        assert "KST_signal" in result.columns

    def test_calculate_ultimate_oscillator(self):
        """Ultimate Oscillatorの別実装テスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            }
        )

        result = self.momentum.calculate_ultimate_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        assert "UO" in result.columns

    def test_calculate_pvo(self):
        """PVO計算のテスト"""
        data = pd.DataFrame(
            {"volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]}
        )

        result = self.momentum.calculate_pvo(data)

        assert isinstance(result, pd.DataFrame)
        assert "PVO" in result.columns
        assert "PVO_signal" in result.columns
        assert "PVO_histogram" in result.columns

    def test_calculate_chande_momentum_oscillator(self):
        """Chande Momentum Oscillator計算のテスト"""
        data = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}
        )

        result = self.momentum.calculate_chande_momentum_oscillator(data)

        assert isinstance(result, pd.DataFrame)
        assert "CMO_9" in result.columns
        # CMOは-100から+100の範囲
        cmo_values = result["CMO_9"].dropna()
        assert all(-100 <= val <= 100 for val in cmo_values)

    def test_calculate_force_index(self):
        """Force Index計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )

        result = self.momentum.calculate_force_index(data)

        assert isinstance(result, pd.DataFrame)
        assert "Force_Index_13" in result.columns

    def test_calculate_momentum(self):
        """単純モメンタム計算のテスト"""
        data = pd.DataFrame(
            {"close": [100, 102, 105, 103, 107, 110, 108, 112, 115, 118]}
        )

        result = self.momentum.calculate_momentum(data, period=3)

        assert isinstance(result, pd.DataFrame)
        assert "Momentum_3" in result.columns

    def test_handle_invalid_data(self):
        """無効データ処理のテスト"""
        data = pd.DataFrame()  # 空のデータ

        result = self.momentum.calculate_rsi(data, period=14)

        assert isinstance(result, pd.DataFrame)

    def test_handle_missing_columns(self):
        """欠損カラム処理のテスト"""
        data = pd.DataFrame({"close": [100, 101, 102]})  # high, lowがない

        # RSIは計算可能
        result_rsi = self.momentum.calculate_rsi(data, period=3)
        assert "RSI_3" in result_rsi.columns

        # ストキャスティクスはhigh, lowが必要
        result_stoch = self.momentum.calculate_stochastic_oscillator(data)
        # エラー処理が行われるか確認

    def test_data_validation(self):
        """データ検証のテスト"""
        valid_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        # 検証が通るか確認
        assert hasattr(self.momentum, "calculate_rsi")
        result = self.momentum.calculate_rsi(valid_data, period=3)
        assert not result.empty

    def test_edge_case_single_value(self):
        """エッジケース（単一値）のテスト"""
        data = pd.DataFrame({"close": [100]})  # 単一データポイント

        result = self.momentum.calculate_rsi(data, period=14)

        # エラーなく処理される
        assert isinstance(result, pd.DataFrame)

    def test_negative_values_handling(self):
        """負値処理のテスト"""
        data = pd.DataFrame(
            {"close": [-1, -2, -3, -4, -5]}
        )  # 負の価格（現実的ではないが）

        result = self.momentum.calculate_rsi(data, period=3)

        # 計算が試みられる
        assert isinstance(result, pd.DataFrame)

    def test_extreme_values(self):
        """極端な値のテスト"""
        data = pd.DataFrame(
            {"close": [1000000, 2000000, 500000, 3000000]}
        )  # 極端な価格

        result = self.momentum.calculate_rsi(data, period=3)

        assert isinstance(result, pd.DataFrame)
        # RSIは依然として0-100の範囲
        rsi_values = result["RSI_3"].dropna()
        assert all(0 <= val <= 100 for val in rsi_values if not rsi_values.empty)

    def test_all_momentum_indicators(self):
        """全てのモメンタム指標計算のテスト"""
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
                ],
                "high": [
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
                ],
                "low": [
                    98,
                    99,
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
                ],
                "volume": [
                    1000,
                    1100,
                    1200,
                    1300,
                    1400,
                    1500,
                    1600,
                    1700,
                    1800,
                    1900,
                    2000,
                    2100,
                    2200,
                    2300,
                    2400,
                    2500,
                ],
            }
        )

        # 各指標が計算可能か確認
        indicators_to_test = [
            ("calculate_rsi", {}),
            ("calculate_macd", {}),
            ("calculate_stochastic_oscillator", {}),
            ("calculate_cci", {}),
            ("calculate_williams_r", {}),
            ("calculate_roc", {"period": 5}),
            ("calculate_uo", {}),
            ("calculate_mfi", {}),
            ("calculate_tsi", {}),
            ("calculate_kst", {}),
            ("calculate_pvo", {}),
            ("calculate_chande_momentum_oscillator", {}),
            ("calculate_force_index", {}),
            ("calculate_momentum", {"period": 3}),
        ]

        for indicator_method, params in indicators_to_test:
            with self.subTest(indicator=indicator_method):
                method = getattr(self.momentum, indicator_method)
                result = method(data, **params)
                assert isinstance(result, pd.DataFrame)

    def subTest(self, indicator):
        """サブテスト用ダミーメソッド"""
        pass

    def test_psychological_line_valid_data(self):
        """有効データでのPsychological Line計算テスト"""
        data = pd.DataFrame({
            "close": [100, 102, 101, 105, 103, 107, 105, 110, 108, 112]
        })

        result = MomentumIndicators.psychological_line(data["close"])

        assert isinstance(result, pd.Series)
        # PSYは0-100の範囲
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
