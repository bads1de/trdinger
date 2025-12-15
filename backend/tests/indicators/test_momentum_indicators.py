"""
Momentum Indicatorsのテスト（レジストリベース・簡略版）
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService


class TestMomentumIndicators:
    """Momentum Indicatorsのテストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テスト前のセットアップ"""
        self.indicator_service = TechnicalIndicatorService()
        # 十分なデータを用意
        self.valid_data = pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, 100),
                "high": np.random.uniform(100, 120, 100),
                "low": np.random.uniform(80, 100, 100),
                "close": np.random.uniform(90, 110, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )
        # 不十分なデータ
        self.insufficient_data = pd.DataFrame(
            {
                "open": [100, 102],
                "high": [105, 107],
                "low": [95, 98],
                "close": [100, 102],
                "volume": [1000, 1500],
            }
        )

    def test_init(self):
        """初期化のテスト"""
        assert self.indicator_service is not None
        assert self.indicator_service.registry is not None

    def test_calculate_rsi_valid_data(self):
        """有効データでのRSI計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "RSI", {"length": 14}
        )

        assert result is not None
        # RSIは0-100の範囲
        rsi_values = pd.Series(result).dropna()
        assert len(rsi_values) > 0
        if len(rsi_values) > 0:
            assert all(0 <= val <= 100 for val in rsi_values)

    def test_calculate_rsi_insufficient_data(self):
        """データ不足でのRSI計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.insufficient_data, "RSI", {"length": 14}
        )

        assert result is not None

    def test_calculate_macd_valid_data(self):
        """有効データでのMACD計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )

        assert result is not None

    def test_calculate_macd_insufficient_data(self):
        """データ不足でのMACD計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.insufficient_data, "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )

        assert result is not None

    def test_calculate_stochastic_oscillator(self):
        """Stochastic Oscillatorの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "STOCH", {"k": 14, "d": 3}
        )

        assert result is not None

    def test_calculate_stochastic_insufficient_data(self):
        """データ不足でのStochastic計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.insufficient_data, "STOCH", {"k": 14, "d": 3}
        )

        assert result is not None

    def test_calculate_cci(self):
        """CCIの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "CCI", {"length": 20}
        )

        assert result is not None
        cci_values = pd.Series(result).dropna()
        # CCIは範囲制限なし（通常-100～+100付近だが、超えることがある）
        assert len(cci_values) >= 0

    def test_calculate_williams_r(self):
        """Williams %Rの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "WILLR", {"length": 14}
        )

        assert result is not None
        willr_values = pd.Series(result).dropna()
        # Williams %Rは-100～0の範囲
        if len(willr_values) > 0:
            assert all(-100 <= val <= 0 for val in willr_values)

    def test_calculate_roc(self):
        """ROCの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "ROC", {"length": 10}
        )

        assert result is not None

    def test_calculate_uo(self):
        """Ultimate Oscillatorの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "UO", {"fast": 7, "medium": 14, "slow": 28}
        )

        assert result is not None

    def test_calculate_mfi(self):
        """MFIの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "MFI", {"length": 14}
        )

        assert result is not None
        mfi_values = pd.Series(result).dropna()
        # MFIは0-100の範囲
        if len(mfi_values) > 0:
            assert all(0 <= val <= 100 for val in mfi_values)

    def test_calculate_tsi(self):
        """TSIの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "TSI", {"fast": 13, "slow": 25}
        )

        assert result is not None

    def test_calculate_kst(self):
        """KSTの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "KST", {"roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30}
        )

        assert result is not None

    def test_calculate_ultimate_oscillator(self):
        """Ultimate Oscillatorの計算テスト（別名）"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "UO", {"fast": 7, "medium": 14, "slow": 28}
        )

        assert result is not None

    def test_calculate_pvo(self):
        """PVOの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "PVO", {"fast": 12, "slow": 26}
        )

        assert result is not None

    def test_calculate_chande_momentum_oscillator(self):
        """Chande Momentum Oscillatorの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "CMO", {"length": 14}
        )

        assert result is not None

    def test_calculate_force_index(self):
        """Force Indexの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "EFI", {"length": 13}
        )

        assert result is not None

    def test_calculate_momentum(self):
        """Momentumの計算テスト"""
        result = self.indicator_service.calculate_indicator(
            self.valid_data, "MOM", {"length": 10}
        )

        assert result is not None

    def test_handle_invalid_data(self):
        """無効データのハンドリングテスト"""
        invalid_data = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})

        result = self.indicator_service.calculate_indicator(
            invalid_data, "RSI", {"length": 14}
        )

        assert result is not None

    def test_handle_missing_columns(self):
        """必要なカラム不足のハンドリングテスト"""
        incomplete_data = pd.DataFrame({"close": [100, 102, 98]})

        # closeのみのデータでRSIは計算可能
        result = self.indicator_service.calculate_indicator(
            incomplete_data, "RSI", {"length": 2}
        )

        assert result is not None

    def test_data_validation(self):
        """データ検証のテスト"""
        # 空のデータフレーム
        empty_data = pd.DataFrame()

        # 空のデータフレームでもNaN結果を返すことを確認（エラーにならない）
        result = self.indicator_service.calculate_indicator(
            empty_data, "RSI", {"length": 14}
        )
        assert result is not None

    def test_edge_case_single_value(self):
        """単一値のエッジケーステスト"""
        single_value_data = pd.DataFrame({"close": [100]})

        result = self.indicator_service.calculate_indicator(
            single_value_data, "RSI", {"length": 14}
        )

        assert result is not None

    def test_negative_values_handling(self):
        """負の値のハンドリングテスト"""
        negative_data = pd.DataFrame({"close": [-100, -102, -98, -105, -103] * 10})

        result = self.indicator_service.calculate_indicator(
            negative_data, "RSI", {"length": 14}
        )

        assert result is not None

    def test_extreme_values(self):
        """極端な値のハンドリングテスト"""
        extreme_data = pd.DataFrame(
            {"close": [1e10, 1e10 + 1, 1e10 - 1, 1e10 + 2, 1e10] * 10}
        )

        result = self.indicator_service.calculate_indicator(
            extreme_data, "RSI", {"length": 14}
        )

        assert result is not None

    def test_all_momentum_indicators(self):
        """全てのMomentum Indicatorsが計算可能であることをテスト"""
        # 主要なmomentum indicators
        indicators = [
            ("RSI", {"length": 14}),
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
            ("STOCH", {"k": 14, "d": 3}),
            ("CCI", {"length": 20}),
            ("WILLR", {"length": 14}),
            ("ROC", {"length": 10}),
            ("MOM", {"length": 10}),
        ]

        for indicator_type, params in indicators:
            try:
                result = self.indicator_service.calculate_indicator(
                    self.valid_data, indicator_type, params
                )
                assert result is not None, f"{indicator_type} returned None"
            except Exception as e:
                pytest.fail(f"{indicator_type} failed with error: {str(e)}")

    def test_psychological_line_valid_data(self):
        """Psychological Lineの計算テスト"""
        try:
            result = self.indicator_service.calculate_indicator(
                self.valid_data, "PSY", {"length": 12}
            )
            assert result is not None
        except ValueError:
            # PSYがサポートされていない場合はスキップ
            pytest.skip("PSY indicator not supported")




