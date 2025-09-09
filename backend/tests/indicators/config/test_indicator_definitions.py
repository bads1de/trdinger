"""
テストケース: テクニカル指標定義のテスト

このモジュールは、indicator_definitions.pyとその関連関数が正しく機能することを検証します。
TDDアプローチに従い、指標の登録、レジストリ機能、設定の妥当性をテストします。
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from app.services.indicators.config.indicator_definitions import (
    setup_momentum_indicators,
    setup_trend_indicators,
    setup_volatility_indicators,
    setup_volume_indicators,
    initialize_all_indicators,
)
from app.services.indicators.config.indicator_config import indicator_registry, IndicatorConfig


class TestIndicatorDefinitions:
    """テクニカル指標定義のテスト"""

    def setup_method(self):
        """各テスト前のセットアップ"""
        # レジストリをクリア
        indicator_registry.reset()

    def teardown_method(self):
        """各テスト後のクリーンアップ"""
        # レジストリをクリア
        indicator_registry.reset()

    def test_setup_momentum_indicators_registers_functions(self):
        """モメンタム指標の関数登録テスト"""
        # セットアップを実行
        setup_momentum_indicators()

        # 登録された指標を確認
        registered_names = list(indicator_registry.get_all_indicators().keys())

        expected_momentum_indicators = [
            'MAD', 'STOCH', 'AO', 'KDJ', 'RVI', 'QQE', 'SMI', 'KST',
            'RSI', 'MACD', 'CCI', 'CMO', 'UO', 'MOM', 'ADX', 'MFI',
            'WILLR', 'AROON', 'AROONOSC', 'DX', 'PLUS_DI', 'MINUS_DI',
            'ROC', 'TRIX', 'ULTOSC', 'BOP', 'APO', 'ADXR', 'STOCHRSI',
            'ROCP', 'ROCR', 'ROCR100', 'PLUS_DM', 'MINUS_DM', 'TSI',
            'RMI', 'CFO', 'CTI', 'DPO', 'CHOP', 'VORTEX'
        ]

        # 少なくとも期待される指標が登録されていることを確認
        for indicator_name in expected_momentum_indicators:
            assert indicator_name in registered_names, f"{indicator_name} should be registered"

    def test_setup_volatility_indicators_registers_functions(self):
        """ボラティリティ指標の関数登録テスト"""
        setup_volatility_indicators()

        registered_names = list(indicator_registry.get_all_indicators().keys())

        expected_volatility_indicators = [
            'ATR', 'NATR', 'TRANGE', 'BBANDS', 'ACCBANDS', 'MASSI',
            'PDIST', 'UI', 'VAR', 'CV', 'IRM', 'KELTNER', 'DONCHIAN', 'SUPERTREND'
        ]

        for indicator_name in expected_volatility_indicators:
            assert indicator_name in registered_names, f"{indicator_name} should be registered"

    def test_setup_volume_indicators_registers_functions(self):
        """出来高指標の関数登録テスト"""
        setup_volume_indicators()

        registered_names = list(indicator_registry.get_all_indicators().keys())

        expected_volume_indicators = [
            'OBV', 'AD', 'ADOSC', 'VP', 'EOM', 'PVT', 'CMF', 'AOBV', 'EFI', 'PVOL', 'PVR'
        ]

        for indicator_name in expected_volume_indicators:
            assert indicator_name in registered_names, f"{indicator_name} should be registered"

    def test_setup_trend_indicators_registers_functions(self):
        """トレンド指標の関数登録テスト"""
        setup_trend_indicators()

        registered_names = list(indicator_registry.get_all_indicators().keys())

        expected_trend_indicators = [
            'SMA', 'EMA', 'WMA', 'TRIMA', 'KAMA', 'TEMA', 'DEMA',
            'ALMA', 'T3', 'HMA', 'RMA', 'SWMA', 'ZLMA', 'MA', 'TLB', # TLB を追加
            'SAR', 'PRICE_EMA_RATIO', 'SMA_SLOPE', 'VWMA', 'FWMA'
        ]

        for indicator_name in expected_trend_indicators:
            assert indicator_name in registered_names, f"{indicator_name} should be registered"

    def test_initialize_all_indicators_comprehensive_registration(self):
        """全指標初期化の総合テスト"""
        initialize_all_indicators()

        registered_indicators = indicator_registry.get_all_indicators()

        # 最低基準数の指標が登録されていることを確認
        min_expected_count = 50  # 全カテゴリを合わせた適正数
        assert len(registered_indicators) >= min_expected_count, \
            f"Expected at least {min_expected_count} indicators, got {len(registered_indicators)}"

        # 主要カテゴリの代表指標が存在することを確認
        essential_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BBANDS', 'OBV']
        for indicator in essential_indicators:
            assert indicator in registered_indicators, \
                f"Essential indicator {indicator} should be registered"

    def test_indicator_config_creation_validation(self):
        """指標設定作成の妥当性テスト"""
        # サンプル指標設定の作成
        config = IndicatorConfig(
            indicator_name="SAMPLE_RSI",
            adapter_function=MagicMock(return_value=pd.Series()),
            required_data=["close"],
            result_type=MagicMock(),  # Mock resut type
            scale_type=MagicMock(),   # Mock scale type
            category="momentum",
        )

        # 設定項目が正しく設定されていることを確認
        assert config.indicator_name == "SAMPLE_RSI"
        assert config.category == "momentum"
        assert "close" in config.required_data

    @patch('app.services.indicators.config.indicator_definitions.logger')
    def test_error_handling_during_registration(self, mock_logger):
        """登録プロセス中のエラー処理テスト"""
        # エラーを投げる関数をモック
        def failing_indicator_function():
            raise Exception("Test function failure")

        with patch('pandas_ta.rsi', side_effect=Exception("Indicator library failure")):
            # エラー発生時にログが記録され、処理が継続されることを確認
            initialize_all_indicators()

            # レジストリが部分的に登録されていることを確認（一部の指標は登録されている）
            registered_count = len(indicator_registry.get_all_indicators())
            assert registered_count > 0, "Should have partial registration even with errors"

    def test_registry_idempotency(self):
        """レジストリの冪等性テスト（複数回の呼び出しで結果が変わらない）"""
        # 初回実行
        initialize_all_indicators()
        first_count = len(indicator_registry.get_all_indicators())

        # 2回目実行
        indicator_registry.reset()
        initialize_all_indicators()
        second_count = len(indicator_registry.get_all_indicators())

        # 結果が一致することを確認
        assert first_count == second_count, "Registry counts should be identical across runs"


class TestMomentumIndicatorsRegistration:
    """モメンタム指標登録詳細テスト"""

    def setup_method(self):
        indicator_registry.reset()

    def test_vortex_indicator_registration(self):
        """VORTEX指標の登録テスト"""
        setup_momentum_indicators()

        indicator = indicator_registry.get_indicator_config("VORTEX")
        assert indicator is not None, "VORTEX should be registered"
        assert indicator.indicator_name == "VORTEX"
        assert indicator.category == "momentum"
        assert set(indicator.required_data) == {"high", "low", "close"}

    def test_cti_indicator_registration(self):
        """CTI指標の登録テスト"""
        setup_momentum_indicators()

        indicator = indicator_registry.get_indicator_config("CTI")
        assert indicator is not None, "CTI should be registered"
        assert indicator.indicator_name == "CTI"
        assert indicator.category == "momentum"
        assert indicator.required_data == ["close"]

    def test_uo_indicator_registration(self):
        """UO指標の登録テスト"""
        setup_momentum_indicators()

        indicator = indicator_registry.get_indicator_config("UO")
        assert indicator is not None, "UO should be registered"
        assert indicator.indicator_name == "UO"
        assert indicator.category == "momentum"
        assert set(indicator.required_data) == {"high", "low", "close"}


class TestVolumeIndicatorsRegistration:
    """出来高指標登録詳細テスト"""

    def setup_method(self):
        indicator_registry.reset()

    def test_adosc_indicator_registration(self):
        """ADOSC指標の登録テスト"""
        setup_volume_indicators()

        indicator = indicator_registry.get_indicator_config("ADOSC")
        assert indicator is not None, "ADOSC should be registered"
        assert indicator.indicator_name == "ADOSC"
        assert indicator.category == "volume"
        assert set(indicator.required_data) == {"high", "low", "close", "volume"}

    def test_aobv_indicator_registration(self):
        """AOBV指標の登録テスト"""
        setup_volume_indicators()

        indicator = indicator_registry.get_indicator_config("AOBV")
        assert indicator is not None, "AOBV should be registered"
        assert indicator.indicator_name == "AOBV"
        assert indicator.category == "volume"
        assert set(indicator.required_data) == {"close", "volume"}

    def test_efi_indicator_registration(self):
        """EFI指標の登録テスト"""
        setup_volume_indicators()

        indicator = indicator_registry.get_indicator_config("EFI")
        assert indicator is not None, "EFI should be registered"
        assert indicator.indicator_name == "EFI"
        assert indicator.category == "volume"
        assert set(indicator.required_data) == {"close", "volume"}


class TestVolatilityIndicatorsRegistration:
    """ボラティリティ指標登録詳細テスト"""

    def setup_method(self):
        indicator_registry.reset()

    def test_supertrend_indicator_registration(self):
        """SUPERTREND指標の登録テスト"""
        setup_volatility_indicators()

        indicator = indicator_registry.get_indicator_config("SUPERTREND")
        assert indicator is not None, "SUPERTREND should be registered"
        assert indicator.indicator_name == "SUPERTREND"
        assert indicator.category == "volatility"
        assert set(indicator.required_data) == {"high", "low", "close"}

    def test_keltner_indicator_registration(self):
        """KELTNER指標の登録テスト"""
        setup_volatility_indicators()

        indicator = indicator_registry.get_indicator_config("KELTNER")
        assert indicator is not None, "KELTNER should be registered"
        assert indicator.indicator_name == "KELTNER"
        assert indicator.category == "volatility"
        assert set(indicator.required_data) == {"high", "low", "close"}

    def test_donchian_indicator_registration(self):
        """DONCHIAN指標の登録テスト"""
        setup_volatility_indicators()

        indicator = indicator_registry.get_indicator_config("DONCHIAN")
        assert indicator is not None, "DONCHIAN should be registered"
        assert indicator.indicator_name == "DONCHIAN"
        assert indicator.category == "volatility"
class TestPandasTaConfig:
    """PANDAS_TA_CONFIGのテスト"""

    def test_pandas_ta_config_no_duplicate_keys(self):
        """PANDAS_TA_CONFIGの重複キーなしテスト"""
        from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG

        keys = list(PANDAS_TA_CONFIG.keys())
        assert len(keys) == len(set(keys)), f"Duplicate keys found in PANDAS_TA_CONFIG: {[k for k in keys if keys.count(k) > 1]}"
class TestPandasTaConfig:
    """PANDAS_TA_CONFIGのテスト"""

    def test_pandas_ta_config_no_duplicate_keys(self):
        """PANDAS_TA_CONFIGの重複キーなしテスト"""
        from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG

        keys = list(PANDAS_TA_CONFIG.keys())
        assert len(keys) == len(set(keys)), f"Duplicate keys found in PANDAS_TA_CONFIG: {[k for k in keys if keys.count(k) > 1]}"