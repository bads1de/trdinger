"""
テスト駆動開発による指標パラメータ修正テスト

このテストは、parameter_manager.pyの修正が正しく機能することを確認します。
"""

import pytest
import pandas as pd
import numpy as np
import inspect
from unittest.mock import Mock, patch

from app.services.indicators.parameter_manager import normalize_params
from app.services.indicators.config.indicator_config import IndicatorConfig, IndicatorResultType, IndicatorScaleType


class TestIndicatorParameterFix:
    """指標パラメータ修正のテスト"""

    def test_stochf_no_length_parameter(self):
        """STOCHFがlengthパラメータを受け付けないことを確認"""
        config = self._create_mock_config("STOCHF")

        # STOCHFのパラメータ（lengthパラメータなし）
        params = {"fastk_period": 5, "fastd_period": 3}

        # normalize_paramsをテスト
        result = normalize_params("STOCHF", params, config)

        # lengthパラメータが追加されていないことを確認
        assert "length" not in result
        assert result["fastk_period"] == 5
        assert result["fastd_period"] == 3

    def test_kst_no_length_parameter(self):
        """KSTがlengthパラメータを受け付けないことを確認"""
        config = self._create_mock_config("KST")

        # KSTのパラメータ（lengthパラメータなし）
        params = {"r1": 10, "r2": 15, "r3": 20, "r4": 30}

        # normalize_paramsをテスト
        result = normalize_params("KST", params, config)

        # lengthパラメータが追加されていないことを確認
        assert "length" not in result
        assert result["r1"] == 10
        assert result["r2"] == 15
        assert result["r3"] == 20
        assert result["r4"] == 30

    def test_linearreg_period_to_length_conversion(self):
        """LINEARREGがperiodパラメータをlengthに変換することを確認"""
        config = self._create_mock_config("LINEARREG")

        # LINEARREGのパラメータ（periodパラメータをlengthに変換）
        params = {"period": 14}

        # normalize_paramsをテスト
        result = normalize_params("LINEARREG", params, config)

        # period -> length 変換が行われていることを確認
        assert "length" in result
        assert result["length"] == 14
        assert "period" not in result  # periodは削除される

    def test_macd_no_length_parameter(self):
        """MACD系指標がlengthパラメータを受け付けないことを確認"""
        for indicator in ["MACD", "MACDEXT", "MACDFIX"]:
            config = self._create_mock_config(indicator)

            # MACD系のパラメータ（lengthパラメータなし）
            params = {"fast": 12, "slow": 26, "signal": 9}

            # normalize_paramsをテスト
            result = normalize_params(indicator, params, config)

            # lengthパラメータが追加されていないことを確認
            assert "length" not in result
            assert result["fast"] == 12
            assert result["slow"] == 26
            assert result["signal"] == 9

    def test_stochrsi_length_parameter_conversion(self):
        """STOCHRSIがperiodパラメータをlengthに変換することを確認"""
        config = self._create_mock_config("STOCHRSI")

        # STOCHRSIのパラメータ（periodパラメータをlengthに変換）
        params = {"period": 14, "fastk_period": 5, "fastd_period": 3}

        # normalize_paramsをテスト
        result = normalize_params("STOCHRSI", params, config)

        # period -> length 変換が行われていることを確認
        assert "length" in result
        assert result["length"] == 14
        assert "period" not in result  # periodは削除される
        assert result["fastk_period"] == 5
        assert result["fastd_period"] == 3

    def _create_mock_config(self, indicator_name: str) -> IndicatorConfig:
        """モック設定を作成"""
        config = Mock(spec=IndicatorConfig)
        config.indicator_name = indicator_name
        config.parameters = {}
        config.param_map = {}
        return config


class TestIndicatorCalculation:
    """実際の指標計算テスト"""

    def test_stoch_function_signature(self):
        """STOCH関数のシグネチャが正しく認識されることを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 実際の関数シグネチャを取得
        sig = inspect.signature(MomentumIndicators.stoch)

        # 必要なパラメータのみが存在することを確認
        params = list(sig.parameters.keys())
        assert "high" in params
        assert "low" in params
        assert "close" in params
        assert "k" in params
        assert "d" in params
        assert "smooth_k" in params
        assert "length" not in params  # lengthパラメータは存在しない

    def test_kst_function_signature(self):
        """KST関数のシグネチャが正しく認識されることを確認"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 実際の関数シグネチャを取得
        sig = inspect.signature(MomentumIndicators.kst)

        # 必要なパラメータのみが存在することを確認
        params = list(sig.parameters.keys())
        assert "data" in params
        assert "r1" in params
        assert "r2" in params
        assert "r3" in params
        assert "r4" in params
        assert "n1" in params
        assert "n2" in params
        assert "n3" in params
        assert "n4" in params
        assert "signal" in params
        assert "length" not in params  # lengthパラメータは存在しない


if __name__ == "__main__":
    pytest.main([__file__, "-v"])