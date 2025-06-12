#!/usr/bin/env python3
"""
EMV (Ease of Movement) 指標のテスト

TDD方式でEMVの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# テスト対象のインポート（実装前なので失敗する予定）
try:
    from app.core.services.indicators.volume_indicators import EMVIndicator
    from app.core.services.indicators.adapters import VolumeAdapter
except ImportError:
    # 実装前なので期待されるエラー
    EMVIndicator = None
    VolumeAdapter = None


class TestEMVIndicator:
    """EMV指標のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # より現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # OHLC データを生成
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]

        return pd.DataFrame(
            {
                "open": prices,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_emv_indicator_initialization(self):
        """EMVIndicatorの初期化テスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()

        # 基本属性の確認
        assert indicator.indicator_type == "EMV"
        assert hasattr(indicator, "supported_periods")
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0

        # 期待される期間が含まれているか
        expected_periods = [14, 20, 30]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_emv_calculation_basic(self, sample_data):
        """EMV基本計算のテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        period = 14

        # EMV計算実行
        result = indicator.calculate(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

        # NaN値の確認（初期値はNaNであるべき）
        assert pd.isna(result.iloc[0])

        # 有効な値の確認
        valid_values = result.dropna()
        assert len(valid_values) > 0
        assert all(isinstance(val, (int, float)) for val in valid_values)

    def test_emv_calculation_different_periods(self, sample_data):
        """異なる期間でのEMV計算テスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        periods = [14, 20, 30]

        results = {}
        for period in periods:
            results[period] = indicator.calculate(sample_data, period)

        # 各期間の結果が計算されていることを確認
        for period in periods:
            assert len(results[period].dropna()) > 0

    def test_emv_adapter_calculation(self, sample_data):
        """VolumeAdapterのEMV計算テスト"""
        if VolumeAdapter is None:
            pytest.skip("VolumeAdapter EMV method not implemented yet")

        period = 14

        # VolumeAdapter経由でのEMV計算
        result = VolumeAdapter.emv(
            sample_data["high"], sample_data["low"], sample_data["volume"], period
        )

        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_emv_mathematical_properties(self, sample_data):
        """EMVの数学的特性のテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        period = 14

        result = indicator.calculate(sample_data, period)
        valid_result = result.dropna()

        # EMVは移動の容易さを示す指標
        # 値が正の場合は上昇が容易、負の場合は下降が容易

        if len(valid_result) > 0:
            # 値が存在することを確認
            assert not valid_result.empty

            # 無限大やNaNでないことを確認
            assert not np.isinf(valid_result).any()
            assert not np.isnan(valid_result).any()

    def test_emv_volume_dependency(self, sample_data):
        """EMVの出来高依存性のテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        period = 14

        # 出来高データが必要
        result = indicator.calculate(sample_data, period)

        # 出来高データなしでエラーになることを確認
        data_without_volume = sample_data.drop("volume", axis=1)

        with pytest.raises((ValueError, KeyError)):
            indicator.calculate(data_without_volume, period)

    def test_emv_high_low_dependency(self, sample_data):
        """EMVの高値・安値依存性のテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        period = 14

        # 高値・安値データが必要
        result = indicator.calculate(sample_data, period)

        # 高値データなしでエラーになることを確認
        data_without_high = sample_data.drop("high", axis=1)

        with pytest.raises((ValueError, KeyError)):
            indicator.calculate(data_without_high, period)

        # 安値データなしでエラーになることを確認
        data_without_low = sample_data.drop("low", axis=1)

        with pytest.raises((ValueError, KeyError)):
            indicator.calculate(data_without_low, period)

    def test_emv_error_handling(self, sample_data):
        """EMVのエラーハンドリングテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        from app.core.services.indicators.adapters.base_adapter import (
            TALibCalculationError,
        )

        indicator = EMVIndicator()

        # 無効な期間でのテスト（負の値）
        try:
            result = indicator.calculate(sample_data, -1)
            # もし例外が発生しなかった場合、結果が空またはエラーであることを確認
            assert result.dropna().empty or len(result.dropna()) == 0
        except (ValueError, TypeError, TALibCalculationError):
            # 期待される例外
            pass

        # 期間0のテスト
        try:
            result = indicator.calculate(sample_data, 0)
            # もし例外が発生しなかった場合、結果が空またはエラーであることを確認
            assert result.dropna().empty or len(result.dropna()) == 0
        except (ValueError, TypeError, TALibCalculationError):
            # 期待される例外
            pass

        # データ不足のテスト
        small_data = sample_data.head(5)
        with pytest.raises((ValueError, IndexError, TALibCalculationError)):
            indicator.calculate(small_data, 20)

    def test_emv_description(self):
        """EMV説明文のテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        description = indicator.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert (
            "EMV" in description
            or "Ease of Movement" in description
            or "移動の容易さ" in description
        )


class TestEMVIntegration:
    """EMV統合テストクラス"""

    def test_emv_in_volume_indicators_info(self):
        """VOLUME_INDICATORS_INFOにEMVが含まれているかテスト"""
        try:
            from app.core.services.indicators.volume_indicators import (
                VOLUME_INDICATORS_INFO,
            )

            assert "EMV" in VOLUME_INDICATORS_INFO
            emv_info = VOLUME_INDICATORS_INFO["EMV"]

            assert "periods" in emv_info
            assert "description" in emv_info
            assert "category" in emv_info
            assert emv_info["category"] == "volume"

        except ImportError:
            pytest.skip("VOLUME_INDICATORS_INFO not available yet")

    def test_emv_factory_function(self):
        """get_volume_indicator関数でEMVが取得できるかテスト"""
        try:
            from app.core.services.indicators.volume_indicators import (
                get_volume_indicator,
            )

            emv_indicator = get_volume_indicator("EMV")
            assert emv_indicator is not None
            assert emv_indicator.indicator_type == "EMV"

        except ImportError:
            pytest.skip("get_volume_indicator not available yet")

    def test_emv_in_main_module(self):
        """メインモジュールからEMVがインポートできるかテスト"""
        try:
            from app.core.services.indicators import EMVIndicator

            indicator = EMVIndicator()
            assert indicator.indicator_type == "EMV"

        except ImportError:
            pytest.skip("EMVIndicator not exported from main module yet")


class TestEMVAlgorithm:
    """EMVアルゴリズムの詳細テスト"""

    @pytest.fixture
    def trend_data(self):
        """トレンドのあるテストデータ"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # 上昇トレンドのデータ
        base_price = 100
        trend = np.linspace(0, 20, 50)  # 20ポイントの上昇トレンド
        noise = np.random.normal(0, 1, 50)
        prices = base_price + trend + noise

        # OHLC データを生成
        highs = [max(p * 1.01, p + 1) for p in prices]
        lows = [min(p * 0.99, p - 1) for p in prices]

        # 出来高データ（価格上昇時に減少傾向 = 移動が容易）
        base_volume = 1000
        volume_trend = np.where(
            np.diff(np.concatenate([[base_price], prices])) > 0,
            np.random.randint(500, 800, 50),  # 上昇時は出来高少なめ
            np.random.randint(1200, 1500, 50),
        )  # 下降時は出来高多め

        return pd.DataFrame(
            {"high": highs, "low": lows, "close": prices, "volume": volume_trend},
            index=dates,
        )

    def test_emv_movement_ease(self, trend_data):
        """EMVの移動容易性測定のテスト"""
        if EMVIndicator is None:
            pytest.skip("EMVIndicator not implemented yet")

        indicator = EMVIndicator()
        period = 14

        result = indicator.calculate(trend_data, period)
        valid_result = result.dropna()

        if len(valid_result) > 5:
            # EMVが計算されていることを確認
            assert not valid_result.empty

            # 上昇トレンドで出来高が少ない場合、EMVは正の値になりやすい
            # （移動が容易であることを示す）
            recent_emv = valid_result.tail(10).mean()

            # 値が計算されていることを確認（具体的な符号は市場条件による）
            assert not np.isnan(recent_emv)
            assert not np.isinf(recent_emv)


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
