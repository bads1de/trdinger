#!/usr/bin/env python3
"""
PVT (Price Volume Trend) 指標のテスト

TDD方式でPVTの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# テスト対象のインポート（実装前なので失敗する予定）
try:
    from app.core.services.indicators.volume_indicators import PVTIndicator
    from app.core.services.indicators.adapters import VolumeAdapter
except ImportError:
    # 実装前なので期待されるエラー
    PVTIndicator = None
    VolumeAdapter = None


class TestPVTIndicator:
    """PVT指標のテストクラス"""

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

        return pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_pvt_indicator_initialization(self):
        """PVTIndicatorの初期化テスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()

        # 基本属性の確認
        assert indicator.indicator_type == "PVT"
        assert hasattr(indicator, "supported_periods")
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0

        # PVTは累積指標なので期間は通常1
        assert 1 in indicator.supported_periods

    def test_pvt_calculation_basic(self, sample_data):
        """PVT基本計算のテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()
        period = 1  # PVTは累積指標

        # PVT計算実行
        result = indicator.calculate(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

        # 最初の値はNaNまたは0であるべき
        assert pd.isna(result.iloc[0]) or result.iloc[0] == 0

        # 有効な値の確認
        valid_values = result.dropna()
        assert len(valid_values) > 0
        assert all(isinstance(val, (int, float)) for val in valid_values)

    def test_pvt_calculation_properties(self, sample_data):
        """PVTの計算特性のテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()
        period = 1

        result = indicator.calculate(sample_data, period)

        # PVTは累積指標なので、一般的に単調増加または減少の傾向を持つ
        valid_result = result.dropna()

        if len(valid_result) > 1:
            # 値が存在することを確認
            assert not valid_result.empty

            # PVTの値が計算されていることを確認
            assert not all(val == 0 for val in valid_result)

    def test_pvt_adapter_calculation(self, sample_data):
        """VolumeAdapterのPVT計算テスト"""
        if VolumeAdapter is None:
            pytest.skip("VolumeAdapter PVT method not implemented yet")

        # VolumeAdapter経由でのPVT計算
        result = VolumeAdapter.pvt(sample_data["close"], sample_data["volume"])

        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_pvt_mathematical_properties(self, sample_data):
        """PVTの数学的特性のテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()
        period = 1

        result = indicator.calculate(sample_data, period)
        valid_result = result.dropna()

        # PVTは出来高と価格変化の関係を示す
        # 価格が上昇し出来高が多い場合、PVTは増加する傾向
        # 価格が下降し出来高が多い場合、PVTは減少する傾向

        if len(valid_result) > 10:
            # 値の範囲が合理的であることを確認
            assert not np.isinf(valid_result).any()
            assert not np.isnan(valid_result).any()

    def test_pvt_volume_dependency(self, sample_data):
        """PVTの出来高依存性のテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()
        period = 1

        # 出来高データが必要
        result = indicator.calculate(sample_data, period)

        # 出来高データなしでエラーになることを確認
        data_without_volume = sample_data.drop("volume", axis=1)

        with pytest.raises((ValueError, KeyError)):
            indicator.calculate(data_without_volume, period)

    def test_pvt_error_handling(self, sample_data):
        """PVTのエラーハンドリングテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        from app.core.services.indicators.adapters.base_adapter import (
            TALibCalculationError,
        )

        indicator = PVTIndicator()

        # PVTは期間1のみサポートしているので、無効な期間のテストは期間2以上で行う
        try:
            # 期間2以上は通常サポートされていない
            result = indicator.calculate(sample_data, 2)
            # もし例外が発生しなかった場合、結果が計算されていることを確認
            assert isinstance(result, pd.Series)
        except (ValueError, TypeError, TALibCalculationError):
            # 期待される例外
            pass

        # データ不足のテスト
        small_data = sample_data.head(1)
        # PVTは最低2個のデータが必要（前日比較のため）
        with pytest.raises((ValueError, IndexError, TALibCalculationError)):
            indicator.calculate(small_data, 1)

    def test_pvt_description(self):
        """PVT説明文のテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()
        description = indicator.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert (
            "PVT" in description
            or "Price Volume Trend" in description
            or "価格出来高トレンド" in description
        )


class TestPVTIntegration:
    """PVT統合テストクラス"""

    def test_pvt_in_volume_indicators_info(self):
        """VOLUME_INDICATORS_INFOにPVTが含まれているかテスト"""
        try:
            from app.core.services.indicators.volume_indicators import (
                VOLUME_INDICATORS_INFO,
            )

            assert "PVT" in VOLUME_INDICATORS_INFO
            pvt_info = VOLUME_INDICATORS_INFO["PVT"]

            assert "periods" in pvt_info
            assert "description" in pvt_info
            assert "category" in pvt_info
            assert pvt_info["category"] == "volume"

        except ImportError:
            pytest.skip("VOLUME_INDICATORS_INFO not available yet")

    def test_pvt_factory_function(self):
        """get_volume_indicator関数でPVTが取得できるかテスト"""
        try:
            from app.core.services.indicators.volume_indicators import (
                get_volume_indicator,
            )

            pvt_indicator = get_volume_indicator("PVT")
            assert pvt_indicator is not None
            assert pvt_indicator.indicator_type == "PVT"

        except ImportError:
            pytest.skip("get_volume_indicator not available yet")

    def test_pvt_in_main_module(self):
        """メインモジュールからPVTがインポートできるかテスト"""
        try:
            from app.core.services.indicators import PVTIndicator

            indicator = PVTIndicator()
            assert indicator.indicator_type == "PVT"

        except ImportError:
            pytest.skip("PVTIndicator not exported from main module yet")


class TestPVTAlgorithm:
    """PVTアルゴリズムの詳細テスト"""

    @pytest.fixture
    def trend_data(self):
        """トレンドのあるテストデータ"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # 上昇トレンドのデータ
        base_price = 100
        trend = np.linspace(0, 20, 50)  # 20ポイントの上昇トレンド
        noise = np.random.normal(0, 1, 50)
        prices = base_price + trend + noise

        # 出来高も増加傾向
        base_volume = 1000
        volume_trend = np.linspace(0, 500, 50)
        volume_noise = np.random.normal(0, 100, 50)
        volumes = base_volume + volume_trend + volume_noise
        volumes = np.maximum(volumes, 100)  # 最小出来高を保証

        return pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

    def test_pvt_trend_following(self, trend_data):
        """PVTのトレンドフォロー特性のテスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        indicator = PVTIndicator()
        period = 1

        result = indicator.calculate(trend_data, period)
        valid_result = result.dropna()

        if len(valid_result) > 10:
            # 上昇トレンドでPVTも上昇傾向にあることを確認
            first_half = valid_result[: len(valid_result) // 2].mean()
            second_half = valid_result[len(valid_result) // 2 :].mean()

            # 上昇トレンドでPVTも増加することを期待
            # （ただし、必ずしも単調増加ではない）
            assert second_half != first_half  # 何らかの変化があることを確認

    def test_pvt_vs_obv_comparison(self, trend_data):
        """PVTとOBVの比較テスト"""
        if PVTIndicator is None:
            pytest.skip("PVTIndicator not implemented yet")

        from app.core.services.indicators.volume_indicators import OBVIndicator

        pvt_indicator = PVTIndicator()
        obv_indicator = OBVIndicator()
        period = 1

        pvt_result = pvt_indicator.calculate(trend_data, period)
        obv_result = obv_indicator.calculate(trend_data, period)

        # 両方とも出来高ベースの指標として計算されることを確認
        assert len(pvt_result.dropna()) > 0
        assert len(obv_result.dropna()) > 0

        # PVTとOBVは異なる計算方法なので、値は異なるはず
        pvt_values = pvt_result.dropna()
        obv_values = obv_result.dropna()

        if len(pvt_values) > 5 and len(obv_values) > 5:
            # 最後の5個の値で比較
            pvt_recent = pvt_values.tail(5).mean()
            obv_recent = obv_values.tail(5).mean()

            # 異なる計算方法なので値は異なるはず
            assert pvt_recent != obv_recent


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
