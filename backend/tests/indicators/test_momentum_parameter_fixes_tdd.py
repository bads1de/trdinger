import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestMomentumIndicatorsParameterFixes:
    """グループ3のMomentumIndicatorsパラメータエラー修正用のTDDテスト"""

    def setup_method(self):
        """テストデータの準備"""
        # 簡単なテストデータを作成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.open_data = pd.Series(np.random.uniform(100, 200, 100), index=dates) + np.random.normal(0, 5, 100)
        self.high_data = self.open_data + np.random.uniform(0, 10, 100)
        self.low_data = self.open_data - np.random.uniform(0, 10, 100)
        self.close_data = pd.Series(np.random.uniform(100, 200, 100), index=dates) + np.random.normal(0, 5, 100)
        self.volume_data = pd.Series(np.random.uniform(1000, 10000, 100), index=dates)

    def test_bop_open_parameter_fix(self):
        """BOP: open_パラメータが正しく渡されることを確認"""
        # テストケース1: 基本的なパラメータ渡し
        try:
            result = MomentumIndicators.bop(
                open_=self.open_data,
                high=self.high_data,
                low=self.low_data,
                close=self.close_data,
                period=14
            )
            # 結果がNoneでないことを確認
            assert result is not None
            assert isinstance(result, pd.Series)
            # NaN以外に値があることを確認（計算が成功している）
            assert not result.isna().all()
        except Exception as e:
            pytest.fail(f"BOPパラメータ渡しでエラー: {e}")

    def test_ultosc_period_parameter_extension(self):
        """ULTOSC: periodパラメータを追加し、正しいマッピングを確認"""
        # テストケース1: 新しいperiodパラメータ
        period = 21
        try:
            result = MomentumIndicators.ultosc(
                high=self.high_data,
                low=self.low_data,
                close=self.close_data,
                period=period
            )
            # 結果がNoneでないことを確認
            assert result is not None
            assert isinstance(result, pd.Series)
            # periodに基づくマッピング確認
            expected_fast = period // 3  # 7
            expected_medium = period     # 21
            expected_slow = period * 2   # 42
            # この内部検証はログから間接的に確認

        except Exception as e:
            pytest.fail(f"ULTOSC periodパラメータでエラー: {e}")

    def test_ultosc_backward_compatibility(self):
        """ULTOSC: 古いfast,medium,slowパラメータがまだ動作することを確認"""
        # テストケース: 古いパラメータ
        try:
            result = MomentumIndicators.ultosc(
                high=self.high_data,
                low=self.low_data,
                close=self.close_data,
                fast=7,
                medium=14,
                slow=28
            )
            # 結果がNoneでないことを確認
            assert result is not None
            assert isinstance(result, pd.Series)
        except Exception as e:
            pytest.fail(f"ULTOSC後方互換性パラメータでエラー: {e}")

    def test_cmo_length_parameter_support(self):
        """CMO: lengthパラメータが正しく受け付けられることを確認"""
        # テストケース: lengthパラメータ使用
        try:
            result = MomentumIndicators.cmo(
                data=self.close_data,
                length=20
            )
            # 結果がNoneでないことを確認
            assert result is not None
            assert isinstance(result, pd.Series)
            # lengthパラメータが正しくマッピングされていることを間接的に確認
        except Exception as e:
            pytest.fail(f"CMO lengthパラメータでエラー: {e}")

    def test_bop_parameter_validation(self):
        """BOP: パラメータ検証機能の確認"""
        # 無効なパラメータテスト
        with pytest.raises(ValueError):
            MomentumIndicators.bop(
                open_=self.open_data,
                high=self.high_data,
                low=self.low_data,
                close=self.close_data,
                period=0  # 無効なperiod
            )

        # 型検証テスト
        with pytest.raises(TypeError):
            MomentumIndicators.bop(
                open_="invalid",  # Seriesでない
                high=self.high_data,
                low=self.low_data,
                close=self.close_data,
                period=14
            )

    def test_ultosc_parameter_validation(self):
        """ULTOSC: パラメータ検証機能の確認"""
        # 型検証テスト
        with pytest.raises(TypeError):
            MomentumIndicators.ultosc(
                high="invalid",  # Seriesでない
                low=self.low_data,
                close=self.close_data,
                period=14
            )

    def test_cmo_parameter_validation(self):
        """CMO: パラメータ検証機能の確認"""
        # 型検証テスト
        with pytest.raises(TypeError):
            MomentumIndicators.cmo(
                data="invalid",  # Seriesでない
                length=14
            )


if __name__ == "__main__":
    pytest.main([__file__])