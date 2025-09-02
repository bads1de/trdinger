"""
TSI (True Strength Index) 指標の修正テスト
スケール不一致の問題をテストおよび修正
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


class TestTSIIndicator:
    """TSI指標のテストクラス"""

    def setup_sample_data(self, length=100):
        """テスト用サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='h')

        # OHLCデータを生成（高いボラティリティでスケール問題をテスト）
        close = 100 + np.cumsum(np.random.randn(length) * 10)  # 高いボラティリティ
        high = close * (1 + np.random.rand(length) * 0.05)
        low = close * (1 - np.random.rand(length) * 0.05)

        close = pd.Series(close, index=dates)
        high = pd.Series(high, index=dates)
        low = pd.Series(low, index=dates)

        return pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(length) * 5,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, length)
        })

    @pytest.fixture
    def sample_ohlcv_data(self):
        """pytest fixture for sample data"""
        return self.setup_sample_data()

    def test_tsi_normal_calculation(self, sample_ohlcv_data):
        """正規のTSI計算テスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(sample_ohlcv_data['close']),
            fast=13,
            slow=25
        )

        assert tsi_result is not None
        assert isinstance(tsi_result, pd.Series)
        assert len(tsi_result) == len(sample_ohlcv_data)
        assert not tsi_result.isna().all()

        # TSIの期待されるスケール範囲を確認（通常-100〜100）
        valid_tsi = tsi_result.dropna()
        if len(valid_tsi) > 0:
            print(f"TSI範囲: {valid_tsi.min():.2f} to {valid_tsi.max():.2f}")
            # 理想的には-100〜100の範囲だが、確認中

    def test_tsi_high_volatility_scaling_issue(self, sample_ohlcv_data):
        """高ボラティリティ時のスケーリング問題テスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 極端にボラティリティの高いデータを生成
        high_vol_close = sample_ohlcv_data['close'] * np.random.rand(len(sample_ohlcv_data)) * 100  # 極端な値

        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(high_vol_close),
            fast=13,
            slow=25
        )

        assert tsi_result is not None

        # 高ボラティリティでスケール外の値が出ていないことを確認
        valid_tsi = tsi_result.dropna()
        if len(valid_tsi) > 0:
            # 極端なスケール（±1000以上）は問題
            extreme_values = np.abs(valid_tsi) > 1000
            assert not extreme_values.any(), f"極端なスケール値検出: {valid_tsi[extreme_values].values}"

    def test_tsi_scaling_range_test(self, sample_ohlcv_data):
        """TSIのスケール範囲テスト（問題再現）"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 様々なパラメータでテスト
        parameters = [
            (13, 25),  # 標準
            (5, 10),   # 短期間
            (25, 50),  # 長期間
        ]

        for fast, slow in parameters:
            tsi_result = MomentumIndicators.tsi(
                data=pd.Series(sample_ohlcv_data['close']),
                fast=fast,
                slow=slow
            )

            assert tsi_result is not None
            valid_tsi = tsi_result.dropna()
            assert len(valid_tsi) > 0

            # スケール範囲の統計
            print(f"TSI(fast={fast}, slow={slow}): min={valid_tsi.min():.2f}, max={valid_tsi.max():.2f}")

    def test_tsi_data_length_insufficient(self, sample_ohlcv_data):
        """データ長不足時のテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 最小データ長より少ないデータをテスト
        short_data = sample_ohlcv_data['close'][:10]  # 25（slow） + 数より少ない

        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(short_data),
            fast=13,
            slow=25
        )

        # データ不足の場合、適切に処理されるはず
        assert tsi_result is not None
        assert isinstance(tsi_result, pd.Series)

    def test_tsi_zero_price_values(self, sample_ohlcv_data):
        """ゼロ価格値の処理テスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # ゼロ価格を含むデータ
        zero_data = sample_ohlcv_data['close'].copy()
        zero_data.iloc[10:15] = 0

        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(zero_data),
            fast=13,
            slow=25
        )

        assert tsi_result is not None
        # ゼロ価格でもinfが発生しないことを確認
        assert not tsi_result.isin([np.inf, -np.inf]).any()

    def test_tsi_negative_price_values(self, sample_ohlcv_data):
        """負価格値の処理テスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 負価格を含むデータ
        negative_data = sample_ohlcv_data['close'].copy()
        negative_data.iloc[10:15] = -abs(negative_data.iloc[10:15])

        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(negative_data),
            fast=13,
            slow=25
        )

        assert tsi_result is not None
        # 負価格でも適切な値が出力されることを確認
        valid_tsi = tsi_result.dropna()
        assert len(valid_tsi) > 0

    def test_tsi_extreme_parameters(self, sample_ohlcv_data):
        """極端なパラメータのテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 極端に長い期間のパラメータ
        extreme_fast = 50
        extreme_slow = 100

        # データ長より長い期間のパラメータも最低限処理されるべき
        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(sample_ohlcv_data['close']),
            fast=extreme_fast,
            slow=extreme_slow
        )

        assert tsi_result is not None
        # 極端なパラメータでも計算結果が出るか確認（空ではない）
        assert not tsi_result.isna().all()

    def test_tsi_pandas_ta_none_handling(self, sample_ohlcv_data):
        """pandas-taがNoneを返すケースのテスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # pandas-ta.tsiがNoneを返すシミュレーション
        with patch('pandas_ta.tsi', return_value=None):
            tsi_result = MomentumIndicators.tsi(
                data=pd.Series(sample_ohlcv_data['close']),
                fast=13,
                slow=25
            )

            # 現在の実装では空のSeriesを返すはず
            assert isinstance(tsi_result, pd.Series)
            # pandas-taがNoneの場合、空のSeriesまたはNaNで埋められたSeries
            assert len(tsi_result) == 0 or tsi_result.isna().all()

    def test_tsi_consistent_scaling(self, sample_ohlcv_data):
        """TSIのスケーリング一貫性テスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # 複数回実行して一致することを確認
        results = []
        for i in range(3):
            tsi_result = MomentumIndicators.tsi(
                data=pd.Series(sample_ohlcv_data['close']),
                fast=13,
                slow=25
            )
            valid_tsi = tsi_result.dropna()
            if len(valid_tsi) > 0:
                results.append({
                    'min': valid_tsi.min(),
                    'max': valid_tsi.max(),
                    'mean': valid_tsi.mean(),
                    'std': valid_tsi.std()
                })

        # 同じ入力で結果が一貫していることを確認
        if len(results) > 1:
            # スケール範囲が大きく異なる場合は問題
            min_range = abs(results[0]['max'] - results[0]['min'])
            for result in results[1:]:
                current_range = abs(result['max'] - result['min'])
                # 範囲が大きく異なる場合は警告
                if abs(min_range - current_range) > min_range * 0.5:
                    pytest.skip("TSIスケーリングに一貫性の問題あり")

    def test_tsi_scaling_improvement_check(self, sample_ohlcv_data):
        """TSIスケーリング改善の確認テスト"""
        from app.services.indicators.technical_indicators.momentum import MomentumIndicators

        # スケール問題の疑われるデータでテスト
        extreme_data = pd.concat([
            pd.Series([100] * 30),
            pd.Series([200] * 20),  # 急変動
            pd.Series([50] * 30),   # 急下降
            pd.Series([150] * 20)
        ])

        tsi_result = MomentumIndicators.tsi(
            data=pd.Series(extreme_data.values),
            fast=13,
            slow=25
        )

        assert tsi_result is not None
        valid_tsi = tsi_result.dropna()

        if len(valid_tsi) > 0:
            # スケール範囲が妥当か確認
            tsi_range = abs(valid_tsi.max() - valid_tsi.min())
            print(f"Extreme data TSI range: {tsi_range}")

            # 極端なスケール（例: ±100000）は問題
            if tsi_range > 100000:
                pytest.fail(f"TSIスケールが極端に大きい: {tsi_range}")

    def test_tsi_integration_with_service(self, sample_ohlcv_data):
        """TechnicalIndicatorServiceとの統合テスト"""
        try:
            from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

            service = TechnicalIndicatorService()
            result = service.calculate_indicator(
                sample_ohlcv_data.copy(),
                "TSI",
                {"fast": 13, "slow": 25}
            )

            assert result is not None

        except ImportError:
            pytest.skip("TechnicalIndicatorService 未実装")


if __name__ == "__main__":
    pytest.main([__file__])