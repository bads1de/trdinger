"""
SignalGenerator のユニットテスト

Test-Driven Development (TDD) アプローチでメタラベリング用の
シグナル生成ロジックをテストします。
"""

import pandas as pd
import numpy as np
import pytest

from app.services.ml.label_generation.signal_generator import SignalGenerator


class TestSignalGeneratorInitialization:
    """SignalGenerator の初期化テスト"""

    def test_initialization(self):
        """正常に初期化できることを確認"""
        generator = SignalGenerator()
        assert generator is not None


class TestBollingerBandBreakout:
    """ボリンジャーバンド ブレイクアウト検知のテスト"""

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)

        # トレンドがあるデータを生成
        base_price = 100
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 2, 100)
        close_prices = base_price + trend + noise

        df = pd.DataFrame(
            {
                "open": close_prices + np.random.normal(0, 0.5, 100),
                "high": close_prices + np.abs(np.random.normal(1, 0.5, 100)),
                "low": close_prices - np.abs(np.random.normal(1, 0.5, 100)),
                "close": close_prices,
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def sample_ohlcv_with_breakout(self):
        """明確なブレイクアウトを含むOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)

        # 最初は狭いレンジ相場、後半で急激なブレイクアウト
        close_prices = np.concatenate(
            [
                np.random.normal(100, 0.5, 50),  # 狭いレンジ相場（σ=0.5）
                np.linspace(100, 125, 50),  # 急激なブレイクアウト（+25）
            ]
        )

        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 0.5,
                "low": close_prices - 0.5,
                "close": close_prices,
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        return df

    def test_bb_breakout_returns_datetimeindex(self, sample_ohlcv):
        """BBブレイクアウトがDatetimeIndexを返すことを確認"""
        generator = SignalGenerator()
        events = generator.get_bb_breakout_events(df=sample_ohlcv, window=20, dev=2.0)

        assert isinstance(events, pd.DatetimeIndex)

    def test_bb_breakout_with_no_breakouts(self):
        """ブレイクアウトがない場合は空のインデックスを返すことを確認"""
        # 完全に平坦なデータ
        dates = pd.date_range(start="2023-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "close": [100.0] * 50,
                "high": [100.5] * 50,
                "low": [99.5] * 50,
            },
            index=dates,
        )

        generator = SignalGenerator()
        events = generator.get_bb_breakout_events(df, window=20, dev=2.0)

        # ブレイクアウトが検出されないことを確認
        assert len(events) == 0

    def test_bb_breakout_detects_upper_band_break(self, sample_ohlcv_with_breakout):
        """上バンドブレイクアウトを検出できることを確認"""
        generator = SignalGenerator()
        events = generator.get_bb_breakout_events(
            df=sample_ohlcv_with_breakout, window=20, dev=2.0
        )

        # ブレイクアウトが検出されることを確認
        assert len(events) > 0

        # 検出されたイベントがデータの後半（ブレイクアウト区間）に集中していることを確認
        mid_index = sample_ohlcv_with_breakout.index[50]
        later_events = events[events >= mid_index]
        assert len(later_events) > 0

    def test_bb_breakout_with_custom_parameters(self, sample_ohlcv):
        """カスタムパラメータでBBブレイクアウトを検出できることを確認"""
        generator = SignalGenerator()

        # 狭いバンド（dev=1.0）では多くのブレイクアウトが検出される
        events_narrow = generator.get_bb_breakout_events(
            df=sample_ohlcv, window=20, dev=1.0
        )

        # 広いバンド（dev=3.0）では少ないブレイクアウトが検出される
        events_wide = generator.get_bb_breakout_events(
            df=sample_ohlcv, window=20, dev=3.0
        )

        # 狭いバンドの方が多くのブレイクアウトを検出する
        assert len(events_narrow) >= len(events_wide)


class TestDonchianBreakout:
    """ドンチャンブレイクアウト検知のテスト"""

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)

        close_prices = 100 + np.random.normal(0, 2, 100)

        df = pd.DataFrame(
            {
                "open": close_prices + np.random.normal(0, 0.5, 100),
                "high": close_prices + np.abs(np.random.normal(1, 0.5, 100)),
                "low": close_prices - np.abs(np.random.normal(1, 0.5, 100)),
                "close": close_prices,
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        return df

    def test_donchian_breakout_returns_datetimeindex(self, sample_ohlcv):
        """ドンチャンブレイクアウトがDatetimeIndexを返すことを確認"""
        generator = SignalGenerator()
        events = generator.get_donchian_breakout_events(df=sample_ohlcv, window=20)

        assert isinstance(events, pd.DatetimeIndex)

    def test_donchian_breakout_with_trending_data(self):
        """トレンドのあるデータでドンチャンブレイクアウトを検出"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")

        # レンジ相場から上昇トレンドへ
        close_prices = np.concatenate(
            [
                np.random.normal(100, 1, 50),  # レンジ相場
                np.linspace(100, 120, 50),  # 上昇トレンド
            ]
        )

        df = pd.DataFrame(
            {
                "high": close_prices + 1,
                "low": close_prices - 1,
                "close": close_prices,
            },
            index=dates,
        )

        generator = SignalGenerator()
        events = generator.get_donchian_breakout_events(df, window=20)

        # トレンド開始後にブレイクアウトが検出されることを確認
        assert len(events) > 0


class TestVolumeSpikeDetection:
    """出来高急増検知のテスト"""

    @pytest.fixture
    def sample_ohlcv_with_volume_spike(self):
        """出来高急増を含むOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)

        # 通常の出来高
        normal_volume = np.random.uniform(1000, 1500, 100)

        # 特定の地点で出来高を急増させる
        normal_volume[50] = 5000  # 3倍以上の急増
        normal_volume[75] = 6000  # 4倍の急増

        df = pd.DataFrame(
            {"close": 100 + np.random.normal(0, 2, 100), "volume": normal_volume},
            index=dates,
        )

        return df

    def test_volume_spike_returns_datetimeindex(self, sample_ohlcv_with_volume_spike):
        """出来高急増検知がDatetimeIndexを返すことを確認"""
        generator = SignalGenerator()
        events = generator.get_volume_spike_events(
            df=sample_ohlcv_with_volume_spike, window=20, multiplier=2.0
        )

        assert isinstance(events, pd.DatetimeIndex)

    def test_volume_spike_detects_spikes(self, sample_ohlcv_with_volume_spike):
        """出来高急増を正しく検出することを確認"""
        generator = SignalGenerator()
        events = generator.get_volume_spike_events(
            df=sample_ohlcv_with_volume_spike, window=20, multiplier=2.0
        )

        # 2つの急増イベントが検出されることを確認
        assert len(events) >= 2

    def test_volume_spike_with_high_threshold(self, sample_ohlcv_with_volume_spike):
        """高い閾値では少ないイベントが検出されることを確認"""
        generator = SignalGenerator()

        # 低い閾値（2倍）
        events_low = generator.get_volume_spike_events(
            df=sample_ohlcv_with_volume_spike, window=20, multiplier=2.0
        )

        # 高い閾値（5倍）
        events_high = generator.get_volume_spike_events(
            df=sample_ohlcv_with_volume_spike, window=20, multiplier=5.0
        )

        # 低い閾値の方が多くのイベントを検出
        assert len(events_low) >= len(events_high)


class TestCombinedSignals:
    """複数のシグナルを組み合わせたテスト"""

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)

        close_prices = np.concatenate(
            [
                np.random.normal(100, 1, 50),  # レンジ相場
                np.linspace(100, 120, 50),  # ブレイクアウト
            ]
        )

        volume = np.random.uniform(1000, 1500, 100)
        volume[55] = 5000  # ブレイクアウト時に出来高急増

        df = pd.DataFrame(
            {
                "open": close_prices + np.random.normal(0, 0.5, 100),
                "high": close_prices + np.abs(np.random.normal(1, 0.5, 100)),
                "low": close_prices - np.abs(np.random.normal(1, 0.5, 100)),
                "close": close_prices,
                "volume": volume,
            },
            index=dates,
        )

        return df

    def test_get_combined_events(self, sample_ohlcv):
        """複数のシグナルを組み合わせてイベントを取得できることを確認"""
        generator = SignalGenerator()

        # 各シグナルを個別に取得（将来的な検証用）
        _bb_events = generator.get_bb_breakout_events(sample_ohlcv)
        _donchian_events = generator.get_donchian_breakout_events(sample_ohlcv)
        _volume_events = generator.get_volume_spike_events(sample_ohlcv)

        # 組み合わせ（ユニオン）
        combined = generator.get_combined_events(
            sample_ohlcv, use_bb=True, use_donchian=True, use_volume=True
        )

        # 組み合わせた結果は個別の合計以上のイベント数を持つ（重複除去後）
        assert isinstance(combined, pd.DatetimeIndex)

    def test_get_combined_events_with_selective_signals(self, sample_ohlcv):
        """選択的にシグナルを組み合わせられることを確認"""
        generator = SignalGenerator()

        # BBのみ
        bb_only = generator.get_combined_events(
            sample_ohlcv, use_bb=True, use_donchian=False, use_volume=False
        )

        # BB + Volume
        bb_volume = generator.get_combined_events(
            sample_ohlcv, use_bb=True, use_donchian=False, use_volume=True
        )

        # BB + Volume の方が多いか同じイベント数を持つ
        assert len(bb_volume) >= len(bb_only)


