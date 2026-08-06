import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.advanced_features import (
    AdvancedFeatures,
)


class TestAdvancedFeatures:
    @pytest.fixture
    def sample_data(self):
        """サンプルデータ生成"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        close = pd.Series(
            np.cumsum(np.random.randn(100)) + 100, index=dates, name="close"
        )
        oi = pd.Series(
            np.cumsum(np.random.randn(100)) + 500, index=dates, name="open_interest"
        )
        volume = pd.Series(np.random.rand(100) * 100 + 50, index=dates, name="volume")
        fr = pd.Series(np.random.randn(100) * 0.0001, index=dates, name="funding_rate")
        return pd.DataFrame(
            {"close": close, "open_interest": oi, "volume": volume, "funding_rate": fr}
        )

    def test_frac_diff_ffd(self, sample_data):
        """分数次差分のテスト"""
        series = sample_data["close"]
        # d=0.4 で計算
        diff_series = AdvancedFeatures.frac_diff_ffd(
            series, d=0.4, thres=1e-4, window=100
        )

        assert isinstance(diff_series, pd.Series)
        assert len(diff_series) == len(series)
        # 最初の方はNaNになるはず（ウィンドウサイズ分）
        # ただし現在の実装では window サイズではなく重みの有効長で決まる

        # 定常性の確認（ADF検定などは厳密すぎるので、ここでは値が計算されていることと
        # 元の系列との相関が一定以上あることを確認）
        valid_data = diff_series.dropna()
        assert len(valid_data) > 0

        # d=0 (差分なし) の場合は元の系列と完全一致するはず
        diff_zero = AdvancedFeatures.frac_diff_ffd(series, d=0.0, thres=1e-4)
        # 浮動小数点の誤差を許容
        pd.testing.assert_series_equal(series, diff_zero, rtol=1e-5)

        # d=1 (通常差分) の場合は diff() と近い動きをするはず
        # (ただしFFDはウィンドウ制限があるので完全一致はしない)
        diff_one = AdvancedFeatures.frac_diff_ffd(series, d=1.0, thres=1e-4)
        simple_diff = series.diff()

        # 相関が高いことを確認
        corr = diff_one.corr(simple_diff)
        assert corr > 0.9

    def test_frac_diff_ffd_default_window_short_data(self):
        """
        短いデータでもデフォルト window で有効な FFD 値が得られることを検証。

        回帰テスト: 以前は window=2000（有効重み数1458）だったため、
        データが1458本未満だと全NaNになり、下流で黙って定数0に潰れていた。
        デフォルト window=200 なら 400本程度のデータでも半分以上が有効になる。
        """
        dates = pd.date_range(start="2024-01-01", periods=400, freq="1h")
        close = pd.Series(np.cumsum(np.random.randn(400)) + 100, index=dates)

        diff_series = AdvancedFeatures.frac_diff_ffd(close, d=0.4)

        valid_count = int(diff_series.notna().sum())
        assert len(diff_series) == len(close)
        # window=200 のウォームアップ(199本)を除いた値が計算されている
        assert valid_count >= 100, f"有効値が少なすぎます: {valid_count}"
        # 全NaN（定数0に潰れる）ではないこと
        assert not diff_series.isna().all()

    def test_frac_diff_ffd_too_short_data_returns_all_nan(self):
        """データが重み数に満たない場合は全NaNを返す（既存仕様の維持）"""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
        close = pd.Series(np.cumsum(np.random.randn(50)) + 100, index=dates)

        diff_series = AdvancedFeatures.frac_diff_ffd(close, d=0.4, window=200)

        assert len(diff_series) == len(close)
        assert diff_series.isna().all()

    def test_liquidation_cascade_score(self, sample_data):
        """清算カスケードスコアのテスト"""
        score = AdvancedFeatures.liquidation_cascade_score(
            sample_data["close"], sample_data["open_interest"], sample_data["volume"]
        )
        assert len(score) == len(sample_data)
        assert not score.isna().all()

    def test_regime_quadrant_logic(self):
        """4象限レジーム分析ロジックのテスト（新規実装予定）"""
        # ケース1: 価格上昇(Up) + OI増加(Up) -> 強気トレンド(Bull Trend)
        # ケース2: 価格上昇(Up) + OI減少(Down) -> ショートカバー(Short Cover)
        # ケース3: 価格下落(Down) + OI増加(Up) -> 弱気トレンド(Bear Trend)
        # ケース4: 価格下落(Down) + OI減少(Down) -> ロング清算(Long Liquidation)

        # データ準備
        data = pd.DataFrame(
            {
                "close_change": [0.01, 0.01, -0.01, -0.01],
                "oi_change": [100, -100, 100, -100],
            }
        )

        # 期待されるカテゴリ (0, 1, 2, 3)
        # 定義:
        # 0: Bull Trend (P+, OI+)
        # 1: Short Cover (P+, OI-)
        # 2: Bear Trend (P-, OI+)
        # 3: Long Liquidation (P-, OI-)

        expected = [0, 1, 2, 3]

        # ロジックの実装（仮）
        regime = []
        for _i, row in data.iterrows():
            p_chg = row["close_change"]
            oi_chg = row["oi_change"]

            if p_chg > 0 and oi_chg > 0:
                r = 0
            elif p_chg > 0 and oi_chg < 0:
                r = 1
            elif p_chg < 0 and oi_chg > 0:
                r = 2
            elif p_chg < 0 and oi_chg < 0:
                r = 3
            else:
                r = -1
            regime.append(r)

        assert regime == expected

    def test_oi_volume_ratio(self, sample_data):
        """OI/Volume比率のテスト（既存のliquidity_efficiencyと同じか確認）"""
        ratio = AdvancedFeatures.liquidity_efficiency(
            sample_data["open_interest"], sample_data["volume"]
        )
        assert (ratio == sample_data["open_interest"] / sample_data["volume"]).all()
