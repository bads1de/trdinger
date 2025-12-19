import pytest
import pandas as pd
import numpy as np
from app.services.ml.label_generation.event_driven import EventDrivenLabelGenerator, BarrierProfile

class TestEventDrivenLabelGenerator:
    @pytest.fixture
    def generator(self):
        return EventDrivenLabelGenerator()

    @pytest.fixture
    def sample_market_data(self):
        """
        テスト用OHLCVデータを生成
        """
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        # 緩やかな上昇トレンド
        prices = 100 + np.linspace(0, 10, n) + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({
            "open": prices - 0.2,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": np.random.rand(n) * 1000
        }, index=dates)
        return df

    def test_generate_labels_basic(self, generator, sample_market_data):
        """基本的なラベル生成フローのテスト"""
        labels_df, info = generator.generate_hrhp_lrlp_labels(sample_market_data)
        
        assert isinstance(labels_df, pd.DataFrame)
        assert "label_hrhp" in labels_df.columns
        assert "label_lrlp" in labels_df.columns
        assert "market_regime" in labels_df.columns
        
        # サンプル数: 最後の1件は先を見れないため n-1 件になる仕様
        assert len(labels_df) == len(sample_market_data) - 1
        
        assert "label_distribution" in info
        assert "active_regime" in info

    def test_generate_labels_with_regimes(self, generator, sample_market_data):
        """異なるレジームラベルを指定した場合"""
        # 0, 1, 2 を混ぜたレジーム
        regimes = [0]*30 + [1]*30 + [2]*40
        labels_df, info = generator.generate_hrhp_lrlp_labels(sample_market_data, regime_labels=regimes)
        
        assert (labels_df["market_regime"].iloc[:30] == 0).all()
        assert (labels_df["market_regime"].iloc[30:60] == 1).all()
        assert info["active_regime"] == regimes[-1]

    def test_profile_overrides(self, generator, sample_market_data):
        """バリア設定の上書きテスト"""
        # 極端に狭いバリアを設定して、必ず接触するようにする
        overrides = {
            "hrhp": {"base_tp": 0.001, "base_sl": 0.001, "holding_period": 5}
        }
        labels_df, _ = generator.generate_hrhp_lrlp_labels(sample_market_data, profile_overrides=overrides)
        
        # バリアが狭いので、中立（0）が減り、1 または -1 が増えるはず
        dist = labels_df["label_hrhp"].value_counts()
        assert dist.get(1, 0) + dist.get(-1, 0) > 0

    def test_ensure_required_columns(self, generator):
        """カラムチェックと正規化のテスト"""
        # 大文字小文字の混在
        df = pd.DataFrame({
            "OPEN": [1], "High": [2], "low": [3], "Close": [4]
        })
        generator._ensure_required_columns(df)
        assert "open" in df.columns
        assert "close" in df.columns

        # 欠損時
        bad_df = pd.DataFrame({"close": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generator._ensure_required_columns(bad_df)

    def test_first_touch_logic(self, generator):
        """トリプルバリアの接触判定ロジックの詳細テスト"""
        # 1. 利確にヒット
        close = np.array([100.0, 101.0, 105.0])
        high = np.array([100.0, 102.0, 106.0])
        low = np.array([100.0, 99.0, 104.0])
        # tp=0.04 (104.0), sl=0.02 (98.0), holding=5
        label = generator._first_touch_label(close, high, low, start_idx=0, tp_mult=0.04, sl_mult=0.02, holding=5)
        assert label == 1

        # 2. 損切にヒット
        close = np.array([100.0, 99.0, 97.0])
        high = np.array([100.0, 100.0, 98.0])
        low = np.array([100.0, 98.5, 96.0])
        # tp=0.05, sl=0.03 (97.0)
        label = generator._first_touch_label(close, high, low, start_idx=0, tp_mult=0.05, sl_mult=0.03, holding=5)
        assert label == -1

        # 3. タイムアウト (どちらにも触れず期間終了)
        close = np.array([100.0, 100.1, 100.2])
        high = np.array([100.0, 101.0, 101.0])
        low = np.array([100.0, 99.0, 99.0])
        label = generator._first_touch_label(close, high, low, start_idx=0, tp_mult=0.1, sl_mult=0.1, holding=2)
        assert label == 0

    def test_simultaneous_hit(self, generator):
        """同じバーで利確と損切の両方に触れた場合、より深く刺さった方を優先する"""
        close = np.array([100.0, 100.0])
        # TP=105, SL=95
        # High=110 (TPから+5), Low=90 (SLから-5) -> 誤差が同じなら損切優先
        high = np.array([100.0, 110.0])
        low = np.array([100.0, 90.0])
        label = generator._first_touch_label(close, high, low, start_idx=0, tp_mult=0.05, sl_mult=0.05, holding=1)
        assert label == -1

        # High=120 (TPから+15), Low=94 (SLから-1) -> TPの方が深く刺さっている
        high = np.array([100.0, 120.0])
        low = np.array([100.0, 94.0])
        label = generator._first_touch_label(close, high, low, start_idx=0, tp_mult=0.05, sl_mult=0.05, holding=1)
        assert label == 1

    def test_zero_price_handling(self, generator):
        """価格が0以下の場合の安全な処理"""
        close = np.array([0.0, 1.0])
        high = np.array([0.0, 1.0])
        low = np.array([0.0, 0.0])
        label = generator._first_touch_label(close, high, low, start_idx=0, tp_mult=0.1, sl_mult=0.1, holding=1)
        assert label == 0

    def test_empty_distribution(self, generator):
        """空データ時の統計サマリー"""
        empty_s = pd.Series([], dtype=int)
        res = generator._distribution_summary(empty_s)
        assert res["positive_ratio"] == 0.0
        assert res["neutral_ratio"] == 0.0
