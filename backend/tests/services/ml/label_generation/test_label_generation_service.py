"""
LabelGenerationService のユニットテスト（メタラベリング対応）

SignalGenerator を使ってイベントに基づいたラベリングをテストします。
"""

import pandas as pd
import numpy as np
import pytest

from app.services.ml.label_generation.label_generation_service import (
    LabelGenerationService,
)


class TestLabelGenerationServiceWithEvents:
    """SignalGenerator を使用したイベントベースのラベル生成テスト"""

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))

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
    def sample_features(self, sample_ohlcv):
        """テスト用の特徴量DataFrameを生成"""
        # 簡単な特徴量を作成
        df = sample_ohlcv.copy()
        df["feature_1"] = df["close"].pct_change(fill_method=None)
        df["feature_2"] = df["volume"].rolling(window=5).mean()

        features = df[["feature_1", "feature_2"]].fillna(0)
        return features

    def test_prepare_labels_without_signal_generator(
        self, sample_features, sample_ohlcv
    ):
        """SignalGenerator なしの従来の動作をテスト"""
        service = LabelGenerationService()

        features_clean, labels_clean = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=sample_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
        )

        # すべての足がラベリング対象になっている（イベントフィルタなし）
        # 実際のサンプル数はTriple Barrierの制約により少なくなる
        assert len(features_clean) > 0
        assert len(labels_clean) == len(features_clean)

    def test_prepare_labels_with_signal_generator(self, sample_features, sample_ohlcv):
        """SignalGenerator ありでイベントフィルタリングをテスト"""
        service = LabelGenerationService()

        # SignalGeneratorを使用してイベントベースのラベリング
        features_clean, labels_clean = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=sample_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
            use_signal_generator=True,
            signal_config={
                "use_bb": True,
                "use_donchian": False,
                "use_volume": False,
                "bb_window": 20,
                "bb_dev": 2.0,
            },
        )

        # イベントフィルタリングにより、サンプル数が減少する
        assert len(features_clean) > 0
        assert len(labels_clean) == len(features_clean)

        # イベントのみがラベリングされているため、全足より少ない
        assert len(features_clean) < len(sample_ohlcv)

    def test_prepare_labels_with_multiple_signals(self, sample_features, sample_ohlcv):
        """複数のシグナルを組み合わせたイベントフィルタリングをテスト"""
        service = LabelGenerationService()

        # 複数のシグナルを組み合わせ
        features_clean, labels_clean = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=sample_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
            use_signal_generator=True,
            signal_config={
                "use_bb": True,
                "use_donchian": True,
                "use_volume": True,
                "bb_window": 20,
                "bb_dev": 2.0,
                "donchian_window": 20,
                "volume_window": 20,
                "volume_multiplier": 2.0,
            },
        )

        # 複数シグナルの組み合わせでも正常に動作
        assert len(features_clean) > 0
        assert len(labels_clean) == len(features_clean)

    def test_prepare_labels_event_filtering_reduces_samples(
        self, sample_features, sample_ohlcv
    ):
        """イベントフィルタリングによりサンプル数が減ることを確認"""
        service = LabelGenerationService()

        # フィルタなし
        features_all, labels_all = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=sample_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
            use_signal_generator=False,
        )

        # フィルタあり
        features_filtered, labels_filtered = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=sample_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
            use_signal_generator=True,
            signal_config={
                "use_bb": True,
                "use_donchian": False,
                "use_volume": False,
                "bb_window": 20,
                "bb_dev": 2.0,
            },
        )

        # イベントフィルタリングによりサンプル数が減少することを確認
        assert len(features_filtered) <= len(features_all)

    def test_prepare_labels_with_no_events_detected(self, sample_features):
        """イベントが1つも検出されない場合の挙動をテスト"""
        # 完全に平坦なデータ（ブレイクアウトが発生しない）
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        flat_ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [100.5] * 100,
                "low": [99.5] * 100,
                "close": [100.0] * 100,
                "volume": [1000.0] * 100,
            },
            index=dates,
        )

        service = LabelGenerationService()

        # イベントが検出されない場合、空のDataFrame/Seriesが返される
        features_clean, labels_clean = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=flat_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
            use_signal_generator=True,
            signal_config={"use_bb": True, "bb_window": 20, "bb_dev": 2.0},
        )

        # イベントが検出されない場合、サンプル数は0
        assert len(features_clean) == 0
        assert len(labels_clean) == 0

    def test_prepare_labels_signal_config_defaults(self, sample_features, sample_ohlcv):
        """signal_config のデフォルト値が正しく適用されることをテスト"""
        service = LabelGenerationService()

        # signal_config を省略した場合、デフォルト設定が適用される
        features_clean, labels_clean = service.prepare_labels(
            features_df=sample_features,
            ohlcv_df=sample_ohlcv,
            threshold_method="TRIPLE_BARRIER",
            pt_factor=1.0,
            sl_factor=1.0,
            use_atr=False,
            use_signal_generator=True,
        )

        # デフォルト設定でも正常に動作
        assert len(features_clean) > 0
        assert len(labels_clean) == len(features_clean)
