import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from backend.app.services.auto_strategy.config.ga import GASettings
from backend.app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from backend.app.services.auto_strategy.genes.strategy import StrategyGene
from backend.app.services.auto_strategy.utils.hybrid_feature_adapter import (
    HybridFeatureAdapter,
)
from backend.app.services.backtest.backtest_data_service import BacktestDataService
from app.services.ml.label_generation import EventDrivenLabelGenerator


def _build_sample_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 98, 99, 100],
            "high": [101, 103, 103, 99, 100, 102],
            "low": [99, 100, 100, 95, 97, 99],
            "close": [100, 102, 101, 97, 99, 101],
            "volume": [1000, 1100, 1050, 1200, 1150, 1180],
            "open_interest": [5000, 5100, 5200, 5300, 5250, 5400],
            "funding_rate": [0.0001, 0.00015, 0.0002, 0.00018, 0.00012, 0.00016],
        },
        index=index,
    )


def test_triple_barrier_labels_basic():
    data = _build_sample_market_data()
    regime_labels = np.array([0, 0, 1, 1, 2, 2])

    generator = EventDrivenLabelGenerator()
    labels_df, info = generator.generate_hrhp_lrlp_labels(
        data,
        regime_labels=regime_labels,
        profile_overrides={
            "hrhp": {"base_tp": 0.01, "base_sl": 0.01, "holding_period": 2},
            "lrlp": {"base_tp": 0.005, "base_sl": 0.005, "holding_period": 2},
        },
    )

    assert "label_hrhp" in labels_df.columns
    assert "label_lrlp" in labels_df.columns
    # 先頭サンプルは上方バリアに先に達する想定
    assert labels_df.iloc[0]["label_hrhp"] == 1
    # 2番目サンプルは下方バリアに先に達する想定
    assert labels_df.iloc[1]["label_hrhp"] == -1
    # プロファイル情報にはレジーム別情報が含まれること
    assert "regime_profiles" in info
    assert set(info["regime_profiles"].keys()) >= {0, 1, 2}


def test_backtest_data_service_event_labels(monkeypatch):
    data_service = BacktestDataService()
    sample_df = _build_sample_market_data()

    integration_mock = MagicMock()
    integration_mock.create_ml_training_dataframe.return_value = sample_df
    # BacktestDataService内部の統合サービスをモックに差し替え
    data_service._integration_service = integration_mock  # type: ignore[attr-defined]

    labeled_df, profile_info = data_service.get_event_labeled_training_data(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date=datetime.datetime(2024, 1, 1),
        end_date=datetime.datetime(2024, 1, 1, 5),
    )

    assert "label_hrhp" in labeled_df.columns
    assert "label_lrlp" in labeled_df.columns
    assert "market_regime" in labeled_df.columns
    assert "label_distribution" in profile_info
    assert "hrhp" in profile_info["label_distribution"]


def test_random_gene_generator_context_injects_regime_thresholds():
    config = GASettings()
    context = {
        "timeframe": "1h",
        "symbol": "BTCUSDT",
        "regime_thresholds": {
            0: {"hrhp": {"take_profit": 0.02, "stop_loss": 0.01}},
            1: {"hrhp": {"take_profit": 0.015, "stop_loss": 0.01}},
        },
    }

    generator = RandomGeneGenerator(config, smart_context=context)
    stored_context = generator.smart_condition_generator.context

    assert stored_context.get("regime_thresholds") == context["regime_thresholds"]


def test_hybrid_feature_adapter_label_features():
    adapter = HybridFeatureAdapter()
    gene = StrategyGene()
    market_df = _build_sample_market_data()

    labels_df = pd.DataFrame(
        {
            "label_hrhp": [1, -1, 0, 1, 0],
            "label_lrlp": [1, 0, -1, 1, 0],
            "market_regime": [0, 0, 1, 1, 2],
        },
        index=market_df.index[:-1],
    )
    sentiment = pd.Series([0.1, -0.2, 0.0, 0.3, -0.1], index=labels_df.index)

    features_df = adapter.gene_to_features(
        gene,
        market_df,
        apply_preprocessing=False,
        label_data=labels_df,
        sentiment_scores=sentiment,
    )

    assert "label_hrhp_signal" in features_df.columns
    assert "oi_pct_change" in features_df.columns
    assert "funding_rate_change" in features_df.columns
    assert "sentiment_smoothed" in features_df.columns
    assert not features_df["sentiment_smoothed"].isna().any()




