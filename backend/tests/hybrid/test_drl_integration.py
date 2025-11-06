"""
DRL連携とウェーブレット拡張のテスト

TDD: まず新機能向けのテストを定義して失敗を確認する。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from app.services.auto_strategy.models.strategy_gene import StrategyGene


@pytest.fixture
def sample_gene() -> StrategyGene:
    """最小構成のStrategyGeneインスタンス"""

    return StrategyGene(id="gene-test-001")


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """波形生成用のOHLCVサンプル"""

    periods = 32
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="H")
    base = np.linspace(42000, 43000, periods)
    data = {
        "timestamp": timestamps,
        "open": base + np.random.randn(periods),
        "high": base + 10 + np.random.randn(periods),
        "low": base - 10 + np.random.randn(periods),
        "close": base + np.random.randn(periods),
        "volume": np.linspace(1000, 2000, periods),
    }
    return pd.DataFrame(data).set_index("timestamp")


class _DummyTrainingService:
    """HybridPredictor用のダミートレーニングサービス"""

    def __init__(
        self,
        trainer_type: str = "single",
        single_model_config: Optional[Dict[str, Any]] = None,
        ensemble_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trainer = Mock()
        self.trainer.is_trained = True
        self.trainer.model = object()

    def generate_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        return {"up": 0.4, "down": 0.3, "range": 0.3}


def test_feature_adapter_adds_wavelet_features(
    sample_gene: StrategyGene, sample_ohlcv: pd.DataFrame
) -> None:
    """ウェーブレット設定が有効な場合に特徴量が追加されることを検証"""

    from app.services.auto_strategy.utils.hybrid_feature_adapter import (
        HybridFeatureAdapter,
    )

    wavelet_config = {
        "enabled": True,
        "base_wavelet": "haar",
        "scales": [2, 4],
        "target_columns": ["close"],
    }

    adapter = HybridFeatureAdapter(wavelet_config=wavelet_config)
    feature_df = adapter.gene_to_features(
        sample_gene, sample_ohlcv, apply_preprocessing=False
    )

    assert "wavelet_close_scale_2" in feature_df.columns
    assert "wavelet_close_scale_4" in feature_df.columns
    assert feature_df["wavelet_close_scale_2"].abs().sum() > 0


def test_hybrid_predictor_blends_drl_signals(sample_ohlcv: pd.DataFrame) -> None:
    """HybridPredictorがDRL出力をウェイト付きで混合することを検証"""

    from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor

    drl_adapter = Mock()
    drl_adapter.predict_signals.return_value = {"up": 0.8, "down": 0.1, "range": 0.1}

    drl_config = {
        "enabled": True,
        "policy_type": "ppo",
        "policy_weight": 0.6,
    }

    predictor = HybridPredictor(
        trainer_type="single",
        model_type="lightgbm",
        drl_config=drl_config,
        training_service_cls=_DummyTrainingService,
        drl_policy_adapter=drl_adapter,
    )

    predictions = predictor.predict(sample_ohlcv)

    expected_up = 0.6 * 0.8 + 0.4 * 0.4
    expected_down = 0.6 * 0.1 + 0.4 * 0.3
    expected_range = 1.0 - expected_up - expected_down

    assert pytest.approx(predictions["up"], rel=1e-5) == expected_up
    assert pytest.approx(predictions["down"], rel=1e-5) == expected_down
    assert pytest.approx(predictions["range"], rel=1e-5) == expected_range
    assert drl_adapter.predict_signals.called
