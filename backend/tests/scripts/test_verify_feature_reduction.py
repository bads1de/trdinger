from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# 実体ファイルをインポートするためにパスを調整
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


SCRIPT_MODULE = "backend.scripts.verify_feature_reduction"
MOCK_MODULES = (
    "database.connection",
    "database.repositories.ohlcv_repository",
    "database.repositories.funding_rate_repository",
    "database.repositories.open_interest_repository",
    "database.repositories.long_short_ratio_repository",
    "app.config.unified_config",
)


@pytest.fixture
def verify_feature_reduction_module(monkeypatch):
    """スクリプトの依存関係をテストごとに隔離して読み込む。"""
    for module_name in MOCK_MODULES:
        monkeypatch.setitem(sys.modules, module_name, MagicMock())

    # 以前の import 結果を使い回さず、モックを反映した状態で再読み込みする。
    sys.modules.pop(SCRIPT_MODULE, None)
    module = importlib.import_module(SCRIPT_MODULE)

    yield module

    sys.modules.pop(SCRIPT_MODULE, None)


@pytest.fixture
def mock_data():
    idx = pd.date_range("2024-01-01", periods=300, freq="1h")
    ohlcv_df = pd.DataFrame(
        {
            "open": np.random.rand(300),
            "high": np.random.rand(300),
            "low": np.random.rand(300),
            "close": np.random.rand(300),
            "volume": np.random.rand(300),
        },
        index=idx,
    )
    return ohlcv_df


def test_prepare_model_data_with_none_ohlcv_1m(
    mock_data, verify_feature_reduction_module
):
    """ohlcv_1m が None の場合にクラッシュせず動作することを確認"""
    module = verify_feature_reduction_module

    with (
        patch.object(module, "fetch_all_data") as mock_fetch,
        patch.object(module, "FeatureEngineeringService") as mock_fe_service_cls,
        patch.object(module, "FeatureSelector") as mock_selector_cls,
    ):

        # ohlcv_1m を None に設定
        mock_fetch.return_value = (mock_data, None, None, None, None)

        # モックサービスの設定
        mock_fe_service = mock_fe_service_cls.return_value
        # superset は index さえ合っていれば良い
        mock_fe_service.create_feature_superset.return_value = pd.DataFrame(
            {"feat1": np.random.rand(300)}, index=mock_data.index
        )
        mock_fe_service.expand_features.return_value = pd.DataFrame(
            {"feat1": np.random.rand(300)}, index=mock_data.index
        )

        # セレクターのモック
        mock_selector = mock_selector_cls.return_value
        mock_selector.fit_transform.return_value = pd.DataFrame(
            {"feat1": np.random.rand(300)}, index=mock_data.index
        )

        # 実行
        with patch.object(module, "triple_barrier_method_preset") as mock_preset:
            mock_preset.return_value = pd.Series([0, 1] * 150, index=mock_data.index)

            result = module.prepare_model_data("SYMBOL", "1h", 100, "triple_barrier")

            assert result is not None
            assert "X_elite" in result
            # aggregate_intraday_features が呼ばれていないことを確認
            assert mock_fe_service.aggregate_intraday_features.called is False


def test_run_analysis_pipeline(verify_feature_reduction_module):
    """run_analysis_pipeline が完走することを確認"""
    module = verify_feature_reduction_module
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    X_elite = pd.DataFrame(
        {
            "feat1": np.linspace(0.0, 1.0, len(index)),
            "feat2": np.linspace(1.0, 2.0, len(index)),
        },
        index=index,
    )
    y_model_all = pd.Series([0, 1] * (len(index) // 2), index=index)
    w_model_all = pd.Series(np.ones(len(index)), index=index)
    prepared_data = {
        "X_elite": X_elite,
        "y_model_all": y_model_all,
        "w_model_all": w_model_all,
        "elite_cols": list(X_elite.columns),
        "X_model_all": X_elite,
        "ohlcv_df": pd.DataFrame(
            {
                "open": np.linspace(100.0, 149.0, len(index)),
                "high": np.linspace(101.0, 150.0, len(index)),
                "low": np.linspace(99.0, 148.0, len(index)),
                "close": np.linspace(100.5, 149.5, len(index)),
                "volume": np.linspace(1000.0, 1049.0, len(index)),
            },
            index=index,
        ),
    }

    with (
        patch.object(module, "prepare_model_data", return_value=prepared_data),
        patch.object(module, "LGBMClassifier") as mock_lgbm_cls,
        patch.object(module, "MetaLabelingService") as mock_meta_service_cls,
    ):
        mock_model = mock_lgbm_cls.return_value
        mock_model.fit.return_value = None
        mock_model.predict_proba.side_effect = lambda data: np.column_stack(
            (
                np.full(len(data), 0.25),
                np.full(len(data), 0.75),
            )
        )

        mock_meta_service = mock_meta_service_cls.return_value
        mock_meta_service.train.return_value = {"status": "skipped", "reason": "mock"}

        # Execute
        res = module.run_analysis_pipeline(
            "SYMBOL", "1h", 100, "triple_barrier", save_json=False
        )

        assert res is not None
        assert res["winner"] in ["Meta Model", "Primary Model"]
