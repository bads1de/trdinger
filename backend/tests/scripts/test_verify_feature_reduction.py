import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# 実体ファイルをインポートするためにパスを調整
import sys
from pathlib import Path

# プロジェクトルート (trading/) を追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# モック
from unittest.mock import MagicMock, patch
sys.modules['database.connection'] = MagicMock()
sys.modules['database.repositories.ohlcv_repository'] = MagicMock()
sys.modules['database.repositories.funding_rate_repository'] = MagicMock()
sys.modules['database.repositories.open_interest_repository'] = MagicMock()
sys.modules['database.repositories.long_short_ratio_repository'] = MagicMock()
sys.modules['app.config.unified_config'] = MagicMock()

# ここでインポートするように変更
# backend/scripts/ をモジュールとして扱えるように
from backend.scripts.verify_feature_reduction import prepare_model_data

@pytest.fixture
def mock_data():
    idx = pd.date_range("2024-01-01", periods=300, freq="1h")
    ohlcv_df = pd.DataFrame({
        "open": np.random.rand(300),
        "high": np.random.rand(300),
        "low": np.random.rand(300),
        "close": np.random.rand(300),
        "volume": np.random.rand(300)
    }, index=idx)
    return ohlcv_df

def test_prepare_model_data_with_none_ohlcv_1m(mock_data):
    """ohlcv_1m が None の場合にクラッシュせず動作することを確認"""
    with patch("backend.scripts.verify_feature_reduction.fetch_all_data") as mock_fetch, \
         patch("backend.scripts.verify_feature_reduction.FeatureEngineeringService") as mock_fe_service_cls, \
         patch("backend.scripts.verify_feature_reduction.FeatureSelector") as mock_selector_cls:
        
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
        with patch("backend.scripts.verify_feature_reduction.triple_barrier_method_preset") as mock_preset:
            mock_preset.return_value = pd.Series([0, 1] * 150, index=mock_data.index)
            
            result = prepare_model_data("SYMBOL", "1h", 100, "triple_barrier")
            
            assert result is not None
            assert "X_elite" in result
            # aggregate_intraday_features が呼ばれていないことを確認
            assert mock_fe_service.aggregate_intraday_features.called is False

def test_run_analysis_pipeline(mock_data):
    """run_analysis_pipeline が完走することを確認"""
    from backend.scripts.verify_feature_reduction import run_analysis_pipeline
    
    with patch("backend.scripts.verify_feature_reduction.fetch_all_data") as mock_fetch, \
         patch("backend.scripts.verify_feature_reduction.FeatureEngineeringService") as mock_fe_service_cls, \
         patch("backend.scripts.verify_feature_reduction.FeatureSelector") as mock_selector_cls, \
         patch("backend.scripts.verify_feature_reduction.triple_barrier_method_preset") as mock_preset:
        
        # 1. Mock fetch_all_data
        mock_fetch.return_value = (mock_data, None, None, None, None)
        
        # 2. Mock FeatureEngineeringService
        mock_fe_service = mock_fe_service_cls.return_value
        mock_fe_service.create_feature_superset.return_value = pd.DataFrame(
            {"feat1": np.random.rand(300)}, index=mock_data.index
        )
        mock_fe_service.expand_features.return_value = pd.DataFrame(
            {"feat1": np.random.rand(300)}, index=mock_data.index
        )
        
        # 3. Mock FeatureSelector
        mock_selector = mock_selector_cls.return_value
        mock_selector.fit_transform.return_value = pd.DataFrame(
            {"feat1": np.random.rand(300)}, index=mock_data.index
        )
        
        # 4. Mock labeling
        mock_preset.return_value = pd.Series([0, 1] * 150, index=mock_data.index)
        
        # Execute
        res = run_analysis_pipeline("SYMBOL", "1h", 100, "triple_barrier", save_json=False)
        
        assert res is not None
        assert res["winner"] in ["Meta Model", "Primary Model"]
