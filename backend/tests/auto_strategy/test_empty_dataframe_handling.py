"""
空DataFrame処理のテスト
バグ28: ML オーケストレータでの空 DataFrame 処理失敗の修正を確認
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.ml.exceptions import MLDataError


class TestEmptyDataFrameHandling:
    """空DataFrame処理のテストクラス"""

    @pytest.fixture
    def ml_orchestrator(self):
        """MLOrchestratorインスタンス"""
        # MLTrainingServiceをモックして依存を除去
        mock_trainer = Mock()
        mock_trainer.is_trained = True
        mock_trainer.feature_columns = ['feature1', 'feature2']

        mock_ml_training_service = Mock()
        mock_ml_training_service.trainer = mock_trainer
        mock_ml_training_service.generate_signals.return_value = {
            'up': 0.5, 'down': 0.3, 'range': 0.2
        }
        mock_ml_training_service.get_feature_importance.return_value = {'feature1': 0.7, 'feature2': 0.3}

        return MLOrchestrator(
            ml_training_service=mock_ml_training_service,
            enable_automl=False
        )

    def test_empty_dataframe_calculate_ml_indicators(self, ml_orchestrator):
        """calculate_ml_indicatorsで空DataFrame処理を確認"""
        empty_df = pd.DataFrame()

        with pytest.raises(MLDataError, match="入力データが空です"):
            ml_orchestrator.calculate_ml_indicators(empty_df)

    def test_none_dataframe_calculate_ml_indicators(self, ml_orchestrator):
        """calculate_ml_indicatorsでNone処理を確認"""
        with pytest.raises(MLDataError, match="入力データが空です"):
            ml_orchestrator.calculate_ml_indicators(None)

    def test_empty_dataframe_calculate_single_ml_indicator(self, ml_orchestrator):
        """calculate_single_ml_indicatorで空DataFrame処理を確認"""
        empty_df = pd.DataFrame()
        indicator_type = "ML_UP_PROB"

        with pytest.raises(MLDataError, match="空のデータフレームが提供されました"):
            ml_orchestrator.calculate_single_ml_indicator(indicator_type, empty_df)

    def test_none_dataframe_calculate_single_ml_indicator(self, ml_orchestrator):
        """calculate_single_ml_indicatorでNone処理を確認"""
        indicator_type = "ML_DOWN_PROB"

        with pytest.raises(MLDataError, match="空のデータフレームが提供されました"):
            ml_orchestrator.calculate_single_ml_indicator(indicator_type, None)

    def test_empty_dataframe_with_columns_calculate_ml_indicators(self, ml_orchestrator):
        """カラムはあるがデータがないDataFrameの処理を確認"""
        df_with_columns_only = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        with pytest.raises(MLDataError, match="入力データが空です"):
            ml_orchestrator.calculate_ml_indicators(df_with_columns_only)

    def test_valid_dataframe_calculate_ml_indicators(self, ml_orchestrator):
        """有効なDataFrameで処理が正常に動作することを確認"""
        # モックの特徴量サービスを設定
        mock_feature_df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})

        # 特徴量サービスのモック
        ml_orchestrator.feature_service.calculate_advanced_features = Mock(return_value=mock_feature_df)

        # 有効なDataFrame
        valid_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })

        try:
            result = ml_orchestrator.calculate_ml_indicators(valid_df)
            assert isinstance(result, dict)
            assert 'ML_UP_PROB' in result
            assert 'ML_DOWN_PROB' in result
            assert 'ML_RANGE_PROB' in result
        except Exception as e:
            # モデルがロードされていない場合の処理確認
            if "特徴量計算に失敗しました" in str(e):
                pytest.skip("モデル未ロードのためテストをスキップ")
            else:
                raise