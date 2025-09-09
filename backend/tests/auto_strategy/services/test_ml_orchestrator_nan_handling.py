"""
ML Orchestratorのnull値処理テスト

バグ19対応: DataFrame処理のNullPointerException修正テスト
TDD適用による実装検証
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator


class TestMLOrchestratorNullHandling:
    """MLOrchestratorのnull値処理テスト"""

    @pytest.fixture
    def orchestrator(self):
        """MLOrchestratorインスタンス生成"""
        # full_dataオプションを有効にしてテスト
        with patch('app.services.auto_strategy.services.ml_orchestrator.BacktestDataService'), \
             patch('app.services.auto_strategy.services.ml_orchestrator.MLTrainingService') as mock_ml:
            mock_ml.return_value.generate_signals.return_value = {"up": 0.5, "down": 0.3, "range": 0.2}
            return MLOrchestrator()

    def test_calculate_target_for_automl_handles_nan_values(self, orchestrator):
        """NaN値を含むDataFrameでのターゲット計算テスト"""
        # テストデータ: NaNを含む価格データ
        data = {
            "open": [1.0, 2.0, 3.0, np.nan, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.2, 2.2, 3.2, np.nan, 5.2],
            "volume": [100, 200, 300, 400, 500]
        }
        df = pd.DataFrame(data)

        # 実行: _calculate_target_for_automlを直接呼び出し
        result = orchestrator._calculate_target_for_automl(df)

        # 検証: Noneが返らず、NaN処理されていること
        assert result is not None, "ターゲット計算がNoneを返しました"
        assert len(result) == len(df), "結果の長さが入力データと一致しません"
        assert not result.isnull().any(), "結果にNaNが残っています"

    def test_calculate_target_for_automl_with_empty_dataframe(self, orchestrator):
        """空DataFrameでのターゲット計算テスト"""
        df = pd.DataFrame()

        # 実行
        result = orchestrator._calculate_target_for_automl(df)

        # 検証: Noneを返すこと
        assert result is None, "空DataFrameでNoneを返すべきです"

    def test_calculate_target_for_automl_without_close_column(self, orchestrator):
        """CloseカラムなしDataFrameでのターゲット計算テスト"""
        data = {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "volume": [100, 200, 300]
        }
        df = pd.DataFrame(data)

        # 実行
        result = orchestrator._calculate_target_for_automl(df)

        # 検証: Noneを返すこと
        assert result is None, "CloseカラムなしでNoneを返すべきです"

    def test_calculate_target_for_automl_pct_change_fillna(self, orchestrator):
        """pct_change後のfillna処理テスト"""
        # 最初の値がNaNになるデータ
        data = {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.0, 1.5, 2.0, 2.5],  # 初期値での変化を明確に
            "volume": [100, 200, 300, 400]
        }
        df = pd.DataFrame(data)

        # 実行
        result = orchestrator._calculate_target_for_automl(df)

        # 検証: pct_change後のfillnaで最初のNaNが処理されている
        assert result is not None
        assert result.iloc[0] == 0.0, "最初の値がfillされていない"
        assert not result.isnull().any(), "NaNがすべて処理されていない"

    @patch('app.services.auto_strategy.services.ml_orchestrator.data_preprocessor')
    def test_calculate_target_for_automl_interpolate_failure(self, mock_preprocessor, orchestrator):
        """interpolate_columns失敗時の処理テスト"""
        # interpolate_columnsがNoneを返すモック
        mock_preprocessor.interpolate_columns.return_value = None

        data = {
            "close": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        df = pd.DataFrame(data)

        # 実行
        result = orchestrator._calculate_target_for_automl(df)

        # 検証: Noneを返すこと
        assert result is None

    def test_calculate_features_null_check(self, orchestrator):
        """特徴量計算でのnullチェックテスト"""
        # Noneを返すモック設定
        with patch.object(orchestrator.feature_service, 'calculate_enhanced_features', return_value=None):
            df = pd.DataFrame({
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
                "volume": [100, 200, 300]
            })

            # 実行
            result = orchestrator._calculate_features(df)

            # 検証: NoneチェックによりNoneを返す
            assert result is None

    def test_ml_orchestrator_chain_operation_with_null_handling(self, orchestrator):
        """chain操作でのnullハンドリング統合テスト"""
        # NaNを含むデータ
        data = {
            "open": [1.0, 2.0, 3.0, np.nan, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.2, 2.2, 3.2, np.nan, 5.2],
            "volume": [100, 200, 300, 400, 500]
        }
        df = pd.DataFrame(data)

        # ML指標計算 (calculate_ml_indicators呼び出し)
        with patch.object(orchestrator, '_get_enhanced_data_with_fr_oi', return_value=df), \
             patch.object(orchestrator, '_calculate_features', return_value=df), \
             patch.object(orchestrator.ml_training_service, 'generate_signals') as mock_predict:
            mock_predict.return_value = {"up": 0.5, "down": 0.3, "range": 0.2}

            # 実行
            result = orchestrator.calculate_ml_indicators(df)

            # 検証: AttributeErrorが発生せず、結果が返されること
            assert isinstance(result, dict)
            assert len(result) == 3  # ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB
            assert all(len(values) == len(df) for values in result.values())

            # プロパティアクセスが有効
            for key, values in result.items():
                assert hasattr(values, 'shape'), f"{key} のshape属性アクセス失敗"
                assert values.shape[0] == len(df), f"{key} の型チェック失敗"