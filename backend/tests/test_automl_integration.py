"""
AutoML統合テスト

MLOrchestratorのAutoML機能統合を検証します。

テスト項目:
1. AutoML機能の有効/無効切り替え
2. EnhancedFeatureEngineeringServiceの正常な呼び出し
3. ターゲット変数の計算
4. AutoML設定の動的変更
5. 基本特徴量とAutoML特徴量の比較
"""

import pytest
import logging
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.ml.feature_engineering.enhanced_feature_engineering_service import EnhancedFeatureEngineeringService
from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService

logger = logging.getLogger(__name__)


class TestAutoMLIntegration:
    """AutoML統合テストクラス"""

    def setup_method(self):
        """テストセットアップ"""
        # テスト用のサンプルデータを作成
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

    def test_automl_enabled_initialization(self):
        """1. AutoML有効での初期化テスト"""
        logger.info("=== AutoML有効での初期化テスト ===")

        # AutoML有効でMLOrchestratorを初期化
        orchestrator = MLOrchestrator(enable_automl=True)
        
        # AutoML機能が有効になっていることを確認
        assert orchestrator.enable_automl == True
        assert isinstance(orchestrator.feature_service, EnhancedFeatureEngineeringService)
        
        # AutoML状態を確認
        status = orchestrator.get_automl_status()
        assert status["enabled"] == True
        assert status["service_type"] == "EnhancedFeatureEngineeringService"
        
        logger.info("✅ AutoML有効での初期化: 成功")

    def test_automl_disabled_initialization(self):
        """2. AutoML無効での初期化テスト"""
        logger.info("=== AutoML無効での初期化テスト ===")

        # AutoML無効でMLOrchestratorを初期化
        orchestrator = MLOrchestrator(enable_automl=False)
        
        # AutoML機能が無効になっていることを確認
        assert orchestrator.enable_automl == False
        assert isinstance(orchestrator.feature_service, FeatureEngineeringService)
        
        # AutoML状態を確認
        status = orchestrator.get_automl_status()
        assert status["enabled"] == False
        assert status["service_type"] == "FeatureEngineeringService"
        
        logger.info("✅ AutoML無効での初期化: 成功")

    def test_automl_toggle(self):
        """3. AutoML機能の動的切り替えテスト"""
        logger.info("=== AutoML機能の動的切り替えテスト ===")

        # 基本機能で開始
        orchestrator = MLOrchestrator(enable_automl=False)
        assert orchestrator.enable_automl == False
        
        # AutoMLに切り替え
        orchestrator.set_automl_enabled(True)
        assert orchestrator.enable_automl == True
        assert isinstance(orchestrator.feature_service, EnhancedFeatureEngineeringService)
        
        # 基本機能に戻す
        orchestrator.set_automl_enabled(False)
        assert orchestrator.enable_automl == False
        assert isinstance(orchestrator.feature_service, FeatureEngineeringService)
        
        logger.info("✅ AutoML機能の動的切り替え: 成功")

    def test_automl_config_creation(self):
        """4. AutoML設定作成テスト"""
        logger.info("=== AutoML設定作成テスト ===")

        orchestrator = MLOrchestrator(enable_automl=False)
        
        # カスタムAutoML設定
        custom_config = {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "fdr_level": 0.01,
                "feature_count_limit": 200,
                "parallel_jobs": 4,
            },
            "autofeat": {
                "enabled": True,
                "max_features": 100,
                "feateng_steps": 3,
                "max_gb": 2.0,
            }
        }
        
        # AutoML設定オブジェクトを作成
        automl_config_obj = orchestrator._create_automl_config_from_dict(custom_config)
        
        # 設定が正しく作成されていることを確認
        assert automl_config_obj.tsfresh.enabled == True
        assert automl_config_obj.tsfresh.fdr_level == 0.01
        assert automl_config_obj.tsfresh.feature_count_limit == 200
        assert automl_config_obj.autofeat.enabled == True
        assert automl_config_obj.autofeat.max_features == 100
        
        logger.info("✅ AutoML設定作成: 成功")

    def test_target_calculation(self):
        """5. ターゲット変数計算テスト"""
        logger.info("=== ターゲット変数計算テスト ===")

        orchestrator = MLOrchestrator(enable_automl=True)
        
        # カラム名を大文字に変換
        df_for_target = self.sample_data.copy()
        df_for_target.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # ターゲット変数を計算
        target = orchestrator._calculate_target_for_automl(df_for_target)
        
        # ターゲット変数が正しく計算されていることを確認
        assert target is not None
        assert len(target) == len(df_for_target)
        assert isinstance(target, pd.Series)
        
        # 価格変化率が計算されていることを確認
        expected_changes = df_for_target['Close'].pct_change().shift(-1).fillna(0)
        pd.testing.assert_series_equal(target, expected_changes, check_names=False)
        
        logger.info("✅ ターゲット変数計算: 成功")

    @patch('app.services.ml.feature_engineering.enhanced_feature_engineering_service.EnhancedFeatureEngineeringService.calculate_enhanced_features')
    def test_enhanced_feature_calculation(self, mock_enhanced_features):
        """6. 拡張特徴量計算の呼び出しテスト"""
        logger.info("=== 拡張特徴量計算の呼び出しテスト ===")

        # モックの設定
        mock_result = self.sample_data.copy()
        mock_result['automl_feature_1'] = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_enhanced_features.return_value = mock_result

        # AutoML有効でオーケストレーターを作成
        orchestrator = MLOrchestrator(enable_automl=True)
        
        # 特徴量計算を実行
        result = orchestrator._calculate_features(self.sample_data)
        
        # EnhancedFeatureEngineeringServiceが呼び出されたことを確認
        mock_enhanced_features.assert_called_once()
        
        # 結果が正しく返されることを確認
        assert result is not None
        assert 'automl_feature_1' in result.columns
        
        logger.info("✅ 拡張特徴量計算の呼び出し: 成功")

    @patch('app.services.ml.feature_engineering.feature_engineering_service.FeatureEngineeringService.calculate_advanced_features')
    def test_basic_feature_calculation(self, mock_basic_features):
        """7. 基本特徴量計算の呼び出しテスト"""
        logger.info("=== 基本特徴量計算の呼び出しテスト ===")

        # モックの設定
        mock_result = self.sample_data.copy()
        mock_result['basic_feature_1'] = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_basic_features.return_value = mock_result

        # AutoML無効でオーケストレーターを作成
        orchestrator = MLOrchestrator(enable_automl=False)
        
        # 特徴量計算を実行
        result = orchestrator._calculate_features(self.sample_data)
        
        # FeatureEngineeringServiceが呼び出されたことを確認
        mock_basic_features.assert_called_once()
        
        # 結果が正しく返されることを確認
        assert result is not None
        assert 'basic_feature_1' in result.columns
        
        logger.info("✅ 基本特徴量計算の呼び出し: 成功")

    def test_automl_status_reporting(self):
        """8. AutoML状態レポートテスト"""
        logger.info("=== AutoML状態レポートテスト ===")

        # AutoML有効でテスト
        orchestrator = MLOrchestrator(enable_automl=True)
        status = orchestrator.get_automl_status()
        
        assert "enabled" in status
        assert "service_type" in status
        assert "config" in status
        assert "available_features" in status
        
        assert status["enabled"] == True
        assert status["service_type"] == "EnhancedFeatureEngineeringService"
        
        # AutoML無効でテスト
        orchestrator.set_automl_enabled(False)
        status = orchestrator.get_automl_status()
        
        assert status["enabled"] == False
        assert status["service_type"] == "FeatureEngineeringService"
        
        logger.info("✅ AutoML状態レポート: 成功")

    def test_automl_integration_summary(self):
        """9. AutoML統合の総合確認"""
        logger.info("=== AutoML統合の総合確認 ===")

        summary = {
            "automl_initialization": True,
            "dynamic_toggle": True,
            "config_management": True,
            "feature_calculation": True,
            "target_calculation": True,
            "status_reporting": True
        }

        try:
            # 各機能の確認
            orchestrator = MLOrchestrator(enable_automl=True)
            assert isinstance(orchestrator.feature_service, EnhancedFeatureEngineeringService)
            
            # 動的切り替えの確認
            orchestrator.set_automl_enabled(False)
            assert isinstance(orchestrator.feature_service, FeatureEngineeringService)
            
            # 設定管理の確認
            config = {"tsfresh": {"enabled": True}, "autofeat": {"enabled": True}}
            orchestrator.set_automl_enabled(True, config)
            assert orchestrator.automl_config == config
            
            # ターゲット計算の確認
            df_test = self.sample_data.copy()
            df_test.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            target = orchestrator._calculate_target_for_automl(df_test)
            assert target is not None
            
            logger.info("✅ AutoML統合の総合確認: 成功")
            logger.info(f"統合機能サマリー: {summary}")
            
        except Exception as e:
            pytest.fail(f"AutoML統合の確認でエラーが発生しました: {e}")


if __name__ == "__main__":
    # テストを直接実行する場合
    pytest.main([__file__, "-v", "--tb=short"])
