"""
ステップ・バイ・ステップ特徴量生成テスト

EnhancedFeatureEngineeringServiceの効率化されたフローを検証します。

テスト項目:
1. ステップ1: 手動特徴量計算
2. ステップ2: TSFresh特徴量 + 特徴量選択
3. ステップ3: AutoFeat特徴量 + 特徴量選択
4. 特徴量選択機能
5. 統合フローの動作確認
"""

import pytest
import logging
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.ml.feature_engineering.enhanced_feature_engineering_service import EnhancedFeatureEngineeringService
from app.services.ml.feature_engineering.automl_features.automl_config import AutoMLConfig

logger = logging.getLogger(__name__)


class TestStepByStepFeatures:
    """ステップ・バイ・ステップ特徴量生成テストクラス"""

    def setup_method(self):
        """テストセットアップ"""
        # DatetimeIndexを持つテスト用のサンプルデータを作成
        dates = pd.date_range('2024-01-01', periods=5, freq='1H')
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

        # ターゲット変数を作成
        self.target = pd.Series([0.1, 0.2, -0.1, 0.3, -0.2], name='target', index=dates)

        # AutoML設定を作成
        self.automl_config = AutoMLConfig.get_financial_optimized_config()

    def test_step1_manual_features(self):
        """1. ステップ1: 手動特徴量計算テスト"""
        logger.info("=== ステップ1: 手動特徴量計算テスト ===")

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # ステップ1を実行
        result_df = service._step1_manual_features(self.sample_data)
        
        # 結果の検証
        assert result_df is not None
        assert len(result_df) == len(self.sample_data)
        assert len(result_df.columns) > len(self.sample_data.columns)  # 特徴量が追加されている
        
        # 統計情報が記録されていることを確認
        assert "manual_features" in service.last_enhancement_stats
        assert "manual_time" in service.last_enhancement_stats
        assert service.last_enhancement_stats["manual_features"] > 0
        
        logger.info("✅ ステップ1: 手動特徴量計算: 成功")

    @patch('app.services.ml.feature_engineering.automl_features.tsfresh_calculator.TSFreshFeatureCalculator.calculate_tsfresh_features')
    def test_step2_tsfresh_features(self, mock_tsfresh):
        """2. ステップ2: TSFresh特徴量 + 特徴量選択テスト"""
        logger.info("=== ステップ2: TSFresh特徴量 + 特徴量選択テスト ===")

        # モックの設定
        mock_result = self.sample_data.copy()
        for i in range(50):  # 50個の特徴量を追加
            mock_result[f'tsfresh_feature_{i}'] = np.random.random(len(self.sample_data))
        mock_tsfresh.return_value = mock_result

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # ステップ2を実行（特徴量数制限あり）
        result_df = service._step2_tsfresh_features(
            self.sample_data, self.target, max_features=20
        )
        
        # 結果の検証
        assert result_df is not None
        assert len(result_df.columns) <= 20  # 特徴量選択により制限されている
        
        # TSFreshが呼び出されたことを確認
        mock_tsfresh.assert_called_once()
        
        # 統計情報が記録されていることを確認
        assert "tsfresh_features" in service.last_enhancement_stats
        assert "tsfresh_time" in service.last_enhancement_stats
        
        logger.info("✅ ステップ2: TSFresh特徴量 + 特徴量選択: 成功")

    @patch('app.services.ml.feature_engineering.automl_features.autofeat_calculator.AutoFeatCalculator.generate_features')
    def test_step3_autofeat_features(self, mock_autofeat):
        """3. ステップ3: AutoFeat特徴量 + 特徴量選択テスト"""
        logger.info("=== ステップ3: AutoFeat特徴量 + 特徴量選択テスト ===")

        # モックの設定
        mock_result = self.sample_data.copy()
        for i in range(30):  # 30個の特徴量を追加
            mock_result[f'autofeat_feature_{i}'] = np.random.random(len(self.sample_data))
        mock_autofeat.return_value = (mock_result, {"generated_features": 30})

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # ステップ3を実行（特徴量数制限あり）
        result_df = service._step3_autofeat_features(
            self.sample_data, self.target, max_features=15
        )
        
        # 結果の検証
        assert result_df is not None
        assert len(result_df.columns) <= 15  # 特徴量選択により制限されている
        
        # AutoFeatが呼び出されたことを確認
        mock_autofeat.assert_called_once()
        
        # 統計情報が記録されていることを確認
        assert "autofeat_features" in service.last_enhancement_stats
        assert "autofeat_time" in service.last_enhancement_stats
        
        logger.info("✅ ステップ3: AutoFeat特徴量 + 特徴量選択: 成功")

    def test_step3_without_target(self):
        """4. ターゲット変数なしでのステップ3テスト"""
        logger.info("=== ターゲット変数なしでのステップ3テスト ===")

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # ターゲット変数なしでステップ3を実行
        result_df = service._step3_autofeat_features(
            self.sample_data, target=None, max_features=15
        )
        
        # 元のDataFrameがそのまま返されることを確認
        pd.testing.assert_frame_equal(result_df, self.sample_data)
        
        logger.info("✅ ターゲット変数なしでのステップ3: 成功")

    def test_feature_selection(self):
        """5. 特徴量選択機能テスト"""
        logger.info("=== 特徴量選択機能テスト ===")

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # 多数の特徴量を持つDataFrameを作成
        large_df = self.sample_data.copy()
        for i in range(50):
            large_df[f'feature_{i}'] = np.random.random(len(self.sample_data))
        
        # 特徴量選択を実行
        selected_df = service._select_top_features(large_df, self.target, max_features=10)
        
        # 結果の検証
        assert selected_df is not None
        assert len(selected_df.columns) == 10  # 指定した数の特徴量が選択されている
        assert len(selected_df) == len(large_df)  # 行数は変わらない
        
        logger.info("✅ 特徴量選択機能: 成功")

    def test_feature_selection_without_target(self):
        """6. ターゲット変数なしでの特徴量選択テスト"""
        logger.info("=== ターゲット変数なしでの特徴量選択テスト ===")

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # ターゲット変数なしで特徴量選択を実行
        result_df = service._select_top_features(self.sample_data, target=None, max_features=3)
        
        # 元のDataFrameがそのまま返されることを確認
        pd.testing.assert_frame_equal(result_df, self.sample_data)
        
        logger.info("✅ ターゲット変数なしでの特徴量選択: 成功")

    @patch('app.services.ml.feature_engineering.automl_features.tsfresh_calculator.TSFreshFeatureCalculator.calculate_tsfresh_features')
    @patch('app.services.ml.feature_engineering.automl_features.autofeat_calculator.AutoFeatCalculator.generate_features')
    def test_integrated_step_by_step_flow(self, mock_autofeat, mock_tsfresh):
        """7. 統合ステップ・バイ・ステップフローテスト"""
        logger.info("=== 統合ステップ・バイ・ステップフローテスト ===")

        # モックの設定
        tsfresh_result = self.sample_data.copy()
        for i in range(20):
            tsfresh_result[f'tsfresh_{i}'] = np.random.random(len(self.sample_data))
        mock_tsfresh.return_value = tsfresh_result

        autofeat_result = tsfresh_result.copy()
        for i in range(15):
            autofeat_result[f'autofeat_{i}'] = np.random.random(len(self.sample_data))
        mock_autofeat.return_value = (autofeat_result, {"generated_features": 15})

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # 統合フローを実行
        result_df = service.calculate_enhanced_features(
            ohlcv_data=self.sample_data,
            target=self.target,
            max_features_per_step=25
        )
        
        # 結果の検証
        assert result_df is not None
        assert len(result_df) == len(self.sample_data)
        assert len(result_df.columns) > len(self.sample_data.columns)
        
        # 統計情報が記録されていることを確認
        stats = service.last_enhancement_stats
        assert "manual_features" in stats
        assert "tsfresh_features" in stats
        assert "autofeat_features" in stats
        assert "total_features" in stats
        assert "total_time" in stats
        assert "processing_method" in stats
        assert stats["processing_method"] == "step_by_step"
        
        # 各ステップが呼び出されたことを確認
        mock_tsfresh.assert_called_once()
        mock_autofeat.assert_called_once()
        
        logger.info("✅ 統合ステップ・バイ・ステップフロー: 成功")

    def test_step_by_step_efficiency(self):
        """8. ステップ・バイ・ステップ効率性テスト"""
        logger.info("=== ステップ・バイ・ステップ効率性テスト ===")

        service = EnhancedFeatureEngineeringService(self.automl_config)
        
        # 処理時間を測定
        import time
        start_time = time.time()
        
        # 手動特徴量のみを計算（AutoMLは無効）
        service.automl_config.tsfresh.enabled = False
        service.automl_config.autofeat.enabled = False
        
        result_df = service.calculate_enhanced_features(
            ohlcv_data=self.sample_data,
            target=self.target
        )
        
        processing_time = time.time() - start_time
        
        # 結果の検証
        assert result_df is not None
        assert processing_time < 10.0  # 10秒以内で完了
        
        # 統計情報の確認
        stats = service.last_enhancement_stats
        assert stats["processing_method"] == "step_by_step"
        assert stats["total_time"] > 0
        
        logger.info(f"✅ ステップ・バイ・ステップ効率性: 成功 (処理時間: {processing_time:.2f}秒)")

    def test_step_by_step_summary(self):
        """9. ステップ・バイ・ステップ機能の総合確認"""
        logger.info("=== ステップ・バイ・ステップ機能の総合確認 ===")

        summary = {
            "step1_manual_features": True,
            "step2_tsfresh_with_selection": True,
            "step3_autofeat_with_selection": True,
            "feature_selection": True,
            "integrated_flow": True,
            "efficiency_optimization": True
        }

        try:
            service = EnhancedFeatureEngineeringService(self.automl_config)
            
            # 各ステップの確認
            step1_result = service._step1_manual_features(self.sample_data)
            assert step1_result is not None
            
            # 特徴量選択の確認
            large_df = self.sample_data.copy()
            for i in range(20):
                large_df[f'test_feature_{i}'] = np.random.random(len(self.sample_data))
            
            selected_df = service._select_top_features(large_df, self.target, max_features=10)
            assert len(selected_df.columns) == 10
            
            # 統計情報の確認
            assert hasattr(service, 'last_enhancement_stats')
            assert isinstance(service.last_enhancement_stats, dict)
            
            logger.info("✅ ステップ・バイ・ステップ機能の総合確認: 成功")
            logger.info(f"機能サマリー: {summary}")
            
        except Exception as e:
            pytest.fail(f"ステップ・バイ・ステップ機能の確認でエラーが発生しました: {e}")


if __name__ == "__main__":
    # テストを直接実行する場合
    pytest.main([__file__, "-v", "--tb=short"])
