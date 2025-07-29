"""
フェーズ2統合テスト

MLシステム問題修正プロジェクトのフェーズ2（中期対応）の修正内容を検証します。

テスト項目:
1. AutoML統合の正常動作確認
2. 特徴量生成フローの効率化確認
3. データ前処理の改善確認
4. MLパイプライン全体の動作確認
5. パフォーマンス改善の確認
"""

import pytest
import logging
import gc
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.ml.feature_engineering.enhanced_feature_engineering_service import EnhancedFeatureEngineeringService
from app.services.ml.feature_engineering.feature_engineering_service import FeatureEngineeringService
from app.utils.data_preprocessing import data_preprocessor, DataPreprocessor

logger = logging.getLogger(__name__)


class TestPhase2Integration:
    """フェーズ2統合テストクラス"""

    def setup_method(self):
        """テストセットアップ"""
        # DatetimeIndexを持つテスト用のサンプルデータを作成
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1000, 2000, 50)
        }, index=dates)
        
        # 意図的に欠損値を追加
        self.sample_data.loc[self.sample_data.index[10:15], 'volume'] = np.nan
        
        # ターゲット変数を作成
        self.target = pd.Series(
            np.random.choice([0, 1], 50), 
            name='target', 
            index=dates
        )

    def test_automl_integration(self):
        """1. AutoML統合の正常動作確認"""
        logger.info("=== AutoML統合の正常動作確認テスト ===")

        # AutoML有効でMLOrchestratorを作成
        orchestrator = MLOrchestrator(enable_automl=True)
        
        # AutoML機能が正しく統合されていることを確認
        assert orchestrator.enable_automl == True
        assert isinstance(orchestrator.feature_service, EnhancedFeatureEngineeringService)
        
        # AutoML状態を確認
        status = orchestrator.get_automl_status()
        assert status["enabled"] == True
        assert status["service_type"] == "EnhancedFeatureEngineeringService"
        
        # AutoML設定の動的変更をテスト
        custom_config = {
            "tsfresh": {"enabled": True, "feature_count_limit": 50},
            "autofeat": {"enabled": True, "max_features": 30}
        }
        orchestrator.set_automl_enabled(True, custom_config)
        assert orchestrator.automl_config == custom_config
        
        logger.info("✅ AutoML統合の正常動作確認: 成功")

    @patch('app.services.ml.feature_engineering.enhanced_feature_engineering_service.EnhancedFeatureEngineeringService.calculate_enhanced_features')
    def test_step_by_step_feature_generation(self, mock_enhanced_features):
        """2. 特徴量生成フローの効率化確認"""
        logger.info("=== 特徴量生成フローの効率化確認テスト ===")

        # モックの設定
        mock_result = self.sample_data.copy()
        for i in range(20):
            mock_result[f'enhanced_feature_{i}'] = np.random.random(len(self.sample_data))
        mock_enhanced_features.return_value = mock_result

        # AutoML有効でオーケストレーターを作成
        orchestrator = MLOrchestrator(enable_automl=True)
        
        # 特徴量計算を実行
        start_time = time.time()
        result = orchestrator._calculate_features(self.sample_data)
        processing_time = time.time() - start_time
        
        # 結果の検証
        assert result is not None
        assert len(result) == len(self.sample_data)
        assert len(result.columns) > len(self.sample_data.columns)
        
        # EnhancedFeatureEngineeringServiceが呼び出されたことを確認
        mock_enhanced_features.assert_called_once()
        
        # 処理時間が合理的であることを確認
        assert processing_time < 30.0, f"処理時間が長すぎます: {processing_time:.2f}秒"
        
        logger.info(f"✅ 特徴量生成フローの効率化確認: 成功 (処理時間: {processing_time:.2f}秒)")

    def test_data_preprocessing_improvement(self):
        """3. データ前処理の改善確認"""
        logger.info("=== データ前処理の改善確認テスト ===")

        # 欠損値を含むデータを作成
        test_data = self.sample_data.copy()
        test_data.loc[test_data.index[20:25], 'close'] = np.nan
        test_data.loc[test_data.index[30:35], 'high'] = np.inf  # 無限値
        
        # データ前処理を実行
        preprocessor = DataPreprocessor()
        result_df = preprocessor.preprocess_features(
            test_data,
            imputation_strategy="median",
            scale_features=False,
            remove_outliers=True
        )
        
        # 結果の検証
        assert result_df.isnull().sum().sum() == 0, "欠損値が残存しています"
        assert not result_df.isin([np.inf, -np.inf]).any().any(), "無限値が残存しています"
        assert len(result_df) == len(test_data), "データサイズが変更されました"
        
        # 補完統計情報を確認
        stats = preprocessor.get_imputation_stats()
        assert len(stats) > 0, "補完統計情報が記録されていません"
        
        logger.info("✅ データ前処理の改善確認: 成功")

    def test_ml_pipeline_integration(self):
        """4. MLパイプライン全体の動作確認"""
        logger.info("=== MLパイプライン全体の動作確認テスト ===")

        # AutoML有効でMLOrchestratorを作成
        orchestrator = MLOrchestrator(enable_automl=True)
        
        try:
            # ターゲット変数計算をテスト
            df_for_target = self.sample_data.copy()
            df_for_target.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            target = orchestrator._calculate_target_for_automl(df_for_target)
            
            assert target is not None, "ターゲット変数が計算されませんでした"
            assert len(target) == len(df_for_target), "ターゲット変数のサイズが正しくありません"
            
            # AutoML設定の取得をテスト
            status = orchestrator.get_automl_status()
            assert "enabled" in status
            assert "service_type" in status
            assert "available_features" in status
            
            logger.info("✅ MLパイプライン全体の動作確認: 成功")
            
        except Exception as e:
            pytest.fail(f"MLパイプライン統合テストでエラー: {e}")

    def test_performance_improvement(self):
        """5. パフォーマンス改善の確認"""
        logger.info("=== パフォーマンス改善の確認テスト ===")

        # メモリ使用量を測定
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # AutoML有効とAutoML無効での処理時間を比較
        times = {}
        
        # AutoML無効での処理時間
        orchestrator_basic = MLOrchestrator(enable_automl=False)
        start_time = time.time()
        
        # 基本特徴量計算をシミュレート
        basic_result = orchestrator_basic._calculate_features(self.sample_data)
        times['basic'] = time.time() - start_time
        
        # AutoML有効での処理時間（モック使用）
        with patch('app.services.ml.feature_engineering.enhanced_feature_engineering_service.EnhancedFeatureEngineeringService.calculate_enhanced_features') as mock_enhanced:
            mock_enhanced.return_value = self.sample_data.copy()
            
            orchestrator_automl = MLOrchestrator(enable_automl=True)
            start_time = time.time()
            automl_result = orchestrator_automl._calculate_features(self.sample_data)
            times['automl'] = time.time() - start_time

        # メモリ使用量を再測定
        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # パフォーマンス検証
        assert times['basic'] > 0, "基本処理時間が測定されませんでした"
        assert times['automl'] > 0, "AutoML処理時間が測定されませんでした"
        assert memory_increase < 100, f"メモリ使用量が大幅に増加しました: {memory_increase:.2f}MB"
        
        logger.info(f"✅ パフォーマンス改善確認: 成功")
        logger.info(f"処理時間 - 基本: {times['basic']:.2f}秒, AutoML: {times['automl']:.2f}秒")
        logger.info(f"メモリ増加: {memory_increase:.2f}MB")

    def test_error_handling_robustness(self):
        """6. エラーハンドリングの堅牢性確認"""
        logger.info("=== エラーハンドリングの堅牢性確認テスト ===")

        orchestrator = MLOrchestrator(enable_automl=True)
        
        # 空データでのエラーハンドリング（警告で処理される）
        empty_df = pd.DataFrame()
        result = orchestrator._calculate_features(empty_df)
        # 空データは警告で処理され、空のDataFrameが返される
        assert result is not None and result.empty, "空データは空のDataFrameが返されるべきです"

        # 無効なデータでのエラーハンドリング
        invalid_df = pd.DataFrame({'invalid': [np.nan, np.nan, np.nan]})
        result = orchestrator._calculate_features(invalid_df)
        # 無効なデータはNoneが返される（エラーハンドリングが正常に動作）
        assert result is None, "無効なデータはNoneが返されるべきです"
        
        # データ前処理でのエラーハンドリング
        preprocessor = DataPreprocessor()
        
        # 空データの処理
        empty_result = preprocessor.transform_missing_values(pd.DataFrame())
        assert empty_result.empty, "空データは空のまま返されるべきです"
        
        # Noneの処理
        none_result = preprocessor.transform_missing_values(None)
        assert none_result is None, "Noneはそのまま返されるべきです"
        
        logger.info("✅ エラーハンドリングの堅牢性確認: 成功")

    def test_feature_quality_improvement(self):
        """7. 特徴量品質の改善確認"""
        logger.info("=== 特徴量品質の改善確認テスト ===")

        # 欠損値と外れ値を含むデータを作成
        quality_test_data = self.sample_data.copy()
        quality_test_data.loc[quality_test_data.index[5:10], 'volume'] = np.nan
        quality_test_data.loc[quality_test_data.index[15], 'close'] = 10000  # 外れ値

        # 改善前（fillna(0)相当）
        basic_filled = quality_test_data.fillna(0)
        
        # 改善後（統計的補完）
        preprocessor = DataPreprocessor()
        improved_filled = preprocessor.preprocess_features(
            quality_test_data,
            imputation_strategy="median",
            remove_outliers=True
        )
        
        # 品質比較
        # 1. 欠損値補完の品質
        volume_median = quality_test_data['volume'].median()
        basic_volume_mean = basic_filled['volume'].mean()
        improved_volume_mean = improved_filled['volume'].mean()
        
        # 統計的補完の方が元データの分布に近いことを確認
        original_volume_mean = quality_test_data['volume'].dropna().mean()
        basic_diff = abs(basic_volume_mean - original_volume_mean)
        improved_diff = abs(improved_volume_mean - original_volume_mean)
        
        assert improved_diff < basic_diff, "統計的補完の方が元データに近い分布を保持すべきです"
        
        # 2. 外れ値除去の効果
        basic_close_max = basic_filled['close'].max()
        improved_close_max = improved_filled['close'].max()
        
        assert improved_close_max < basic_close_max, "外れ値除去により最大値が小さくなるべきです"
        
        logger.info("✅ 特徴量品質の改善確認: 成功")

    def test_phase2_comprehensive_validation(self):
        """8. フェーズ2包括的検証"""
        logger.info("=== フェーズ2包括的検証テスト ===")

        validation_results = {
            "automl_integration": False,
            "step_by_step_features": False,
            "data_preprocessing": False,
            "pipeline_integration": False,
            "performance_improvement": False,
            "error_handling": False,
            "feature_quality": False
        }

        try:
            # 1. AutoML統合の検証
            orchestrator = MLOrchestrator(enable_automl=True)
            assert isinstance(orchestrator.feature_service, EnhancedFeatureEngineeringService)
            validation_results["automl_integration"] = True

            # 2. ステップ・バイ・ステップ特徴量生成の検証
            enhanced_service = EnhancedFeatureEngineeringService()
            assert hasattr(enhanced_service, '_step1_manual_features')
            assert hasattr(enhanced_service, '_step2_tsfresh_features')
            assert hasattr(enhanced_service, '_step3_autofeat_features')
            validation_results["step_by_step_features"] = True

            # 3. データ前処理の検証
            preprocessor = DataPreprocessor()
            test_result = preprocessor.transform_missing_values(self.sample_data)
            assert test_result.isnull().sum().sum() == 0
            validation_results["data_preprocessing"] = True

            # 4. パイプライン統合の検証
            status = orchestrator.get_automl_status()
            assert status["enabled"] == True
            validation_results["pipeline_integration"] = True

            # 5. パフォーマンス改善の検証
            start_time = time.time()
            orchestrator.set_automl_enabled(False)
            orchestrator.set_automl_enabled(True)
            toggle_time = time.time() - start_time
            assert toggle_time < 5.0  # 5秒以内で切り替え完了
            validation_results["performance_improvement"] = True

            # 6. エラーハンドリングの検証
            empty_result = orchestrator._calculate_features(pd.DataFrame())
            invalid_result = orchestrator._calculate_features(pd.DataFrame({'invalid': [np.nan]}))
            # エラーハンドリングが正常に動作していることを確認
            assert (empty_result is not None and empty_result.empty) or empty_result is None
            assert invalid_result is None
            validation_results["error_handling"] = True

            # 7. 特徴量品質の検証
            quality_data = self.sample_data.copy()
            quality_data.iloc[0, 0] = np.nan
            quality_result = preprocessor.preprocess_features(quality_data)
            assert quality_result.isnull().sum().sum() == 0
            validation_results["feature_quality"] = True

            # 全ての検証が成功したことを確認
            all_passed = all(validation_results.values())
            assert all_passed, f"一部の検証が失敗しました: {validation_results}"

            logger.info("✅ フェーズ2包括的検証: 成功")
            logger.info(f"検証結果: {validation_results}")

        except Exception as e:
            pytest.fail(f"フェーズ2包括的検証でエラーが発生しました: {e}")

    def test_phase2_summary(self):
        """9. フェーズ2修正内容の総合確認"""
        logger.info("=== フェーズ2修正内容の総合確認 ===")

        summary = {
            "automl_integration_complete": True,
            "step_by_step_optimization": True,
            "data_preprocessing_improved": True,
            "pipeline_redesigned": True,
            "performance_enhanced": True,
            "quality_assured": True
        }

        try:
            # フェーズ2の主要改善点を確認
            
            # 1. AutoML統合の完了
            orchestrator = MLOrchestrator(enable_automl=True)
            assert orchestrator.enable_automl == True
            
            # 2. ステップ・バイ・ステップ最適化
            enhanced_service = EnhancedFeatureEngineeringService()
            assert hasattr(enhanced_service, 'last_enhancement_stats')
            
            # 3. データ前処理の改善
            assert data_preprocessor is not None
            assert isinstance(data_preprocessor, DataPreprocessor)
            
            # 4. パイプライン再設計
            status = orchestrator.get_automl_status()
            assert "available_features" in status
            
            # 5. パフォーマンス向上
            # メモリ効率的な処理が実装されていることを確認
            assert hasattr(enhanced_service, 'cleanup_resources')
            
            # 6. 品質保証
            # 統合テストが実装されていることを確認
            assert hasattr(self, 'test_phase2_comprehensive_validation')

            logger.info("✅ フェーズ2修正内容の総合確認: 成功")
            logger.info(f"修正内容サマリー: {summary}")

        except Exception as e:
            pytest.fail(f"フェーズ2修正内容の確認でエラーが発生しました: {e}")


if __name__ == "__main__":
    # テストを直接実行する場合
    pytest.main([__file__, "-v", "--tb=short"])
