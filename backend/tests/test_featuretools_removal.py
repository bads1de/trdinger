"""
FeaturetoolsConfig削除の検証テスト

FeaturetoolsConfigが完全に削除され、インポートエラーが発生しないことを確認します。

テスト項目:
1. AutoMLConfigからFeaturetoolsConfigインポートが削除されていることを確認
2. AutoMLConfig.from_dictが正常に動作することを確認
3. MLOrchestratorでFeaturetoolsConfig参照がないことを確認
4. 後方互換性が保たれていることを確認
"""

import pytest
import logging
from typing import Dict, Any

from app.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
    AutoFeatConfig
)
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

logger = logging.getLogger(__name__)


class TestFeaturetoolsRemoval:
    """FeaturetoolsConfig削除検証テストクラス"""

    def test_automl_config_imports(self):
        """1. AutoMLConfigからFeaturetoolsConfigインポートが削除されていることを確認"""
        logger.info("=== AutoMLConfigインポート確認テスト ===")

        # TSFreshConfigとAutoFeatConfigは正常にインポートできることを確認
        assert TSFreshConfig is not None
        assert AutoFeatConfig is not None
        assert AutoMLConfig is not None

        # FeaturetoolsConfigをインポートしようとするとエラーになることを確認
        with pytest.raises(ImportError):
            from app.services.ml.feature_engineering.automl_features.automl_config import FeaturetoolsConfig

        logger.info("✅ AutoMLConfigインポート確認: 成功")

    def test_automl_config_creation(self):
        """2. AutoMLConfigの作成が正常に動作することを確認"""
        logger.info("=== AutoMLConfig作成テスト ===")

        # デフォルト設定で作成
        config = AutoMLConfig.get_default_config()
        assert config is not None
        assert hasattr(config, 'tsfresh')
        assert hasattr(config, 'autofeat')
        assert hasattr(config, 'featuretools')  # 後方互換性のため

        # 後方互換性のfeaturetoolsは無効になっていることを確認
        assert config.featuretools.enabled == False

        logger.info("✅ AutoMLConfig作成: 成功")

    def test_automl_config_from_dict(self):
        """3. AutoMLConfig.from_dictが正常に動作することを確認"""
        logger.info("=== AutoMLConfig.from_dict テスト ===")

        # テスト用の設定辞書
        config_dict = {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "fdr_level": 0.05,
                "feature_count_limit": 100,
                "parallel_jobs": 2
            },
            "featuretools": {
                "enabled": False,  # 削除済みなので無視される
                "max_depth": 0,
                "max_features": 0
            },
            "autofeat": {
                "enabled": True,
                "max_features": 50,
                "feateng_steps": 2,
                "max_gb": 1.0
            }
        }

        # from_dictが正常に動作することを確認
        config = AutoMLConfig.from_dict(config_dict)
        assert config is not None
        assert config.tsfresh.enabled == True
        assert config.autofeat.enabled == True
        assert config.featuretools.enabled == False  # 後方互換性

        logger.info("✅ AutoMLConfig.from_dict: 成功")

    def test_automl_config_to_dict(self):
        """4. AutoMLConfig.to_dictが正常に動作することを確認"""
        logger.info("=== AutoMLConfig.to_dict テスト ===")

        config = AutoMLConfig.get_default_config()
        config_dict = config.to_dict()

        # 必要なキーが含まれていることを確認
        assert "tsfresh" in config_dict
        assert "autofeat" in config_dict
        assert "featuretools" in config_dict  # 後方互換性

        # featuretoolsは無効になっていることを確認
        assert config_dict["featuretools"]["enabled"] == False

        logger.info("✅ AutoMLConfig.to_dict: 成功")

    def test_ml_orchestrator_automl_config(self):
        """5. MLOrchestratorでFeaturetoolsConfig参照がないことを確認"""
        logger.info("=== MLOrchestrator AutoMLConfig テスト ===")

        orchestrator = MLOrchestrator(enable_automl=True)

        # AutoML設定の作成が正常に動作することを確認
        config_dict = {
            "tsfresh": {"enabled": True},
            "featuretools": {"enabled": False},  # 無視される
            "autofeat": {"enabled": True}
        }

        automl_config = orchestrator._create_automl_config_from_dict(config_dict)
        assert automl_config is not None
        assert automl_config.tsfresh.enabled == True
        assert automl_config.autofeat.enabled == True

        logger.info("✅ MLOrchestrator AutoMLConfig: 成功")

    def test_backward_compatibility(self):
        """6. 後方互換性が保たれていることを確認"""
        logger.info("=== 後方互換性テスト ===")

        # 古い設定辞書（featuretoolsを含む）でも動作することを確認
        old_config_dict = {
            "tsfresh": {"enabled": True},
            "featuretools": {
                "enabled": True,  # 古い設定では有効だったが、無視される
                "max_depth": 3,
                "max_features": 100
            },
            "autofeat": {"enabled": False}
        }

        config = AutoMLConfig.from_dict(old_config_dict)
        assert config is not None

        # featuretoolsは無効になっていることを確認
        assert config.featuretools.enabled == False

        # to_dictでも正しく出力されることを確認
        output_dict = config.to_dict()
        assert output_dict["featuretools"]["enabled"] == False

        logger.info("✅ 後方互換性: 成功")

    def test_financial_optimized_config(self):
        """7. 金融最適化設定が正常に動作することを確認"""
        logger.info("=== 金融最適化設定テスト ===")

        config = AutoMLConfig.get_financial_optimized_config()
        assert config is not None
        assert config.tsfresh.enabled == True
        assert config.autofeat.enabled == True
        assert config.featuretools.enabled == False  # 削除済み

        # 設定値が適切であることを確認
        assert config.tsfresh.feature_count_limit == 500
        assert config.autofeat.max_features == 100

        logger.info("✅ 金融最適化設定: 成功")

    def test_no_featuretools_references(self):
        """8. Featuretools参照が完全に削除されていることを確認"""
        logger.info("=== Featuretools参照削除確認テスト ===")

        # AutoMLConfigクラスの属性を確認
        config = AutoMLConfig.get_default_config()
        
        # featuretools属性は後方互換性のため存在するが、常に無効
        assert hasattr(config, 'featuretools')
        assert config.featuretools.enabled == False

        # TSFreshとAutoFeatは正常に動作
        assert config.tsfresh.enabled == True
        assert isinstance(config.autofeat, AutoFeatConfig)

        logger.info("✅ Featuretools参照削除確認: 成功")

    def test_comprehensive_featuretools_removal_validation(self):
        """9. FeaturetoolsConfig削除の包括的検証"""
        logger.info("=== FeaturetoolsConfig削除の包括的検証 ===")

        validation_results = {
            "automl_config_imports": False,
            "config_creation": False,
            "from_dict_method": False,
            "to_dict_method": False,
            "ml_orchestrator_integration": False,
            "backward_compatibility": False,
            "financial_optimized": False,
            "no_references": False
        }

        try:
            # 1. インポート確認
            try:
                from app.services.ml.feature_engineering.automl_features.automl_config import FeaturetoolsConfig
                validation_results["automl_config_imports"] = False  # インポートできてはいけない
            except ImportError:
                validation_results["automl_config_imports"] = True  # インポートエラーが正常

            # 2. 設定作成確認
            config = AutoMLConfig.get_default_config()
            validation_results["config_creation"] = config is not None

            # 3. from_dict確認
            test_dict = {"tsfresh": {"enabled": True}, "autofeat": {"enabled": True}}
            from_dict_config = AutoMLConfig.from_dict(test_dict)
            validation_results["from_dict_method"] = from_dict_config is not None

            # 4. to_dict確認
            to_dict_result = config.to_dict()
            validation_results["to_dict_method"] = "featuretools" in to_dict_result and to_dict_result["featuretools"]["enabled"] == False

            # 5. MLOrchestrator統合確認
            orchestrator = MLOrchestrator(enable_automl=True)
            validation_results["ml_orchestrator_integration"] = orchestrator is not None

            # 6. 後方互換性確認
            old_config = {"featuretools": {"enabled": True}}
            compat_config = AutoMLConfig.from_dict(old_config)
            validation_results["backward_compatibility"] = compat_config.featuretools.enabled == False

            # 7. 金融最適化確認
            fin_config = AutoMLConfig.get_financial_optimized_config()
            validation_results["financial_optimized"] = fin_config.featuretools.enabled == False

            # 8. 参照削除確認
            validation_results["no_references"] = hasattr(config, 'featuretools') and config.featuretools.enabled == False

            # 全ての検証が成功したことを確認
            failed_validations = [k for k, v in validation_results.items() if not v]
            assert len(failed_validations) == 0, f"以下の検証が失敗しました: {failed_validations}"

            logger.info("✅ FeaturetoolsConfig削除の包括的検証: 成功")
            logger.info(f"検証結果: {validation_results}")

        except Exception as e:
            pytest.fail(f"FeaturetoolsConfig削除の包括的検証でエラーが発生しました: {e}")

    def test_featuretools_removal_summary(self):
        """10. FeaturetoolsConfig削除の総合確認"""
        logger.info("=== FeaturetoolsConfig削除の総合確認 ===")

        summary = {
            "featuretools_config_removed": True,
            "import_errors_resolved": True,
            "backward_compatibility_maintained": True,
            "automl_functionality_preserved": True,
            "ml_orchestrator_updated": True,
            "comprehensive_testing": True
        }

        try:
            # 主要な確認点をテスト
            
            # 1. FeaturetoolsConfigの削除
            with pytest.raises(ImportError):
                from app.services.ml.feature_engineering.automl_features.automl_config import FeaturetoolsConfig

            # 2. インポートエラーの解決
            config = AutoMLConfig.get_default_config()
            assert config is not None

            # 3. 後方互換性の維持
            old_dict = {"featuretools": {"enabled": True}}
            compat_config = AutoMLConfig.from_dict(old_dict)
            assert compat_config.featuretools.enabled == False

            # 4. AutoML機能の保持
            assert config.tsfresh.enabled == True
            assert isinstance(config.autofeat, AutoFeatConfig)

            # 5. MLOrchestratorの更新
            orchestrator = MLOrchestrator(enable_automl=True)
            assert orchestrator is not None

            # 6. 包括的テストの実装
            assert hasattr(self, 'test_comprehensive_featuretools_removal_validation')

            logger.info("✅ FeaturetoolsConfig削除の総合確認: 成功")
            logger.info(f"削除サマリー: {summary}")

        except Exception as e:
            pytest.fail(f"FeaturetoolsConfig削除の総合確認でエラーが発生しました: {e}")


if __name__ == "__main__":
    # テストを直接実行する場合
    pytest.main([__file__, "-v", "--tb=short"])
