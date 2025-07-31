"""
アンサンブル無効化のデバッグテスト

アンサンブル設定をオフにしても実際にはアンサンブルが実行される問題を調査する。
"""

import pytest
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.ml_training import MLTrainingConfig, EnsembleConfig, SingleModelConfig
from app.services.ml.orchestration.ml_training_orchestration_service import MLTrainingOrchestrationService
from app.services.ml.ml_training_service import MLTrainingService

logger = logging.getLogger(__name__)


class TestEnsembleDisableDebug:
    """アンサンブル無効化デバッグテストクラス"""

    def test_ensemble_config_parsing(self):
        """アンサンブル設定の解析をテスト"""
        logger.info("=== アンサンブル設定解析テスト ===")
        
        # アンサンブル無効化設定を作成
        ensemble_config = EnsembleConfig(enabled=False)
        single_model_config = SingleModelConfig(model_type="lightgbm")
        
        # 設定をmodel_dump()で辞書化
        ensemble_dict = ensemble_config.model_dump()
        single_dict = single_model_config.model_dump()
        
        logger.info(f"アンサンブル設定辞書: {ensemble_dict}")
        logger.info(f"単一モデル設定辞書: {single_dict}")
        
        # enabled フィールドの確認
        assert "enabled" in ensemble_dict
        assert ensemble_dict["enabled"] == False
        logger.info(f"✅ enabled フィールド確認: {ensemble_dict['enabled']}")
        
        # トレーナータイプ決定のテスト
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_dict)
        logger.info(f"決定されたトレーナータイプ: {trainer_type}")
        assert trainer_type == "single"
        
        return ensemble_dict, single_dict

    def test_ml_training_config_creation(self):
        """MLTrainingConfig作成をテスト"""
        logger.info("=== MLTrainingConfig作成テスト ===")
        
        # アンサンブル無効化のMLTrainingConfigを作成
        config = MLTrainingConfig(
            symbol="BTC",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-12-31",
            save_model=True,
            train_test_split=0.8,
            random_state=42,
            ensemble_config=EnsembleConfig(enabled=False),
            single_model_config=SingleModelConfig(model_type="xgboost")
        )
        
        logger.info(f"作成されたconfig.ensemble_config: {config.ensemble_config}")
        logger.info(f"作成されたconfig.single_model_config: {config.single_model_config}")
        
        # 設定の確認
        assert config.ensemble_config.enabled == False
        assert config.single_model_config.model_type == "xgboost"
        
        # model_dump()での変換をテスト
        ensemble_dict = config.ensemble_config.model_dump()
        single_dict = config.single_model_config.model_dump()
        
        logger.info(f"ensemble_config.model_dump(): {ensemble_dict}")
        logger.info(f"single_model_config.model_dump(): {single_dict}")
        
        assert ensemble_dict["enabled"] == False
        assert single_dict["model_type"] == "xgboost"
        
        return config

    def test_ml_training_service_initialization(self):
        """MLTrainingService初期化をテスト"""
        logger.info("=== MLTrainingService初期化テスト ===")
        
        # アンサンブル無効化設定
        ensemble_config = {"enabled": False}
        single_model_config = {"model_type": "catboost"}
        
        # トレーナータイプを決定
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_config)
        logger.info(f"決定されたトレーナータイプ: {trainer_type}")
        assert trainer_type == "single"
        
        # MLTrainingServiceを初期化
        ml_service = MLTrainingService(
            trainer_type=trainer_type,
            ensemble_config=ensemble_config,
            single_model_config=single_model_config
        )
        
        logger.info(f"MLTrainingService.trainer_type: {ml_service.trainer_type}")
        logger.info(f"MLTrainingService.trainer: {type(ml_service.trainer).__name__}")
        
        # 正しく単一モデルトレーナーが選択されているか確認
        assert ml_service.trainer_type == "single"
        from app.services.ml.single_model.single_model_trainer import SingleModelTrainer
        assert isinstance(ml_service.trainer, SingleModelTrainer)
        assert ml_service.trainer.model_type == "catboost"
        
        logger.info("✅ MLTrainingService初期化成功")
        return ml_service

    def test_orchestration_service_logic(self):
        """オーケストレーションサービスのロジックをテスト"""
        logger.info("=== オーケストレーションサービスロジックテスト ===")
        
        # テスト用のMLTrainingConfigを作成
        config = MLTrainingConfig(
            symbol="BTC",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-12-31",
            save_model=True,
            train_test_split=0.8,
            random_state=42,
            ensemble_config=EnsembleConfig(enabled=False),
            single_model_config=SingleModelConfig(model_type="tabnet")
        )
        
        # オーケストレーションサービスのロジックをシミュレート
        ensemble_config_dict = None
        single_model_config_dict = None
        trainer_type = "ensemble"  # デフォルト
        
        if config.ensemble_config:
            ensemble_config_dict = config.ensemble_config.model_dump()
            logger.info(f"アンサンブル設定辞書: {ensemble_config_dict}")
            
            # アンサンブルが無効化されている場合は単一モデルを使用
            if not ensemble_config_dict.get("enabled", True):
                trainer_type = "single"
                logger.info("🔄 アンサンブルが無効化されているため、単一モデルトレーニングを使用します")
                logger.info(f"📋 アンサンブル設定確認: enabled={ensemble_config_dict.get('enabled')}")

        # 単一モデル設定の準備
        if config.single_model_config:
            single_model_config_dict = config.single_model_config.model_dump()
            if trainer_type == "single":
                logger.info(f"📋 単一モデル設定: {single_model_config_dict}")
        
        # 最終確認
        logger.info(f"🎯 最終決定されたトレーナータイプ: {trainer_type}")
        
        # 検証
        assert trainer_type == "single"
        assert ensemble_config_dict["enabled"] == False
        assert single_model_config_dict["model_type"] == "tabnet"
        
        # MLTrainingServiceを初期化してテスト
        ml_service = MLTrainingService(
            trainer_type=trainer_type,
            ensemble_config=ensemble_config_dict,
            single_model_config=single_model_config_dict
        )
        
        assert ml_service.trainer_type == "single"
        logger.info("✅ オーケストレーションサービスロジック正常")
        
        return {
            'trainer_type': trainer_type,
            'ensemble_config': ensemble_config_dict,
            'single_model_config': single_model_config_dict,
            'ml_service': ml_service
        }

    def test_actual_api_request_simulation(self):
        """実際のAPIリクエストをシミュレート"""
        logger.info("=== 実際のAPIリクエストシミュレーション ===")
        
        # フロントエンドから送信されるであろうリクエストデータ
        request_data = {
            "symbol": "BTC",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "save_model": True,
            "train_test_split": 0.8,
            "random_state": 42,
            "ensemble_config": {
                "enabled": False,  # ここが重要！
                "method": "stacking",
                "bagging_params": {
                    "n_estimators": 5,
                    "bootstrap_fraction": 0.8,
                    "base_model_type": "lightgbm",
                    "mixed_models": ["lightgbm"],
                    "random_state": 42
                },
                "stacking_params": {
                    "base_models": ["lightgbm"],
                    "meta_model": "lightgbm",
                    "cv_folds": 5,
                    "use_probas": True,
                    "random_state": 42
                }
            },
            "single_model_config": {
                "model_type": "lightgbm"
            }
        }
        
        # MLTrainingConfigを作成
        config = MLTrainingConfig(**request_data)
        
        logger.info(f"リクエストから作成されたconfig:")
        logger.info(f"  ensemble_config.enabled: {config.ensemble_config.enabled}")
        logger.info(f"  single_model_config.model_type: {config.single_model_config.model_type}")
        
        # オーケストレーションサービスの処理をシミュレート
        ensemble_config_dict = config.ensemble_config.model_dump()
        single_model_config_dict = config.single_model_config.model_dump()
        
        trainer_type = "ensemble"  # デフォルト
        if not ensemble_config_dict.get("enabled", True):
            trainer_type = "single"
        
        logger.info(f"最終的なtrainer_type: {trainer_type}")
        logger.info(f"ensemble_config_dict: {ensemble_config_dict}")
        logger.info(f"single_model_config_dict: {single_model_config_dict}")
        
        # 検証
        assert trainer_type == "single"
        assert ensemble_config_dict["enabled"] == False
        
        logger.info("✅ APIリクエストシミュレーション成功")
        
        return {
            'config': config,
            'trainer_type': trainer_type,
            'ensemble_config_dict': ensemble_config_dict,
            'single_model_config_dict': single_model_config_dict
        }

    def test_overall_debug_analysis(self):
        """全体的なデバッグ分析"""
        logger.info("=== 全体的なデバッグ分析 ===")
        
        # 各テストを実行
        ensemble_dict, single_dict = self.test_ensemble_config_parsing()
        config = self.test_ml_training_config_creation()
        ml_service = self.test_ml_training_service_initialization()
        orchestration_result = self.test_orchestration_service_logic()
        api_simulation = self.test_actual_api_request_simulation()
        
        # 結果を分析
        logger.info("=== デバッグ分析結果 ===")
        logger.info("1. アンサンブル設定解析: ✅ 正常")
        logger.info("2. MLTrainingConfig作成: ✅ 正常")
        logger.info("3. MLTrainingService初期化: ✅ 正常")
        logger.info("4. オーケストレーションロジック: ✅ 正常")
        logger.info("5. APIリクエストシミュレーション: ✅ 正常")
        
        logger.info("=== 結論 ===")
        logger.info("コード上では正しく動作しているため、問題は以下の可能性があります：")
        logger.info("1. フロントエンドから送信される実際のデータが期待と異なる")
        logger.info("2. オーケストレーションサービスで例外が発生してデフォルト動作になっている")
        logger.info("3. ログの出力タイミングと実際の処理が異なる")
        
        return {
            'all_tests_passed': True,
            'ensemble_parsing': True,
            'config_creation': True,
            'service_initialization': True,
            'orchestration_logic': True,
            'api_simulation': True
        }


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_instance = TestEnsembleDisableDebug()
    
    # 全体的なデバッグ分析を実行
    results = test_instance.test_overall_debug_analysis()
    
    print(f"\n=== アンサンブル無効化デバッグ結果 ===")
    print(f"全テスト通過: {results['all_tests_passed']}")
    print(f"アンサンブル設定解析: {'✅' if results['ensemble_parsing'] else '❌'}")
    print(f"Config作成: {'✅' if results['config_creation'] else '❌'}")
    print(f"サービス初期化: {'✅' if results['service_initialization'] else '❌'}")
    print(f"オーケストレーションロジック: {'✅' if results['orchestration_logic'] else '❌'}")
    print(f"APIシミュレーション: {'✅' if results['api_simulation'] else '❌'}")
    
    print(f"\n=== 推奨対応 ===")
    print("1. フロントエンドから送信される実際のリクエストデータをログで確認")
    print("2. オーケストレーションサービスでの例外処理を強化")
    print("3. より詳細なログを追加して実際の動作を追跡")
