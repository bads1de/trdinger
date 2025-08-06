"""
単一モデルトレーニングAPI機能のテスト

APIエンドポイントが単一モデル設定を正しく処理し、
アンサンブル無効化時に単一モデルトレーニングが実行されることを確認する。
"""

import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.ml_training import SingleModelConfig, EnsembleConfig
from app.services.ml.ml_training_service import MLTrainingService

logger = logging.getLogger(__name__)


class TestSingleModelAPI:
    """単一モデルトレーニングAPI機能のテストクラス"""

    def test_single_model_config_validation(self):
        """SingleModelConfigの検証をテスト"""
        logger.info("=== SingleModelConfig検証テスト ===")
        
        # 有効な設定
        valid_config = SingleModelConfig(model_type="lightgbm")
        assert valid_config.model_type == "lightgbm"
        logger.info("✅ 有効なSingleModelConfig作成成功")
        
        # 各モデルタイプをテスト
        model_types = ["lightgbm", "xgboost", "catboost", "tabnet"]
        for model_type in model_types:
            config = SingleModelConfig(model_type=model_type)
            assert config.model_type == model_type
            logger.info(f"✅ {model_type.upper()}設定作成成功")
        
        return True

    def test_ensemble_config_disabled(self):
        """アンサンブル無効化設定をテスト"""
        logger.info("=== アンサンブル無効化設定テスト ===")
        
        # アンサンブル無効化設定
        ensemble_config = EnsembleConfig(enabled=False)
        assert ensemble_config.enabled == False
        logger.info("✅ アンサンブル無効化設定作成成功")
        
        # トレーナータイプの自動決定をテスト
        trainer_type = MLTrainingService.determine_trainer_type(
            ensemble_config.dict()
        )
        assert trainer_type == "single"
        logger.info("✅ アンサンブル無効化時の自動決定: single")
        
        return True

    def test_api_request_structure(self):
        """API リクエスト構造をテスト"""
        logger.info("=== APIリクエスト構造テスト ===")
        
        # 単一モデル用のリクエストデータを作成
        request_data = {
            "symbol": "BTC",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "save_model": True,
            "train_test_split": 0.8,
            "random_state": 42,
            "ensemble_config": {
                "enabled": False,  # アンサンブル無効化
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
                "model_type": "lightgbm"  # 単一モデル設定
            }
        }
        
        # 設定の検証
        ensemble_config = EnsembleConfig(**request_data["ensemble_config"])
        single_model_config = SingleModelConfig(**request_data["single_model_config"])
        
        assert ensemble_config.enabled == False
        assert single_model_config.model_type == "lightgbm"
        
        logger.info("✅ APIリクエスト構造検証成功")
        logger.info(f"   - アンサンブル有効: {ensemble_config.enabled}")
        logger.info(f"   - 単一モデルタイプ: {single_model_config.model_type}")
        
        return request_data

    def test_ml_training_service_integration(self):
        """MLTrainingServiceとの統合をテスト"""
        logger.info("=== MLTrainingService統合テスト ===")
        
        # アンサンブル無効化 + 単一モデル設定
        ensemble_config = {"enabled": False}
        single_model_config = {"model_type": "xgboost"}
        
        # トレーナータイプを決定
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_config)
        assert trainer_type == "single"
        
        # MLTrainingServiceを初期化
        ml_service = MLTrainingService(
            trainer_type=trainer_type,
            ensemble_config=ensemble_config,
            single_model_config=single_model_config
        )
        
        assert ml_service.trainer_type == "single"
        assert ml_service.trainer.model_type == "xgboost"
        
        logger.info("✅ MLTrainingService統合成功")
        logger.info(f"   - トレーナータイプ: {ml_service.trainer_type}")
        logger.info(f"   - モデルタイプ: {ml_service.trainer.model_type}")
        
        return ml_service

    def test_available_models_api_simulation(self):
        """利用可能モデル取得APIのシミュレーション"""
        logger.info("=== 利用可能モデル取得APIシミュレーション ===")
        
        # MLTrainingServiceから利用可能モデルを取得
        available_models = MLTrainingService.get_available_single_models()
        
        # API レスポンス形式をシミュレート
        api_response = {
            "success": True,
            "available_models": available_models,
            "message": f"{len(available_models)}個のモデルが利用可能です"
        }
        
        assert api_response["success"] == True
        assert len(api_response["available_models"]) > 0
        assert "lightgbm" in api_response["available_models"]
        
        logger.info("✅ 利用可能モデル取得APIシミュレーション成功")
        logger.info(f"   - 利用可能モデル数: {len(available_models)}")
        logger.info(f"   - モデル一覧: {available_models}")
        
        return api_response

    def test_different_model_types(self):
        """異なるモデルタイプでの動作をテスト"""
        logger.info("=== 異なるモデルタイプでの動作テスト ===")
        
        available_models = MLTrainingService.get_available_single_models()
        test_results = {}
        
        for model_type in available_models:
            try:
                # 単一モデル設定
                single_model_config = {"model_type": model_type}
                
                # MLTrainingServiceを初期化
                ml_service = MLTrainingService(
                    trainer_type="single",
                    single_model_config=single_model_config
                )
                
                # 初期化成功を確認
                assert ml_service.trainer_type == "single"
                assert ml_service.trainer.model_type == model_type
                
                test_results[model_type] = "成功"
                logger.info(f"✅ {model_type.upper()}モデル初期化成功")
                
            except Exception as e:
                test_results[model_type] = f"失敗: {e}"
                logger.error(f"❌ {model_type.upper()}モデル初期化失敗: {e}")
        
        # 成功率を計算
        success_count = sum(1 for result in test_results.values() if result == "成功")
        success_rate = success_count / len(test_results) if test_results else 0
        
        logger.info(f"モデルタイプテスト成功率: {success_rate*100:.1f}% ({success_count}/{len(test_results)})")
        
        return test_results

    def test_configuration_priority(self):
        """設定の優先順位をテスト"""
        logger.info("=== 設定優先順位テスト ===")
        
        # ケース1: アンサンブル有効 → アンサンブルトレーニング
        ensemble_enabled = {"enabled": True}
        single_model_config = {"model_type": "lightgbm"}
        
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_enabled)
        assert trainer_type == "ensemble"
        logger.info("✅ アンサンブル有効時: ensembleが優先")
        
        # ケース2: アンサンブル無効 → 単一モデルトレーニング
        ensemble_disabled = {"enabled": False}
        
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_disabled)
        assert trainer_type == "single"
        logger.info("✅ アンサンブル無効時: singleが選択")
        
        # ケース3: 設定なし → デフォルト（アンサンブル）
        trainer_type = MLTrainingService.determine_trainer_type(None)
        assert trainer_type == "ensemble"
        logger.info("✅ 設定なし時: ensembleがデフォルト")
        
        return True

    def test_overall_api_functionality(self):
        """全体的なAPI機能をテスト"""
        logger.info("=== 全体的なAPI機能テスト ===")
        
        # 各テストを実行
        config_validation = self.test_single_model_config_validation()
        ensemble_disabled = self.test_ensemble_config_disabled()
        request_structure = self.test_api_request_structure()
        ml_service_integration = self.test_ml_training_service_integration()
        available_models_api = self.test_available_models_api_simulation()
        model_types_test = self.test_different_model_types()
        priority_test = self.test_configuration_priority()
        
        # 総合スコアを計算
        api_score = 0
        
        # 設定検証（最大15点）
        if config_validation:
            api_score += 15
        
        # アンサンブル無効化（最大15点）
        if ensemble_disabled:
            api_score += 15
        
        # リクエスト構造（最大20点）
        if request_structure:
            api_score += 20
        
        # MLTrainingService統合（最大20点）
        if ml_service_integration:
            api_score += 20
        
        # 利用可能モデルAPI（最大10点）
        if available_models_api and available_models_api["success"]:
            api_score += 10
        
        # 異なるモデルタイプ（最大15点）
        success_count = sum(1 for result in model_types_test.values() if result == "成功")
        total_models = len(model_types_test)
        if total_models > 0:
            model_score = (success_count / total_models) * 15
            api_score += model_score
        
        # 設定優先順位（最大5点）
        if priority_test:
            api_score += 5
        
        logger.info(f"API機能スコア: {api_score:.1f}/100")
        
        if api_score >= 90:
            logger.info("🎉 優秀なAPI機能が確認されました")
        elif api_score >= 80:
            logger.info("✅ 良好なAPI機能が確認されました")
        elif api_score >= 70:
            logger.info("✅ 基本的なAPI機能が確認されました")
        else:
            logger.warning("⚠️ API機能に改善が必要です")
        
        return {
            'api_score': api_score,
            'config_validation': config_validation,
            'ensemble_disabled': ensemble_disabled,
            'request_structure': request_structure is not None,
            'ml_service_integration': ml_service_integration is not None,
            'available_models_count': len(available_models_api["available_models"]) if available_models_api else 0,
            'model_types_success_rate': success_count / total_models if total_models > 0 else 0,
            'priority_test': priority_test
        }


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_instance = TestSingleModelAPI()
    
    # 全体的なAPI機能を検証
    results = test_instance.test_overall_api_functionality()
    
    print(f"\n=== 単一モデルトレーニングAPI機能テスト結果 ===")
    print(f"API機能スコア: {results['api_score']:.1f}/100")
    print(f"設定検証: {'成功' if results['config_validation'] else '失敗'}")
    print(f"アンサンブル無効化: {'成功' if results['ensemble_disabled'] else '失敗'}")
    print(f"リクエスト構造: {'成功' if results['request_structure'] else '失敗'}")
    print(f"MLTrainingService統合: {'成功' if results['ml_service_integration'] else '失敗'}")
    print(f"利用可能モデル数: {results['available_models_count']}")
    print(f"モデルタイプ成功率: {results['model_types_success_rate']*100:.1f}%")
    print(f"設定優先順位: {'成功' if results['priority_test'] else '失敗'}")
