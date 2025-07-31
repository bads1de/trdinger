"""
単一モデルトレーニング機能のテスト

アンサンブル機能をオフにして単一モデルでのトレーニングが
正常に動作することを確認する。
"""

import pytest
import pandas as pd
import numpy as np
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.single_model.single_model_trainer import SingleModelTrainer
from app.services.ml.ml_training_service import MLTrainingService

logger = logging.getLogger(__name__)


class TestSingleModelTraining:
    """単一モデルトレーニング機能のテストクラス"""

    def generate_test_data(self):
        """テスト用のデータを生成"""
        np.random.seed(42)
        n_samples = 200
        
        # 特徴量データを生成
        features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples),
            'feature5': np.random.normal(0, 1, n_samples),
        })
        
        # ターゲットデータを生成（3クラス分類）
        target = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
        target = pd.Series(target)
        
        return features, target

    def test_single_model_trainer_initialization(self):
        """SingleModelTrainerの初期化をテスト"""
        logger.info("=== SingleModelTrainer初期化テスト ===")
        
        # 利用可能なモデルを取得
        available_models = SingleModelTrainer.get_available_models()
        logger.info(f"利用可能なモデル: {available_models}")
        
        assert len(available_models) > 0, "利用可能なモデルが見つかりません"
        
        # 各モデルタイプで初期化をテスト
        for model_type in available_models:
            try:
                trainer = SingleModelTrainer(model_type=model_type)
                assert trainer.model_type == model_type
                logger.info(f"✅ {model_type.upper()}トレーナーの初期化成功")
            except Exception as e:
                logger.error(f"❌ {model_type.upper()}トレーナーの初期化失敗: {e}")
                raise
        
        return available_models

    def test_single_model_training(self):
        """単一モデルでのトレーニングをテスト"""
        logger.info("=== 単一モデルトレーニングテスト ===")
        
        # テストデータを生成
        X, y = self.generate_test_data()
        
        # 訓練・テストデータに分割
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # 利用可能なモデルを取得
        available_models = SingleModelTrainer.get_available_models()
        
        # 最初の利用可能なモデルでテスト
        if available_models:
            model_type = available_models[0]
            logger.info(f"テスト対象モデル: {model_type.upper()}")
            
            try:
                # トレーナーを初期化
                trainer = SingleModelTrainer(model_type=model_type)
                
                # トレーニングを実行
                result = trainer._train_model_impl(X_train, X_test, y_train, y_test)
                
                # 結果を検証
                assert "model_type" in result
                assert result["model_type"] == model_type
                assert "training_samples" in result
                assert result["training_samples"] == len(X_train)
                assert "test_samples" in result
                assert result["test_samples"] == len(X_test)
                
                logger.info(f"✅ {model_type.upper()}モデルのトレーニング成功")
                logger.info(f"   - 訓練サンプル数: {result['training_samples']}")
                logger.info(f"   - テストサンプル数: {result['test_samples']}")
                logger.info(f"   - 特徴量数: {result['feature_count']}")
                
                # 予測をテスト
                predictions = trainer.predict(X_test)
                assert predictions.shape == (len(X_test), 3), f"予測形状が不正: {predictions.shape}"
                
                # 予測確率の合計が1に近いことを確認
                prob_sums = np.sum(predictions, axis=1)
                assert np.allclose(prob_sums, 1.0, atol=1e-6), "予測確率の合計が1になっていません"
                
                logger.info(f"✅ {model_type.upper()}モデルの予測成功")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ {model_type.upper()}モデルのトレーニング失敗: {e}")
                raise
        else:
            pytest.skip("利用可能なモデルがありません")

    def test_ml_training_service_single_mode(self):
        """MLTrainingServiceでの単一モデルモードをテスト"""
        logger.info("=== MLTrainingService単一モードテスト ===")
        
        # 利用可能なモデルを取得
        available_models = MLTrainingService.get_available_single_models()
        logger.info(f"MLTrainingService利用可能モデル: {available_models}")
        
        if available_models:
            model_type = available_models[0]
            
            # 単一モデル設定
            single_model_config = {"model_type": model_type}
            
            try:
                # MLTrainingServiceを単一モードで初期化
                ml_service = MLTrainingService(
                    trainer_type="single",
                    single_model_config=single_model_config
                )
                
                assert ml_service.trainer_type == "single"
                assert isinstance(ml_service.trainer, SingleModelTrainer)
                assert ml_service.trainer.model_type == model_type
                
                logger.info(f"✅ MLTrainingService単一モード初期化成功: {model_type.upper()}")
                
                return ml_service
                
            except Exception as e:
                logger.error(f"❌ MLTrainingService単一モード初期化失敗: {e}")
                raise
        else:
            pytest.skip("利用可能なモデルがありません")

    def test_trainer_type_determination(self):
        """トレーナータイプの自動決定をテスト"""
        logger.info("=== トレーナータイプ自動決定テスト ===")
        
        # アンサンブル有効の場合
        ensemble_config_enabled = {"enabled": True}
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_config_enabled)
        assert trainer_type == "ensemble", f"期待値: ensemble, 実際: {trainer_type}"
        logger.info("✅ アンサンブル有効時: ensembleトレーナー選択")
        
        # アンサンブル無効の場合
        ensemble_config_disabled = {"enabled": False}
        trainer_type = MLTrainingService.determine_trainer_type(ensemble_config_disabled)
        assert trainer_type == "single", f"期待値: single, 実際: {trainer_type}"
        logger.info("✅ アンサンブル無効時: singleトレーナー選択")
        
        # 設定なしの場合（デフォルト）
        trainer_type = MLTrainingService.determine_trainer_type(None)
        assert trainer_type == "ensemble", f"期待値: ensemble, 実際: {trainer_type}"
        logger.info("✅ 設定なし時: ensembleトレーナー選択（デフォルト）")

    def test_model_info_and_compatibility(self):
        """モデル情報と互換性をテスト"""
        logger.info("=== モデル情報・互換性テスト ===")
        
        available_models = SingleModelTrainer.get_available_models()
        
        if available_models:
            model_type = available_models[0]
            
            # トレーナーを初期化
            trainer = SingleModelTrainer(model_type=model_type)
            
            # 初期状態のモデル情報を確認
            info = trainer.get_model_info()
            assert info["model_type"] == model_type
            assert info["is_trained"] == False
            assert info["trainer_type"] == "single_model"
            
            logger.info(f"✅ 初期モデル情報確認: {info}")
            
            # テストデータでトレーニング
            X, y = self.generate_test_data()
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            trainer._train_model_impl(X_train, X_test, y_train, y_test)
            
            # トレーニング後のモデル情報を確認
            info_after = trainer.get_model_info()
            assert info_after["is_trained"] == True
            assert info_after["feature_count"] == len(X.columns)
            
            logger.info(f"✅ トレーニング後モデル情報確認: {info_after}")

    def test_overall_single_model_functionality(self):
        """全体的な単一モデル機能をテスト"""
        logger.info("=== 全体的な単一モデル機能テスト ===")
        
        # 各テストを実行
        available_models = self.test_single_model_trainer_initialization()
        training_result = self.test_single_model_training()
        ml_service = self.test_ml_training_service_single_mode()
        self.test_trainer_type_determination()
        self.test_model_info_and_compatibility()
        
        # 総合評価
        functionality_score = 0
        
        # 利用可能モデル数（最大20点）
        if len(available_models) >= 3:
            functionality_score += 20
        elif len(available_models) >= 2:
            functionality_score += 15
        elif len(available_models) >= 1:
            functionality_score += 10
        
        # トレーニング成功（最大30点）
        if training_result and "model_type" in training_result:
            functionality_score += 30
        
        # MLTrainingService統合（最大25点）
        if ml_service and ml_service.trainer_type == "single":
            functionality_score += 25
        
        # 自動決定機能（最大15点）
        functionality_score += 15  # test_trainer_type_determinationが成功した場合
        
        # 互換性（最大10点）
        functionality_score += 10  # test_model_info_and_compatibilityが成功した場合
        
        logger.info(f"単一モデル機能スコア: {functionality_score}/100")
        
        if functionality_score >= 90:
            logger.info("🎉 優秀な単一モデル機能が確認されました")
        elif functionality_score >= 70:
            logger.info("✅ 良好な単一モデル機能が確認されました")
        elif functionality_score >= 50:
            logger.info("⚠️ 基本的な単一モデル機能が確認されました")
        else:
            logger.warning("❌ 単一モデル機能に問題があります")
        
        return {
            'functionality_score': functionality_score,
            'available_models': available_models,
            'training_result': training_result,
            'ml_service_initialized': ml_service is not None
        }


if __name__ == "__main__":
    # テストを直接実行する場合
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_instance = TestSingleModelTraining()
    
    # 全体的な機能を検証
    results = test_instance.test_overall_single_model_functionality()
    
    print(f"\n=== 単一モデルトレーニング機能テスト結果 ===")
    print(f"機能スコア: {results['functionality_score']}/100")
    print(f"利用可能モデル数: {len(results['available_models'])}")
    print(f"利用可能モデル: {results['available_models']}")
    print(f"MLTrainingService統合: {'成功' if results['ml_service_initialized'] else '失敗'}")
