"""
ML統合テスト - BaseResourceManagerの実装確認

実際のMLクラス（BaseMLTrainer、EnsembleTrainer、MLTrainingService）で
BaseResourceManagerが正しく動作することを確認するテスト
"""

import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.common.base_resource_manager import (
    CleanupLevel,
    managed_ml_operation,
)
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.services.ml.ml_training_service import MLTrainingService

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_base_ml_trainer_resource_management():
    """BaseMLTrainerのリソース管理テスト"""
    logger.info("=== BaseMLTrainerのリソース管理テスト ===")

    # EnsembleTrainerを使用（BaseMLTrainerの具象実装）
    ensemble_config = {
        "method": "bagging",
        "bagging_params": {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
        },
    }
    trainer = EnsembleTrainer(ensemble_config)

    # BaseResourceManagerのメソッドが利用可能か確認
    assert hasattr(trainer, "cleanup_resources")
    assert hasattr(trainer, "set_cleanup_level")
    assert hasattr(trainer, "add_cleanup_callback")
    assert hasattr(trainer, "is_cleaned_up")

    # クリーンアップレベルの設定テスト
    trainer.set_cleanup_level(CleanupLevel.THOROUGH)

    # コールバック追加テスト
    callback_executed = False

    def test_callback():
        nonlocal callback_executed
        callback_executed = True
        logger.info("テストコールバック実行")

    trainer.add_cleanup_callback(test_callback)

    # クリーンアップ実行
    stats = trainer.cleanup_resources()

    # 結果確認
    assert trainer.is_cleaned_up()
    assert callback_executed
    assert "level" in stats
    assert "cleaned_components" in stats

    logger.info(f"クリーンアップ統計: {stats}")
    logger.info("✅ BaseMLTrainerのリソース管理テスト完了")


def test_ensemble_trainer_resource_management():
    """EnsembleTrainerのリソース管理テスト"""
    logger.info("=== EnsembleTrainerのリソース管理テスト ===")

    ensemble_config = {
        "method": "bagging",
        "bagging_params": {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
        },
    }
    trainer = EnsembleTrainer(ensemble_config)

    # アンサンブル固有の属性を設定
    trainer.ensemble_model = "test_model"  # ダミーモデル

    # クリーンアップ実行
    stats = trainer.cleanup_resources(CleanupLevel.STANDARD)

    # アンサンブルモデルがクリアされていることを確認
    assert trainer.ensemble_model is None
    assert trainer.is_cleaned_up()

    logger.info(f"クリーンアップ統計: {stats}")
    logger.info("✅ EnsembleTrainerのリソース管理テスト完了")


def test_ml_training_service_resource_management():
    """MLTrainingServiceのリソース管理テスト"""
    logger.info("=== MLTrainingServiceのリソース管理テスト ===")

    service = MLTrainingService(trainer_type="ensemble")

    # BaseResourceManagerのメソッドが利用可能か確認
    assert hasattr(service, "cleanup_resources")
    assert hasattr(service, "set_cleanup_level")

    # クリーンアップ実行
    stats = service.cleanup_resources(CleanupLevel.MINIMAL)

    # 結果確認
    assert service.is_cleaned_up()
    assert "level" in stats

    logger.info(f"クリーンアップ統計: {stats}")
    logger.info("✅ MLTrainingServiceのリソース管理テスト完了")


def test_context_manager_with_ml_classes():
    """MLクラスでのコンテキストマネージャーテスト"""
    logger.info("=== MLクラスでのコンテキストマネージャーテスト ===")

    # EnsembleTrainerでのテスト
    ensemble_config = {
        "method": "bagging",
        "bagging_params": {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
        },
    }
    trainer = EnsembleTrainer(ensemble_config)

    with trainer as managed_trainer:
        assert managed_trainer is trainer
        assert not trainer.is_cleaned_up()

        # ダミーデータを設定
        trainer.ensemble_model = "test_model"

    # with文を抜けた後、自動的にクリーンアップされている
    assert trainer.is_cleaned_up()
    assert trainer.ensemble_model is None

    logger.info("✅ MLクラスでのコンテキストマネージャーテスト完了")


def test_managed_ml_operation_with_real_classes():
    """実際のMLクラスでのmanaged_ml_operationテスト"""
    logger.info("=== 実際のMLクラスでのmanaged_ml_operationテスト ===")

    service = MLTrainingService(trainer_type="ensemble")

    with managed_ml_operation(
        service, "ML学習サービステスト", CleanupLevel.THOROUGH
    ) as managed_service:
        assert managed_service is service
        assert not service.is_cleaned_up()

        # 何らかの処理をシミュレート
        logger.info("ML処理をシミュレート中...")

    # 操作完了後、自動的にクリーンアップされている
    assert service.is_cleaned_up()

    logger.info("✅ 実際のMLクラスでのmanaged_ml_operationテスト完了")


def test_inheritance_chain():
    """継承チェーンのテスト"""
    logger.info("=== 継承チェーンのテスト ===")

    ensemble_config = {
        "method": "bagging",
        "bagging_params": {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
        },
    }
    trainer = EnsembleTrainer(ensemble_config)

    # 継承関係の確認
    from app.services.ml.common.base_resource_manager import BaseResourceManager

    assert isinstance(trainer, BaseResourceManager)

    # メソッド解決順序の確認
    mro = EnsembleTrainer.__mro__
    logger.info(f"メソッド解決順序: {[cls.__name__ for cls in mro]}")

    # BaseResourceManagerが継承チェーンに含まれていることを確認
    assert BaseResourceManager in mro

    logger.info("✅ 継承チェーンのテスト完了")


def test_cleanup_level_propagation():
    """クリーンアップレベルの伝播テスト"""
    logger.info("=== クリーンアップレベルの伝播テスト ===")

    service = MLTrainingService(trainer_type="ensemble")

    # 各レベルでのクリーンアップテスト
    for level in [CleanupLevel.MINIMAL, CleanupLevel.STANDARD, CleanupLevel.THOROUGH]:
        # 新しいサービスインスタンスを作成
        test_service = MLTrainingService(trainer_type="ensemble")

        # クリーンアップ実行
        stats = test_service.cleanup_resources(level)

        # レベルが正しく設定されていることを確認
        assert stats["level"] == level.value

        logger.info(f"レベル {level.value} でのクリーンアップ完了")

    logger.info("✅ クリーンアップレベルの伝播テスト完了")


def test_error_resilience():
    """エラー耐性テスト"""
    logger.info("=== エラー耐性テスト ===")

    ensemble_config = {
        "method": "bagging",
        "bagging_params": {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
        },
    }
    trainer = EnsembleTrainer(ensemble_config)

    # 無効なクリーンアップコールバックを追加
    def error_callback():
        raise Exception("テスト用エラー")

    trainer.add_cleanup_callback(error_callback)

    # エラーが発生してもクリーンアップは完了する
    stats = trainer.cleanup_resources()

    # エラーが記録されているが、クリーンアップは完了している
    assert len(stats.get("errors", [])) > 0
    assert trainer.is_cleaned_up()

    logger.info("✅ エラー耐性テスト完了")


def test_memory_tracking():
    """メモリ追跡テスト"""
    logger.info("=== メモリ追跡テスト ===")

    ensemble_config = {
        "method": "bagging",
        "bagging_params": {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
        },
    }
    trainer = EnsembleTrainer(ensemble_config)

    # クリーンアップ実行
    stats = trainer.cleanup_resources()

    # メモリ関連の統計が含まれていることを確認
    assert "memory_before" in stats
    assert "memory_after" in stats
    assert "memory_freed" in stats

    logger.info(
        f"メモリ統計: 解放前={stats['memory_before']:.2f}MB, "
        f"解放後={stats['memory_after']:.2f}MB, "
        f"解放量={stats['memory_freed']:.2f}MB"
    )

    logger.info("✅ メモリ追跡テスト完了")


def main():
    """メインテスト実行"""
    logger.info("ML統合テストを開始")

    try:
        test_base_ml_trainer_resource_management()
        test_ensemble_trainer_resource_management()
        test_ml_training_service_resource_management()
        test_context_manager_with_ml_classes()
        test_managed_ml_operation_with_real_classes()
        test_inheritance_chain()
        test_cleanup_level_propagation()
        test_error_resilience()
        test_memory_tracking()

        logger.info("🎉 すべてのML統合テストが正常に完了しました！")
        logger.info("2.21のリファクタリング実装が正常に動作しています。")

    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
