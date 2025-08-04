"""
BaseResourceManagerの実装テスト

2.21のリファクタリング実装が正しく動作することを確認するテストスクリプト
"""

import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.common.base_resource_manager import (
    BaseResourceManager, 
    CleanupLevel, 
    ResourceManagedOperation,
    managed_ml_operation
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResourceManager(BaseResourceManager):
    """テスト用のリソースマネージャー実装"""
    
    def __init__(self):
        super().__init__()
        self.test_data = {"models": [], "cache": {}, "temp_files": []}
        self.cleanup_called = {"temp_files": False, "cache": False, "models": False}
        
    def _cleanup_temporary_files(self, level: CleanupLevel):
        """テスト用一時ファイルクリーンアップ"""
        logger.info(f"一時ファイルクリーンアップ実行（レベル: {level.value}）")
        self.test_data["temp_files"].clear()
        self.cleanup_called["temp_files"] = True
        
    def _cleanup_cache(self, level: CleanupLevel):
        """テスト用キャッシュクリーンアップ"""
        logger.info(f"キャッシュクリーンアップ実行（レベル: {level.value}）")
        self.test_data["cache"].clear()
        self.cleanup_called["cache"] = True
        
    def _cleanup_models(self, level: CleanupLevel):
        """テスト用モデルクリーンアップ"""
        logger.info(f"モデルクリーンアップ実行（レベル: {level.value}）")
        self.test_data["models"].clear()
        self.cleanup_called["models"] = True
        
    def add_test_data(self):
        """テスト用データを追加"""
        self.test_data["models"] = ["model1", "model2"]
        self.test_data["cache"] = {"key1": "value1", "key2": "value2"}
        self.test_data["temp_files"] = ["temp1.txt", "temp2.txt"]


def test_basic_cleanup():
    """基本的なクリーンアップ機能のテスト"""
    logger.info("=== 基本的なクリーンアップ機能のテスト ===")
    
    manager = TestResourceManager()
    manager.add_test_data()
    
    # クリーンアップ前の状態確認
    assert len(manager.test_data["models"]) == 2
    assert len(manager.test_data["cache"]) == 2
    assert len(manager.test_data["temp_files"]) == 2
    
    # クリーンアップ実行
    stats = manager.cleanup_resources()
    
    # クリーンアップ後の状態確認
    assert len(manager.test_data["models"]) == 0
    assert len(manager.test_data["cache"]) == 0
    assert len(manager.test_data["temp_files"]) == 0
    assert all(manager.cleanup_called.values())
    
    logger.info(f"クリーンアップ統計: {stats}")
    logger.info("✅ 基本的なクリーンアップ機能のテスト完了")


def test_cleanup_levels():
    """クリーンアップレベルのテスト"""
    logger.info("=== クリーンアップレベルのテスト ===")
    
    for level in [CleanupLevel.MINIMAL, CleanupLevel.STANDARD, CleanupLevel.THOROUGH]:
        logger.info(f"レベル {level.value} のテスト")
        
        manager = TestResourceManager()
        manager.add_test_data()
        manager.set_cleanup_level(level)
        
        stats = manager.cleanup_resources()
        
        # すべてのレベルで基本的なクリーンアップは実行される
        assert all(manager.cleanup_called.values())
        assert stats["level"] == level.value
        
        logger.info(f"レベル {level.value} 完了")
    
    logger.info("✅ クリーンアップレベルのテスト完了")


def test_context_manager():
    """コンテキストマネージャーのテスト"""
    logger.info("=== コンテキストマネージャーのテスト ===")
    
    manager = TestResourceManager()
    manager.add_test_data()
    
    # with文を使用したテスト
    with manager as managed:
        assert managed is manager
        # with文内では何もクリーンアップされていない
        assert len(manager.test_data["models"]) == 2
    
    # with文を抜けた後、自動的にクリーンアップされている
    assert len(manager.test_data["models"]) == 0
    assert all(manager.cleanup_called.values())
    
    logger.info("✅ コンテキストマネージャーのテスト完了")


def test_resource_managed_operation():
    """ResourceManagedOperationのテスト"""
    logger.info("=== ResourceManagedOperationのテスト ===")
    
    manager = TestResourceManager()
    manager.add_test_data()
    
    # ResourceManagedOperationを使用したテスト
    with ResourceManagedOperation(manager, "テスト操作", CleanupLevel.STANDARD) as managed:
        assert managed is manager
        # 操作中はクリーンアップされていない
        assert len(manager.test_data["models"]) == 2
    
    # 操作完了後、自動的にクリーンアップされている
    assert len(manager.test_data["models"]) == 0
    assert all(manager.cleanup_called.values())
    
    logger.info("✅ ResourceManagedOperationのテスト完了")


def test_managed_ml_operation():
    """managed_ml_operation関数のテスト"""
    logger.info("=== managed_ml_operation関数のテスト ===")
    
    manager = TestResourceManager()
    manager.add_test_data()
    
    # managed_ml_operation関数を使用したテスト
    with managed_ml_operation(manager, "ML操作テスト", CleanupLevel.THOROUGH) as managed:
        assert managed is manager
        # 操作中はクリーンアップされていない
        assert len(manager.test_data["models"]) == 2
    
    # 操作完了後、自動的にクリーンアップされている
    assert len(manager.test_data["models"]) == 0
    assert all(manager.cleanup_called.values())
    
    logger.info("✅ managed_ml_operation関数のテスト完了")


def test_cleanup_callbacks():
    """クリーンアップコールバックのテスト"""
    logger.info("=== クリーンアップコールバックのテスト ===")
    
    manager = TestResourceManager()
    manager.add_test_data()
    
    # コールバック実行フラグ
    callback_executed = {"callback1": False, "callback2": False}
    
    def callback1():
        callback_executed["callback1"] = True
        logger.info("コールバック1実行")
        
    def callback2():
        callback_executed["callback2"] = True
        logger.info("コールバック2実行")
    
    # コールバックを追加
    manager.add_cleanup_callback(callback1)
    manager.add_cleanup_callback(callback2)
    
    # クリーンアップ実行
    manager.cleanup_resources()
    
    # コールバックが実行されたことを確認
    assert all(callback_executed.values())
    
    logger.info("✅ クリーンアップコールバックのテスト完了")


def test_error_handling():
    """エラーハンドリングのテスト"""
    logger.info("=== エラーハンドリングのテスト ===")
    
    class ErrorTestResourceManager(TestResourceManager):
        def _cleanup_models(self, level: CleanupLevel):
            raise Exception("テスト用エラー")
    
    manager = ErrorTestResourceManager()
    manager.add_test_data()
    
    # エラーが発生してもクリーンアップは継続される
    stats = manager.cleanup_resources()
    
    # エラーが記録されている
    assert len(stats["errors"]) > 0
    assert "テスト用エラー" in str(stats["errors"])
    
    # 他のクリーンアップは正常に実行されている
    assert manager.cleanup_called["temp_files"]
    assert manager.cleanup_called["cache"]
    
    logger.info("✅ エラーハンドリングのテスト完了")


def main():
    """メインテスト実行"""
    logger.info("BaseResourceManagerの実装テストを開始")
    
    try:
        test_basic_cleanup()
        test_cleanup_levels()
        test_context_manager()
        test_resource_managed_operation()
        test_managed_ml_operation()
        test_cleanup_callbacks()
        test_error_handling()
        
        logger.info("🎉 すべてのテストが正常に完了しました！")
        
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
