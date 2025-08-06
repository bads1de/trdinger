#!/usr/bin/env python3
"""
MLモデル全削除機能の手動テストスクリプト

このスクリプトは、全削除機能が正しく実装されているかを確認するために使用します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
from app.services.ml.orchestration.ml_management_orchestration_service import MLManagementOrchestrationService

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_delete_all_models():
    """全削除機能のテスト"""
    logger.info("=== MLモデル全削除機能テスト開始 ===")
    
    try:
        # サービスインスタンス作成
        service = MLManagementOrchestrationService()
        
        # 現在のモデル一覧を取得
        logger.info("現在のモデル一覧を取得中...")
        models_before = await service.get_formatted_models()
        logger.info(f"削除前のモデル数: {len(models_before.get('models', []))}")
        
        if models_before.get('models'):
            for model in models_before['models']:
                logger.info(f"  - {model.get('name', 'Unknown')} ({model.get('path', 'Unknown path')})")
        
        # 全削除実行
        logger.info("全削除を実行中...")
        result = await service.delete_all_models()
        
        logger.info(f"削除結果: {result}")
        
        # 削除後のモデル一覧を取得
        logger.info("削除後のモデル一覧を取得中...")
        models_after = await service.get_formatted_models()
        logger.info(f"削除後のモデル数: {len(models_after.get('models', []))}")
        
        # 結果検証
        if result['success']:
            logger.info("✅ 全削除機能は正常に動作しています")
            logger.info(f"削除されたモデル数: {result.get('deleted_count', 0)}")
            if result.get('failed_count', 0) > 0:
                logger.warning(f"削除に失敗したモデル数: {result.get('failed_count', 0)}")
                logger.warning(f"失敗したモデル: {result.get('failed_models', [])}")
        else:
            logger.error("❌ 全削除機能でエラーが発生しました")
            
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

async def test_delete_all_models_no_models():
    """モデルがない場合の全削除テスト"""
    logger.info("=== モデルなし状態での全削除テスト開始 ===")
    
    try:
        service = MLManagementOrchestrationService()
        
        # 全削除実行（モデルがない状態）
        result = await service.delete_all_models()
        
        logger.info(f"削除結果: {result}")
        
        # 結果検証
        if result['success'] and result.get('deleted_count', 0) == 0:
            logger.info("✅ モデルなし状態での全削除は正常に動作しています")
        else:
            logger.error("❌ モデルなし状態での全削除で予期しない結果が返されました")
            
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")

async def main():
    """メインテスト実行"""
    logger.info("MLモデル全削除機能の統合テストを開始します")
    
    # テスト1: 通常の全削除
    await test_delete_all_models()
    
    print("\n" + "="*50 + "\n")
    
    # テスト2: モデルなし状態での全削除
    await test_delete_all_models_no_models()
    
    logger.info("🎉 すべてのテストが完了しました")

if __name__ == "__main__":
    asyncio.run(main())
