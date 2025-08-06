#!/usr/bin/env python3
"""
全削除APIエンドポイントのテストスクリプト

このスクリプトは、ルーティングの修正後に全削除APIが正しく動作することを確認します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from fastapi.testclient import TestClient
from app.main import app

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_delete_all_models_endpoint():
    """全削除APIエンドポイントのテスト"""
    logger.info("=== 全削除APIエンドポイントテスト開始 ===")
    
    try:
        # TestClientを作成
        client = TestClient(app)
        
        # 全削除APIを呼び出し
        response = client.delete("/api/ml/models/all")
        
        logger.info(f"レスポンスステータス: {response.status_code}")
        logger.info(f"レスポンス内容: {response.text}")
        
        # ステータスコードが200であることを確認
        assert response.status_code == 200, f"期待されるステータス: 200, 実際: {response.status_code}"
        
        # レスポンスがJSONであることを確認
        response_data = response.json()
        logger.info(f"レスポンスJSON: {response_data}")
        
        # 必要なフィールドが含まれていることを確認
        assert "success" in response_data, "レスポンスに 'success' フィールドがありません"
        assert "message" in response_data, "レスポンスに 'message' フィールドがありません"
        assert "deleted_count" in response_data, "レスポンスに 'deleted_count' フィールドがありません"
        
        logger.info("✅ 全削除APIエンドポイントテスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_delete_still_works():
    """個別削除APIが引き続き動作することを確認"""
    logger.info("=== 個別削除APIテスト開始 ===")
    
    try:
        # TestClientを作成
        client = TestClient(app)
        
        # 存在しないモデルIDで個別削除APIを呼び出し（404エラーが期待される）
        response = client.delete("/api/ml/models/nonexistent_model")
        
        logger.info(f"レスポンスステータス: {response.status_code}")
        logger.info(f"レスポンス内容: {response.text}")
        
        # 404エラーが返されることを確認（モデルが存在しないため）
        assert response.status_code == 404, f"期待されるステータス: 404, 実際: {response.status_code}"
        
        logger.info("✅ 個別削除APIテスト成功（404エラーが正しく返される）")
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_routing_distinction():
    """ルーティングの区別が正しく動作することを確認"""
    logger.info("=== ルーティング区別テスト開始 ===")
    
    try:
        # TestClientを作成
        client = TestClient(app)
        
        # 1. 全削除API（/models/all）
        response_all = client.delete("/api/ml/models/all")
        logger.info(f"全削除API - ステータス: {response_all.status_code}")
        
        # 2. 個別削除API（/models/specific_id）
        response_individual = client.delete("/api/ml/models/specific_model_id")
        logger.info(f"個別削除API - ステータス: {response_individual.status_code}")
        
        # 両方とも適切に処理されることを確認
        # 全削除は200（成功）、個別削除は404（モデルが存在しない）が期待される
        assert response_all.status_code == 200, f"全削除API - 期待: 200, 実際: {response_all.status_code}"
        assert response_individual.status_code == 404, f"個別削除API - 期待: 404, 実際: {response_individual.status_code}"
        
        # レスポンス内容の確認
        all_data = response_all.json()
        individual_data = response_individual.json()
        
        # 全削除のレスポンスには deleted_count が含まれる
        assert "deleted_count" in all_data, "全削除APIのレスポンスに deleted_count がありません"
        
        # 個別削除のエラーレスポンスには detail が含まれる
        assert "detail" in individual_data, "個別削除APIのエラーレスポンスに detail がありません"
        
        logger.info("✅ ルーティング区別テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    logger.info("全削除APIエンドポイントのルーティング修正テストを開始します")
    
    tests = [
        test_delete_all_models_endpoint,
        test_individual_delete_still_works,
        test_routing_distinction
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print("\n" + "="*50)
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*50)
    logger.info(f"テスト結果: 成功 {passed}件, 失敗 {failed}件")
    
    if failed == 0:
        logger.info("🎉 すべてのテストが成功しました！")
        logger.info("全削除APIのルーティングが正しく修正されています。")
    else:
        logger.error("❌ 一部のテストが失敗しました。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
