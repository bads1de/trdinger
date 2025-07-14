"""
実際のベイジアン最適化API呼び出しテスト

フロントエンドから実際に呼び出されるAPIエンドポイントをテストして、
DB保存が正しく動作するかを確認します。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
import json
import time
import sqlite3
from datetime import datetime

def check_database_before_after(test_name):
    """テスト前後のDB状況を確認"""
    db_path = "C:/Users/buti3/trading/backend/trdinger.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM bayesian_optimization_results")
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute("""
                SELECT id, profile_name, model_type, best_score, created_at 
                FROM bayesian_optimization_results 
                ORDER BY created_at DESC 
                LIMIT 3
            """)
            records = cursor.fetchall()
            print(f"{test_name} - レコード数: {count}")
            for record in records:
                print(f"  ID: {record[0]}, Name: {record[1]}, Model: {record[2]}, Score: {record[3]}, Created: {record[4]}")
        else:
            print(f"{test_name} - レコード数: {count} (空)")
        
        conn.close()
        return count
        
    except Exception as e:
        print(f"DB確認エラー: {e}")
        return -1

def test_backend_api_direct():
    """バックエンドAPIを直接呼び出しテスト"""
    print("\n=== バックエンドAPI直接呼び出しテスト ===")
    
    # テスト前のDB状況
    initial_count = check_database_before_after("テスト前")
    
    # バックエンドAPIエンドポイント
    url = "http://localhost:8000/api/bayesian-optimization/ml-hyperparameters"
    
    # リクエストデータ（プロファイル保存あり）
    request_data = {
        "model_type": "LightGBM",
        "n_calls": 5,
        "save_as_profile": True,
        "profile_name": "real_api_test_profile",
        "profile_description": "実際のAPI呼び出しテスト用プロファイル"
    }
    
    print(f"リクエストURL: {url}")
    print(f"リクエストデータ: {json.dumps(request_data, indent=2)}")
    
    try:
        # APIを呼び出し
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"レスポンス成功: {result.get('success', False)}")
            print(f"メッセージ: {result.get('message', 'N/A')}")
            
            if result.get('success') and 'result' in result:
                api_result = result['result']
                print(f"ベストスコア: {api_result.get('best_score', 'N/A')}")
                print(f"評価回数: {api_result.get('total_evaluations', 'N/A')}")
                
                if 'saved_profile_id' in api_result:
                    print(f"✅ 保存されたプロファイルID: {api_result['saved_profile_id']}")
                else:
                    print("❌ saved_profile_idが結果に含まれていません")
            
        else:
            print(f"❌ APIエラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
        
        # テスト後のDB状況
        time.sleep(1)  # DB書き込み完了を待つ
        final_count = check_database_before_after("テスト後")
        
        if final_count > initial_count:
            print("✅ DB保存成功")
        else:
            print("❌ DB保存されませんでした")
        
        return final_count > initial_count
        
    except requests.exceptions.ConnectionError:
        print("❌ バックエンドサーバーに接続できません")
        print("   サーバーが起動していることを確認してください")
        return False
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        return False

def test_frontend_api_route():
    """フロントエンドAPIルート経由テスト"""
    print("\n=== フロントエンドAPIルート経由テスト ===")
    
    # テスト前のDB状況
    initial_count = check_database_before_after("テスト前")
    
    # フロントエンドAPIエンドポイント
    url = "http://localhost:3000/api/bayesian-optimization/ml-hyperparameters"
    
    # リクエストデータ
    request_data = {
        "model_type": "XGBoost",
        "n_calls": 5,
        "save_as_profile": True,
        "profile_name": "frontend_api_test_profile",
        "profile_description": "フロントエンドAPI経由テスト用プロファイル"
    }
    
    print(f"リクエストURL: {url}")
    print(f"リクエストデータ: {json.dumps(request_data, indent=2)}")
    
    try:
        # APIを呼び出し
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"レスポンス成功: {result.get('success', False)}")
            print(f"メッセージ: {result.get('message', 'N/A')}")
            
            if result.get('success') and 'result' in result:
                api_result = result['result']
                print(f"ベストスコア: {api_result.get('best_score', 'N/A')}")
                
                if 'saved_profile_id' in api_result:
                    print(f"✅ 保存されたプロファイルID: {api_result['saved_profile_id']}")
                else:
                    print("❌ saved_profile_idが結果に含まれていません")
        else:
            print(f"❌ APIエラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
        
        # テスト後のDB状況
        time.sleep(1)
        final_count = check_database_before_after("テスト後")
        
        if final_count > initial_count:
            print("✅ DB保存成功")
        else:
            print("❌ DB保存されませんでした")
        
        return final_count > initial_count
        
    except requests.exceptions.ConnectionError:
        print("❌ フロントエンドサーバーに接続できません")
        print("   フロントエンドサーバーが起動していることを確認してください")
        return False
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        return False

def test_api_without_save():
    """プロファイル保存なしでのAPI呼び出しテスト"""
    print("\n=== プロファイル保存なしAPI呼び出しテスト ===")
    
    # テスト前のDB状況
    initial_count = check_database_before_after("テスト前")
    
    # バックエンドAPIエンドポイント
    url = "http://localhost:8000/api/bayesian-optimization/ml-hyperparameters"
    
    # リクエストデータ（プロファイル保存なし）
    request_data = {
        "model_type": "RandomForest",
        "n_calls": 3,
        "save_as_profile": False  # 保存しない
    }
    
    print(f"リクエストデータ: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"レスポンス成功: {result.get('success', False)}")
            
            if result.get('success') and 'result' in result:
                api_result = result['result']
                
                if 'saved_profile_id' not in api_result:
                    print("✅ saved_profile_idが含まれていません（正常）")
                else:
                    print("❌ saved_profile_idが含まれています（異常）")
        
        # テスト後のDB状況
        time.sleep(1)
        final_count = check_database_before_after("テスト後")
        
        if final_count == initial_count:
            print("✅ DB保存されませんでした（正常）")
        else:
            print("❌ DB保存されました（異常）")
        
    except requests.exceptions.ConnectionError:
        print("❌ バックエンドサーバーに接続できません")
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")

def check_server_status():
    """サーバーの起動状況を確認"""
    print("\n=== サーバー起動状況確認 ===")
    
    # バックエンドサーバー確認
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ バックエンドサーバー (port 8000) 起動中")
        else:
            print(f"⚠️ バックエンドサーバー応答異常: {response.status_code}")
    except:
        print("❌ バックエンドサーバー (port 8000) 未起動")
    
    # フロントエンドサーバー確認
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ フロントエンドサーバー (port 3000) 起動中")
        else:
            print(f"⚠️ フロントエンドサーバー応答異常: {response.status_code}")
    except:
        print("❌ フロントエンドサーバー (port 3000) 未起動")

def cleanup_test_data():
    """テストデータのクリーンアップ"""
    print("\n=== テストデータクリーンアップ ===")
    
    db_path = "C:/Users/buti3/trading/backend/trdinger.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # テスト用プロファイルを削除
        cursor.execute("""
            DELETE FROM bayesian_optimization_results 
            WHERE profile_name LIKE '%test%' 
            OR profile_name LIKE '%api%'
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"削除されたテストレコード数: {deleted_count}")
        
    except Exception as e:
        print(f"クリーンアップエラー: {e}")

def main():
    """メイン実行関数"""
    print("実際のベイジアン最適化API呼び出しテスト開始")
    print("=" * 60)
    
    # サーバー起動状況確認
    check_server_status()
    
    # 初期DB状況確認
    print("\n=== 初期DB状況 ===")
    check_database_before_after("初期状況")
    
    # テスト実行
    backend_success = test_backend_api_direct()
    frontend_success = test_frontend_api_route()
    test_api_without_save()
    
    # 最終DB状況確認
    print("\n=== 最終DB状況 ===")
    check_database_before_after("最終状況")
    
    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    print(f"バックエンドAPI直接: {'✅ 成功' if backend_success else '❌ 失敗'}")
    print(f"フロントエンドAPI経由: {'✅ 成功' if frontend_success else '❌ 失敗'}")
    
    if not backend_success and not frontend_success:
        print("\n⚠️ 両方のテストが失敗しました。以下を確認してください：")
        print("1. バックエンドサーバーが起動しているか")
        print("2. フロントエンドサーバーが起動しているか")
        print("3. APIエンドポイントが正しく設定されているか")
        print("4. データベース接続が正常か")
    
    # クリーンアップ
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    print("テスト完了")

if __name__ == "__main__":
    main()
