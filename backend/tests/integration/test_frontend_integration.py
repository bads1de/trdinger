"""
フロントエンド統合テスト

修正後のフロントエンドからベイジアン最適化を実行して、
DB保存が正しく動作するかをテストします。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
import json
import time
import sqlite3
from datetime import datetime

def check_database_status():
    """データベースの状況を確認"""
    db_path = "C:/Users/buti3/trading/backend/trdinger.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM bayesian_optimization_results")
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute("""
                SELECT id, profile_name, model_type, best_score, created_at, save_as_profile, target_model_type
                FROM bayesian_optimization_results 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            records = cursor.fetchall()
            print(f"レコード数: {count}")
            print("最新レコード:")
            for record in records:
                print(f"  ID: {record[0]}, Name: {record[1]}, Model: {record[2]}, Score: {record[3]}")
                print(f"      Created: {record[4]}, SaveAsProfile: {record[5]}, TargetModel: {record[6]}")
        else:
            print(f"レコード数: {count} (空)")
        
        conn.close()
        return count
        
    except Exception as e:
        print(f"DB確認エラー: {e}")
        return -1

def test_frontend_bayesian_optimization_with_profile():
    """フロントエンド経由でプロファイル保存ありのベイジアン最適化テスト"""
    print("\n=== フロントエンド経由プロファイル保存テスト ===")
    
    # テスト前のDB状況
    initial_count = check_database_status()
    print(f"テスト前のレコード数: {initial_count}")
    
    # フロントエンドAPIエンドポイント（修正後）
    url = "http://localhost:3000/api/bayesian-optimization/ml-hyperparameters"
    
    # リクエストデータ（プロファイル保存あり）
    request_data = {
        "optimization_type": "ml",
        "model_type": "LightGBM",
        "n_calls": 8,
        "optimization_config": {
            "acq_func": "EI",
            "n_initial_points": 5,
            "random_state": 42
        },
        "save_as_profile": True,
        "profile_name": "frontend_integration_test_profile",
        "profile_description": "フロントエンド統合テスト用プロファイル"
    }
    
    print(f"リクエストURL: {url}")
    print(f"リクエストデータ:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        # APIを呼び出し
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        print(f"\nレスポンスステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"レスポンス成功: {result.get('success', False)}")
            print(f"メッセージ: {result.get('message', 'N/A')}")
            
            if result.get('success') and 'result' in result:
                api_result = result['result']
                print(f"ベストスコア: {api_result.get('best_score', 'N/A')}")
                print(f"評価回数: {api_result.get('total_evaluations', 'N/A')}")
                print(f"最適化時間: {api_result.get('optimization_time', 'N/A')}秒")
                
                if 'saved_profile_id' in api_result:
                    print(f"✅ 保存されたプロファイルID: {api_result['saved_profile_id']}")
                    return api_result['saved_profile_id']
                else:
                    print("❌ saved_profile_idが結果に含まれていません")
                    print(f"結果の内容: {list(api_result.keys())}")
            else:
                print("❌ 結果が正常に取得できませんでした")
                print(f"レスポンス内容: {result}")
        else:
            print(f"❌ APIエラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
        
        # テスト後のDB状況
        time.sleep(2)  # DB書き込み完了を待つ
        final_count = check_database_status()
        print(f"\nテスト後のレコード数: {final_count}")
        
        if final_count > initial_count:
            print("✅ フロントエンド経由でDB保存成功")
            return True
        else:
            print("❌ フロントエンド経由でDB保存されませんでした")
            return False
        
    except requests.exceptions.ConnectionError:
        print("❌ フロントエンドサーバーに接続できません")
        print("   フロントエンドサーバーが起動していることを確認してください")
        return False
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        return False

def test_frontend_bayesian_optimization_without_profile():
    """フロントエンド経由でプロファイル保存なしのベイジアン最適化テスト"""
    print("\n=== フロントエンド経由プロファイル保存なしテスト ===")
    
    # テスト前のDB状況
    initial_count = check_database_status()
    print(f"テスト前のレコード数: {initial_count}")
    
    # フロントエンドAPIエンドポイント
    url = "http://localhost:3000/api/bayesian-optimization/ml-hyperparameters"
    
    # リクエストデータ（プロファイル保存なし）
    request_data = {
        "optimization_type": "ml",
        "model_type": "XGBoost",
        "n_calls": 5,
        "optimization_config": {
            "acq_func": "EI",
            "n_initial_points": 3,
            "random_state": 42
        },
        "save_as_profile": False
    }
    
    print(f"リクエストデータ:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        print(f"\nレスポンスステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"レスポンス成功: {result.get('success', False)}")
            
            if result.get('success') and 'result' in result:
                api_result = result['result']
                print(f"ベストスコア: {api_result.get('best_score', 'N/A')}")
                
                if 'saved_profile_id' not in api_result:
                    print("✅ saved_profile_idが含まれていません（正常）")
                else:
                    print("❌ saved_profile_idが含まれています（異常）")
        
        # テスト後のDB状況
        time.sleep(2)
        final_count = check_database_status()
        print(f"\nテスト後のレコード数: {final_count}")
        
        if final_count == initial_count:
            print("✅ プロファイル保存なしでDB保存されませんでした（正常）")
            return True
        else:
            print("❌ プロファイル保存なしでもDB保存されました（異常）")
            return False
        
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        return False

def test_direct_backend_comparison():
    """バックエンド直接呼び出しとの比較テスト"""
    print("\n=== バックエンド直接呼び出し比較テスト ===")
    
    # バックエンドAPIエンドポイント
    url = "http://localhost:8000/api/bayesian-optimization/ml-hyperparameters"
    
    # リクエストデータ
    request_data = {
        "model_type": "RandomForest",
        "n_calls": 5,
        "save_as_profile": True,
        "profile_name": "backend_direct_comparison_test",
        "profile_description": "バックエンド直接比較テスト"
    }
    
    print(f"バックエンド直接呼び出しテスト:")
    print(f"リクエストURL: {url}")
    
    try:
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"レスポンス成功: {result.get('success', False)}")
            
            if result.get('success') and 'result' in result:
                api_result = result['result']
                if 'saved_profile_id' in api_result:
                    print(f"✅ バックエンド直接: 保存されたプロファイルID: {api_result['saved_profile_id']}")
                    return True
                else:
                    print("❌ バックエンド直接: saved_profile_idが含まれていません")
        else:
            print(f"❌ バックエンド直接APIエラー: {response.status_code}")
        
        return False
        
    except Exception as e:
        print(f"❌ バックエンド直接API呼び出しエラー: {e}")
        return False

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
            OR profile_name LIKE '%integration%'
            OR profile_name LIKE '%comparison%'
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"削除されたテストレコード数: {deleted_count}")
        
    except Exception as e:
        print(f"クリーンアップエラー: {e}")

def main():
    """メイン実行関数"""
    print("フロントエンド統合テスト開始")
    print("=" * 60)
    
    # 初期DB状況確認
    print("=== 初期DB状況 ===")
    check_database_status()
    
    # テスト実行
    frontend_with_profile = test_frontend_bayesian_optimization_with_profile()
    frontend_without_profile = test_frontend_bayesian_optimization_without_profile()
    backend_direct = test_direct_backend_comparison()
    
    # 最終DB状況確認
    print("\n=== 最終DB状況 ===")
    check_database_status()
    
    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    print(f"フロントエンド（プロファイル保存あり）: {'✅ 成功' if frontend_with_profile else '❌ 失敗'}")
    print(f"フロントエンド（プロファイル保存なし）: {'✅ 成功' if frontend_without_profile else '❌ 失敗'}")
    print(f"バックエンド直接: {'✅ 成功' if backend_direct else '❌ 失敗'}")
    
    if frontend_with_profile and frontend_without_profile:
        print("\n🎉 フロントエンド統合テスト成功！")
        print("   プロファイル保存機能が正常に動作しています。")
    else:
        print("\n⚠️ フロントエンド統合テストで問題が発見されました。")
        if not frontend_with_profile:
            print("   - プロファイル保存ありでDB保存されませんでした")
        if not frontend_without_profile:
            print("   - プロファイル保存なしで予期しない動作をしました")
    
    # クリーンアップ
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    print("テスト完了")

if __name__ == "__main__":
    main()
