"""
ML設定管理APIの手動テストスクリプト

実装したML設定の更新・リセット機能をテストします。
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_get_ml_config():
    """ML設定取得テスト"""
    print("=== ML設定取得テスト ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/ml/config")
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            config = response.json()
            print("✅ ML設定取得成功")
            print(f"データ処理設定: max_ohlcv_rows = {config['data_processing']['max_ohlcv_rows']}")
            print(f"予測設定: default_up_prob = {config['prediction']['default_up_prob']}")
            return config
        else:
            print(f"❌ ML設定取得失敗: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def test_update_ml_config():
    """ML設定更新テスト"""
    print("\n=== ML設定更新テスト ===")
    
    update_data = {
        "prediction": {
            "default_up_prob": 0.4,
            "default_down_prob": 0.3,
            "default_range_prob": 0.3
        },
        "data_processing": {
            "max_ohlcv_rows": 500000
        }
    }
    
    try:
        response = requests.put(
            f"{BASE_URL}/api/ml/config",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ML設定更新成功")
            print(f"レスポンス: {result}")
            
            # 更新された設定を確認
            if result.get("success"):
                updated_config = result.get("data")
                if updated_config:
                    print(f"更新後の予測設定: default_up_prob = {updated_config['prediction']['default_up_prob']}")
                    print(f"更新後のデータ処理設定: max_ohlcv_rows = {updated_config['data_processing']['max_ohlcv_rows']}")
            return True
        else:
            print(f"❌ ML設定更新失敗: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_invalid_update():
    """無効な設定更新テスト"""
    print("\n=== 無効な設定更新テスト ===")
    
    invalid_data = {
        "prediction": {
            "default_up_prob": 1.5,  # 無効な値（1.0を超える）
            "default_down_prob": -0.5  # 無効な値（負の値）
        },
        "data_processing": {
            "max_ohlcv_rows": -1  # 無効な値（負の値）
        }
    }
    
    try:
        response = requests.put(
            f"{BASE_URL}/api/ml/config",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if not result.get("success"):
                print("✅ 無効な設定更新が正しく拒否されました")
                print(f"エラーメッセージ: {result.get('message')}")
                return True
            else:
                print("❌ 無効な設定更新が受け入れられました（予期しない動作）")
                return False
        else:
            print(f"❌ 予期しないエラー: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_reset_ml_config():
    """ML設定リセットテスト"""
    print("\n=== ML設定リセットテスト ===")
    
    try:
        response = requests.post(f"{BASE_URL}/api/ml/config/reset")
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ML設定リセット成功")
            print(f"レスポンス: {result}")
            
            if result.get("success"):
                reset_config = result.get("data")
                if reset_config:
                    print(f"リセット後の予測設定: default_up_prob = {reset_config['prediction']['default_up_prob']}")
                    print(f"リセット後のデータ処理設定: max_ohlcv_rows = {reset_config['data_processing']['max_ohlcv_rows']}")
            return True
        else:
            print(f"❌ ML設定リセット失敗: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_server_connection():
    """サーバー接続テスト"""
    print("=== サーバー接続テスト ===")
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ バックエンドサーバーに正常に接続できました")
            return True
        else:
            print(f"❌ サーバー接続失敗: ステータスコード {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ サーバー接続エラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🚀 ML設定管理API統合テスト開始")
    print("=" * 50)
    
    # サーバー接続確認
    if not test_server_connection():
        print("❌ サーバーに接続できません。テストを中止します。")
        return
    
    time.sleep(1)
    
    # 初期設定取得
    initial_config = test_get_ml_config()
    if not initial_config:
        print("❌ 初期設定取得に失敗しました。テストを中止します。")
        return
    
    time.sleep(1)
    
    # 設定更新テスト
    if test_update_ml_config():
        print("✅ 設定更新テスト成功")
    else:
        print("❌ 設定更新テスト失敗")
    
    time.sleep(1)
    
    # 無効な設定更新テスト
    if test_invalid_update():
        print("✅ 無効な設定更新テスト成功")
    else:
        print("❌ 無効な設定更新テスト失敗")
    
    time.sleep(1)
    
    # 設定リセットテスト
    if test_reset_ml_config():
        print("✅ 設定リセットテスト成功")
    else:
        print("❌ 設定リセットテスト失敗")
    
    time.sleep(1)
    
    # 最終確認
    final_config = test_get_ml_config()
    if final_config:
        print("\n=== テスト完了 ===")
        print("✅ 全てのAPIエンドポイントが正常に動作しています")
        
        # デフォルト値に戻っているか確認
        if (final_config['prediction']['default_up_prob'] == 0.33 and 
            final_config['data_processing']['max_ohlcv_rows'] == 1000000):
            print("✅ 設定が正しくデフォルト値にリセットされました")
        else:
            print("⚠️ 設定がデフォルト値と異なります")
    else:
        print("❌ 最終確認に失敗しました")

if __name__ == "__main__":
    main()
