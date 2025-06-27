#!/usr/bin/env python3
"""
ストラテジービルダー機能の最終統合テスト
"""

import sys
import subprocess
import time

def run_test_script(script_name, description):
    """テストスクリプトを実行"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ 失敗")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ タイムアウト")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def check_server_status():
    """サーバーの状態確認"""
    print("\n🔍 サーバー状態確認")
    try:
        import requests
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ サーバーが正常に動作中")
            return True
        else:
            print(f"⚠️ サーバー応答異常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ サーバー接続エラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 ストラテジービルダー機能の最終統合テストを開始します")
    print("="*80)
    
    # テスト結果を記録
    test_results = []
    
    # 1. データベース初期化テスト
    test_results.append(run_test_script(
        "init_db.py",
        "データベース初期化テスト"
    ))
    
    # 2. ユニットテスト
    test_results.append(run_test_script(
        "run_tests.py",
        "ユニットテスト実行"
    ))
    
    # 3. 戦略保存機能テスト
    test_results.append(run_test_script(
        "test_strategy_save.py",
        "戦略保存機能テスト"
    ))
    
    # 4. サーバー状態確認
    server_ok = check_server_status()
    test_results.append(server_ok)
    
    # 5. APIエンドポイントテスト（サーバーが動作中の場合）
    if server_ok:
        test_results.append(run_test_script(
            "test_api_endpoints.py",
            "APIエンドポイント統合テスト"
        ))
    else:
        print("⚠️ サーバーが動作していないため、APIテストをスキップします")
        test_results.append(False)
    
    # 6. バックテスト統合テスト
    test_results.append(run_test_script(
        "test_backtest_integration.py",
        "バックテスト統合テスト"
    ))
    
    # 結果サマリー
    print("\n" + "="*80)
    print("📊 最終統合テスト結果サマリー")
    print("="*80)
    
    test_names = [
        "データベース初期化",
        "ユニットテスト",
        "戦略保存機能",
        "サーバー状態確認",
        "APIエンドポイント",
        "バックテスト統合"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{i+1:2d}. {name:<20}: {status}")
    
    success_count = sum(test_results)
    total_count = len(test_results)
    
    print(f"\n📈 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 すべてのテストが成功しました！")
        print("✨ ストラテジービルダー機能の実装が完了しました")
        print("\n📋 実装された機能:")
        print("   • 58種類のテクニカル指標の管理")
        print("   • ユーザー定義戦略の作成・保存・管理")
        print("   • 戦略設定の検証とエラーハンドリング")
        print("   • StrategyGene形式への変換")
        print("   • バックテストシステムとの統合")
        print("   • RESTful APIエンドポイント")
        print("   • 包括的なユニットテスト")
        
        print("\n🔗 利用可能なAPIエンドポイント:")
        print("   • GET  /api/strategy-builder/indicators")
        print("   • POST /api/strategy-builder/validate")
        print("   • POST /api/strategy-builder/save")
        print("   • GET  /api/strategy-builder/strategies")
        print("   • GET  /api/strategy-builder/strategies/{id}")
        print("   • PUT  /api/strategy-builder/strategies/{id}")
        print("   • DELETE /api/strategy-builder/strategies/{id}")
        
        return True
    else:
        print(f"\n⚠️ {total_count - success_count}個のテストが失敗しました")
        print("詳細は上記のテスト結果を確認してください")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
