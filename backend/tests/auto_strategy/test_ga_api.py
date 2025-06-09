#!/usr/bin/env python3
"""
GA機能のAPI統合テスト

FastAPIアプリケーションのGA関連エンドポイントをテストします。
"""

import sys
import traceback
import logging
from fastapi.testclient import TestClient

# ログレベルを設定
logging.basicConfig(level=logging.INFO)


def test_ga_functionality():
    """GA機能の動作をテスト"""

    print("=== GA機能動作テスト開始 ===")

    try:
        # アプリケーションのインポート
        print("1. アプリケーションをインポート中...")
        from app.main import app

        client = TestClient(app)
        print("✅ アプリケーション準備完了")

        # 戦略テスト
        print("\n2. 単一戦略テスト...")
        test_strategy_data = {
            "strategy_gene": {
                "id": "test_strategy_001",
                "indicators": [
                    {"type": "SMA", "parameters": {"period": 20}, "enabled": True},
                    {"type": "RSI", "parameters": {"period": 14}, "enabled": True},
                ],
                "entry_conditions": [
                    {"left_operand": "RSI_14", "operator": "<", "right_operand": 30}
                ],
                "exit_conditions": [
                    {"left_operand": "RSI_14", "operator": ">", "right_operand": 70}
                ],
                "risk_management": {"stop_loss": 0.02, "take_profit": 0.05},
                "metadata": {},
            },
            "backtest_config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.001,
            },
        }

        response = client.post(
            "/api/auto-strategy/test-strategy", json=test_strategy_data
        )
        print(f"✅ 戦略テスト: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   成功: {data.get('success', False)}")
            if data.get("result"):
                print("   戦略テスト結果あり")
        else:
            print(f"   エラー: {response.text}")

        # 小規模GA実行テスト
        print("\n3. 小規模GA実行テスト...")
        ga_config = {
            "experiment_name": "Test_GA_Experiment",
            "base_config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.001,
            },
            "ga_config": {
                "population_size": 3,  # 非常に小さい個体数
                "generations": 2,  # 非常に少ない世代数
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "elite_size": 1,
                "max_indicators": 2,  # 指標数を制限
                "allowed_indicators": ["SMA", "RSI"],  # 指標を制限
                "fitness_weights": {
                    "total_return": 0.3,
                    "sharpe_ratio": 0.4,
                    "max_drawdown": 0.2,
                    "win_rate": 0.1,
                },
                "fitness_constraints": {
                    "min_trades": 1,
                    "max_drawdown_limit": 0.9,
                    "min_sharpe_ratio": -10.0,
                },
            },
        }

        response = client.post("/api/auto-strategy/generate", json=ga_config)
        print(f"✅ GA生成開始: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   成功: {data.get('success', False)}")
            experiment_id = data.get("experiment_id")
            print(f"   実験ID: {experiment_id}")

            if experiment_id:
                # 進捗確認（短時間）
                print("\n4. 進捗確認テスト...")
                import time

                for i in range(5):  # 最大5回チェック
                    time.sleep(2)  # 2秒待機

                    response = client.get(
                        f"/api/auto-strategy/experiments/{experiment_id}/progress"
                    )
                    if response.status_code == 200:
                        progress_data = response.json()
                        if progress_data.get("success") and progress_data.get(
                            "progress"
                        ):
                            progress = progress_data["progress"]
                            status = progress.get("status", "unknown")
                            current_gen = progress.get("current_generation", 0)
                            total_gen = progress.get("total_generations", 0)
                            best_fitness = progress.get("best_fitness", 0)

                            print(
                                f"   進捗 {i+1}: 世代{current_gen}/{total_gen}, 状態:{status}, フィットネス:{best_fitness:.4f}"
                            )

                            if status in ["completed", "error"]:
                                print(f"   実験終了: {status}")

                                if status == "completed":
                                    # 結果取得
                                    print("\n5. 結果取得テスト...")
                                    response = client.get(
                                        f"/api/auto-strategy/experiments/{experiment_id}/results"
                                    )
                                    if response.status_code == 200:
                                        result_data = response.json()
                                        print(
                                            f"   結果取得成功: {result_data.get('success', False)}"
                                        )
                                        if result_data.get("result"):
                                            result = result_data["result"]
                                            print(
                                                f"   最高フィットネス: {result.get('best_fitness', 0):.4f}"
                                            )
                                            print(
                                                f"   実行時間: {result.get('execution_time', 0):.1f}秒"
                                            )
                                    else:
                                        print(f"   結果取得エラー: {response.text}")
                                break
                        else:
                            print(f"   進捗データなし: {progress_data}")
                    else:
                        print(f"   進捗確認エラー: {response.status_code}")

                # 実験一覧確認
                print("\n6. 実験一覧確認...")
                response = client.get("/api/auto-strategy/experiments")
                if response.status_code == 200:
                    experiments = response.json()
                    print(f"   実験数: {len(experiments)}")
                    for exp in experiments[-3:]:  # 最新3件
                        print(
                            f"   - ID: {exp.get('id', 'N/A')}, 名前: {exp.get('name', 'N/A')}, 状態: {exp.get('status', 'N/A')}"
                        )
                else:
                    print(f"   実験一覧取得エラー: {response.status_code}")
        else:
            print(f"   エラー: {response.text}")

        print("\n=== GA機能動作テスト完了 ===")
        return True

    except Exception as e:
        print(f"❌ GA機能テストエラー: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ga_functionality()
    if not success:
        sys.exit(1)
    print("✅ GA機能テスト成功")
