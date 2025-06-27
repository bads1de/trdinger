#!/usr/bin/env python3
"""
戦略保存機能のテストスクリプト
"""

from database.connection import get_db
from app.core.services.strategy_builder_service import StrategyBuilderService


def test_strategy_save():
    # データベースセッションを取得
    db = next(get_db())

    try:
        service = StrategyBuilderService(db)

        # テスト用の戦略設定
        strategy_config = {
            "indicators": [
                {
                    "type": "SMA",
                    "parameters": {"period": 20},
                    "enabled": True,
                    "json_config": {
                        "indicator_name": "SMA",
                        "parameters": {"period": 20},
                    },
                }
            ],
            "entry_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": ">", "value": 100}
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": "<", "value": 95}
            ],
        }

        print("=== 戦略設定の検証 ===")
        is_valid, errors = service.validate_strategy_config(strategy_config)
        print(f"検証結果: {is_valid}")
        if not is_valid:
            print(f"エラー: {errors}")
            return False

        print("=== 戦略の保存 ===")
        user_strategy = service.save_strategy(
            name="テスト戦略",
            description="統合テスト用の戦略",
            strategy_config=strategy_config,
        )

        if user_strategy:
            print(f"戦略保存成功: ID={user_strategy.id}")
            print(f"  名前: {user_strategy.name}")
            print(f"  説明: {user_strategy.description}")
            print(f"  作成日時: {user_strategy.created_at}")

            print("=== 保存済み戦略の取得 ===")
            strategies = service.get_strategies()
            print(f"保存済み戦略数: {len(strategies)}")
            for strategy in strategies:
                print(f"  - ID={strategy.id}, 名前={strategy.name}")

            return True
        else:
            print("戦略保存失敗")
            return False

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = test_strategy_save()
    if success:
        print("\n戦略保存機能のテストが成功しました")
    else:
        print("\n戦略保存機能のテストが失敗しました")
