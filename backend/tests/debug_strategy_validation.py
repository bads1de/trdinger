#!/usr/bin/env python3
"""
戦略検証デバッグスクリプト

StrategyBuilderServiceの検証プロセスを詳細にデバッグします。
"""

from database.connection import get_db
from app.core.services.strategy_builder_service import StrategyBuilderService
from app.core.services.auto_strategy.models.strategy_gene import Condition, StrategyGene, IndicatorGene
from app.core.services.auto_strategy.models.gene_validation import GeneValidator

def debug_strategy_validation():
    """戦略検証のデバッグ"""
    
    print("=== 戦略検証デバッグ ===")
    
    # データベースセッションを取得
    db = next(get_db())
    
    try:
        # サービスを作成
        service = StrategyBuilderService(db)
        
        # テストデータ（フロントエンドから送信される形式）
        strategy_config = {
            "indicators": [
                {
                    "type": "SMA",
                    "parameters": {"period": 20},
                    "enabled": True,
                    "json_config": {
                        "indicator_name": "SMA",
                        "parameters": {"period": 20}
                    }
                },
                {
                    "type": "RSI",
                    "parameters": {"period": 14},
                    "enabled": True,
                    "json_config": {
                        "indicator_name": "RSI",
                        "parameters": {"period": 14}
                    }
                }
            ],
            "entry_conditions": [
                {
                    "id": "condition_1",
                    "type": "threshold",
                    "indicator1": "RSI",
                    "operator": "<",
                    "value": 30,
                    "logicalOperator": "AND"
                }
            ],
            "exit_conditions": [
                {
                    "id": "condition_3",
                    "type": "threshold",
                    "indicator1": "RSI",
                    "operator": ">",
                    "value": 70,
                    "logicalOperator": "OR"
                }
            ]
        }
        
        print("テストデータ:")
        print(f"indicators: {len(strategy_config['indicators'])} 個")
        print(f"entry_conditions: {len(strategy_config['entry_conditions'])} 個")
        print(f"exit_conditions: {len(strategy_config['exit_conditions'])} 個")
        print()
        
        # 戦略設定の検証
        print("--- 戦略設定検証 ---")
        is_valid, errors = service.validate_strategy_config(strategy_config)
        
        print(f"検証結果: {is_valid}")
        if errors:
            print("エラー:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("エラーなし")
        print()
        
        # 手動で条件を作成してテスト
        print("--- 手動条件作成テスト ---")
        
        # フロントエンドの条件を手動で変換
        entry_condition = strategy_config["entry_conditions"][0]
        print(f"フロントエンド条件: {entry_condition}")
        
        # 手動変換
        condition_obj = Condition(
            left_operand=entry_condition["indicator1"],  # "RSI"
            operator=entry_condition["operator"],        # "<"
            right_operand=entry_condition["value"]       # 30
        )
        
        print(f"変換後条件: left_operand={condition_obj.left_operand}, operator={condition_obj.operator}, right_operand={condition_obj.right_operand}")
        
        # バリデーターで検証
        validator = GeneValidator()
        is_condition_valid = validator.validate_condition(condition_obj)
        print(f"条件検証結果: {is_condition_valid}")
        
        if not is_condition_valid:
            print("詳細チェック:")
            print(f"  - 演算子有効性: {condition_obj.operator in validator.valid_operators}")
            print(f"  - 左オペランド有効性: {validator._is_valid_operand(condition_obj.left_operand)}")
            print(f"  - 右オペランド有効性: {validator._is_valid_operand(condition_obj.right_operand)}")
            
            if isinstance(condition_obj.left_operand, str):
                print(f"  - 左オペランド指標名チェック: {validator._is_indicator_name(condition_obj.left_operand)}")
                print(f"  - 左オペランドデータソースチェック: {condition_obj.left_operand in validator.valid_data_sources}")
        
        print()
        
        # StrategyGeneを手動で作成してテスト
        print("--- StrategyGene手動作成テスト ---")
        
        indicators = [
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True,
                json_config={"indicator_name": "SMA", "parameters": {"period": 20}}
            ),
            IndicatorGene(
                type="RSI",
                parameters={"period": 14},
                enabled=True,
                json_config={"indicator_name": "RSI", "parameters": {"period": 14}}
            )
        ]
        
        entry_conditions = [condition_obj]
        
        exit_condition_data = strategy_config["exit_conditions"][0]
        exit_condition_obj = Condition(
            left_operand=exit_condition_data["indicator1"],
            operator=exit_condition_data["operator"],
            right_operand=exit_condition_data["value"]
        )
        exit_conditions = [exit_condition_obj]
        
        strategy_gene = StrategyGene(
            id="test_strategy",
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions
        )
        
        # 検証実行
        is_gene_valid, gene_errors = strategy_gene.validate()
        
        print(f"StrategyGene検証結果: {is_gene_valid}")
        if gene_errors:
            print("エラー:")
            for error in gene_errors:
                print(f"  - {error}")
        else:
            print("エラーなし")
            
    finally:
        db.close()

if __name__ == "__main__":
    debug_strategy_validation()
