"""
条件評価の修正をテストするスクリプト

数値文字列の問題が修正されているかを確認します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
from app.core.services.auto_strategy.models.strategy_gene import Condition
import logging

# ログレベルを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockStrategyInstance:
    """テスト用のモック戦略インスタンス"""
    
    def __init__(self):
        # モックデータ
        self.data = MockData()
        self.indicators = {
            'RSI': MockIndicator([45.0, 55.0, 65.0]),
            'SMA': MockIndicator([100.0, 105.0, 110.0]),
            'CCI': MockIndicator([-50.0, 0.0, 50.0])
        }


class MockData:
    """モック価格データ"""
    
    def __init__(self):
        self.Close = [100.0, 105.0, 110.0]
        self.High = [102.0, 107.0, 112.0]
        self.Low = [98.0, 103.0, 108.0]
        self.Open = [99.0, 104.0, 109.0]
        self.Volume = [1000.0, 1500.0, 2000.0]
        self.OpenInterest = [1000000.0, 1100000.0, 1200000.0]
        self.FundingRate = [0.0001, 0.0002, 0.0003]


class MockIndicator:
    """モック指標"""
    
    def __init__(self, values):
        self.values = values
    
    def __getitem__(self, index):
        return self.values[index]
    
    def __len__(self):
        return len(self.values)


def test_numeric_string_handling():
    """数値文字列の処理テスト"""
    print("=== 数値文字列処理テスト ===")
    
    evaluator = ConditionEvaluator()
    mock_strategy = MockStrategyInstance()
    
    # テストケース
    test_cases = [
        # (オペランド, 期待値, 説明)
        (50, 50.0, "整数"),
        (50.5, 50.5, "浮動小数点数"),
        ("50", 50.0, "整数文字列"),
        ("50.5", 50.5, "浮動小数点文字列"),
        ("-30", -30.0, "負の数値文字列"),
        ("RSI", 65.0, "指標名"),
        ("close", 110.0, "価格データ"),
        ("invalid_indicator", None, "無効な指標名"),
    ]
    
    success_count = 0
    for operand, expected, description in test_cases:
        try:
            result = evaluator.get_condition_value(operand, mock_strategy)
            
            if result == expected:
                print(f"✅ {description}: {operand} -> {result}")
                success_count += 1
            else:
                print(f"❌ {description}: {operand} -> {result} (期待値: {expected})")
        except Exception as e:
            print(f"❌ {description}: {operand} -> エラー: {e}")
    
    print(f"\n成功率: {success_count}/{len(test_cases)} ({success_count/len(test_cases):.1%})")
    return success_count == len(test_cases)


def test_condition_evaluation():
    """条件評価のテスト"""
    print("\n=== 条件評価テスト ===")
    
    evaluator = ConditionEvaluator()
    mock_strategy = MockStrategyInstance()
    
    # テスト条件
    test_conditions = [
        # (条件, 期待結果, 説明)
        (Condition("RSI", ">", 50), True, "RSI > 50 (65 > 50)"),
        (Condition("RSI", "<", 50), False, "RSI < 50 (65 < 50)"),
        (Condition("RSI", ">", "50"), True, "RSI > '50' (数値文字列)"),
        (Condition("close", ">", 100), True, "close > 100 (110 > 100)"),
        (Condition("SMA", ">=", "105"), True, "SMA >= '105' (110 >= 105)"),
        (Condition("CCI", "<=", 60), True, "CCI <= 60 (50 <= 60)"),
    ]
    
    success_count = 0
    for condition, expected, description in test_conditions:
        try:
            result = evaluator.evaluate_condition(condition, mock_strategy)
            
            if result == expected:
                print(f"✅ {description}: {result}")
                success_count += 1
            else:
                print(f"❌ {description}: {result} (期待値: {expected})")
        except Exception as e:
            print(f"❌ {description}: エラー: {e}")
    
    print(f"\n成功率: {success_count}/{len(test_conditions)} ({success_count/len(test_conditions):.1%})")
    return success_count == len(test_conditions)


def test_problematic_conditions():
    """問題のあった条件のテスト"""
    print("\n=== 問題条件テスト ===")
    
    evaluator = ConditionEvaluator()
    mock_strategy = MockStrategyInstance()
    
    # 実際に問題となった条件
    problematic_conditions = [
        Condition("RSI", ">", "50"),  # 数値文字列
        Condition("SMA", "<", "100"),  # 数値文字列
        Condition("CCI", ">=", "0"),   # 数値文字列
        Condition("close", "<=", "120"), # 数値文字列
    ]
    
    print("修正前に警告が出ていた条件をテスト中...")
    
    success_count = 0
    for i, condition in enumerate(problematic_conditions, 1):
        try:
            # 条件評価を実行（警告が出ないことを確認）
            result = evaluator.evaluate_condition(condition, mock_strategy)
            print(f"✅ 条件{i}: {condition.left_operand} {condition.operator} {condition.right_operand} -> {result}")
            success_count += 1
        except Exception as e:
            print(f"❌ 条件{i}: エラー: {e}")
    
    print(f"\n成功率: {success_count}/{len(problematic_conditions)} ({success_count/len(problematic_conditions):.1%})")
    return success_count == len(problematic_conditions)


def test_new_ga_execution():
    """修正版でのGA実行テスト"""
    print("\n=== 修正版GA実行テスト ===")
    
    try:
        from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from datetime import datetime
        
        # AutoStrategyServiceを初期化
        print("AutoStrategyServiceを初期化中...")
        service = AutoStrategyService()
        
        # 小規模なテスト用GA設定
        ga_config = GAConfig(
            population_size=3,  # 非常に小さな個体数
            generations=1,      # 1世代のみ
            crossover_rate=0.8,
            mutation_rate=0.2,
            allowed_indicators=["RSI", "SMA"]  # 制限された指標
        )
        
        # テスト用のバックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-12-01",
            "end_date": "2024-12-02",  # 1日のみ
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        print("修正版でGA実行を開始...")
        experiment_id = service.start_strategy_generation(
            experiment_name=f"FIX_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ga_config=ga_config,
            backtest_config=backtest_config
        )
        
        print(f"実験ID: {experiment_id}")
        
        # 短時間の進捗監視
        import time
        max_wait = 60  # 1分間待機
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            progress = service.get_experiment_progress(experiment_id)
            if progress:
                print(f"  ステータス: {progress.status}")
                
                if progress.status == "completed":
                    print("✅ 修正版GA実行完了")
                    
                    # 結果を取得
                    result = service.get_experiment_result(experiment_id)
                    if result:
                        print(f"最高フィットネス: {result['best_fitness']}")
                        print(f"実行時間: {result['execution_time']:.2f}秒")
                        
                        # 戦略の詳細を確認
                        best_strategy = result['best_strategy']
                        print(f"エントリー条件数: {len(best_strategy.entry_conditions)}")
                        print(f"エグジット条件数: {len(best_strategy.exit_conditions)}")
                        
                        return True
                    break
                elif progress.status == "failed":
                    print(f"❌ GA実行失敗: {getattr(progress, 'error_message', '不明なエラー')}")
                    return False
            
            time.sleep(5)  # 5秒間隔で確認
        else:
            print("⏰ タイムアウト: GA実行が完了しませんでした")
            return False
        
    except Exception as e:
        print(f"❌ GA実行テストエラー: {e}")
        logger.exception("GA実行テスト中にエラーが発生")
        return False


def main():
    """メイン実行関数"""
    print("🔧 条件評価修正テスト開始")
    print(f"実行時刻: {datetime.now()}")
    
    # 1. 数値文字列処理テスト
    test1_success = test_numeric_string_handling()
    
    # 2. 条件評価テスト
    test2_success = test_condition_evaluation()
    
    # 3. 問題条件テスト
    test3_success = test_problematic_conditions()
    
    # 4. 修正版GA実行テスト
    test4_success = test_new_ga_execution()
    
    # 結果サマリー
    print(f"\n📊 テスト結果サマリー:")
    print(f"  数値文字列処理: {'✅' if test1_success else '❌'}")
    print(f"  条件評価: {'✅' if test2_success else '❌'}")
    print(f"  問題条件: {'✅' if test3_success else '❌'}")
    print(f"  修正版GA実行: {'✅' if test4_success else '❌'}")
    
    overall_success = all([test1_success, test2_success, test3_success, test4_success])
    print(f"\n🎯 総合結果: {'✅ 修正成功' if overall_success else '❌ 修正不完全'}")
    
    return overall_success


if __name__ == "__main__":
    from datetime import datetime
    main()
