"""
最終修正版のテストスクリプト

スケール不一致問題を解決した修正版をテストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
from app.core.services.auto_strategy.utils.operand_grouping import operand_grouping_system
from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from datetime import datetime
import time
import logging

# ログレベルを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_improved_condition_generation():
    """改良された条件生成のテスト"""
    print("=== 改良された条件生成テスト ===")
    
    # テスト用の設定
    config = {
        "min_indicators": 2,
        "max_indicators": 3,
        "min_conditions": 1,
        "max_conditions": 2
    }
    
    generator = RandomGeneGenerator(config)
    
    # テスト用の指標
    test_indicators = [
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="CCI", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="MACD", parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}, enabled=True),
    ]
    
    print("修正版で条件生成をテスト中...")
    
    # 統計を収集
    numeric_comparisons = 0
    indicator_comparisons = 0
    scale_mismatches = 0
    total_conditions = 0
    
    condition_patterns = {}
    
    for i in range(50):  # より多くのサンプルでテスト
        condition = generator._generate_single_condition(test_indicators, "entry")
        total_conditions += 1
        
        left = condition.left_operand
        op = condition.operator
        right = condition.right_operand
        
        pattern = f"{left} {op} {type(right).__name__}"
        condition_patterns[pattern] = condition_patterns.get(pattern, 0) + 1
        
        if isinstance(right, (int, float)):
            numeric_comparisons += 1
            print(f"  {i+1}. {left} {op} {right} (数値)")
        else:
            indicator_comparisons += 1
            
            # 互換性をチェック
            compatibility = operand_grouping_system.get_compatibility_score(left, right)
            
            if compatibility < 0.8:
                scale_mismatches += 1
                print(f"  {i+1}. {left} {op} {right} (互換性: {compatibility:.2f}) ⚠️")
            else:
                print(f"  {i+1}. {left} {op} {right} (互換性: {compatibility:.2f}) ✅")
    
    print(f"\n📊 改良版テスト結果:")
    print(f"  総条件数: {total_conditions}")
    print(f"  数値比較: {numeric_comparisons} ({numeric_comparisons/total_conditions:.1%})")
    print(f"  指標比較: {indicator_comparisons} ({indicator_comparisons/total_conditions:.1%})")
    print(f"  スケール不一致: {scale_mismatches} ({scale_mismatches/total_conditions:.1%})")
    
    print(f"\n📋 条件パターン:")
    for pattern, count in sorted(condition_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}回")
    
    # 成功判定
    success = (
        numeric_comparisons / total_conditions >= 0.7 and  # 70%以上が数値比較
        scale_mismatches / total_conditions <= 0.1  # スケール不一致が10%以下
    )
    
    if success:
        print(f"  ✅ 改良版が正常に動作しています")
    else:
        print(f"  ❌ 改良版に問題があります")
    
    return success


def test_final_auto_strategy():
    """最終修正版でのオートストラテジー実行テスト"""
    print(f"\n=== 最終修正版オートストラテジー実行テスト ===")
    
    try:
        # AutoStrategyServiceを初期化
        print("AutoStrategyServiceを初期化中...")
        service = AutoStrategyService()
        
        # 小規模だが実用的なGA設定
        ga_config = GAConfig(
            population_size=3,   # 小さな個体数
            generations=2,       # 少ない世代数
            crossover_rate=0.8,
            mutation_rate=0.2,
            allowed_indicators=["RSI", "SMA", "CCI", "MACD"]  # 基本的な指標
        )
        
        # 適切な期間のバックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-10-01",  # 1ヶ月間
            "end_date": "2024-10-31",
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        print("最終修正版GA実行を開始...")
        print(f"期間: {backtest_config['start_date']} - {backtest_config['end_date']}")
        
        experiment_id = service.start_strategy_generation(
            experiment_name=f"FINAL_FIX_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ga_config=ga_config,
            backtest_config=backtest_config
        )
        
        print(f"実験ID: {experiment_id}")
        
        # 進捗監視
        print("進捗監視中...")
        max_wait = 180  # 3分間待機
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            progress = service.get_experiment_progress(experiment_id)
            if progress:
                if progress.status == "completed":
                    print("✅ 最終修正版GA実行完了")
                    
                    # 結果を取得
                    result = service.get_experiment_result(experiment_id)
                    if result:
                        print(f"\n📊 実行結果:")
                        print(f"  最高フィットネス: {result['best_fitness']:.4f}")
                        print(f"  実行時間: {result['execution_time']:.2f}秒")
                        
                        # 戦略の詳細を確認
                        best_strategy = result['best_strategy']
                        print(f"\n🏆 最優秀戦略:")
                        print(f"  指標数: {len(best_strategy.indicators)}")
                        print(f"  エントリー条件数: {len(best_strategy.entry_conditions)}")
                        print(f"  エグジット条件数: {len(best_strategy.exit_conditions)}")
                        
                        # 条件の詳細分析
                        print(f"\n📈 エントリー条件分析:")
                        for i, condition in enumerate(best_strategy.entry_conditions, 1):
                            left = condition.left_operand
                            op = condition.operator
                            right = condition.right_operand
                            
                            if isinstance(right, (int, float)):
                                print(f"    {i}. {left} {op} {right} (数値比較)")
                            else:
                                compatibility = operand_grouping_system.get_compatibility_score(left, right)
                                print(f"    {i}. {left} {op} {right} (互換性: {compatibility:.2f})")
                        
                        # バックテスト結果の確認
                        check_final_backtest_results(experiment_id)
                        
                        return True
                    break
                elif progress.status == "failed":
                    error_msg = getattr(progress, 'error_message', '不明なエラー')
                    print(f"❌ GA実行失敗: {error_msg}")
                    return False
            
            time.sleep(10)  # 10秒間隔で確認
        else:
            print("⏰ タイムアウト: GA実行が完了しませんでした")
            return False
        
    except Exception as e:
        print(f"❌ 最終修正版GA実行エラー: {e}")
        logger.exception("最終修正版GA実行中にエラーが発生")
        return False


def check_final_backtest_results(experiment_id):
    """最終バックテスト結果の確認"""
    print(f"\n📊 最終バックテスト結果確認:")
    
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect("trdinger.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 最新の結果を取得
        cursor.execute("""
            SELECT strategy_name, performance_metrics, created_at
            FROM backtest_results 
            WHERE strategy_name LIKE '%FINAL_FIX%'
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        
        if result:
            print(f"  戦略名: {result['strategy_name']}")
            print(f"  作成日時: {result['created_at']}")
            
            if result['performance_metrics']:
                metrics = json.loads(result['performance_metrics'])
                total_trades = metrics.get('total_trades', 0)
                total_return = metrics.get('total_return', 0)
                win_rate = metrics.get('win_rate', 0)
                
                print(f"  📈 パフォーマンス:")
                print(f"    総取引数: {total_trades}")
                print(f"    総リターン: {total_return:.4f} ({total_return*100:.2f}%)")
                print(f"    勝率: {win_rate:.4f}" if win_rate and str(win_rate) != 'nan' else "    勝率: N/A")
                
                if total_trades > 0:
                    print(f"    🎉 最終修正版で取引が発生しました！")
                    print(f"    🎯 取引回数0問題が解決されました！")
                else:
                    print(f"    ❌ 最終修正版でも取引回数0")
        else:
            print(f"  最終修正版の結果が見つかりませんでした")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 最終バックテスト結果確認エラー: {e}")


def main():
    """メイン実行関数"""
    print("🔧 最終修正版テスト開始")
    print(f"実行時刻: {datetime.now()}")
    
    # 1. 改良された条件生成のテスト
    test1_success = test_improved_condition_generation()
    
    # 2. 最終修正版オートストラテジー実行テスト
    test2_success = test_final_auto_strategy()
    
    # 結果サマリー
    print(f"\n📊 最終テスト結果サマリー:")
    print(f"  条件生成改良: {'✅' if test1_success else '❌'}")
    print(f"  最終GA実行: {'✅' if test2_success else '❌'}")
    
    overall_success = test1_success and test2_success
    print(f"\n🎯 最終結果: {'✅ 取引回数0問題解決' if overall_success else '❌ 問題未解決'}")
    
    return overall_success


if __name__ == "__main__":
    main()
