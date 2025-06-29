"""
実行されたオートストラテジーの詳細を調査するスクリプト

取引回数0問題の原因を特定するため、実際に生成された戦略の詳細を確認します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import GeneratedStrategyRepository
from database.repositories.backtest_result_repository import BacktestResultRepository
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
from app.core.services.auto_strategy.utils.operand_grouping import operand_grouping_system
import json
from datetime import datetime, timedelta


def find_recent_strategy():
    """最近実行された戦略を検索"""
    print("=== 最近実行された戦略の検索 ===")
    
    db = SessionLocal()
    try:
        exp_repo = GAExperimentRepository(db)
        strategy_repo = GeneratedStrategyRepository(db)
        
        # 最近の実験を取得
        recent_experiments = exp_repo.get_recent_experiments(limit=10)
        print(f"最近の実験数: {len(recent_experiments)}")
        
        for exp in recent_experiments:
            print(f"\n実験: {exp.name}")
            print(f"  ID: {exp.id}")
            print(f"  ステータス: {exp.status}")
            print(f"  作成日時: {exp.created_at}")
            print(f"  完了日時: {exp.completed_at}")
            
            # この実験の戦略を取得
            strategies = strategy_repo.get_strategies_by_experiment(exp.id)
            print(f"  生成戦略数: {len(strategies)}")
            
            if strategies:
                # 最高フィットネスの戦略を詳細分析
                best_strategy = max(strategies, key=lambda s: s.fitness_score or 0)
                print(f"  最高フィットネス: {best_strategy.fitness_score}")
                
                # 戦略の詳細分析
                if "AUTO_STRATEGY_GA_2025-06-29_BTC_USDT" in exp.name:
                    print(f"\n🎯 対象戦略を発見: {exp.name}")
                    analyze_strategy_details(best_strategy)
                    return best_strategy
        
        return None
        
    finally:
        db.close()


def analyze_strategy_details(strategy_record):
    """戦略の詳細分析"""
    print(f"\n=== 戦略詳細分析 ===")
    print(f"戦略ID: {strategy_record.id}")
    print(f"フィットネス: {strategy_record.fitness_score}")
    print(f"世代: {strategy_record.generation}")
    
    # 遺伝子データを解析
    gene_data = strategy_record.gene_data
    if not gene_data:
        print("❌ 遺伝子データが見つかりません")
        return
    
    try:
        # StrategyGeneオブジェクトに復元
        strategy_gene = StrategyGene.from_dict(gene_data)
        
        print(f"\n📊 戦略構成:")
        print(f"  指標数: {len(strategy_gene.indicators)}")
        print(f"  エントリー条件数: {len(strategy_gene.entry_conditions)}")
        print(f"  エグジット条件数: {len(strategy_gene.exit_conditions)}")
        
        # 指標の詳細
        print(f"\n🔧 使用指標:")
        for i, indicator in enumerate(strategy_gene.indicators, 1):
            print(f"  {i}. {indicator.type} - {indicator.parameters}")
        
        # エントリー条件の詳細分析
        print(f"\n📈 エントリー条件:")
        analyze_conditions(strategy_gene.entry_conditions, "エントリー")
        
        # エグジット条件の詳細分析
        print(f"\n📉 エグジット条件:")
        analyze_conditions(strategy_gene.exit_conditions, "エグジット")
        
        # バックテスト結果の確認
        check_backtest_results(strategy_record.id)
        
    except Exception as e:
        print(f"❌ 戦略分析エラー: {e}")
        print(f"生データ: {json.dumps(gene_data, indent=2, ensure_ascii=False)}")


def analyze_conditions(conditions, condition_type):
    """条件の詳細分析"""
    if not conditions:
        print(f"  {condition_type}条件がありません")
        return
    
    for i, condition in enumerate(conditions, 1):
        print(f"  {i}. {condition.left_operand} {condition.operator} {condition.right_operand}")
        
        # 条件の妥当性をチェック
        if isinstance(condition.right_operand, str):
            # 指標同士の比較の場合
            compatibility = operand_grouping_system.get_compatibility_score(
                condition.left_operand, condition.right_operand
            )
            
            left_group = operand_grouping_system.get_operand_group(condition.left_operand)
            right_group = operand_grouping_system.get_operand_group(condition.right_operand)
            
            print(f"     左: {condition.left_operand} ({left_group.value})")
            print(f"     右: {condition.right_operand} ({right_group.value})")
            print(f"     互換性スコア: {compatibility:.2f}")
            
            if compatibility <= 0.3:
                print(f"     ⚠️ スケール不一致の可能性")
            elif compatibility >= 0.8:
                print(f"     ✅ 高い互換性")
            else:
                print(f"     🔶 中程度の互換性")
        else:
            # 数値との比較の場合
            print(f"     数値比較: {condition.right_operand}")
            
            # 数値の妥当性をチェック
            left_group = operand_grouping_system.get_operand_group(condition.left_operand)
            if left_group.value == "percentage_0_100":
                if not (0 <= condition.right_operand <= 100):
                    print(f"     ⚠️ 0-100%指標に対する範囲外の値")
            elif "FundingRate" in condition.left_operand:
                if not (-0.01 <= condition.right_operand <= 0.01):
                    print(f"     ⚠️ FundingRateに対する非現実的な値")


def check_backtest_results(strategy_id):
    """バックテスト結果の確認"""
    print(f"\n📊 バックテスト結果確認:")
    
    db = SessionLocal()
    try:
        backtest_repo = BacktestResultRepository(db)
        
        # この戦略に関連するバックテスト結果を検索
        # 注意: strategy_idとbacktest結果の関連付け方法を確認する必要があります
        print(f"  戦略ID {strategy_id} のバックテスト結果を検索中...")
        
        # 最近のバックテスト結果を取得
        recent_results = backtest_repo.get_recent_results(limit=5)
        print(f"  最近のバックテスト結果数: {len(recent_results)}")
        
        for result in recent_results:
            if result.strategy_name and "AUTO_STRATEGY_GA" in result.strategy_name:
                print(f"\n  📈 関連結果: {result.strategy_name}")
                print(f"    実行日時: {result.created_at}")
                
                # パフォーマンス指標を確認
                if result.performance_metrics:
                    metrics = result.performance_metrics
                    print(f"    総取引数: {metrics.get('total_trades', 'N/A')}")
                    print(f"    総リターン: {metrics.get('total_return', 'N/A')}")
                    print(f"    勝率: {metrics.get('win_rate', 'N/A')}")
                    print(f"    最大ドローダウン: {metrics.get('max_drawdown', 'N/A')}")
                    
                    # 取引回数0の問題を確認
                    total_trades = metrics.get('total_trades', 0)
                    if total_trades == 0:
                        print(f"    ❌ 取引回数0問題を確認")
                    else:
                        print(f"    ✅ 取引が実行されています")
        
    finally:
        db.close()


def test_condition_generation():
    """修正されたコードで条件生成をテスト"""
    print(f"\n=== 修正版条件生成テスト ===")
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene
        
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
        ]
        
        print("修正版で条件生成をテスト中...")
        
        # 複数回条件生成をテスト
        scale_mismatches = 0
        numerical_conditions = 0
        total_conditions = 0
        
        for i in range(20):
            condition = generator._generate_single_condition(test_indicators, "entry")
            total_conditions += 1
            
            print(f"  {i+1}. {condition.left_operand} {condition.operator} {condition.right_operand}")
            
            if isinstance(condition.right_operand, (int, float)):
                numerical_conditions += 1
                print(f"     → 数値比較")
            else:
                compatibility = operand_grouping_system.get_compatibility_score(
                    condition.left_operand, condition.right_operand
                )
                print(f"     → 互換性スコア: {compatibility:.2f}")
                
                if compatibility <= 0.3:
                    scale_mismatches += 1
                    print(f"     → ⚠️ スケール不一致")
        
        print(f"\n📊 テスト結果:")
        print(f"  総条件数: {total_conditions}")
        print(f"  数値比較: {numerical_conditions} ({numerical_conditions/total_conditions:.1%})")
        print(f"  スケール不一致: {scale_mismatches} ({scale_mismatches/total_conditions:.1%})")
        
        if scale_mismatches / total_conditions <= 0.25:
            print(f"  ✅ 修正版が正常に動作しています")
        else:
            print(f"  ❌ 修正版が適用されていない可能性があります")
            
    except Exception as e:
        print(f"❌ 条件生成テストエラー: {e}")


def main():
    """メイン実行関数"""
    print("🔍 オートストラテジー詳細調査開始")
    print(f"実行時刻: {datetime.now()}")
    
    # 1. 最近の戦略を検索
    strategy = find_recent_strategy()
    
    # 2. 修正版の動作確認
    test_condition_generation()
    
    print(f"\n🔍 調査完了")


if __name__ == "__main__":
    main()
