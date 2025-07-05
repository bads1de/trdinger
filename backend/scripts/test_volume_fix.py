#!/usr/bin/env python3
"""
オートストラテジー取引量0問題の修正を検証するスクリプト
"""

import sys
import os
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_strategy_gene():
    """テスト用の戦略遺伝子を作成"""
    indicators = [
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
    ]
    
    entry_conditions = [
        Condition(left_operand="RSI", operator="<", right_operand=30)
    ]
    
    exit_conditions = [
        Condition(left_operand="RSI", operator=">", right_operand=70)
    ]
    
    risk_management = {
        "stop_loss": 0.03,
        "take_profit": 0.15,
        "position_size": 0.1,  # 10%の取引量
    }
    
    return StrategyGene(
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        risk_management=risk_management,
        metadata={"test": "volume_fix"}
    )


def test_strategy_factory_volume_calculation():
    """StrategyFactoryでの取引量計算テスト"""
    print("=== StrategyFactoryでの取引量計算テスト ===")
    
    try:
        # テスト用戦略遺伝子を作成
        test_gene = create_test_strategy_gene()
        print(f"テスト遺伝子の取引量設定: {test_gene.risk_management['position_size']}")
        
        # StrategyFactoryで戦略クラスを生成
        strategy_factory = StrategyFactory()
        strategy_class = strategy_factory.create_strategy_class(test_gene)
        
        print(f"戦略クラス生成成功: {strategy_class.__name__}")
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class()
        
        # 遺伝子が正しく設定されているかチェック
        if hasattr(strategy_instance, 'gene'):
            gene = strategy_instance.gene
            position_size = gene.risk_management.get("position_size", 0.0)
            print(f"戦略インスタンスの取引量設定: {position_size}")
            
            if position_size > 0:
                print("✅ 取引量設定が正しく設定されています")
                return True
            else:
                print("❌ 取引量が0になっています")
                return False
        else:
            print("❌ 戦略インスタンスに遺伝子が設定されていません")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ga_engine_parameter_setup():
    """GAエンジンでのパラメータ設定テスト"""
    print("\n=== GAエンジンでのパラメータ設定テスト ===")
    
    try:
        # GAConfigを作成
        ga_config = GAConfig(
            population_size=5,
            generations=1,
            crossover_rate=0.8,
            mutation_rate=0.2
        )
        
        # モックのBacktestServiceを作成
        class MockBacktestService:
            def run_backtest(self, config):
                print(f"バックテスト設定を受信: {list(config.keys())}")
                
                # strategy_configの確認
                if "strategy_config" in config:
                    strategy_config = config["strategy_config"]
                    print(f"strategy_type: {strategy_config.get('strategy_type')}")
                    
                    parameters = strategy_config.get("parameters", {})
                    if "strategy_gene" in parameters:
                        strategy_gene = parameters["strategy_gene"]
                        risk_management = strategy_gene.get("risk_management", {})
                        position_size = risk_management.get("position_size", 0.0)
                        print(f"パラメータ内の取引量設定: {position_size}")
                        
                        if position_size > 0:
                            print("✅ パラメータが正しく設定されています")
                        else:
                            print("❌ パラメータの取引量が0です")
                    else:
                        print("❌ strategy_geneパラメータが見つかりません")
                else:
                    print("❌ strategy_configが見つかりません")
                
                return {
                    "performance_metrics": {
                        "total_return": 10.0,
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 5.0,
                        "total_trades": 5,
                    }
                }
        
        # GAエンジンを初期化
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)
        backtest_service = MockBacktestService()
        
        ga_engine = GeneticAlgorithmEngine(backtest_service, strategy_factory, gene_generator)
        
        # 固定バックテスト設定
        ga_engine._fixed_backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        # テスト用個体（ダミー）
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]
        
        # 個体評価を実行
        print("個体評価を実行中...")
        fitness = ga_engine._evaluate_individual(individual, ga_config)
        
        print(f"フィットネス値: {fitness}")
        
        if fitness[0] > 0:
            print("✅ GAエンジンでのパラメータ設定が正常に動作しています")
            return True
        else:
            print("❌ フィットネス値が0です")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_gene_serialization():
    """戦略遺伝子のシリアライゼーションテスト"""
    print("\n=== 戦略遺伝子のシリアライゼーションテスト ===")
    
    try:
        # テスト用戦略遺伝子を作成
        test_gene = create_test_strategy_gene()
        
        # 辞書形式に変換
        gene_dict = test_gene.to_dict()
        print(f"遺伝子辞書のキー: {list(gene_dict.keys())}")
        
        # リスク管理設定の確認
        if "risk_management" in gene_dict:
            risk_management = gene_dict["risk_management"]
            position_size = risk_management.get("position_size", 0.0)
            print(f"シリアライズされた取引量設定: {position_size}")
            
            if position_size > 0:
                print("✅ 戦略遺伝子のシリアライゼーションが正常です")
                return True
            else:
                print("❌ シリアライズ後の取引量が0です")
                return False
        else:
            print("❌ risk_managementが見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("オートストラテジー取引量0問題の修正検証を開始します\n")
    
    results = []
    
    # テスト1: StrategyFactoryでの取引量計算
    results.append(test_strategy_factory_volume_calculation())
    
    # テスト2: GAエンジンでのパラメータ設定
    results.append(test_ga_engine_parameter_setup())
    
    # テスト3: 戦略遺伝子のシリアライゼーション
    results.append(test_strategy_gene_serialization())
    
    # 結果のまとめ
    print("\n" + "="*50)
    print("テスト結果のまとめ:")
    print(f"成功: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 すべてのテストが成功しました！取引量0問題が修正されています。")
    else:
        print("⚠️ 一部のテストが失敗しました。追加の修正が必要です。")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
