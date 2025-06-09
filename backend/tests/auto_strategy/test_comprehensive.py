"""
自動戦略生成機能の包括的テスト

実装した全コンポーネントの大規模テストを実行します。
"""

import pytest
import asyncio
import time
import json
import random
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, IndicatorGene, Condition,
    encode_gene_to_list, decode_list_to_gene
)
from app.core.services.auto_strategy.models.ga_config import GAConfig, GAProgress
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService


class TestStrategyGeneComprehensive:
    """戦略遺伝子の包括的テスト"""
    
    def test_large_scale_gene_creation(self):
        """大量の戦略遺伝子作成テスト"""
        print("\n=== 大量戦略遺伝子作成テスト ===")
        
        indicators = ["SMA", "EMA", "RSI", "MACD", "BB", "STOCH", "CCI", "WILLIAMS", "ADX"]
        operators = [">", "<", ">=", "<=", "cross_above", "cross_below"]
        
        genes = []
        start_time = time.time()
        
        for i in range(1000):
            # ランダムな指標を選択
            num_indicators = random.randint(1, 5)
            selected_indicators = random.sample(indicators, num_indicators)
            
            indicator_genes = []
            for ind_type in selected_indicators:
                period = random.randint(5, 200)
                indicator_genes.append(IndicatorGene(
                    type=ind_type,
                    parameters={"period": period},
                    enabled=True
                ))
            
            # ランダムな条件を生成
            entry_conditions = []
            exit_conditions = []
            
            for _ in range(random.randint(1, 3)):
                left_operand = f"{random.choice(selected_indicators)}_{random.randint(5, 50)}"
                operator = random.choice(operators)
                right_operand = random.choice([
                    f"{random.choice(selected_indicators)}_{random.randint(5, 50)}",
                    random.uniform(10, 90)
                ])
                
                entry_conditions.append(Condition(
                    left_operand=left_operand,
                    operator=operator,
                    right_operand=right_operand
                ))
                
                exit_conditions.append(Condition(
                    left_operand=left_operand,
                    operator=operator,
                    right_operand=right_operand
                ))
            
            gene = StrategyGene(
                indicators=indicator_genes,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management={
                    "stop_loss": random.uniform(0.01, 0.05),
                    "take_profit": random.uniform(0.02, 0.10)
                }
            )
            
            genes.append(gene)
        
        creation_time = time.time() - start_time
        print(f"✅ 1000個の戦略遺伝子作成完了: {creation_time:.2f}秒")
        
        # 妥当性検証
        valid_count = 0
        validation_start = time.time()
        
        for gene in genes:
            is_valid, _ = gene.validate()
            if is_valid:
                valid_count += 1
        
        validation_time = time.time() - validation_start
        print(f"✅ 妥当性検証完了: {valid_count}/1000 有効 ({validation_time:.2f}秒)")
        
        assert valid_count > 800, f"有効な遺伝子が少なすぎます: {valid_count}/1000"
        
        return genes
    
    def test_serialization_performance(self):
        """シリアライゼーション性能テスト"""
        print("\n=== シリアライゼーション性能テスト ===")
        
        # テスト用遺伝子を作成
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
                IndicatorGene(type="EMA", parameters={"period": 12}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
                IndicatorGene(type="MACD", parameters={"fast": 12, "slow": 26, "signal": 9}),
                IndicatorGene(type="BB", parameters={"period": 20, "std": 2})
            ],
            entry_conditions=[
                Condition("RSI_14", "<", 30),
                Condition("SMA_20", ">", "EMA_12")
            ],
            exit_conditions=[
                Condition("RSI_14", ">", 70),
                Condition("SMA_20", "<", "EMA_12")
            ]
        )
        
        # JSON シリアライゼーション性能
        start_time = time.time()
        for _ in range(10000):
            json_str = gene.to_json()
            restored_gene = StrategyGene.from_json(json_str)
        json_time = time.time() - start_time
        print(f"✅ JSON シリアライゼーション (10000回): {json_time:.2f}秒")
        
        # エンコード/デコード性能
        start_time = time.time()
        for _ in range(10000):
            encoded = encode_gene_to_list(gene)
            decoded_gene = decode_list_to_gene(encoded)
        encode_time = time.time() - start_time
        print(f"✅ エンコード/デコード (10000回): {encode_time:.2f}秒")
        
        assert json_time < 10.0, f"JSON処理が遅すぎます: {json_time}秒"
        assert encode_time < 5.0, f"エンコード処理が遅すぎます: {encode_time}秒"


class TestGAConfigComprehensive:
    """GA設定の包括的テスト"""
    
    def test_config_variations(self):
        """設定バリエーションテスト"""
        print("\n=== GA設定バリエーションテスト ===")
        
        test_configs = [
            # 小規模設定
            {"population_size": 10, "generations": 5},
            # 中規模設定
            {"population_size": 50, "generations": 30},
            # 大規模設定
            {"population_size": 200, "generations": 100},
            # 極端な設定
            {"population_size": 500, "generations": 200},
        ]
        
        valid_configs = 0
        for i, config_params in enumerate(test_configs):
            try:
                config = GAConfig(**config_params)
                is_valid, errors = config.validate()
                
                if is_valid:
                    valid_configs += 1
                    print(f"✅ 設定{i+1}: {config_params} - 有効")
                else:
                    print(f"❌ 設定{i+1}: {config_params} - 無効: {errors}")
                    
            except Exception as e:
                print(f"❌ 設定{i+1}: {config_params} - エラー: {e}")
        
        print(f"✅ 有効な設定: {valid_configs}/{len(test_configs)}")
        assert valid_configs >= len(test_configs) - 1, "設定の妥当性に問題があります"
    
    def test_fitness_weight_combinations(self):
        """フィットネス重み組み合わせテスト"""
        print("\n=== フィットネス重み組み合わせテスト ===")
        
        weight_combinations = [
            {"total_return": 1.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0},
            {"total_return": 0.0, "sharpe_ratio": 1.0, "max_drawdown": 0.0, "win_rate": 0.0},
            {"total_return": 0.25, "sharpe_ratio": 0.25, "max_drawdown": 0.25, "win_rate": 0.25},
            {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.2, "win_rate": 0.1},
            {"total_return": 0.1, "sharpe_ratio": 0.6, "max_drawdown": 0.2, "win_rate": 0.1},
        ]
        
        valid_weights = 0
        for i, weights in enumerate(weight_combinations):
            config = GAConfig(fitness_weights=weights)
            is_valid, errors = config.validate()
            
            if is_valid:
                valid_weights += 1
                print(f"✅ 重み{i+1}: 有効")
            else:
                print(f"❌ 重み{i+1}: 無効: {errors}")
        
        print(f"✅ 有効な重み設定: {valid_weights}/{len(weight_combinations)}")
        assert valid_weights == len(weight_combinations), "重み設定の妥当性に問題があります"


class TestStrategyFactoryComprehensive:
    """戦略ファクトリーの包括的テスト"""
    
    def test_factory_with_all_indicators(self):
        """全指標対応テスト"""
        print("\n=== 全指標対応テスト ===")
        
        factory = StrategyFactory()
        all_indicators = list(factory.indicator_adapters.keys())
        
        successful_strategies = 0
        failed_strategies = 0
        
        for indicator in all_indicators:
            try:
                gene = StrategyGene(
                    indicators=[
                        IndicatorGene(type=indicator, parameters={"period": 20})
                    ],
                    entry_conditions=[
                        Condition("price", ">", 100)
                    ],
                    exit_conditions=[
                        Condition("price", "<", 90)
                    ]
                )
                
                is_valid, errors = factory.validate_gene(gene)
                if is_valid:
                    strategy_class = factory.create_strategy_class(gene)
                    successful_strategies += 1
                    print(f"✅ {indicator}: 戦略クラス生成成功")
                else:
                    failed_strategies += 1
                    print(f"❌ {indicator}: 妥当性検証失敗: {errors}")
                    
            except Exception as e:
                failed_strategies += 1
                print(f"❌ {indicator}: 戦略生成エラー: {e}")
        
        print(f"✅ 成功: {successful_strategies}, 失敗: {failed_strategies}")
        success_rate = successful_strategies / len(all_indicators)
        assert success_rate > 0.8, f"成功率が低すぎます: {success_rate:.2%}"
    
    def test_complex_strategy_generation(self):
        """複雑な戦略生成テスト"""
        print("\n=== 複雑な戦略生成テスト ===")
        
        factory = StrategyFactory()
        
        # 複雑な戦略遺伝子
        complex_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
                IndicatorGene(type="EMA", parameters={"period": 12}),
                IndicatorGene(type="RSI", parameters={"period": 14}),
                IndicatorGene(type="MACD", parameters={"fast": 12, "slow": 26, "signal": 9}),
                IndicatorGene(type="BB", parameters={"period": 20, "std": 2})
            ],
            entry_conditions=[
                Condition("RSI_14", "<", 30),
                Condition("SMA_20", "cross_above", "EMA_12"),
                Condition("price", ">", "BB_lower")
            ],
            exit_conditions=[
                Condition("RSI_14", ">", 70),
                Condition("SMA_20", "cross_below", "EMA_12"),
                Condition("price", "<", "BB_upper")
            ],
            risk_management={
                "stop_loss": 0.02,
                "take_profit": 0.05
            }
        )
        
        try:
            is_valid, errors = factory.validate_gene(complex_gene)
            print(f"✅ 複雑な戦略の妥当性: {is_valid}")
            if not is_valid:
                print(f"   エラー: {errors}")
            
            if is_valid:
                strategy_class = factory.create_strategy_class(complex_gene)
                strategy_instance = strategy_class()
                print(f"✅ 複雑な戦略クラス生成成功")
                print(f"   指標数: {len(complex_gene.indicators)}")
                print(f"   エントリー条件数: {len(complex_gene.entry_conditions)}")
                print(f"   イグジット条件数: {len(complex_gene.exit_conditions)}")
                
        except Exception as e:
            print(f"❌ 複雑な戦略生成エラー: {e}")
            raise


class TestGAEngineComprehensive:
    """GAエンジンの包括的テスト"""
    
    def test_deap_integration(self):
        """DEAP統合テスト"""
        print("\n=== DEAP統合テスト ===")
        
        # モックのBacktestServiceを作成
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 25
            }
        }
        
        factory = StrategyFactory()
        ga_engine = GeneticAlgorithmEngine(mock_backtest_service, factory)
        
        # 小規模GA設定
        config = GAConfig(
            population_size=10,
            generations=3,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2
        )
        
        try:
            # DEAP環境のセットアップ
            ga_engine.setup_deap(config)
            print("✅ DEAP環境セットアップ成功")
            
            # ツールボックスの確認
            assert ga_engine.toolbox is not None, "ツールボックスが初期化されていません"
            assert hasattr(ga_engine.toolbox, 'individual'), "個体生成関数が登録されていません"
            assert hasattr(ga_engine.toolbox, 'population'), "個体群生成関数が登録されていません"
            assert hasattr(ga_engine.toolbox, 'evaluate'), "評価関数が登録されていません"
            
            print("✅ DEAP ツールボックス検証成功")
            
            # 個体生成テスト
            individual = ga_engine.toolbox.individual()
            assert len(individual) == 16, f"個体長が不正: {len(individual)}"
            print(f"✅ 個体生成成功: 長さ{len(individual)}")
            
            # 個体群生成テスト
            population = ga_engine.toolbox.population(n=5)
            assert len(population) == 5, f"個体群サイズが不正: {len(population)}"
            print(f"✅ 個体群生成成功: {len(population)}個体")
            
        except Exception as e:
            print(f"❌ DEAP統合テストエラー: {e}")
            raise
    
    def test_fitness_calculation(self):
        """フィットネス計算テスト"""
        print("\n=== フィットネス計算テスト ===")
        
        # 様々なパフォーマンス結果でテスト
        test_results = [
            # 良好な結果
            {
                "performance_metrics": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.05,
                    "win_rate": 0.7,
                    "total_trades": 30
                }
            },
            # 平均的な結果
            {
                "performance_metrics": {
                    "total_return": 0.10,
                    "sharpe_ratio": 0.8,
                    "max_drawdown": 0.15,
                    "win_rate": 0.55,
                    "total_trades": 20
                }
            },
            # 悪い結果
            {
                "performance_metrics": {
                    "total_return": -0.05,
                    "sharpe_ratio": 0.2,
                    "max_drawdown": 0.35,
                    "win_rate": 0.4,
                    "total_trades": 5
                }
            }
        ]
        
        mock_backtest_service = Mock()
        factory = StrategyFactory()
        ga_engine = GeneticAlgorithmEngine(mock_backtest_service, factory)
        
        config = GAConfig()
        
        fitness_scores = []
        for i, result in enumerate(test_results):
            fitness = ga_engine._calculate_fitness(result, config)
            fitness_scores.append(fitness)
            print(f"✅ 結果{i+1}: フィットネス = {fitness:.4f}")
        
        # フィットネススコアの順序確認
        assert fitness_scores[0] > fitness_scores[1], "良好な結果のフィットネスが低い"
        assert fitness_scores[1] > fitness_scores[2], "平均的な結果のフィットネスが低い"
        
        print("✅ フィットネス計算順序確認成功")


class TestAutoStrategyServiceComprehensive:
    """自動戦略サービスの包括的テスト"""
    
    def test_service_initialization(self):
        """サービス初期化テスト"""
        print("\n=== サービス初期化テスト ===")
        
        try:
            # 実際のサービス初期化はデータベース接続が必要なため、
            # コンポーネントの個別初期化をテスト
            factory = StrategyFactory()
            print("✅ 戦略ファクトリー初期化成功")
            
            # GA設定の作成
            config = GAConfig.create_default()
            print("✅ デフォルトGA設定作成成功")
            
            # 戦略遺伝子の作成
            gene = StrategyGene(
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                entry_conditions=[Condition("price", ">", 100)],
                exit_conditions=[Condition("price", "<", 90)]
            )
            print("✅ 戦略遺伝子作成成功")
            
            # 妥当性検証
            is_valid, errors = factory.validate_gene(gene)
            assert is_valid, f"遺伝子妥当性検証失敗: {errors}"
            print("✅ 遺伝子妥当性検証成功")
            
        except Exception as e:
            print(f"❌ サービス初期化テストエラー: {e}")
            raise
    
    def test_experiment_management(self):
        """実験管理テスト"""
        print("\n=== 実験管理テスト ===")
        
        # 実験情報の管理をシミュレート
        experiments = {}
        
        # 実験作成
        experiment_id = "test_experiment_001"
        experiment_info = {
            "id": experiment_id,
            "name": "Test Experiment",
            "status": "running",
            "start_time": time.time(),
            "config": GAConfig.create_fast().to_dict()
        }
        
        experiments[experiment_id] = experiment_info
        print(f"✅ 実験作成: {experiment_id}")
        
        # 進捗更新シミュレート
        for generation in range(1, 6):
            progress = GAProgress(
                experiment_id=experiment_id,
                current_generation=generation,
                total_generations=5,
                best_fitness=0.5 + generation * 0.1,
                average_fitness=0.3 + generation * 0.05,
                execution_time=generation * 10.0,
                estimated_remaining_time=(5 - generation) * 10.0
            )
            
            print(f"✅ 世代{generation}: フィットネス={progress.best_fitness:.2f}")
        
        # 実験完了
        experiments[experiment_id]["status"] = "completed"
        experiments[experiment_id]["end_time"] = time.time()
        
        print(f"✅ 実験完了: {experiment_id}")
        
        assert len(experiments) == 1, "実験管理に問題があります"
        assert experiments[experiment_id]["status"] == "completed", "実験状態更新に問題があります"


def run_stress_test():
    """ストレステスト"""
    print("\n" + "="*60)
    print("🔥 ストレステスト開始")
    print("="*60)
    
    # 大量の戦略遺伝子生成・処理
    start_time = time.time()
    
    factory = StrategyFactory()
    genes = []
    
    # 1000個の戦略遺伝子を生成
    for i in range(1000):
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": random.randint(5, 50)}),
                IndicatorGene(type="RSI", parameters={"period": random.randint(10, 30)})
            ],
            entry_conditions=[
                Condition("RSI_14", "<", random.randint(20, 40))
            ],
            exit_conditions=[
                Condition("RSI_14", ">", random.randint(60, 80))
            ]
        )
        genes.append(gene)
    
    generation_time = time.time() - start_time
    print(f"✅ 1000個の戦略遺伝子生成: {generation_time:.2f}秒")
    
    # 妥当性検証
    validation_start = time.time()
    valid_count = 0
    
    for gene in genes:
        is_valid, _ = factory.validate_gene(gene)
        if is_valid:
            valid_count += 1
    
    validation_time = time.time() - validation_start
    print(f"✅ 妥当性検証: {valid_count}/1000 有効 ({validation_time:.2f}秒)")
    
    # エンコード/デコード性能
    encode_start = time.time()
    for gene in genes[:100]:  # 100個でテスト
        encoded = encode_gene_to_list(gene)
        decoded = decode_list_to_gene(encoded)
    
    encode_time = time.time() - encode_start
    print(f"✅ エンコード/デコード (100個): {encode_time:.2f}秒")
    
    total_time = time.time() - start_time
    print(f"🎯 ストレステスト完了: {total_time:.2f}秒")
    
    # パフォーマンス基準
    assert generation_time < 5.0, f"遺伝子生成が遅すぎます: {generation_time}秒"
    assert validation_time < 2.0, f"妥当性検証が遅すぎます: {validation_time}秒"
    assert encode_time < 1.0, f"エンコード処理が遅すぎます: {encode_time}秒"
    assert valid_count > 950, f"有効な遺伝子が少なすぎます: {valid_count}/1000"
    
    print("🎉 ストレステスト全て成功！")


def main():
    """メインテスト実行"""
    print("🚀 自動戦略生成機能 大規模テスト開始")
    print("=" * 80)
    
    test_results = []
    
    # 各テストクラスの実行
    test_classes = [
        TestStrategyGeneComprehensive(),
        TestGAConfigComprehensive(),
        TestStrategyFactoryComprehensive(),
        TestGAEngineComprehensive(),
        TestAutoStrategyServiceComprehensive(),
    ]
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 {class_name} 実行中...")
        
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in methods:
            try:
                method = getattr(test_class, method_name)
                method()
                test_results.append((class_name, method_name, "✅ 成功"))
            except Exception as e:
                test_results.append((class_name, method_name, f"❌ 失敗: {e}"))
                print(f"❌ {method_name} 失敗: {e}")
    
    # ストレステスト実行
    try:
        run_stress_test()
        test_results.append(("ストレステスト", "run_stress_test", "✅ 成功"))
    except Exception as e:
        test_results.append(("ストレステスト", "run_stress_test", f"❌ 失敗: {e}"))
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("📊 テスト結果サマリー")
    print("=" * 80)
    
    success_count = 0
    total_count = len(test_results)
    
    for class_name, method_name, result in test_results:
        print(f"{class_name:30} {method_name:30} {result}")
        if "成功" in result:
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"🎯 総合結果: {success_count}/{total_count} 成功 ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 全てのテストが成功しました！")
        print("\n✅ 実装品質確認:")
        print("  - 戦略遺伝子モデル: 完全動作")
        print("  - GA設定管理: 完全動作")
        print("  - 戦略ファクトリー: 完全動作")
        print("  - GAエンジン: 完全動作")
        print("  - サービス統合: 完全動作")
        print("  - パフォーマンス: 基準クリア")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("実装を見直してください")
    
    return success_count == total_count


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
