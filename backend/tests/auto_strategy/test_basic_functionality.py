"""
自動戦略生成機能の基本動作テスト

実装した基盤コンポーネントの基本的な動作を確認します。
"""

import pytest
import json
from typing import Dict, Any

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, IndicatorGene, Condition,
    encode_gene_to_list, decode_list_to_gene
)
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory


class TestStrategyGene:
    """戦略遺伝子のテスト"""
    
    def test_strategy_gene_creation(self):
        """戦略遺伝子の作成テスト"""
        # 指標遺伝子の作成
        sma_indicator = IndicatorGene(
            type="SMA",
            parameters={"period": 20},
            enabled=True
        )
        
        rsi_indicator = IndicatorGene(
            type="RSI", 
            parameters={"period": 14},
            enabled=True
        )
        
        # 条件の作成
        entry_condition = Condition(
            left_operand="RSI_14",
            operator="<",
            right_operand=30
        )
        
        exit_condition = Condition(
            left_operand="RSI_14",
            operator=">",
            right_operand=70
        )
        
        # 戦略遺伝子の作成
        gene = StrategyGene(
            indicators=[sma_indicator, rsi_indicator],
            entry_conditions=[entry_condition],
            exit_conditions=[exit_condition],
            risk_management={"stop_loss": 0.02, "take_profit": 0.05}
        )
        
        # 基本的な検証
        assert len(gene.indicators) == 2
        assert len(gene.entry_conditions) == 1
        assert len(gene.exit_conditions) == 1
        assert gene.id is not None
        
        # 妥当性検証
        is_valid, errors = gene.validate()
        assert is_valid, f"Validation errors: {errors}"
    
    def test_strategy_gene_serialization(self):
        """戦略遺伝子のシリアライゼーションテスト"""
        # 戦略遺伝子の作成
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
                IndicatorGene(type="RSI", parameters={"period": 14})
            ],
            entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ]
        )
        
        # 辞書変換
        gene_dict = gene.to_dict()
        assert isinstance(gene_dict, dict)
        assert "indicators" in gene_dict
        assert "entry_conditions" in gene_dict
        
        # JSON変換
        gene_json = gene.to_json()
        assert isinstance(gene_json, str)
        
        # 復元テスト
        restored_gene = StrategyGene.from_dict(gene_dict)
        assert len(restored_gene.indicators) == len(gene.indicators)
        assert len(restored_gene.entry_conditions) == len(gene.entry_conditions)
        
        # JSON復元テスト
        json_restored_gene = StrategyGene.from_json(gene_json)
        assert len(json_restored_gene.indicators) == len(gene.indicators)
    
    def test_gene_encoding_decoding(self):
        """遺伝子エンコード/デコードテスト"""
        # 戦略遺伝子の作成
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
                IndicatorGene(type="RSI", parameters={"period": 14})
            ],
            entry_conditions=[
                Condition(left_operand="SMA_20", operator=">", right_operand="SMA_50")
            ],
            exit_conditions=[
                Condition(left_operand="SMA_20", operator="<", right_operand="SMA_50")
            ]
        )
        
        # エンコード
        encoded = encode_gene_to_list(gene)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
        # デコード
        decoded_gene = decode_list_to_gene(encoded)
        assert isinstance(decoded_gene, StrategyGene)
        assert len(decoded_gene.indicators) > 0


class TestGAConfig:
    """GA設定のテスト"""
    
    def test_ga_config_creation(self):
        """GA設定の作成テスト"""
        config = GAConfig(
            population_size=50,
            generations=30,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=5
        )
        
        assert config.population_size == 50
        assert config.generations == 30
        assert config.crossover_rate == 0.8
        
        # 妥当性検証
        is_valid, errors = config.validate()
        assert is_valid, f"Validation errors: {errors}"
    
    def test_ga_config_presets(self):
        """GA設定プリセットのテスト"""
        # デフォルト設定
        default_config = GAConfig.create_default()
        is_valid, _ = default_config.validate()
        assert is_valid
        
        # 高速設定
        fast_config = GAConfig.create_fast()
        is_valid, _ = fast_config.validate()
        assert is_valid
        assert fast_config.population_size < default_config.population_size
        
        # 徹底設定
        thorough_config = GAConfig.create_thorough()
        is_valid, _ = thorough_config.validate()
        assert is_valid
        assert thorough_config.population_size > default_config.population_size
    
    def test_ga_config_serialization(self):
        """GA設定のシリアライゼーションテスト"""
        config = GAConfig.create_default()
        
        # 辞書変換
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "population_size" in config_dict
        
        # JSON変換
        config_json = config.to_json()
        assert isinstance(config_json, str)
        
        # 復元テスト
        restored_config = GAConfig.from_dict(config_dict)
        assert restored_config.population_size == config.population_size
        
        # JSON復元テスト
        json_restored_config = GAConfig.from_json(config_json)
        assert json_restored_config.population_size == config.population_size


class TestStrategyFactory:
    """戦略ファクトリーのテスト"""
    
    def test_strategy_factory_creation(self):
        """戦略ファクトリーの作成テスト"""
        factory = StrategyFactory()
        assert factory is not None
        assert hasattr(factory, 'indicator_adapters')
        assert len(factory.indicator_adapters) > 0
    
    def test_gene_validation(self):
        """遺伝子妥当性検証テスト"""
        factory = StrategyFactory()
        
        # 有効な遺伝子
        valid_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}),
                IndicatorGene(type="RSI", parameters={"period": 14})
            ],
            entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ]
        )
        
        is_valid, errors = factory.validate_gene(valid_gene)
        assert is_valid, f"Validation errors: {errors}"
        
        # 無効な遺伝子（未対応指標）
        invalid_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="UNKNOWN_INDICATOR", parameters={"period": 20})
            ],
            entry_conditions=[
                Condition(left_operand="price", operator=">", right_operand=100)
            ],
            exit_conditions=[
                Condition(left_operand="price", operator="<", right_operand=90)
            ]
        )
        
        is_valid, errors = factory.validate_gene(invalid_gene)
        assert not is_valid
        assert len(errors) > 0
    
    def test_strategy_class_creation(self):
        """戦略クラス生成テスト"""
        factory = StrategyFactory()
        
        # 簡単な戦略遺伝子
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20})
            ],
            entry_conditions=[
                Condition(left_operand="price", operator=">", right_operand=100)
            ],
            exit_conditions=[
                Condition(left_operand="price", operator="<", right_operand=90)
            ]
        )
        
        # 戦略クラス生成
        strategy_class = factory.create_strategy_class(gene)
        
        # 基本的な検証
        assert strategy_class is not None
        assert hasattr(strategy_class, 'init')
        assert hasattr(strategy_class, 'next')
        
        # インスタンス作成テスト
        strategy_instance = strategy_class()
        assert strategy_instance is not None
        assert hasattr(strategy_instance, 'gene')
        assert strategy_instance.gene == gene


if __name__ == "__main__":
    # 直接実行用
    print("=== 自動戦略生成機能 基本動作テスト ===")
    
    # 戦略遺伝子テスト
    print("\n1. 戦略遺伝子テスト")
    test_gene = TestStrategyGene()
    test_gene.test_strategy_gene_creation()
    test_gene.test_strategy_gene_serialization()
    test_gene.test_gene_encoding_decoding()
    print("   ✅ 戦略遺伝子テスト完了")
    
    # GA設定テスト
    print("\n2. GA設定テスト")
    test_config = TestGAConfig()
    test_config.test_ga_config_creation()
    test_config.test_ga_config_presets()
    test_config.test_ga_config_serialization()
    print("   ✅ GA設定テスト完了")
    
    # 戦略ファクトリーテスト
    print("\n3. 戦略ファクトリーテスト")
    test_factory = TestStrategyFactory()
    test_factory.test_strategy_factory_creation()
    test_factory.test_gene_validation()
    test_factory.test_strategy_class_creation()
    print("   ✅ 戦略ファクトリーテスト完了")
    
    print("\n🎉 全ての基本動作テストが完了しました！")
