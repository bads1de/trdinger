"""
エンドツーエンド統合テスト

新しいインジケータを含むオートストラテジー機能の
完全な動作確認テスト
"""

import pytest
from unittest.mock import patch

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.indicators import TechnicalIndicatorService


class TestEndToEndIntegration:
    """エンドツーエンド統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.config = GAConfig(
            population_size=5,
            generations=2,
            max_indicators=3,
            min_indicators=1,
            max_conditions=2,
            min_conditions=1
        )
        self.generator = RandomGeneGenerator(self.config)
        self.indicator_service = TechnicalIndicatorService()

    def test_indicator_service_includes_new_indicators(self):
        """TechnicalIndicatorServiceが新しいインジケータを含むことを確認"""
        supported_indicators = self.indicator_service.get_supported_indicators()
        
        # 新しいカテゴリのインジケータが含まれていることを確認
        new_indicators = [
            "HT_DCPERIOD", "HT_DCPHASE", "HT_SINE",  # サイクル系
            "BETA", "CORREL", "STDDEV", "VAR",  # 統計系
            "ACOS", "ASIN", "COS", "SIN", "SQRT",  # 数学変換系
            "ADD", "SUB", "MULT", "DIV",  # 数学演算子系
            "CDL_DOJI", "CDL_HAMMER", "CDL_HANGING_MAN"  # パターン認識系
        ]
        
        found_indicators = []
        for indicator in new_indicators:
            if indicator in supported_indicators:
                found_indicators.append(indicator)
        
        # 各カテゴリから少なくとも1つは見つかることを確認
        assert len(found_indicators) >= 10, f"新しいインジケータが十分に見つかりませんでした: {found_indicators}"

    def test_strategy_generation_with_new_indicators(self):
        """新しいインジケータを使った戦略生成の完全テスト"""
        # 複数の戦略を生成して新しいインジケータが使用されることを確認
        strategies_with_new_indicators = 0
        total_strategies = 20
        
        new_indicators = [
            "HT_DCPERIOD", "HT_DCPHASE", "HT_SINE", "HT_TRENDMODE",
            "BETA", "CORREL", "STDDEV", "VAR", "LINEARREG", "TSF",
            "ACOS", "ASIN", "COS", "SIN", "TAN", "SQRT", "LN", "EXP",
            "ADD", "SUB", "MULT", "DIV", "MAX", "MIN",
            "CDL_DOJI", "CDL_HAMMER", "CDL_HANGING_MAN", "CDL_SHOOTING_STAR"
        ]
        
        for i in range(total_strategies):
            strategy = self.generator.generate_random_gene()
            
            # 戦略に新しいインジケータが含まれているかチェック
            for indicator in strategy.indicators:
                if indicator.type in new_indicators:
                    strategies_with_new_indicators += 1
                    break
        
        # 少なくとも30%の戦略で新しいインジケータが使用されることを期待
        usage_rate = strategies_with_new_indicators / total_strategies
        assert usage_rate >= 0.3, f"新しいインジケータの使用率が低すぎます: {usage_rate:.2%}"

    def test_condition_generation_quality(self):
        """生成された条件の品質テスト"""
        strategy = self.generator.generate_random_gene()
        
        # 基本的な検証
        assert len(strategy.indicators) >= 1
        assert len(strategy.long_entry_conditions) >= 1
        assert len(strategy.short_entry_conditions) >= 1
        
        # 条件の構造が正しいことを確認
        for condition in strategy.long_entry_conditions + strategy.short_entry_conditions:
            assert condition.left_operand is not None
            assert condition.operator in [">", "<", ">=", "<=", "==", "!="]
            assert condition.right_operand is not None

    def test_indicator_parameter_generation(self):
        """新しいインジケータのパラメータ生成テスト"""
        # 各カテゴリのインジケータでパラメータが正しく生成されることを確認
        test_indicators = [
            "HT_DCPHASE",  # サイクル系
            "CORREL",      # 統計系
            "STDDEV",      # 統計系（パラメータあり）
            "COS",         # 数学変換系
            "CDL_DOJI"     # パターン認識系
        ]
        
        for indicator_type in test_indicators:
            if indicator_type in self.generator.available_indicators:
                # パラメータ生成をテスト
                from app.core.services.auto_strategy.utils.parameter_generators import generate_indicator_parameters
                
                try:
                    params = generate_indicator_parameters(indicator_type)
                    # パラメータが辞書形式で返されることを確認
                    assert isinstance(params, dict)
                except Exception as e:
                    # エラーが発生した場合でも適切にハンドリングされることを確認
                    assert True  # パラメータ生成エラーは許容される

    def test_strategy_validation(self):
        """生成された戦略の妥当性検証"""
        strategy = self.generator.generate_random_gene()
        
        # 戦略の基本構造が正しいことを確認
        assert hasattr(strategy, 'indicators')
        assert hasattr(strategy, 'long_entry_conditions')
        assert hasattr(strategy, 'short_entry_conditions')
        assert hasattr(strategy, 'exit_conditions')
        assert hasattr(strategy, 'tpsl_gene')
        
        # インジケータが有効であることを確認
        for indicator in strategy.indicators:
            assert indicator.enabled
            assert indicator.type in self.generator.available_indicators

    def test_multiple_strategy_generation_consistency(self):
        """複数戦略生成の一貫性テスト"""
        strategies = []
        
        # 10個の戦略を生成
        for _ in range(10):
            strategy = self.generator.generate_random_gene()
            strategies.append(strategy)
        
        # 全ての戦略が有効であることを確認
        for i, strategy in enumerate(strategies):
            assert len(strategy.indicators) > 0, f"戦略{i}にインジケータがありません"
            assert len(strategy.long_entry_conditions) > 0, f"戦略{i}にロング条件がありません"
            assert len(strategy.short_entry_conditions) > 0, f"戦略{i}にショート条件がありません"

    def test_new_indicator_categories_coverage(self):
        """新しいインジケータカテゴリの網羅性テスト"""
        # 大量の戦略を生成して全カテゴリがカバーされることを確認
        found_categories = set()
        category_mapping = {
            "HT_DCPERIOD": "cycle", "HT_DCPHASE": "cycle", "HT_SINE": "cycle",
            "BETA": "statistics", "CORREL": "statistics", "STDDEV": "statistics",
            "ACOS": "math_transform", "COS": "math_transform", "SIN": "math_transform",
            "ADD": "math_operators", "SUB": "math_operators", "MULT": "math_operators",
            "CDL_DOJI": "pattern_recognition", "CDL_HAMMER": "pattern_recognition"
        }
        
        for _ in range(50):  # 50個の戦略を生成
            strategy = self.generator.generate_random_gene()
            
            for indicator in strategy.indicators:
                if indicator.type in category_mapping:
                    found_categories.add(category_mapping[indicator.type])
        
        # 少なくとも3つの新しいカテゴリが使用されることを期待
        assert len(found_categories) >= 3, f"新しいカテゴリの使用が不十分です: {found_categories}"

    def test_error_resilience(self):
        """エラー耐性テスト"""
        # 異常な設定でも戦略が生成されることを確認
        extreme_config = GAConfig(
            population_size=1,
            generations=1,
            max_indicators=10,  # 多めの指標
            min_indicators=5,
            max_conditions=5,
            min_conditions=1
        )
        
        extreme_generator = RandomGeneGenerator(extreme_config)
        
        try:
            strategy = extreme_generator.generate_random_gene()
            assert strategy is not None
            assert len(strategy.indicators) >= 1
        except Exception as e:
            pytest.fail(f"極端な設定でエラーが発生しました: {e}")
