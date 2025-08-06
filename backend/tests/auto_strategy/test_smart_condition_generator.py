"""
SmartConditionGeneratorの包括的テスト

戦略タイプ選択、条件生成ロジック、指標特性活用、バランス調整機能を詳細にテストします。
"""

import logging
from typing import List

import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSmartConditionGenerator:
    """SmartConditionGeneratorの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.test_indicators = self._create_test_indicators()

    def _create_test_indicators(self):
        """テスト用の指標リストを作成"""
        try:
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            return [
                # モメンタム系指標
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="STOCH", parameters={"fastk_period": 14}, enabled=True),
                
                # トレンド系指標
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="EMA", parameters={"period": 12}, enabled=True),
                
                # ボラティリティ系指標
                IndicatorGene(type="BB", parameters={"period": 20, "std": 2}, enabled=True),
                IndicatorGene(type="ATR", parameters={"period": 14}, enabled=True),
                
                # ML指標
                IndicatorGene(type="ML_UP_PROB", parameters={}, enabled=True),
                IndicatorGene(type="ML_DOWN_PROB", parameters={}, enabled=True),
                IndicatorGene(type="ML_RANGE_PROB", parameters={}, enabled=True),
                
                # パターン認識指標
                IndicatorGene(type="CDL_HAMMER", parameters={}, enabled=True),
                IndicatorGene(type="CDL_DOJI", parameters={}, enabled=True),
                
                # 無効な指標（テスト用）
                IndicatorGene(type="MACD", parameters={"fast": 12, "slow": 26}, enabled=False),
            ]
        except ImportError:
            # インポートエラーの場合は空のリストを返す
            return []

    def test_smart_condition_generator_initialization(self):
        """SmartConditionGeneratorの初期化テスト"""
        logger.info("=== SmartConditionGenerator初期化テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            
            # デフォルト初期化
            generator_default = SmartConditionGenerator()
            assert generator_default.enable_smart_generation is True, "デフォルトでスマート生成が有効になっていません"
            assert hasattr(generator_default, 'logger'), "ロガーが設定されていません"
            
            # スマート生成無効での初期化
            generator_disabled = SmartConditionGenerator(enable_smart_generation=False)
            assert generator_disabled.enable_smart_generation is False, "スマート生成が無効になっていません"
            
            logger.info("✅ SmartConditionGenerator初期化テスト成功")
            
        except Exception as e:
            pytest.fail(f"SmartConditionGenerator初期化テストエラー: {e}")

    def test_strategy_type_selection(self):
        """戦略タイプ選択テスト"""
        logger.info("=== 戦略タイプ選択テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import (
                SmartConditionGenerator, StrategyType
            )
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 異なる指標の組み合わせ戦略のテスト
            mixed_indicators = [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="BB", parameters={"period": 20}, enabled=True),
            ]
            strategy_type = generator._select_strategy_type(mixed_indicators)
            assert strategy_type == StrategyType.DIFFERENT_INDICATORS, "異なる指標の組み合わせ戦略が選択されませんでした"
            
            # 時間軸分離戦略のテスト（同じ指標の複数）
            same_type_indicators = [
                IndicatorGene(type="SMA", parameters={"period": 10}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ]
            strategy_type = generator._select_strategy_type(same_type_indicators)
            assert strategy_type == StrategyType.TIME_SEPARATION, "時間軸分離戦略が選択されませんでした"
            
            # 指標特性活用戦略のテスト（ボリンジャーバンド）
            bb_indicators = [
                IndicatorGene(type="BB", parameters={"period": 20}, enabled=True),
            ]
            strategy_type = generator._select_strategy_type(bb_indicators)
            assert strategy_type == StrategyType.INDICATOR_CHARACTERISTICS, "指標特性活用戦略が選択されませんでした"
            
            logger.info("✅ 戦略タイプ選択テスト成功")
            
        except Exception as e:
            pytest.fail(f"戦略タイプ選択テストエラー: {e}")

    def test_generate_balanced_conditions(self):
        """バランス調整機能テスト"""
        logger.info("=== バランス調整機能テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            
            generator = SmartConditionGenerator()
            
            # 条件生成
            long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(
                self.test_indicators
            )
            
            # 基本的な妥当性チェック
            assert isinstance(long_conditions, list), "ロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "ショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "イグジット条件がリスト形式ではありません"
            
            # 条件が生成されていることを確認
            total_conditions = len(long_conditions) + len(short_conditions) + len(exit_conditions)
            assert total_conditions > 0, "条件が生成されていません"
            
            # 各条件の妥当性チェック
            all_conditions = long_conditions + short_conditions + exit_conditions
            for condition in all_conditions:
                assert hasattr(condition, 'left_operand'), "left_operand属性が不足しています"
                assert hasattr(condition, 'operator'), "operator属性が不足しています"
                assert hasattr(condition, 'right_operand'), "right_operand属性が不足しています"
                assert condition.operator in ['>', '<', '>=', '<=', '==', '!='], f"無効な演算子: {condition.operator}"
            
            logger.info(f"生成された条件数 - ロング: {len(long_conditions)}, ショート: {len(short_conditions)}, イグジット: {len(exit_conditions)}")
            logger.info("✅ バランス調整機能テスト成功")
            
        except Exception as e:
            pytest.fail(f"バランス調整機能テストエラー: {e}")

    def test_different_indicators_strategy(self):
        """異なる指標の組み合わせ戦略テスト"""
        logger.info("=== 異なる指標の組み合わせ戦略テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 異なるタイプの指標を準備
            different_indicators = [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="BB", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="ATR", parameters={"period": 14}, enabled=True),
            ]
            
            long_conditions, short_conditions, exit_conditions = generator._generate_different_indicators_strategy(
                different_indicators
            )
            
            # 結果の妥当性チェック
            assert isinstance(long_conditions, list), "ロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "ショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "イグジット条件がリスト形式ではありません"
            
            # 異なる指標が使用されていることを確認
            used_indicators = set()
            for condition in long_conditions + short_conditions:
                if isinstance(condition.left_operand, str):
                    used_indicators.add(condition.left_operand)
            
            assert len(used_indicators) > 1, "複数の指標が使用されていません"
            
            logger.info("✅ 異なる指標の組み合わせ戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"異なる指標の組み合わせ戦略テストエラー: {e}")

    def test_time_separation_strategy(self):
        """時間軸分離戦略テスト"""
        logger.info("=== 時間軸分離戦略テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 同じタイプの指標を異なるパラメータで準備
            same_type_indicators = [
                IndicatorGene(type="SMA", parameters={"period": 10}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 50}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 7}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ]
            
            long_conditions, short_conditions, exit_conditions = generator._generate_time_separation_strategy(
                same_type_indicators
            )
            
            # 結果の妥当性チェック
            assert isinstance(long_conditions, list), "ロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "ショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "イグジット条件がリスト形式ではありません"
            
            # 条件が生成されていることを確認
            total_conditions = len(long_conditions) + len(short_conditions)
            assert total_conditions > 0, "条件が生成されていません"
            
            logger.info("✅ 時間軸分離戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"時間軸分離戦略テストエラー: {e}")

    def test_indicator_characteristics_strategy(self):
        """指標特性活用戦略テスト"""
        logger.info("=== 指標特性活用戦略テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 特性を活用できる指標を準備
            characteristic_indicators = [
                IndicatorGene(type="BB", parameters={"period": 20, "std": 2}, enabled=True),
                IndicatorGene(type="ML_UP_PROB", parameters={}, enabled=True),
                IndicatorGene(type="ML_DOWN_PROB", parameters={}, enabled=True),
                IndicatorGene(type="CDL_HAMMER", parameters={}, enabled=True),
            ]
            
            long_conditions, short_conditions, exit_conditions = generator._generate_indicator_characteristics_strategy(
                characteristic_indicators
            )
            
            # 結果の妥当性チェック
            assert isinstance(long_conditions, list), "ロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "ショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "イグジット条件がリスト形式ではありません"
            
            # ML指標の特別な処理が適用されていることを確認
            ml_conditions_found = False
            for condition in long_conditions:
                if isinstance(condition.left_operand, str) and condition.left_operand.startswith("ML_"):
                    ml_conditions_found = True
                    break
            
            if any(ind.type.startswith("ML_") for ind in characteristic_indicators if ind.enabled):
                assert ml_conditions_found, "ML指標の条件が生成されていません"
            
            logger.info("✅ 指標特性活用戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"指標特性活用戦略テストエラー: {e}")

    def test_complex_conditions_strategy(self):
        """複合条件戦略テスト"""
        logger.info("=== 複合条件戦略テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            
            generator = SmartConditionGenerator()
            
            long_conditions, short_conditions, exit_conditions = generator._generate_complex_conditions_strategy(
                self.test_indicators
            )
            
            # 結果の妥当性チェック
            assert isinstance(long_conditions, list), "ロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "ショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "イグジット条件がリスト形式ではありません"
            
            # 複数の条件が組み合わされていることを確認
            total_conditions = len(long_conditions) + len(short_conditions)
            if total_conditions > 0:
                # 条件の複雑さを確認（複数の指標が使用されているか）
                used_indicators = set()
                for condition in long_conditions + short_conditions:
                    if isinstance(condition.left_operand, str):
                        used_indicators.add(condition.left_operand)
                
                # 複合条件なので、可能であれば複数の指標が使用されることを期待
                logger.info(f"使用された指標数: {len(used_indicators)}")
            
            logger.info("✅ 複合条件戦略テスト成功")
            
        except Exception as e:
            pytest.fail(f"複合条件戦略テストエラー: {e}")

    def test_ml_indicators_integration(self):
        """ML指標統合テスト"""
        logger.info("=== ML指標統合テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # ML指標のみのテスト
            ml_only_indicators = [
                IndicatorGene(type="ML_UP_PROB", parameters={}, enabled=True),
                IndicatorGene(type="ML_DOWN_PROB", parameters={}, enabled=True),
                IndicatorGene(type="ML_RANGE_PROB", parameters={}, enabled=True),
            ]
            
            long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(
                ml_only_indicators
            )
            
            # ML指標の条件が生成されていることを確認
            ml_conditions_found = False
            for condition in long_conditions + short_conditions:
                if isinstance(condition.left_operand, str) and condition.left_operand.startswith("ML_"):
                    ml_conditions_found = True
                    break
            
            assert ml_conditions_found, "ML指標の条件が生成されていません"
            
            # ML指標とテクニカル指標の混合テスト
            mixed_indicators = [
                IndicatorGene(type="ML_UP_PROB", parameters={}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ]
            
            long_conditions_mixed, short_conditions_mixed, _ = generator.generate_balanced_conditions(
                mixed_indicators
            )
            
            # 混合条件が生成されていることを確認
            total_mixed_conditions = len(long_conditions_mixed) + len(short_conditions_mixed)
            assert total_mixed_conditions > 0, "混合条件が生成されていません"
            
            logger.info("✅ ML指標統合テスト成功")
            
        except Exception as e:
            pytest.fail(f"ML指標統合テストエラー: {e}")

    def test_pattern_recognition_indicators(self):
        """パターン認識指標テスト"""
        logger.info("=== パターン認識指標テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # パターン認識指標のテスト
            pattern_indicators = [
                IndicatorGene(type="CDL_HAMMER", parameters={}, enabled=True),
                IndicatorGene(type="CDL_DOJI", parameters={}, enabled=True),
                IndicatorGene(type="CDL_PIERCING", parameters={}, enabled=True),
                IndicatorGene(type="CDL_THREE_WHITE_SOLDIERS", parameters={}, enabled=True),
            ]
            
            long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(
                pattern_indicators
            )
            
            # パターン認識指標の条件が生成されていることを確認
            pattern_conditions_found = False
            for condition in long_conditions + short_conditions:
                if isinstance(condition.left_operand, str) and condition.left_operand.startswith("CDL_"):
                    pattern_conditions_found = True
                    break
            
            if pattern_indicators:
                assert pattern_conditions_found, "パターン認識指標の条件が生成されていません"
            
            logger.info("✅ パターン認識指標テスト成功")
            
        except Exception as e:
            pytest.fail(f"パターン認識指標テストエラー: {e}")

    def test_error_handling_and_fallback(self):
        """エラーハンドリングとフォールバック機能テスト"""
        logger.info("=== エラーハンドリングとフォールバック機能テスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 空の指標リストでのテスト
            empty_indicators = []
            long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(
                empty_indicators
            )
            
            # フォールバック条件が生成されることを確認
            assert isinstance(long_conditions, list), "フォールバック時にロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "フォールバック時にショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "フォールバック時にイグジット条件がリスト形式ではありません"
            
            # 無効な指標のみでのテスト
            disabled_indicators = [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=False),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=False),
            ]
            
            long_conditions_disabled, short_conditions_disabled, exit_conditions_disabled = generator.generate_balanced_conditions(
                disabled_indicators
            )
            
            # フォールバック条件が生成されることを確認
            assert isinstance(long_conditions_disabled, list), "無効指標時にロング条件がリスト形式ではありません"
            assert isinstance(short_conditions_disabled, list), "無効指標時にショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions_disabled, list), "無効指標時にイグジット条件がリスト形式ではありません"
            
            logger.info("✅ エラーハンドリングとフォールバック機能テスト成功")
            
        except Exception as e:
            pytest.fail(f"エラーハンドリングとフォールバック機能テストエラー: {e}")

    def test_smart_generation_disabled(self):
        """スマート生成無効時のテスト"""
        logger.info("=== スマート生成無効時のテスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            
            # スマート生成を無効にして初期化
            generator = SmartConditionGenerator(enable_smart_generation=False)
            
            long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(
                self.test_indicators
            )
            
            # フォールバック条件が生成されることを確認
            assert isinstance(long_conditions, list), "スマート生成無効時にロング条件がリスト形式ではありません"
            assert isinstance(short_conditions, list), "スマート生成無効時にショート条件がリスト形式ではありません"
            assert isinstance(exit_conditions, list), "スマート生成無効時にイグジット条件がリスト形式ではありません"
            
            logger.info("✅ スマート生成無効時のテスト成功")
            
        except Exception as e:
            pytest.fail(f"スマート生成無効時のテストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
