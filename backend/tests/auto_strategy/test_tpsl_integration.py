"""
TP/SL整合性テスト

SmartConditionGeneratorとTP/SL自動設定機能との整合性を確認
ショートオーダーの検証条件（TP < LIMIT < SL）の遵守を検証
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory


class TestTPSLIntegration:
    """TP/SL整合性テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.smart_generator = SmartConditionGenerator(enable_smart_generation=True)
        self.tpsl_calculator = TPSLCalculator()

    def create_test_data(self):
        """テスト用データを作成"""
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(100)],
            'High': [101 + i * 0.1 for i in range(100)],
            'Low': [99 + i * 0.1 for i in range(100)],
            'Close': [100.5 + i * 0.1 for i in range(100)],
            'Volume': [1000] * 100
        }, index=dates)
        return data

    def test_tpsl_gene_integration(self):
        """TP/SL遺伝子との統合テスト"""
        # SmartConditionGeneratorで条件を生成
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.smart_generator.generate_balanced_conditions(indicators)

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_reward_ratio=2.0,
            enabled=True
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id="test_tpsl_integration",
            indicators=indicators,
            entry_conditions=long_conds,  # 後方互換性
            long_entry_conditions=long_conds,
            short_entry_conditions=short_conds,
            exit_conditions=exit_conds,
            tpsl_gene=tpsl_gene,
            risk_management={
                "stop_loss": 0.03,
                "take_profit": 0.06,
                "position_size": 0.1
            }
        )

        # TP/SL遺伝子が有効な場合、exit_conditionsは空であることを確認
        assert len(exit_conds) == 0, "TP/SL遺伝子が有効な場合、exit_conditionsは空である必要があります"

        # TP/SL遺伝子が正しく設定されていることを確認
        assert strategy_gene.tpsl_gene.enabled is True
        assert strategy_gene.tpsl_gene.stop_loss_pct == 0.03
        assert strategy_gene.tpsl_gene.take_profit_pct == 0.06

    def test_short_order_validation(self):
        """ショートオーダーの検証条件（TP < LIMIT < SL）テスト"""
        current_price = 100.0

        # ショートポジション用のTP/SL計算
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,  # 3%
            take_profit_pct=0.02,  # 2%
            enabled=True
        )

        # ショートポジション（position_direction = -1.0）
        sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_prices(
            current_price=current_price,
            stop_loss_pct=tpsl_gene.stop_loss_pct,
            take_profit_pct=tpsl_gene.take_profit_pct,
            gene=None,
            risk_management={},
            position_direction=-1.0  # ショートポジション
        )

        # ショートオーダーの検証条件：TP < LIMIT < SL
        limit_price = current_price  # エントリー価格

        print(f"ショートオーダー検証:")
        print(f"  TP: {tp_price}")
        print(f"  LIMIT: {limit_price}")
        print(f"  SL: {sl_price}")

        # ショートポジションの場合：
        # - Take Profit は現在価格より低い（利益確定）
        # - Stop Loss は現在価格より高い（損失限定）
        assert tp_price < limit_price, f"TP ({tp_price}) < LIMIT ({limit_price}) の条件が満たされていません"
        assert limit_price < sl_price, f"LIMIT ({limit_price}) < SL ({sl_price}) の条件が満たされていません"

        # 全体の条件：TP < LIMIT < SL
        assert tp_price < limit_price < sl_price, "ショートオーダーの検証条件 TP < LIMIT < SL が満たされていません"

    def test_long_order_validation(self):
        """ロングオーダーの検証条件（SL < LIMIT < TP）テスト"""
        current_price = 100.0

        # ロングポジション用のTP/SL計算
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,  # 3%
            take_profit_pct=0.05,  # 5%
            enabled=True
        )

        # ロングポジション（position_direction = 1.0）
        sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_prices(
            current_price=current_price,
            stop_loss_pct=tpsl_gene.stop_loss_pct,
            take_profit_pct=tpsl_gene.take_profit_pct,
            gene=None,
            risk_management={},
            position_direction=1.0  # ロングポジション
        )

        # ロングオーダーの検証条件：SL < LIMIT < TP
        limit_price = current_price  # エントリー価格

        print(f"ロングオーダー検証:")
        print(f"  SL: {sl_price}")
        print(f"  LIMIT: {limit_price}")
        print(f"  TP: {tp_price}")

        # ロングポジションの場合：
        # - Stop Loss は現在価格より低い（損失限定）
        # - Take Profit は現在価格より高い（利益確定）
        assert sl_price < limit_price, f"SL ({sl_price}) < LIMIT ({limit_price}) の条件が満たされていません"
        assert limit_price < tp_price, f"LIMIT ({limit_price}) < TP ({tp_price}) の条件が満たされていません"

        # 全体の条件：SL < LIMIT < TP
        assert sl_price < limit_price < tp_price, "ロングオーダーの検証条件 SL < LIMIT < TP が満たされていません"

    def test_risk_management_integration(self):
        """資金管理ロジックとの統合テスト"""
        # SmartConditionGeneratorで条件を生成
        indicators = [
            IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.smart_generator.generate_balanced_conditions(indicators)

        # 資金管理設定
        risk_management = {
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "position_size": 0.1,
            "max_risk_per_trade": 0.02
        }

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=risk_management["stop_loss"],
            take_profit_pct=risk_management["take_profit"],
            risk_reward_ratio=2.0,
            enabled=True
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id="test_risk_management",
            indicators=indicators,
            entry_conditions=long_conds,
            long_entry_conditions=long_conds,
            short_entry_conditions=short_conds,
            exit_conditions=exit_conds,
            tpsl_gene=tpsl_gene,
            risk_management=risk_management
        )

        # 資金管理設定が正しく統合されていることを確認
        assert strategy_gene.risk_management["stop_loss"] == tpsl_gene.stop_loss_pct
        assert strategy_gene.risk_management["take_profit"] == tpsl_gene.take_profit_pct
        assert strategy_gene.risk_management["position_size"] == 0.1

        # TP/SL計算が資金管理設定と整合していることを確認
        current_price = 100.0
        sl_price, tp_price = self.tpsl_calculator.calculate_tpsl_prices(
            current_price=current_price,
            stop_loss_pct=tpsl_gene.stop_loss_pct,
            take_profit_pct=tpsl_gene.take_profit_pct,
            gene=strategy_gene,
            risk_management=risk_management,
            position_direction=1.0
        )

        # 計算されたTP/SLが期待値と一致することを確認
        expected_sl = current_price * (1 - risk_management["stop_loss"])
        expected_tp = current_price * (1 + risk_management["take_profit"])

        assert abs(sl_price - expected_sl) < 0.01, f"SL価格が期待値と異なります: {sl_price} vs {expected_sl}"
        assert abs(tp_price - expected_tp) < 0.01, f"TP価格が期待値と異なります: {tp_price} vs {expected_tp}"

    def test_strategy_factory_integration(self):
        """StrategyFactoryとの統合テスト"""
        # SmartConditionGeneratorで条件を生成
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="ADX", parameters={"period": 14}, enabled=True)
        ]

        long_conds, short_conds, exit_conds = self.smart_generator.generate_balanced_conditions(indicators)

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            enabled=True
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            id="test_strategy_factory",
            indicators=indicators,
            entry_conditions=long_conds,
            long_entry_conditions=long_conds,
            short_entry_conditions=short_conds,
            exit_conditions=exit_conds,
            tpsl_gene=tpsl_gene,
            risk_management={
                "stop_loss": 0.03,
                "take_profit": 0.06,
                "position_size": 0.1
            }
        )

        # StrategyFactoryで戦略クラスを作成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)

        # 戦略クラスが正常に作成されることを確認
        assert strategy_class is not None
        assert hasattr(strategy_class, 'next')
        assert hasattr(strategy_class, '_check_long_entry_conditions')
        assert hasattr(strategy_class, '_check_short_entry_conditions')
        assert hasattr(strategy_class, '_check_exit_conditions')

        # 戦略遺伝子が正しく設定されていることを確認
        test_data = self.create_test_data()
        strategy_instance = strategy_class()
        strategy_instance.gene = strategy_gene

        # TP/SL遺伝子が有効な場合、exit条件チェックがスキップされることを確認
        # （実際のバックテストデータが必要なため、ここでは基本的な検証のみ）
        assert strategy_instance.gene.tpsl_gene.enabled is True

    def test_multiple_strategies_consistency(self):
        """複数戦略での一貫性テスト"""
        # 異なる指標セットで複数の戦略を生成し、TP/SL整合性を確認
        test_cases = [
            [IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)],
            [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)],
            [IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)],
            [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]
        ]

        for i, indicators in enumerate(test_cases):
            # SmartConditionGeneratorで条件を生成
            long_conds, short_conds, exit_conds = self.smart_generator.generate_balanced_conditions(indicators)

            # TP/SL遺伝子を作成
            tpsl_gene = TPSLGene(
                method=TPSLMethod.FIXED_PERCENTAGE,
                stop_loss_pct=0.03,
                take_profit_pct=0.05,
                enabled=True
            )

            # 戦略遺伝子を作成
            strategy_gene = StrategyGene(
                id=f"test_consistency_{i}",
                indicators=indicators,
                entry_conditions=long_conds,
                long_entry_conditions=long_conds,
                short_entry_conditions=short_conds,
                exit_conditions=exit_conds,
                tpsl_gene=tpsl_gene,
                risk_management={
                    "stop_loss": 0.03,
                    "take_profit": 0.05,
                    "position_size": 0.1
                }
            )

            # 基本的な整合性チェック
            assert len(long_conds) > 0, f"戦略 {i}: ロング条件が生成されませんでした"
            assert len(short_conds) > 0, f"戦略 {i}: ショート条件が生成されませんでした"
            assert len(exit_conds) == 0, f"戦略 {i}: TP/SL有効時はexit条件は空である必要があります"
            assert strategy_gene.tpsl_gene.enabled is True, f"戦略 {i}: TP/SL遺伝子が有効でありません"

            # TP/SL価格計算の整合性チェック
            current_price = 100.0

            # ロングポジション
            sl_long, tp_long = self.tpsl_calculator.calculate_tpsl_prices(
                current_price=current_price,
                stop_loss_pct=tpsl_gene.stop_loss_pct,
                take_profit_pct=tpsl_gene.take_profit_pct,
                gene=strategy_gene,
                risk_management=strategy_gene.risk_management,
                position_direction=1.0
            )

            # ショートポジション
            sl_short, tp_short = self.tpsl_calculator.calculate_tpsl_prices(
                current_price=current_price,
                stop_loss_pct=tpsl_gene.stop_loss_pct,
                take_profit_pct=tpsl_gene.take_profit_pct,
                gene=strategy_gene,
                risk_management=strategy_gene.risk_management,
                position_direction=-1.0
            )

            # ロング: SL < LIMIT < TP
            assert sl_long < current_price < tp_long, f"戦略 {i}: ロングTP/SL順序が正しくありません"

            # ショート: TP < LIMIT < SL
            assert tp_short < current_price < sl_short, f"戦略 {i}: ショートTP/SL順序が正しくありません"


if __name__ == "__main__":
    test = TestTPSLIntegration()
    test.setup_method()

    print("=== TP/SL整合性テスト開始 ===")

    try:
        test.test_tpsl_gene_integration()
        print("✅ TP/SL遺伝子統合テスト成功")

        test.test_short_order_validation()
        print("✅ ショートオーダー検証テスト成功")

        test.test_long_order_validation()
        print("✅ ロングオーダー検証テスト成功")

        test.test_risk_management_integration()
        print("✅ 資金管理統合テスト成功")

        test.test_strategy_factory_integration()
        print("✅ StrategyFactory統合テスト成功")

        test.test_multiple_strategies_consistency()
        print("✅ 複数戦略一貫性テスト成功")

        print("\n🎉 全てのTP/SL整合性テストが成功しました！")

    except Exception as e:
        print(f"\n🚨 TP/SL整合性テストでエラーが発生しました: {e}")
        raise