#!/usr/bin/env python3
"""
指標計算テスト - 戦略ファクトリー統合版
"""

from app.services.auto_strategy.services.indicator_service import (
    IndicatorCalculator,
)
from app.services.auto_strategy.models.gene_strategy import IndicatorGene, StrategyGene
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
import pandas as pd
import numpy as np

print("=== 戦略ファクトリー統合テスト ===")

# テストデータ作成
data = type("Data", (), {})()
df = pd.DataFrame(
    {
        "Open": np.linspace(50, 60, 100),
        "High": np.linspace(51, 61, 100),
        "Low": np.linspace(49, 59, 100),
        "Close": np.linspace(50, 60, 100),
        "Volume": np.full(100, 1000),
    }
)
data.df = df

# 単一SMA指標のみを持つ戦略遺伝子作成
from app.services.auto_strategy.models.gene_strategy import Condition
from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)

sma_gene = IndicatorGene(type="SMA", parameters={"period": 14}, enabled=True)

# 最小限の条件を追加（SMAを使った簡単な条件）
dummy_condition = Condition(left_operand="SMA", operator=">", right_operand="close")

# ポジションサイジング遺伝子を作成
position_sizing_gene = PositionSizingGene(
    method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1
)

strategy_gene = StrategyGene(
    indicators=[sma_gene],
    long_entry_conditions=[dummy_condition],
    short_entry_conditions=[],
    exit_conditions=[dummy_condition],  # 同じ条件を使用
    risk_management={"stop_loss": 0.02, "take_profit": 0.04},
    position_sizing_gene=position_sizing_gene,
)

print(f"作成した戦略遺伝子: {strategy_gene.indicators[0].type}")

# 戦略ファクトリーでクラス作成
factory = StrategyFactory()
try:
    strategy_class = factory.create_strategy_class(strategy_gene)
    print("戦略クラス作成成功")

    # backtesting.pyシミュレーション
    print("backtesting.pyシミュレーション開始...")

    # backtesting.pyのBacktestクラスをシミュレート
    import backtesting

    # 実際のBacktestを使用してテスト
    bt = backtesting.Backtest(data.df, strategy_class)

    print("Backtest作成成功")

    # 戦略インスタンスを取得（backtesting内部で作成される）
    # 実際のbacktesting環境では、init()は自動的に呼ばれる

    # 短期間のバックテストを実行して指標登録を確認
    try:
        # 戦略遺伝子をパラメータとして渡してテスト実行
        result = bt.run(strategy_gene=strategy_gene)
        print("✅ バックテスト実行成功")

        # バックテスト結果の確認
        print(f"バックテスト結果: {result}")
        print(f"総リターン: {result.get('Return [%]', 'N/A')}")
        print(f"取引数: {result.get('# Trades', 'N/A')}")

        # 戦略インスタンスを取得
        strategy_instance = bt._strategy

        # 詳細な属性確認
        all_attrs = dir(strategy_instance)
        public_attrs = [attr for attr in all_attrs if not attr.startswith("_")]
        private_attrs = [
            attr
            for attr in all_attrs
            if attr.startswith("_") and not attr.startswith("__")
        ]

        print(f"パブリック属性: {public_attrs}")
        print(f"プライベート属性: {private_attrs}")

        # 戦略遺伝子確認
        if hasattr(strategy_instance, "strategy_gene"):
            print(f"戦略遺伝子存在: {strategy_instance.strategy_gene}")
            if hasattr(strategy_instance.strategy_gene, "indicators"):
                print(
                    f"指標遺伝子: {[ind.type for ind in strategy_instance.strategy_gene.indicators]}"
                )

        # SMA確認
        if hasattr(strategy_instance, "SMA"):
            print(f"✅ SMA登録成功: {type(strategy_instance.SMA)}")
            try:
                sma_value = strategy_instance.SMA
                print(
                    f"SMA値の一部: {sma_value[-5:] if hasattr(sma_value, '__getitem__') else sma_value}"
                )
            except Exception as e:
                print(f"SMA値取得エラー: {e}")
        else:
            print("❌ SMA登録失敗")

        # I()メソッドで登録された指標を確認
        print("\n=== I()メソッドで登録された指標確認 ===")
        for attr_name in public_attrs:
            attr_value = getattr(strategy_instance, attr_name)
            if hasattr(attr_value, "__call__") and attr_name not in [
                "buy",
                "sell",
                "init",
                "next",
                "I",
            ]:
                print(f"関数属性: {attr_name} = {attr_value}")
            elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                print(
                    f"配列属性: {attr_name} = {type(attr_value)} (長さ: {len(attr_value)})"
                )
            elif hasattr(attr_value, "__len__") and not isinstance(attr_value, str):
                try:
                    print(
                        f"配列風属性: {attr_name} = {type(attr_value)} (長さ: {len(attr_value)})"
                    )
                except:
                    print(f"その他属性: {attr_name} = {type(attr_value)}")

        # Strategy.I()メソッドの詳細テスト
        print("\n=== Strategy.I()メソッド詳細テスト ===")

        # I()メソッドの引数を調査
        import inspect

        i_method = strategy_instance.I
        sig = inspect.signature(i_method)
        print(f"I()メソッドのシグネチャ: {sig}")
        print(f"I()メソッドのパラメータ: {list(sig.parameters.keys())}")

        try:
            # 手動でI()メソッドを使って指標を登録してみる
            import numpy as np

            test_data = np.array([1, 2, 3, 4, 5])

            def test_indicator_func():
                return test_data

            # 様々な引数パターンでテスト
            print("\n--- パターン1: func のみ ---")
            try:
                test_result1 = strategy_instance.I(test_indicator_func)
                print(f"成功: {type(test_result1)}")
            except Exception as e1:
                print(f"失敗: {e1}")

            print("\n--- パターン2: func + name ---")
            try:
                test_result2 = strategy_instance.I(
                    test_indicator_func, name="TEST_INDICATOR"
                )
                print(f"成功: {type(test_result2)}")
            except Exception as e2:
                print(f"失敗: {e2}")

            print("\n--- パターン3: 引数なし ---")
            try:
                test_result3 = strategy_instance.I()
                print(f"成功: {type(test_result3)}")
            except Exception as e3:
                print(f"失敗: {e3}")

        except Exception as e:
            print(f"Strategy.I()テストエラー: {e}")
            import traceback

            traceback.print_exc()

        # 手動でSMAを確認
        print("\n=== 手動SMA確認 ===")
        try:
            manual_sma = getattr(strategy_instance, "SMA", None)
            print(f"手動SMA取得: {manual_sma}")
            if manual_sma is not None:
                print(f"手動SMA型: {type(manual_sma)}")
        except Exception as e:
            print(f"手動SMA取得エラー: {e}")

    except Exception as e:
        print(f"バックテスト実行エラー: {e}")
        import traceback

        traceback.print_exc()

except Exception as e:
    print(f"戦略作成失敗: {e}")
    import traceback

    traceback.print_exc()

print("\n=== 直接IndicatorCalculatorテスト ===")


# 偽の戦略インスタンス作成（backtesting.pyのI()メソッドをシミュレート）
class FakeStrategy:
    def __init__(self, data):
        self.data = data

    def I(
        self,
        func,
        *args,
        name=None,
        plot=True,
        overlay=None,
        color=None,
        scatter=False,
        **kwargs,
    ):
        """backtesting.pyのStrategy.I()メソッドをシミュレート"""
        _ = name, plot, overlay, color, scatter  # 未使用パラメータ
        return func(*args, **kwargs)


strategy = FakeStrategy(data)
calc = IndicatorCalculator()

# SMAテスト
print("=== SMA テスト ===")
try:
    calc.init_indicator(sma_gene, strategy)
    print(f'SMA登録後の属性確認: hasattr(strategy, "SMA") = {hasattr(strategy, "SMA")}')
    if hasattr(strategy, "SMA"):
        print(f"SMA値: {strategy.SMA}")
except Exception as e:
    print(f"SMA初期化失敗: {e}")
    import traceback

    traceback.print_exc()
