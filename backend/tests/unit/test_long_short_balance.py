"""
ロング・ショートバランステスト

戦略がロングオンリーになっていないかを確認します。
"""

import pytest
import sys
import os
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig


def create_test_data():
    """テスト用データを作成"""
    dates = pd.date_range('2020-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'Open': [100 + i * 0.1 for i in range(100)],
        'High': [101 + i * 0.1 for i in range(100)],
        'Low': [99 + i * 0.1 for i in range(100)],
        'Close': [100.5 + i * 0.1 for i in range(100)],
        'Volume': [1000] * 100
    }, index=dates)
    return data


def test_long_short_condition_evaluation():
    """ロング・ショート条件評価テスト"""
    print("\n=== ロング・ショート条件評価テスト ===")
    
    # TP/SL遺伝子を作成
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        enabled=True
    )

    # RSIベースの戦略を作成
    strategy_gene = StrategyGene(
        id="test_rsi_strategy",
        indicators=[
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="RSI_14", operator="<", right_operand=50)
        ],
        long_entry_conditions=[
            Condition(left_operand="RSI_14", operator="<", right_operand=30)  # 売られすぎでロング
        ],
        short_entry_conditions=[
            Condition(left_operand="RSI_14", operator=">", right_operand=70)  # 買われすぎでショート
        ],
        exit_conditions=[],
        tpsl_gene=tpsl_gene,
        risk_management={"position_size": 0.1}
    )
    
    print(f"戦略ID: {strategy_gene.id}")
    print(f"ロング条件: {[f'{c.left_operand} {c.operator} {c.right_operand}' for c in strategy_gene.long_entry_conditions]}")
    print(f"ショート条件: {[f'{c.left_operand} {c.operator} {c.right_operand}' for c in strategy_gene.short_entry_conditions]}")
    print(f"ロング・ショート分離: {strategy_gene.has_long_short_separation()}")
    
    # StrategyFactoryで戦略クラスを作成
    factory = StrategyFactory()
    strategy_class = factory.create_strategy_class(strategy_gene)
    
    # テストデータを作成
    data = create_test_data()
    
    # RSI計算（簡易版）
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    data["RSI_14"] = calculate_rsi(data["Close"])
    
    # 戦略インスタンスを作成
    strategy_instance = strategy_class(data=data, params={})
    strategy_instance.indicators = {"RSI_14": data["RSI_14"]}
    
    # 異なるRSI値でテスト
    test_cases = [
        (25, "売られすぎ（ロング期待）", True, False),
        (75, "買われすぎ（ショート期待）", False, True),
        (50, "中立", False, False),
    ]
    
    for rsi_value, description, expected_long, expected_short in test_cases:
        print(f"\n--- RSI={rsi_value} ({description}) ---")
        
        # RSI値を設定
        data.loc[data.index[-1], "RSI_14"] = rsi_value
        strategy_instance.indicators = {"RSI_14": data["RSI_14"]}
        
        # 条件評価
        long_result = strategy_instance._check_long_entry_conditions()
        short_result = strategy_instance._check_short_entry_conditions()
        
        print(f"ロング条件結果: {long_result} (期待値: {expected_long})")
        print(f"ショート条件結果: {short_result} (期待値: {expected_short})")
        
        # アサーション
        assert long_result == expected_long, f"RSI={rsi_value}でロング条件の結果が期待値と異なります"
        assert short_result == expected_short, f"RSI={rsi_value}でショート条件の結果が期待値と異なります"
        
        print(f"✅ RSI={rsi_value}: 期待通りの結果")
    
    print("✅ ロング・ショート条件評価テスト成功")


def test_random_strategy_long_short_balance():
    """ランダム戦略のロング・ショートバランステスト"""
    print("\n=== ランダム戦略のロング・ショートバランステスト ===")
    
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    factory = StrategyFactory()
    
    long_only_count = 0
    short_only_count = 0
    both_count = 0
    neither_count = 0
    total_strategies = 10
    
    for i in range(total_strategies):
        print(f"\n--- 戦略 {i+1} ---")
        
        # ランダム戦略を生成
        strategy_gene = generator.generate_random_gene()
        strategy_class = factory.create_strategy_class(strategy_gene)
        
        # テストデータを作成
        data = create_test_data()
        
        # 指標を計算（簡易版）
        if any(ind.type == "RSI" for ind in strategy_gene.indicators):
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50)
            
            data["RSI_14"] = calculate_rsi(data["Close"])
        
        if any(ind.type == "SMA" for ind in strategy_gene.indicators):
            for ind in strategy_gene.indicators:
                if ind.type == "SMA":
                    period = ind.parameters.get("period", 20)
                    data[f"SMA_{period}"] = data["Close"].rolling(window=period).mean()
        
        # 戦略インスタンスを作成
        strategy_instance = strategy_class(data=data, params={})
        
        # 指標を設定
        indicators = {}
        for ind in strategy_gene.indicators:
            if ind.type == "RSI":
                indicators["RSI_14"] = data["RSI_14"]
            elif ind.type == "SMA":
                period = ind.parameters.get("period", 20)
                indicators[f"SMA_{period}"] = data[f"SMA_{period}"]
        
        strategy_instance.indicators = indicators
        
        # 複数の市場状況でテスト
        long_triggers = 0
        short_triggers = 0
        test_points = 10
        
        for j in range(test_points):
            # データポイントを変更
            idx = data.index[-(j+1)]
            
            # RSI値を変更してテスト
            if "RSI_14" in data.columns:
                # 様々なRSI値でテスト
                rsi_values = [20, 30, 40, 50, 60, 70, 80]
                rsi_value = rsi_values[j % len(rsi_values)]
                data.loc[idx, "RSI_14"] = rsi_value
                strategy_instance.indicators["RSI_14"] = data["RSI_14"]
            
            # 条件評価
            long_result = strategy_instance._check_long_entry_conditions()
            short_result = strategy_instance._check_short_entry_conditions()
            
            if long_result:
                long_triggers += 1
            if short_result:
                short_triggers += 1
        
        print(f"ロングトリガー: {long_triggers}/{test_points}")
        print(f"ショートトリガー: {short_triggers}/{test_points}")
        
        # 分類
        if long_triggers > 0 and short_triggers == 0:
            long_only_count += 1
            print("🔴 ロングオンリー戦略")
        elif short_triggers > 0 and long_triggers == 0:
            short_only_count += 1
            print("🔵 ショートオンリー戦略")
        elif long_triggers > 0 and short_triggers > 0:
            both_count += 1
            print("🟢 ロング・ショート両対応戦略")
        else:
            neither_count += 1
            print("⚪ 条件が満たされない戦略")
    
    print(f"\n=== 結果サマリー ===")
    print(f"ロングオンリー: {long_only_count}/{total_strategies} ({long_only_count/total_strategies*100:.1f}%)")
    print(f"ショートオンリー: {short_only_count}/{total_strategies} ({short_only_count/total_strategies*100:.1f}%)")
    print(f"ロング・ショート両対応: {both_count}/{total_strategies} ({both_count/total_strategies*100:.1f}%)")
    print(f"条件が満たされない: {neither_count}/{total_strategies} ({neither_count/total_strategies*100:.1f}%)")
    
    # 問題の判定
    if long_only_count > total_strategies * 0.7:
        print("🚨 警告: ロングオンリー戦略が多すぎます！")
        return False
    elif both_count < total_strategies * 0.3:
        print("🚨 警告: ロング・ショート両対応戦略が少なすぎます！")
        return False
    else:
        print("✅ ロング・ショートバランスは適切です")
        return True


def test_specific_condition_logic():
    """特定条件ロジックテスト"""
    print("\n=== 特定条件ロジックテスト ===")
    
    # TP/SL遺伝子を作成
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        enabled=True
    )

    # SMAベースの戦略を作成
    strategy_gene = StrategyGene(
        id="test_sma_strategy",
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")  # 価格が移動平均上でロング
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20")  # 価格が移動平均下でショート
        ],
        exit_conditions=[],
        tpsl_gene=tpsl_gene,
        risk_management={"position_size": 0.1}
    )
    
    print(f"戦略ID: {strategy_gene.id}")
    print(f"ロング条件: {[f'{c.left_operand} {c.operator} {c.right_operand}' for c in strategy_gene.long_entry_conditions]}")
    print(f"ショート条件: {[f'{c.left_operand} {c.operator} {c.right_operand}' for c in strategy_gene.short_entry_conditions]}")
    
    # StrategyFactoryで戦略クラスを作成
    factory = StrategyFactory()
    strategy_class = factory.create_strategy_class(strategy_gene)
    
    # テストデータを作成
    data = create_test_data()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    
    # 戦略インスタンスを作成
    strategy_instance = strategy_class(data=data, params={})
    strategy_instance.indicators = {"SMA_20": data["SMA_20"]}
    
    # 異なる価格状況でテスト
    test_cases = [
        (105, 100, "価格が移動平均上（ロング期待）", True, False),
        (95, 100, "価格が移動平均下（ショート期待）", False, True),
        (100, 100, "価格が移動平均と同じ", False, False),
    ]
    
    for close_price, sma_value, description, expected_long, expected_short in test_cases:
        print(f"\n--- {description} ---")
        
        # 価格とSMA値を設定
        data.loc[data.index[-1], "Close"] = close_price
        data.loc[data.index[-1], "SMA_20"] = sma_value
        strategy_instance.indicators = {"SMA_20": data["SMA_20"]}
        
        # 条件評価
        long_result = strategy_instance._check_long_entry_conditions()
        short_result = strategy_instance._check_short_entry_conditions()
        
        print(f"Close={close_price}, SMA={sma_value}")
        print(f"ロング条件結果: {long_result} (期待値: {expected_long})")
        print(f"ショート条件結果: {short_result} (期待値: {expected_short})")
        
        # アサーション
        assert long_result == expected_long, f"Close={close_price}, SMA={sma_value}でロング条件の結果が期待値と異なります"
        assert short_result == expected_short, f"Close={close_price}, SMA={sma_value}でショート条件の結果が期待値と異なります"
        
        print(f"✅ {description}: 期待通りの結果")
    
    print("✅ 特定条件ロジックテスト成功")


def test_smart_condition_generator_balance():
    """SmartConditionGeneratorのロング・ショートバランステスト"""
    print("\n=== SmartConditionGeneratorバランステスト ===")

    generator = SmartConditionGenerator(enable_smart_generation=True)

    # 様々な指標の組み合わせでテスト
    test_cases = [
        {
            "name": "異なる指標の組み合わせ",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]
        },
        {
            "name": "時間軸分離戦略",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 7}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 21}, enabled=True)
            ]
        },
        {
            "name": "ボリンジャーバンド特性活用",
            "indicators": [
                IndicatorGene(type="BB", parameters={"period": 20}, enabled=True)
            ]
        },
        {
            "name": "ADX特性活用",
            "indicators": [
                IndicatorGene(type="ADX", parameters={"period": 14}, enabled=True)
            ]
        },
        {
            "name": "複合条件戦略",
            "indicators": [
                IndicatorGene(type="CCI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="STOCH", parameters={"period": 14}, enabled=True)
            ]
        }
    ]

    balanced_strategies = 0
    total_strategies = 0

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")

        # 複数回生成してバランスを確認
        for i in range(10):
            long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(test_case['indicators'])

            # 基本的な検証
            assert len(long_conds) > 0, f"ロング条件が生成されませんでした: {test_case['name']}"
            assert len(short_conds) > 0, f"ショート条件が生成されませんでした: {test_case['name']}"

            # 条件の論理的整合性を確認
            is_balanced = True

            # 同一指標での相反条件チェック（計画書で指摘された問題）
            long_indicators = set()
            short_indicators = set()

            for cond in long_conds:
                if "_" in str(cond.left_operand):
                    indicator_name = str(cond.left_operand).split("_")[0]
                    long_indicators.add((indicator_name, cond.operator, cond.right_operand))

            for cond in short_conds:
                if "_" in str(cond.left_operand):
                    indicator_name = str(cond.left_operand).split("_")[0]
                    short_indicators.add((indicator_name, cond.operator, cond.right_operand))

            # 同一指標で同じ条件が使用されていないかチェック
            overlapping_conditions = long_indicators.intersection(short_indicators)
            if overlapping_conditions:
                print(f"⚠️  同一条件が検出されました: {overlapping_conditions}")
                is_balanced = False

            # ADXの誤用チェック（同じ条件をロング・ショート両方に設定）
            adx_conditions = []
            for cond in long_conds + short_conds:
                if "ADX_" in str(cond.left_operand):
                    adx_conditions.append((cond.operator, cond.right_operand))

            if len(adx_conditions) > 1 and len(set(adx_conditions)) == 1:
                print(f"⚠️  ADXで同一条件が検出されました: {adx_conditions}")
                is_balanced = False

            total_strategies += 1
            if is_balanced:
                balanced_strategies += 1
                print(f"✅ 戦略 {i+1}: バランス良好")
            else:
                print(f"❌ 戦略 {i+1}: バランス問題あり")

    # バランス率を計算
    balance_rate = (balanced_strategies / total_strategies) * 100 if total_strategies > 0 else 0

    print(f"\n=== SmartConditionGeneratorバランステスト結果 ===")
    print(f"総戦略数: {total_strategies}")
    print(f"バランス良好戦略数: {balanced_strategies}")
    print(f"バランス率: {balance_rate:.1f}%")

    # 目標値（60%以上）を達成しているかチェック
    target_rate = 60.0
    if balance_rate >= target_rate:
        print(f"🎉 目標達成！ バランス率 {balance_rate:.1f}% >= {target_rate}%")
        return True
    else:
        print(f"🚨 目標未達成！ バランス率 {balance_rate:.1f}% < {target_rate}%")
        return False





if __name__ == "__main__":
    test_long_short_condition_evaluation()
    test_specific_condition_logic()
    balance_result = test_random_strategy_long_short_balance()
    smart_balance_result = test_smart_condition_generator_balance()

    if balance_result and smart_balance_result:
        print("\n🎉 全てのロング・ショートバランステストが成功しました！")
    else:
        print("\n🚨 ロング・ショートバランスに問題があります！")
