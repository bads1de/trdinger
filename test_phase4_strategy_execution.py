#!/usr/bin/env python3
"""
Phase 4 新規指標戦略実行テスト
PLUS_DI, MINUS_DI, ROCP, ROCR, STOCHF指標を使用した戦略の実行テスト
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def create_test_data():
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    base_price = 50000
    returns = np.random.normal(0, 0.02, 200)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, 200)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 200),
    }, index=dates)

def create_phase4_strategy_genes():
    """Phase 4指標を使用した戦略遺伝子を作成"""
    from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
    
    strategies = []
    
    # 戦略1: PLUS_DI + ROCP
    strategy1 = StrategyGene(
        id="phase4_strategy_1",
        indicators=[
            IndicatorGene(
                type="PLUS_DI",
                parameters={"period": 14},
                enabled=True
            ),
            IndicatorGene(
                type="ROCP",
                parameters={"period": 10},
                enabled=True
            )
        ],
        entry_conditions=[
            Condition(
                left_operand="PLUS_DI_14",
                operator=">",
                right_operand=25.0
            )
        ],
        exit_conditions=[
            Condition(
                left_operand="ROCP_10",
                operator="<",
                right_operand=-2.0
            )
        ],
        risk_management={
            "stop_loss": 0.02,
            "take_profit": 0.03
        }
    )
    strategies.append(("PLUS_DI + ROCP", strategy1))
    
    # 戦略2: MINUS_DI + ROCR
    strategy2 = StrategyGene(
        id="phase4_strategy_2",
        indicators=[
            IndicatorGene(
                type="MINUS_DI",
                parameters={"period": 14},
                enabled=True
            ),
            IndicatorGene(
                type="ROCR",
                parameters={"period": 10},
                enabled=True
            )
        ],
        entry_conditions=[
            Condition(
                left_operand="MINUS_DI_14",
                operator="<",
                right_operand=20.0
            )
        ],
        exit_conditions=[
            Condition(
                left_operand="ROCR_10",
                operator=">",
                right_operand=1.02
            )
        ],
        risk_management={
            "stop_loss": 0.02,
            "take_profit": 0.03
        }
    )
    strategies.append(("MINUS_DI + ROCR", strategy2))
    
    # 戦略3: STOCHF + SMA
    strategy3 = StrategyGene(
        id="phase4_strategy_3",
        indicators=[
            IndicatorGene(
                type="STOCHF",
                parameters={"period": 5, "fastd_period": 3, "fastd_matype": 0},
                enabled=True
            ),
            IndicatorGene(
                type="SMA",
                parameters={"period": 20},
                enabled=True
            )
        ],
        entry_conditions=[
            Condition(
                left_operand="STOCHF_K_5_3",
                operator=">",
                right_operand=30.0
            )
        ],
        exit_conditions=[
            Condition(
                left_operand="close",
                operator="<",
                right_operand="SMA_20"
            )
        ],
        risk_management={
            "stop_loss": 0.02,
            "take_profit": 0.03
        }
    )
    strategies.append(("STOCHF + SMA", strategy3))
    
    return strategies

def test_phase4_strategy_execution():
    """Phase 4指標戦略実行テスト"""
    print("🧪 Phase 4 新規指標戦略実行テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        
        # テストデータ準備
        print("1️⃣ テストデータ準備")
        print("-" * 50)
        df = create_test_data()
        print(f"✅ テストデータ生成完了: {len(df)} 日分")
        
        # Phase 4指標戦略遺伝子作成
        print("\n2️⃣ Phase 4 指標戦略遺伝子作成")
        print("-" * 50)
        strategies = create_phase4_strategy_genes()
        print(f"✅ {len(strategies)} 個のPhase4戦略を作成")
        
        # StrategyFactory初期化
        print("\n3️⃣ 戦略ファクトリー初期化")
        print("-" * 50)
        factory = StrategyFactory()
        print("✅ StrategyFactory初期化完了")
        
        # 戦略検証・実行テスト
        print("\n4️⃣ 戦略検証・実行テスト")
        print("-" * 50)
        
        success_count = 0
        
        for strategy_name, gene in strategies:
            print(f"\n📊 戦略テスト: {strategy_name}")
            print("-" * 30)
            
            try:
                # 遺伝子検証
                is_valid, errors = factory.validate_gene(gene)
                if not is_valid:
                    print(f"❌ 遺伝子検証失敗: {errors}")
                    continue
                
                print("✅ 遺伝子検証成功")
                
                # 戦略クラス生成
                strategy_class = factory.create_strategy_class(gene)
                print("✅ 戦略クラス生成成功")
                
                # 指標計算テスト
                print("📈 指標計算テスト:")
                for indicator in gene.indicators:
                    if indicator.enabled:
                        indicator_type = indicator.type
                        parameters = indicator.parameters
                        
                        if indicator_type == "PLUS_DI":
                            result = factory.indicator_adapters[indicator_type](
                                df["high"], df["low"], df["close"], parameters["period"]
                            )
                        elif indicator_type == "MINUS_DI":
                            result = factory.indicator_adapters[indicator_type](
                                df["high"], df["low"], df["close"], parameters["period"]
                            )
                        elif indicator_type in ["ROCP", "ROCR"]:
                            result = factory.indicator_adapters[indicator_type](
                                df["close"], parameters["period"]
                            )
                        elif indicator_type == "STOCHF":
                            result = factory.indicator_adapters[indicator_type](
                                df["high"], df["low"], df["close"], 
                                parameters["period"], parameters["fastd_period"], parameters["fastd_matype"]
                            )
                        elif indicator_type == "SMA":
                            result = factory.indicator_adapters[indicator_type](
                                df["close"], parameters["period"]
                            )
                        
                        if isinstance(result, dict):
                            # STOCHFの場合
                            for key, series in result.items():
                                valid_count = len(series.dropna())
                                print(f"  ✅ {indicator_type}_{key}: {valid_count} 個の有効値")
                        else:
                            # 単一Seriesの場合
                            valid_count = len(result.dropna())
                            print(f"  ✅ {indicator_type}: {valid_count} 個の有効値")
                
                print("✅ 全指標計算成功")
                success_count += 1
                
            except Exception as e:
                print(f"❌ 戦略テスト失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # テスト結果サマリー
        print("\n5️⃣ テスト結果サマリー")
        print("-" * 50)
        print(f"📊 成功した戦略: {success_count}/{len(strategies)}")
        
        if success_count == len(strategies):
            print("🎉 全てのPhase4戦略が正常に動作しました！")
            return True
        elif success_count > 0:
            print("⚠️  一部の戦略でエラーが発生しました")
            return False
        else:
            print("❌ 全ての戦略でエラーが発生しました")
            return False
        
    except Exception as e:
        print(f"❌ Phase 4戦略実行テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("🚀 Phase 4 新規指標戦略実行テスト開始")
    print("=" * 70)
    
    # テスト実行
    test_result = test_phase4_strategy_execution()
    
    print("\n" + "=" * 70)
    print("📊 最終結果")
    print("=" * 70)
    
    if test_result:
        print("🎉 Phase 4 新規指標の戦略実行テストが成功しました！")
        print("✅ PLUS_DI, MINUS_DI, ROCP, ROCR, STOCHF が正常に動作します")
        return True
    else:
        print("❌ 戦略実行テストが失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
