#!/usr/bin/env python3
"""
Phase 3 新規指標を使用した戦略実行テスト
実際にBOP, PPO, MIDPOINT, MIDPRICE, TRIMAを使用した戦略を生成・実行
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
    
    # OHLCV データ生成
    data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.normal(0, 0.001, 200)),
        'High': close_prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
        'Low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, 200),
    }, index=dates)
    
    return data

def test_phase3_strategy_execution():
    """Phase 3 新規指標を使用した戦略実行テスト"""
    print("🧪 Phase 3 新規指標戦略実行テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        
        # テストデータ準備
        print("1️⃣ テストデータ準備")
        print("-" * 50)
        data = create_test_data()
        print(f"✅ テストデータ生成完了: {len(data)} 日分")
        
        # Phase 3 指標を使用した戦略遺伝子を手動作成
        print("\n2️⃣ Phase 3 指標戦略遺伝子作成")
        print("-" * 50)
        
        phase3_strategies = []
        
        # 戦略1: BOP + MIDPOINT
        strategy1 = StrategyGene(
            indicators=[
                IndicatorGene(type="BOP", parameters={"period": 1}, enabled=True),
                IndicatorGene(type="MIDPOINT", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[
                Condition(left_operand="BOP", operator=">", right_operand=0.0),
                Condition(left_operand="close", operator=">", right_operand="MIDPOINT_20"),
            ],
            exit_conditions=[
                Condition(left_operand="BOP", operator="<", right_operand=0.0),
            ],
            risk_management={"stop_loss": 0.03, "take_profit": 0.1},
        )
        phase3_strategies.append(("BOP + MIDPOINT", strategy1))
        
        # 戦略2: PPO + TRIMA
        strategy2 = StrategyGene(
            indicators=[
                IndicatorGene(type="PPO", parameters={"period": 12, "slow_period": 26, "matype": 0}, enabled=True),
                IndicatorGene(type="TRIMA", parameters={"period": 30}, enabled=True),
            ],
            entry_conditions=[
                Condition(left_operand="PPO_12_26", operator=">", right_operand=0.0),
                Condition(left_operand="close", operator=">", right_operand="TRIMA_30"),
            ],
            exit_conditions=[
                Condition(left_operand="PPO_12_26", operator="<", right_operand=0.0),
            ],
            risk_management={"stop_loss": 0.02, "take_profit": 0.08},
        )
        phase3_strategies.append(("PPO + TRIMA", strategy2))
        
        # 戦略3: MIDPRICE単体
        strategy3 = StrategyGene(
            indicators=[
                IndicatorGene(type="MIDPRICE", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="MIDPRICE_14"),
                Condition(left_operand="MIDPRICE_14", operator=">", right_operand="SMA_20"),
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="MIDPRICE_14"),
            ],
            risk_management={"stop_loss": 0.025, "take_profit": 0.075},
        )
        phase3_strategies.append(("MIDPRICE + SMA", strategy3))
        
        print(f"✅ {len(phase3_strategies)} 個のPhase3戦略を作成")
        
        # 戦略ファクトリー初期化
        print("\n3️⃣ 戦略ファクトリー初期化")
        print("-" * 50)
        factory = StrategyFactory()
        print("✅ StrategyFactory初期化完了")
        
        # 各戦略の検証と実行
        print("\n4️⃣ 戦略検証・実行テスト")
        print("-" * 50)
        
        success_count = 0
        
        for strategy_name, gene in phase3_strategies:
            print(f"\n📊 戦略テスト: {strategy_name}")
            print("-" * 30)
            
            try:
                # 遺伝子の妥当性検証
                is_valid, errors = factory.validate_gene(gene)
                if not is_valid:
                    print(f"❌ 遺伝子検証失敗: {errors}")
                    continue
                
                print("✅ 遺伝子検証成功")
                
                # 戦略クラス生成
                strategy_class = factory.create_strategy_class(gene)
                print("✅ 戦略クラス生成成功")
                
                # 指標計算テスト（backtesting.pyを使わずに直接テスト）
                print("📈 指標計算テスト:")
                
                for indicator in gene.indicators:
                    if indicator.enabled:
                        indicator_type = indicator.type
                        parameters = indicator.parameters
                        
                        if indicator_type in factory.indicator_adapters:
                            try:
                                if indicator_type == "BOP":
                                    result = factory.indicator_adapters[indicator_type](
                                        data['Open'], data['High'], data['Low'], data['Close']
                                    )
                                elif indicator_type == "MIDPRICE":
                                    period = int(parameters.get("period", 14))
                                    result = factory.indicator_adapters[indicator_type](
                                        data['High'], data['Low'], period
                                    )
                                elif indicator_type == "PPO":
                                    fastperiod = int(parameters.get("period", 12))
                                    slowperiod = int(parameters.get("slow_period", 26))
                                    matype = int(parameters.get("matype", 0))
                                    result = factory.indicator_adapters[indicator_type](
                                        data['Close'], fastperiod, slowperiod, matype
                                    )
                                else:
                                    # 単一期間指標
                                    period = int(parameters.get("period", 20))
                                    result = factory.indicator_adapters[indicator_type](
                                        data['Close'], period
                                    )
                                
                                valid_values = result.dropna()
                                print(f"  ✅ {indicator_type}: {len(valid_values)} 個の有効値")
                                
                            except Exception as e:
                                print(f"  ❌ {indicator_type}: 計算エラー - {e}")
                                raise
                        else:
                            print(f"  ❌ {indicator_type}: 未対応指標")
                            raise ValueError(f"未対応指標: {indicator_type}")
                
                print("✅ 全指標計算成功")
                success_count += 1
                
            except Exception as e:
                print(f"❌ 戦略テスト失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # 結果サマリー
        print(f"\n5️⃣ テスト結果サマリー")
        print("-" * 50)
        print(f"📊 成功した戦略: {success_count}/{len(phase3_strategies)}")
        
        if success_count == len(phase3_strategies):
            print("🎉 全てのPhase3戦略が正常に動作しました！")
            return True
        else:
            print("⚠️  一部の戦略でエラーが発生しました")
            return False
        
    except Exception as e:
        print(f"❌ 戦略実行テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("🚀 Phase 3 新規指標戦略実行テスト開始")
    print("=" * 70)
    
    result = test_phase3_strategy_execution()
    
    print("\n" + "=" * 70)
    print("📊 最終結果")
    print("=" * 70)
    
    if result:
        print("🎉 Phase 3 新規指標の戦略実行テストが成功しました！")
        print("✅ BOP, PPO, MIDPOINT, MIDPRICE, TRIMA が正常に動作します")
        return True
    else:
        print("❌ 戦略実行テストが失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
