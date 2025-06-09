#!/usr/bin/env python3
"""
自動戦略生成機能のデバッグテスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_debug_ga():
    """GA機能をデバッグ"""
    print("🔍 自動戦略生成機能 デバッグテスト")
    print("=" * 60)
    
    try:
        # 1. 必要なモジュールのインポート
        print("1. モジュールインポート中...")
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition, decode_list_to_gene
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
        from app.core.services.backtest_service import BacktestService
        from app.core.services.backtest_data_service import BacktestDataService
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.connection import SessionLocal
        print("  ✅ インポート完了")
        
        # 2. 手動で戦略遺伝子を作成
        print("\n2. 手動戦略遺伝子作成中...")
        manual_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ],
            exit_conditions=[
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ]
        )
        print(f"  ✅ 手動戦略作成: {len(manual_gene.indicators)}個の指標")
        
        # 3. 戦略ファクトリーテスト
        print("\n3. 戦略ファクトリーテスト中...")
        factory = StrategyFactory()
        
        # 妥当性チェック
        is_valid, errors = factory.validate_gene(manual_gene)
        print(f"  妥当性チェック: {is_valid}")
        if not is_valid:
            print(f"  エラー: {errors}")
        
        # 戦略クラス生成
        try:
            strategy_class = factory.create_strategy_class(manual_gene)
            print(f"  ✅ 戦略クラス生成成功: {strategy_class.__name__}")
        except Exception as e:
            print(f"  ❌ 戦略クラス生成失敗: {e}")
            return False
        
        # 4. バックテストサービステスト
        print("\n4. バックテストサービステスト中...")
        db = SessionLocal()
        try:
            ohlcv_repo = OHLCVRepository(db)
            data_service = BacktestDataService(ohlcv_repo)
            backtest_service = BacktestService(data_service)
            
            # バックテスト設定
            backtest_config = {
                "strategy_name": "Debug_Test_Strategy",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-04-09",
                "initial_capital": 100000,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "GENERATED_TEST",
                    "parameters": {"strategy_gene": manual_gene.to_dict()},
                },
            }
            
            print(f"  バックテスト設定: {backtest_config['symbol']} {backtest_config['timeframe']}")
            
            # バックテスト実行
            try:
                result = backtest_service.run_backtest(backtest_config)
                print(f"  ✅ バックテスト実行成功")
                
                # 結果の詳細表示
                if "performance_metrics" in result:
                    metrics = result["performance_metrics"]
                    print(f"    総リターン: {metrics.get('total_return', 0):.2%}")
                    print(f"    取引回数: {metrics.get('total_trades', 0)}")
                    print(f"    勝率: {metrics.get('win_rate', 0):.2%}")
                    print(f"    シャープレシオ: {metrics.get('sharpe_ratio', 0):.4f}")
                    print(f"    最大ドローダウン: {metrics.get('max_drawdown', 0):.2%}")
                else:
                    print(f"    ⚠️ performance_metricsが見つかりません")
                    print(f"    結果キー: {list(result.keys())}")
                
            except Exception as e:
                print(f"  ❌ バックテスト実行失敗: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        finally:
            db.close()
        
        # 5. デコード機能テスト
        print("\n5. デコード機能テスト中...")
        
        # ランダムな数値リストを生成
        import random
        random_list = [random.uniform(0, 1) for _ in range(16)]
        print(f"  ランダムリスト: {random_list[:5]}...")
        
        # デコード
        decoded_gene = decode_list_to_gene(random_list)
        print(f"  デコード結果:")
        print(f"    指標数: {len(decoded_gene.indicators)}")
        for i, indicator in enumerate(decoded_gene.indicators):
            print(f"      {i+1}. {indicator.type} - {indicator.parameters}")
        
        print(f"    エントリー条件数: {len(decoded_gene.entry_conditions)}")
        for i, condition in enumerate(decoded_gene.entry_conditions):
            print(f"      {i+1}. {condition}")
        
        print(f"    エグジット条件数: {len(decoded_gene.exit_conditions)}")
        for i, condition in enumerate(decoded_gene.exit_conditions):
            print(f"      {i+1}. {condition}")
        
        # 6. デコードされた戦略のテスト
        print("\n6. デコードされた戦略のテスト中...")
        if decoded_gene.indicators:
            try:
                decoded_strategy_class = factory.create_strategy_class(decoded_gene)
                print(f"  ✅ デコード戦略クラス生成成功")
                
                # バックテスト実行
                decoded_config = backtest_config.copy()
                decoded_config["strategy_name"] = "Decoded_Test_Strategy"
                decoded_config["strategy_config"]["parameters"]["strategy_gene"] = decoded_gene.to_dict()
                
                decoded_result = backtest_service.run_backtest(decoded_config)
                print(f"  ✅ デコード戦略バックテスト成功")
                
                if "performance_metrics" in decoded_result:
                    metrics = decoded_result["performance_metrics"]
                    print(f"    総リターン: {metrics.get('total_return', 0):.2%}")
                    print(f"    取引回数: {metrics.get('total_trades', 0)}")
                else:
                    print(f"    ⚠️ performance_metricsが見つかりません")
                
            except Exception as e:
                print(f"  ❌ デコード戦略テスト失敗: {e}")
        else:
            print(f"  ⚠️ デコードされた戦略に指標がありません")
        
        print("\n✅ デバッグテスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ デバッグテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_debug_ga()
