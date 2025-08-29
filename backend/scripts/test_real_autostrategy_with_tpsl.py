#!/usr/bin/env python3
"""
本物のオートストラテジー実行スクリプト
TPSL統合を現実のワークフローで検証
GAを実行し、生成された戦略にTPSLが反映されているか確認
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_and_run_autostrategy() -> Dict[str, Any]:
    """
    本物のAutoStrategyServiceを使ってGAを実行し、TPSL統合を検証

    Returns:
        生成結果とTPSL検証結果を含む辞書
    """
    from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
    from database.connection import SessionLocal
    from app.services.auto_strategy.models.strategy_models import TPSLMethod

    logger.info("=== REAL AUTOSTRATEGY TPSL INTEGRATION TEST ===")
    logger.info("AutoStrategyServiceを使ったGA実行 & TPSL統合検証を開始")

    db = SessionLocal()
    try:
        # AutoStrategyServiceの初期化
        service = AutoStrategyService(db)

        # GA設定 - シンプルな設定でTPSL統合をテスト
        ga_config = {
            "population_size": 5,  # 小さめにしておく
            "generations": 3,      # 短くする
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "tournament_size": 3,
            "max_indicators": 2,  # シンプルに
        }

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "initial_capital": 100000.0,
            "commission_rate": 0.00055
        }

        logger.info("GA設定:")
        logger.info(f"  Population: {ga_config['population_size']}")
        logger.info(f"  Generations: {ga_config['generations']}")
        logger.info(f"  Max Indicators: {ga_config['max_indicators']}")

        logger.info("バックテスト設定:")
        logger.info(f"  Symbol: {backtest_config['symbol']}")
        logger.info(f"  Timeframe: {backtest_config['timeframe']}")
        logger.info(f"  Period: {backtest_config['start_date']} to {backtest_config['end_date']}")

        # GA実行
        logger.info("=== GA実行開始 ===")
        result = service.run_autostrategy_generation(
            ga_config=ga_config,
            backtest_config=backtest_config,
            db=db
        )

        logger.info("=== GA実行完了 ===")

        # 結果の検証
        if isinstance(result, dict) and result.get("best_strategy"):
            strategy = result["best_strategy"]

            logger.info("=== ベストストラテジーTPSL検証 ===")

            # TPSL Geneの確認
            if hasattr(strategy, 'tpsl_gene') and strategy.tpsl_gene:
                tpsl_gene = strategy.tpsl_gene
                logger.info(f"TPSL Method: {getattr(tpsl_gene.method, 'value', tpsl_gene.method) if hasattr(tpsl_gene.method, 'value') else tpsl_gene.method}")
                logger.info(f"TPSL Stop Loss: {tpsl_gene.stop_loss_pct}")
                logger.info(f"TPSL Take Profit: {tpsl_gene.take_profit_pct}")

                if hasattr(tpsl_gene, 'risk_reward_ratio'):
                    logger.info(f"TPSL Risk/Reward Ratio: {tpsl_gene.risk_reward_ratio}")

                # TPSL統合されているかの判定
                tpsl_integrated = True
                if hasattr(tpsl_gene, 'method'):
                    method_valid = str(tpsl_gene.method).replace('TPSLMethod.', '') in ['RISK_REWARD_RATIO', 'STATISTICAL', 'VOLATILITY_BASED', 'FIXED_PERCENTAGE']
                    tpsl_integrated = method_valid

                # TPSL価格計算テスト
                tpsl_verification = test_tpsl_calculation(strategy)
                current_price = 50000.0

                return {
                    "status": "success",
                    "ga_result": {
                        "generations_completed": result.get("generations_completed", 0),
                        "population_size": result.get("population_size", 0),
                        "best_fitness": result.get("best_fitness", 0.0)
                    },
                    "strategy": {
                        "indicators_count": len(strategy.indicators) if hasattr(strategy, 'indicators') else 0,
                        "has_tpsl_gene": hasattr(strategy, 'tpsl_gene') and strategy.tpsl_gene is not None,
                        "tpsl_method": getattr(tpsl_gene.method, 'value', str(tpsl_gene.method)) if hasattr(tpsl_gene.method, 'value') else str(tpsl_gene.method),
                        "tpsl_stop_loss": tpsl_gene.stop_loss_pct,
                        "tpsl_take_profit": tpsl_gene.take_profit_pct,
                        "tpsl_integrated": tpsl_integrated
                    },
                    "tpsl_calculation": tpsl_verification,
                    "verification": {
                        "ga_execution_successful": True,
                        "strategy_generated": True,
                        "tpsl_attached": tpsl_gene is not None,
                        "tpsl_integration_verified": tpsl_integrated and tpsl_verification["status"] == "success",
                        "price_calculation_works": tpsl_verification["status"] == "success"
                    }
                }
            else:
                logger.warning("戦略にTPSL Geneが含まれていません")
                return {
                    "status": "partial_failure",
                    "error": "No TPSL Gene found in strategy",
                    "strategy": {
                        "indicators_count": len(strategy.indicators) if hasattr(strategy, 'indicators') else 0,
                        "has_tpsl_gene": False
                    },
                    "verification": {
                        "tpsl_integration_verified": False
                    }
                }
        else:
            logger.error("GA結果が正しい形式ではありません")
            logger.error(f"結果タイプ: {type(result)}")
            if isinstance(result, dict):
                logger.error(f"結果キー: {list(result.keys())}")
            return {
                "status": "failure",
                "error": "Invalid GA result format",
                "result_type": str(type(result)),
                "verification": {
                    "ga_execution_successful": False
                }
            }

    except Exception as e:
        logger.error(f"オートストラテジー実行エラー: {e}")
        import traceback
        logger.error(f"詳細:\n{traceback.format_exc()}")
        return {
            "status": "failure",
            "error": str(e),
            "verification": {
                "ga_execution_successful": False
            }
        }

    finally:
        db.close()


def test_tpsl_calculation(strategy) -> Dict[str, Any]:
    """
    生成された戦略のTPSL計算をテスト

    Args:
        strategy: 生成されたストラテジー

    Returns:
        TPSL計算結果
    """
    from app.services.auto_strategy.services.tpsl_service import TPSLService

    try:
        current_price = 50000.0
        service = TPSLService()

        # TPSL Serviceを使って価格計算
        from app.services.auto_strategy.services.tpsl_service import TPSLService
        tpsl_service = TPSLService()
        sl_price, tp_price = tpsl_service.calculate_tpsl_prices(
            current_price=current_price,
            tpsl_gene=strategy.tpsl_gene,
            position_direction=1.0
        )

        logger.info(f"TPSL計算結果: SL={sl_price}, TP={tp_price}")
        logger.info(f"価格変換: {current_price} -> SL={sl_price}, TP={tp_price}")

        success = sl_price is not None and tp_price is not None

        return {
            "status": "success" if success else "failure",
            "current_price": current_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_pct": (current_price - sl_price) / current_price if sl_price else None,
            "tp_pct": (tp_price - current_price) / current_price if tp_price else None,
            "price_calculation_works": success
        }

    except Exception as e:
        logger.error(f"TPSL計算エラー: {e}")
        return {
            "status": "failure",
            "error": str(e),
            "price_calculation_works": False
        }


def main():
    """
    メイン実行関数
    """
    print("=" * 100)
    print("本物のオートストラテジー実行スクリプト")
    print("TPSL統合をリアルワークフローで検証")
    print("=" * 100)

    try:
        # 本物のオートストラテジー実行
        result = create_and_run_autostrategy()

        # 結果出力
        output_file = "real_autostrategy_tpsl_verification.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print("\n" + "=" * 100)
        print("実行結果")
        print("=" * 100)

        if result["status"] == "success":
            print("[SUCCESS] TPSL統合検証成功")
            print("詳細:")
            print(f"  - GA実行: {result['ga_result']['generations_completed']}世代完了")
            print(f"  - 戦略生成: 成功、{result['strategy']['indicators_count']}指標使用")
            print(f"  - TPSL Method: {result['strategy']['tpsl_method']}")
            print(f"  - SL/TP: {result['strategy']['tpsl_stop_loss']:.1%} / {result['strategy']['tpsl_take_profit']:.1%}")
            print(f"  - 価格計算: {result['tpsl_calculation']['sl_price']} / {result['tpsl_calculation']['tp_price']}")

            if result['verification']['tpsl_integration_verified']:
                print("\n[SUCCESS] TPSL統合は完全に機能しています!")
                return_code = 0
            else:
                print("\n[WARNING] TPSL統合に問題が見つかりました")
                return_code = 1

        elif result["status"] == "partial_failure":
            print(f"[PARTIAL] {result.get('error', '一部の機能に問題')}")
            return_code = 1

        else:
            print(f"[FAILURE] {result.get('error', '実行に失敗')}")
            return_code = 1

        print(f"\n結果ファイル: {os.path.abspath(output_file)}")
        print("=" * 100)

        return return_code

    except Exception as e:
        print(f"\n[ERROR] エラー発生: {e}")
        import traceback
        print(f"詳細:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)