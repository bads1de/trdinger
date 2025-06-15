"""
オートストラテジー生成スクリプト

実際のオートストラテジー戦略を生成し、バックテストを実行してデータベースに保存します。
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from database.connection import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_ga_config() -> dict:
    """
    サンプルGA設定を作成

    Returns:
        GA設定辞書
    """
    return {
        "population_size": 20,
        "generations": 10,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "elite_size": 2,
        "fitness_weights": {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        },
        "fitness_constraints": {"min_trades": 10, "max_drawdown_limit": 0.3},
        "max_indicators": 5,
        "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"],
    }


def create_sample_backtest_config() -> dict:
    """
    サンプルバックテスト設定を作成

    Returns:
        バックテスト設定辞書
    """
    # 過去3ヶ月のデータを使用
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
    }


async def generate_strategies():
    """
    オートストラテジー戦略を生成
    """
    try:
        logger.info("オートストラテジー生成開始")

        # AutoStrategyServiceを初期化
        auto_strategy_service = AutoStrategyService()

        # GA設定を作成
        ga_config_dict = create_sample_ga_config()
        ga_config = GAConfig.from_dict(ga_config_dict)

        # バックテスト設定を作成
        backtest_config = create_sample_backtest_config()

        # 設定の検証
        is_valid, errors = ga_config.validate()
        if not is_valid:
            logger.error(f"GA設定が無効です: {', '.join(errors)}")
            return False

        logger.info("GA設定が有効です。戦略生成を開始します...")

        # 戦略生成を開始
        experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_name=f"自動生成実験_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ga_config=ga_config,
            backtest_config=backtest_config,
        )

        logger.info(f"戦略生成が開始されました。実験ID: {experiment_id}")

        # 進捗を監視
        max_wait_time = 300  # 5分
        wait_interval = 10  # 10秒間隔
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            progress = auto_strategy_service.get_experiment_progress(experiment_id)

            if progress:
                logger.info(
                    f"進捗: {progress.progress:.1%} "
                    f"({progress.current_generation}/{progress.total_generations}世代) "
                    f"最高フィットネス: {progress.best_fitness:.4f if progress.best_fitness else 'N/A'}"
                )

                if progress.status == "completed":
                    logger.info("戦略生成が完了しました！")
                    break
                elif progress.status == "failed":
                    logger.error("戦略生成が失敗しました")
                    return False

            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval

        if elapsed_time >= max_wait_time:
            logger.warning(
                "タイムアウトしましたが、バックグラウンドで実行は継続されます"
            )

        # 結果を取得
        results = auto_strategy_service.get_experiment_results(experiment_id)

        if results and results.get("success"):
            strategies = results.get("strategies", [])
            logger.info(f"生成された戦略数: {len(strategies)}")

            # 上位戦略の情報を表示
            for i, strategy in enumerate(strategies[:5]):
                logger.info(
                    f"戦略 {i+1}: フィットネス={strategy.get('fitness_score', 0):.4f}, "
                    f"リターン={strategy.get('performance', {}).get('total_return', 0):.2%}"
                )

            return True
        else:
            logger.error("戦略生成結果の取得に失敗しました")
            return False

    except Exception as e:
        logger.error(f"戦略生成エラー: {e}")
        return False


def test_single_strategy():
    """
    単一戦略のテスト実行
    """
    try:
        logger.info("単一戦略テスト開始")

        # AutoStrategyServiceを初期化
        auto_strategy_service = AutoStrategyService()

        # テスト用の戦略遺伝子を作成
        test_gene_data = {
            "indicators": [
                {"type": "SMA", "parameters": {"period": 10}, "enabled": True},
                {"type": "SMA", "parameters": {"period": 30}, "enabled": True},
                {"type": "RSI", "parameters": {"period": 14}, "enabled": True},
            ],
            "entry_conditions": [
                {"left_operand": "SMA_10", "operator": ">", "right_operand": "SMA_30"},
                {"left_operand": "RSI_14", "operator": "<", "right_operand": 70},
            ],
            "exit_conditions": [
                {"left_operand": "SMA_10", "operator": "<", "right_operand": "SMA_30"}
            ],
            "risk_management": {
                "position_size": 0.2,
                "stop_loss": 0.03,
                "take_profit": 0.06,
            },
        }

        # バックテスト設定
        backtest_config = create_sample_backtest_config()

        # StrategyGeneオブジェクトを作成
        strategy_gene = StrategyGene.from_dict(test_gene_data)

        # テスト実行
        result = auto_strategy_service.test_strategy_generation(
            strategy_gene, backtest_config
        )

        if result.get("success"):
            logger.info("単一戦略テストが成功しました")
            performance = result.get("performance", {})
            logger.info(f"パフォーマンス: {performance}")
            return True
        else:
            logger.error(
                f"単一戦略テストが失敗しました: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        logger.error(f"単一戦略テストエラー: {e}")
        return False


async def main():
    """
    メイン実行関数
    """
    logger.info("=== オートストラテジー生成スクリプト開始 ===")

    # まず単一戦略をテスト
    logger.info("Step 1: 単一戦略テスト")
    if not test_single_strategy():
        logger.error("単一戦略テストが失敗しました。処理を中断します。")
        return

    logger.info("単一戦略テストが成功しました。")

    # 実際の戦略生成を実行
    logger.info("Step 2: オートストラテジー生成")
    success = await generate_strategies()

    if success:
        logger.info("=== オートストラテジー生成が完了しました ===")
        logger.info("生成された戦略は以下のAPIで確認できます:")
        logger.info("- GET /api/strategies/unified")
        logger.info("- GET /api/strategies/auto-generated")
    else:
        logger.error("=== オートストラテジー生成が失敗しました ===")


if __name__ == "__main__":
    asyncio.run(main())
