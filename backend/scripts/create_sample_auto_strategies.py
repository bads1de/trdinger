"""
サンプルオートストラテジーデータ作成スクリプト

フロントエンド表示テスト用のサンプルデータを作成します。
"""

import sys
import os
from datetime import datetime, timedelta
import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.models import GAExperiment, GeneratedStrategy, BacktestResult
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_ga_experiment():
    """サンプルGA実験を作成"""
    db = SessionLocal()
    try:
        experiment = GAExperiment(
            name="サンプル実験_SMA_RSI戦略",
            config={
                "population_size": 20,
                "generations": 10,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "elite_size": 2,
                "fitness_weights": {
                    "total_return": 0.4,
                    "sharpe_ratio": 0.3,
                    "max_drawdown": 0.2,
                    "win_rate": 0.1
                },
                "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"]
            },
            status="completed",
            progress=1.0,
            best_fitness=0.85,
            total_generations=10,
            current_generation=10,
            completed_at=datetime.now()
        )
        
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        
        logger.info(f"GA実験を作成しました: {experiment.id}")
        return experiment
        
    except Exception as e:
        db.rollback()
        logger.error(f"GA実験作成エラー: {e}")
        raise
    finally:
        db.close()


def create_sample_backtest_results():
    """サンプルバックテスト結果を作成"""
    db = SessionLocal()
    try:
        results = []
        
        # サンプル結果データ
        sample_results = [
            {
                "strategy_name": "GA生成戦略_SMA_RSI_001",
                "performance": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.08,
                    "win_rate": 0.65,
                    "profit_factor": 1.8,
                    "total_trades": 45,
                    "winning_trades": 29,
                    "losing_trades": 16,
                    "avg_win": 0.025,
                    "avg_loss": -0.015
                }
            },
            {
                "strategy_name": "GA生成戦略_EMA_MACD_002",
                "performance": {
                    "total_return": 0.22,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.12,
                    "win_rate": 0.58,
                    "profit_factor": 2.1,
                    "total_trades": 38,
                    "winning_trades": 22,
                    "losing_trades": 16,
                    "avg_win": 0.032,
                    "avg_loss": -0.018
                }
            },
            {
                "strategy_name": "GA生成戦略_BB_RSI_003",
                "performance": {
                    "total_return": 0.08,
                    "sharpe_ratio": 0.9,
                    "max_drawdown": 0.05,
                    "win_rate": 0.72,
                    "profit_factor": 1.4,
                    "total_trades": 52,
                    "winning_trades": 37,
                    "losing_trades": 15,
                    "avg_win": 0.018,
                    "avg_loss": -0.012
                }
            }
        ]
        
        for i, result_data in enumerate(sample_results):
            result = BacktestResult(
                strategy_name=result_data["strategy_name"],
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now(),
                initial_capital=10000.0,
                commission_rate=0.001,
                config_json={
                    "strategy_type": "AUTO_GENERATED",
                    "parameters": {
                        "indicators": ["SMA", "RSI"] if i == 0 else ["EMA", "MACD"] if i == 1 else ["BB", "RSI"]
                    }
                },
                performance_metrics=result_data["performance"],
                equity_curve=[
                    {"timestamp": (datetime.now() - timedelta(days=90-j)).isoformat(), "equity": 10000 + j * 50}
                    for j in range(90)
                ],
                trade_history=[
                    {
                        "entry_time": (datetime.now() - timedelta(days=80-k*2)).isoformat(),
                        "exit_time": (datetime.now() - timedelta(days=79-k*2)).isoformat(),
                        "entry_price": 65000 + k * 100,
                        "exit_price": 65000 + k * 100 + (50 if k % 2 == 0 else -30),
                        "pnl": 50 if k % 2 == 0 else -30,
                        "return_pct": 0.0008 if k % 2 == 0 else -0.0005
                    }
                    for k in range(20)
                ]
            )
            
            db.add(result)
            results.append(result)
        
        db.commit()
        
        for result in results:
            db.refresh(result)
            
        logger.info(f"バックテスト結果を作成しました: {len(results)} 件")
        return results
        
    except Exception as e:
        db.rollback()
        logger.error(f"バックテスト結果作成エラー: {e}")
        raise
    finally:
        db.close()


def create_sample_generated_strategies(experiment_id, backtest_results):
    """サンプル生成戦略を作成"""
    db = SessionLocal()
    try:
        strategies = []
        
        # サンプル戦略データ
        sample_strategies = [
            {
                "gene_data": {
                    "id": "sample_001",
                    "indicators": [
                        {"type": "SMA", "parameters": {"period": 10}, "enabled": True},
                        {"type": "SMA", "parameters": {"period": 30}, "enabled": True},
                        {"type": "RSI", "parameters": {"period": 14}, "enabled": True}
                    ],
                    "entry_conditions": [
                        {"left_operand": "SMA_10", "operator": ">", "right_operand": "SMA_30"},
                        {"left_operand": "RSI_14", "operator": "<", "right_operand": 70}
                    ],
                    "exit_conditions": [
                        {"left_operand": "SMA_10", "operator": "<", "right_operand": "SMA_30"}
                    ],
                    "risk_management": {
                        "position_size": 0.2,
                        "stop_loss": 0.03,
                        "take_profit": 0.06
                    },
                    "metadata": {}
                },
                "generation": 8,
                "fitness_score": 0.85,
                "backtest_result_id": 0
            },
            {
                "gene_data": {
                    "id": "sample_002",
                    "indicators": [
                        {"type": "EMA", "parameters": {"period": 12}, "enabled": True},
                        {"type": "EMA", "parameters": {"period": 26}, "enabled": True},
                        {"type": "MACD", "parameters": {"fast": 12, "slow": 26, "signal": 9}, "enabled": True}
                    ],
                    "entry_conditions": [
                        {"left_operand": "EMA_12", "operator": ">", "right_operand": "EMA_26"},
                        {"left_operand": "MACD", "operator": ">", "right_operand": 0}
                    ],
                    "exit_conditions": [
                        {"left_operand": "EMA_12", "operator": "<", "right_operand": "EMA_26"}
                    ],
                    "risk_management": {
                        "position_size": 0.25,
                        "stop_loss": 0.025,
                        "take_profit": 0.05
                    },
                    "metadata": {}
                },
                "generation": 9,
                "fitness_score": 0.92,
                "backtest_result_id": 1
            },
            {
                "gene_data": {
                    "id": "sample_003",
                    "indicators": [
                        {"type": "BB", "parameters": {"period": 20, "std": 2}, "enabled": True},
                        {"type": "RSI", "parameters": {"period": 14}, "enabled": True}
                    ],
                    "entry_conditions": [
                        {"left_operand": "close", "operator": "<", "right_operand": "BB_lower"},
                        {"left_operand": "RSI_14", "operator": "<", "right_operand": 30}
                    ],
                    "exit_conditions": [
                        {"left_operand": "close", "operator": ">", "right_operand": "BB_upper"}
                    ],
                    "risk_management": {
                        "position_size": 0.15,
                        "stop_loss": 0.02,
                        "take_profit": 0.04
                    },
                    "metadata": {}
                },
                "generation": 10,
                "fitness_score": 0.78,
                "backtest_result_id": 2
            }
        ]
        
        for i, strategy_data in enumerate(sample_strategies):
            strategy = GeneratedStrategy(
                experiment_id=experiment_id,
                gene_data=strategy_data["gene_data"],
                generation=strategy_data["generation"],
                fitness_score=strategy_data["fitness_score"],
                parent_ids=[],
                backtest_result_id=backtest_results[i].id if i < len(backtest_results) else None
            )
            
            db.add(strategy)
            strategies.append(strategy)
        
        db.commit()
        
        for strategy in strategies:
            db.refresh(strategy)
            
        logger.info(f"生成戦略を作成しました: {len(strategies)} 件")
        return strategies
        
    except Exception as e:
        db.rollback()
        logger.error(f"生成戦略作成エラー: {e}")
        raise
    finally:
        db.close()


def main():
    """メイン実行関数"""
    logger.info("=== サンプルオートストラテジーデータ作成開始 ===")
    
    try:
        # 1. GA実験を作成
        logger.info("Step 1: GA実験作成")
        experiment = create_sample_ga_experiment()
        
        # 2. バックテスト結果を作成
        logger.info("Step 2: バックテスト結果作成")
        backtest_results = create_sample_backtest_results()
        
        # 3. 生成戦略を作成
        logger.info("Step 3: 生成戦略作成")
        strategies = create_sample_generated_strategies(experiment.id, backtest_results)
        
        logger.info("=== サンプルデータ作成完了 ===")
        logger.info(f"作成されたデータ:")
        logger.info(f"- GA実験: 1件")
        logger.info(f"- バックテスト結果: {len(backtest_results)}件")
        logger.info(f"- 生成戦略: {len(strategies)}件")
        logger.info("")
        logger.info("以下のAPIでデータを確認できます:")
        logger.info("- GET /api/strategies/unified")
        logger.info("- GET /api/strategies/auto-generated")
        logger.info("- GET /api/strategies/stats")
        
    except Exception as e:
        logger.error(f"サンプルデータ作成エラー: {e}")
        raise


if __name__ == "__main__":
    main()
