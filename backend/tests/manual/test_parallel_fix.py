
import logging
import sys
import os
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core.parallel_evaluator import ParallelEvaluator
from app.services.auto_strategy.core.evaluation_worker import initialize_worker_process, worker_evaluate_individual
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.backtest.backtest_service import BacktestService
from deap import creator, base

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_deap_minimal(config):
    """DEAPのIndividualクラスをモジュールレベルで定義（Pickle対策）"""
    from app.services.auto_strategy.genes import StrategyGene
    
    # 既に存在する場合は削除して再作成（テスト用）
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual
        
    creator.create("FitnessMulti", base.Fitness, weights=tuple(config.objective_weights))
    creator.create("Individual", StrategyGene, fitness=creator.FitnessMulti)

def test_parallel_execution():
    logger.info("Starting Parallel Execution Test on Windows...")
    
    config = GAConfig(
        population_size=4,
        generations=1,
        enable_parallel_evaluation=True,
        max_evaluation_workers=2
    )
    
    # DEAPセットアップ
    setup_deap_minimal(config)
    
    # ダミーのバックテスト設定
    backtest_config = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "4h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
        "initial_capital": 100000.0
    }
    
    # 個体の生成
    generator = RandomGeneGenerator(config)
    population = []
    for _ in range(config.population_size):
        gene = generator.generate_random_gene()
        # StrategyGeneをIndividualに変換
        from dataclasses import fields
        gene_dict = {f.name: getattr(gene, f.name) for f in fields(gene)}
        ind = creator.Individual(**gene_dict)
        population.append(ind)
    
    logger.info(f"Generated {len(population)} individuals.")

    # 並列評価器の設定
    # 注意: Windowsでは top-level 関数である必要がある
    evaluator = ParallelEvaluator(
        evaluate_func=worker_evaluate_individual,
        max_workers=2,
        worker_initializer=initialize_worker_process,
        worker_initargs=(backtest_config, config, {}),
        use_process_pool=True
    )
    
    try:
        evaluator.start()
        logger.info("Executor started. Running evaluation...")
        
        fitnesses = evaluator.evaluate_population(population)
        
        for i, fit in enumerate(fitnesses):
            logger.info(f"Individual {i} fitness: {fit}")
            
        if any(f[0] > 0 or f[0] == 0 for f in fitnesses):
            logger.info("SUCCESS: Parallel evaluation completed without PickleError!")
        else:
            logger.error("FAILED: Evaluation returned invalid results.")
            
    except Exception as e:
        logger.error(f"FATAL ERROR during parallel test: {e}", exc_info=True)
    finally:
        evaluator.shutdown()

if __name__ == "__main__":
    # Windows multiprocessing の要件
    import multiprocessing
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    
    test_parallel_execution()
