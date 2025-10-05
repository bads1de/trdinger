#!/usr/bin/env python3

"""

多様性進化デモスクリプト

GeneticAlgorithmEngineを使って10世代進化させ、各世代のfitnessメトリクスを出力

"""

import random

import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from deap import tools

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.core.evolution_runner import EvolutionRunner

from app.services.auto_strategy.config.ga_runtime import GAConfig

from app.services.backtest.backtest_service import BacktestService

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator

# 簡易的な設定

class DiversityEvolutionRunner(EvolutionRunner):

    def run_single_objective_evolution(self, population, config, halloffame=None):

        # 初期評価

        population = self._evaluate_population(population)

        logbook = tools.Logbook()

        for gen in range(config.generations):

            # メトリクス計算

            fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]

            if fitnesses:

                mean_fitness = np.mean(fitnesses)

                variance = np.var(fitnesses)

                # Silhouette score

                X = np.array(fitnesses).reshape(-1, 1)

                if len(set(fitnesses)) > 1 and len(fitnesses) > 1:

                    try:

                        kmeans = KMeans(n_clusters=min(3, len(fitnesses)//2 or 1), random_state=42, n_init=10)

                        labels = kmeans.fit_predict(X)

                        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0

                    except:

                        sil = 0.0

                else:

                    sil = 0.0

                print(f"Generation {gen+1}: Mean Fitness={mean_fitness:.2f}, Variance={variance:.2f}, Silhouette={sil:.2f}")

            else:

                print(f"Generation {gen+1}: No valid fitness")

            # 進化の続き (EvolutionRunnerのコードをコピー)

            if config.enable_fitness_sharing and self.fitness_sharing:

                population = self.fitness_sharing.apply_fitness_sharing(population)

            offspring = list(self.toolbox.map(self.toolbox.clone, population))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < config.crossover_rate:

                    self.toolbox.mate(child1, child2)

                    del child1.fitness.values

                    del child2.fitness.values

            for mutant in offspring:

                if random.random() < config.mutation_rate:

                    self.toolbox.mutate(mutant)

                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):

                ind.fitness.values = fit

            population[:] = self.toolbox.select(offspring + population, len(population))

            record = self.stats.compile(population) if self.stats else {}

            logbook.record(gen=gen, **record)

            if halloffame is not None:

                halloffame.update(population)

        return population, logbook

def run_diversity_evolution_demo():

    # GA設定

    config = GAConfig(

        generations=10,

        population_size=50,

        crossover_rate=0.7,

        mutation_rate=0.2,

        enable_fitness_sharing=True,

        # 他のデフォルト

    )

    # backtest_serviceなど初期化 (簡易)

    backtest_service = BacktestService()

    strategy_factory = StrategyFactory()

    gene_generator = RandomGeneGenerator(config)

    engine = GeneticAlgorithmEngine(backtest_service, strategy_factory, gene_generator)

    # runnerを置き換え

    original_create_runner = engine._create_evolution_runner

    def create_diversity_runner(toolbox, stats, population):

        runner = original_create_runner(toolbox, stats, population)

        runner.__class__ = DiversityEvolutionRunner

        return runner

    engine._create_evolution_runner = create_diversity_runner

    backtest_config = {

        'start_date': '2023-01-01',

        'end_date': '2023-12-31',

        'symbol': 'BTCUSDT',

        'timeframe': '1h',

        'initial_capital': 10000,

        'commission_rate': 0.001

    }

    result = engine.run_evolution(config, backtest_config)

if __name__ == "__main__":

    run_diversity_evolution_demo()