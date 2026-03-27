"""
GA設定プリセット

よく使うGA設定の組み合わせをプリセットとして提供する。
プリセットはGAConfigのファクトリ関数として機能し、
個別フィールドの上書きも可能。
"""

from typing import Optional

from .ga import GAConfig


class GAPresets:
    """GA設定プリセットファクトリ。"""

    @staticmethod
    def quick_scan() -> GAConfig:
        """高速スキャン用（粗い探索、短時間）。

        パラメータチューニングとWFAを無効にし、小規模な集団で
        素早く有望な戦略領域を特定する用途。
        """
        return GAConfig(
            population_size=50,
            generations=20,
            max_indicators=5,
            max_conditions=2,
            enable_parameter_tuning=False,
            enable_walk_forward=False,
            use_seed_strategies=True,
            seed_injection_rate=0.15,
            enable_parallel_evaluation=True,
        )

    @staticmethod
    def thorough_search() -> GAConfig:
        """徹底探索用（精密な探索、長時間）。

        大規模な集団と多数の世代で広範囲を探索し、
        エリート個体に対してOptunaチューニングとWFA検証を実施する。
        """
        return GAConfig(
            population_size=200,
            generations=100,
            max_indicators=10,
            max_conditions=3,
            crossover_rate=0.85,
            mutation_rate=0.15,
            elite_size=20,
            enable_parameter_tuning=True,
            tuning_n_trials=100,
            tuning_use_wfa=True,
            enable_walk_forward=True,
            wfa_n_folds=5,
            wfa_train_ratio=0.7,
            use_seed_strategies=True,
            seed_injection_rate=0.1,
            enable_fitness_sharing=True,
            enable_parallel_evaluation=True,
        )

    @staticmethod
    def multi_objective() -> GAConfig:
        """多目的最適化用（NSGA-II）。

        リターン、シャープレシオ、ドローダウンを同時に最適化する。
        """
        return GAConfig(
            population_size=150,
            generations=80,
            enable_multi_objective=True,
            objectives=[
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
            ],
            objective_weights=[1.0, 1.0, 1.0],
            enable_parameter_tuning=False,
            use_seed_strategies=True,
            enable_fitness_sharing=True,
            enable_parallel_evaluation=True,
        )

    @staticmethod
    def short_term() -> GAConfig:
        """短期トレード用プリセット。

        高頻度取引に適した設定。トレード頻度ペナルティを緩和し、
        より.Aggressiveなパラメータ範囲を使用する。
        """
        config = GAConfig(
            population_size=100,
            generations=50,
            max_indicators=6,
            use_seed_strategies=True,
            seed_injection_rate=0.1,
            enable_parameter_tuning=True,
            tuning_n_trials=30,
            enable_parallel_evaluation=True,
        )
        # 短期向けフィットネス重み
        config.fitness_weights = {
            "total_return": 0.25,
            "sharpe_ratio": 0.30,
            "max_drawdown": 0.15,
            "win_rate": 0.10,
            "balance_score": 0.10,
            "ulcer_index_penalty": 0.05,
            "trade_frequency_penalty": 0.05,
        }
        return config

    @staticmethod
    def long_term() -> GAConfig:
        """長期トレード用プリセット。

        低頻度取引に適した設定。ドローダウンとUlcer Indexの
        ペナルティを強化し、安定性を重視する。
        """
        config = GAConfig(
            population_size=150,
            generations=80,
            max_indicators=8,
            max_conditions=3,
            use_seed_strategies=True,
            seed_injection_rate=0.1,
            enable_parameter_tuning=True,
            tuning_n_trials=50,
            enable_walk_forward=True,
            wfa_n_folds=4,
            enable_parallel_evaluation=True,
        )
        # 長期向けフィットネス重み
        config.fitness_weights = {
            "total_return": 0.15,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.20,
            "win_rate": 0.10,
            "balance_score": 0.10,
            "ulcer_index_penalty": 0.15,
            "trade_frequency_penalty": 0.05,
        }
        return config

    @staticmethod
    def get_preset(name: str) -> Optional[GAConfig]:
        """プリセット名からGAConfigを取得する。

        Args:
            name: プリセット名（quick_scan, thorough_search, multi_objective,
                  short_term, long_term）

        Returns:
            GAConfig インスタンス、または不明な名前の場合はNone
        """
        presets = {
            "quick_scan": GAPresets.quick_scan,
            "thorough_search": GAPresets.thorough_search,
            "multi_objective": GAPresets.multi_objective,
            "short_term": GAPresets.short_term,
            "long_term": GAPresets.long_term,
        }
        factory = presets.get(name)
        return factory() if factory else None
