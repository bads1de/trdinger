"""
評価戦略モジュール

OOS (Out-of-Sample) 検証、Walk-Forward 分析などの
評価戦略ルーティングを担当します。
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple

import pandas as pd

from ..config.ga import GAConfig

if TYPE_CHECKING:
    from .individual_evaluator import IndividualEvaluator

logger = logging.getLogger(__name__)


class EvaluationStrategy:
    """
    評価戦略ルーティングクラス

    IndividualEvaluator から委譲を受け、設定に応じて
    通常評価、OOS検証、Walk-Forward 分析のいずれかに振り分けます。
    """

    def __init__(self, evaluator: "IndividualEvaluator") -> None:
        self._evaluator = evaluator

    def execute(
        self, gene: Any, base_backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """
        設定に応じた評価戦略を実行します。

        Args:
            gene: 評価対象の遺伝子
            base_backtest_config: 固定的なバックテスト設定
            config: GA 実行設定

        Returns:
            算出された適応度（タプル）
        """
        if getattr(config, "enable_walk_forward", False):
            return self._evaluate_with_walk_forward(
                gene, base_backtest_config, config
            )

        oos_ratio = getattr(config, "oos_split_ratio", 0.0)
        oos_weight = getattr(config, "oos_fitness_weight", 0.5)

        if oos_ratio > 0.0:
            return self._evaluate_with_oos(
                gene, base_backtest_config, config, oos_ratio, oos_weight
            )
        else:
            return self._evaluator._perform_single_evaluation(
                gene, base_backtest_config, config
            )

    def _evaluate_with_oos(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """
        Out-of-Sample (OOS) 検証を含む評価を実行します。
        """
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                return self._evaluator._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            total_duration = end_date - start_date
            train_duration = total_duration * (1.0 - oos_ratio)

            split_date = start_date + train_duration

            start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
            split_str = split_date.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

            is_config = base_backtest_config.copy()
            is_config["start_date"] = start_str
            is_config["end_date"] = split_str
            is_fitness = self._evaluator._perform_single_evaluation(
                gene, is_config, config
            )

            oos_config = base_backtest_config.copy()
            oos_config["start_date"] = split_str
            oos_config["end_date"] = end_str
            oos_fitness = self._evaluator._perform_single_evaluation(
                gene, oos_config, config
            )

            combined_fitness = []
            for f_is, f_oos in zip(is_fitness, oos_fitness):
                combined = f_is * (1.0 - oos_weight) + f_oos * oos_weight
                combined_fitness.append(max(0.0, combined))

            logger.info(
                f"OOS評価完了: IS={is_fitness}, OOS={oos_fitness}, Combined={combined_fitness}"
            )
            return tuple(combined_fitness)

        except Exception as e:
            logger.error(f"OOS評価中エラー: {e}")
            return self._evaluator._perform_single_evaluation(
                gene, base_backtest_config, config
            )

    def _evaluate_with_walk_forward(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """
        Walk-Forward Analysis による評価
        """
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                logger.warning("WFA: 期間が不明なため通常評価にフォールバック")
                return self._evaluator._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            n_folds = getattr(config, "wfa_n_folds", 5)
            train_ratio = getattr(config, "wfa_train_ratio", 0.7)
            anchored = getattr(config, "wfa_anchored", False)

            total_duration = end_date - start_date
            fold_duration = total_duration / n_folds

            oos_fitness_values = []

            for fold_idx in range(n_folds):
                if anchored:
                    fold_train_start = start_date
                else:
                    fold_train_start = start_date + (fold_duration * fold_idx)

                fold_end = start_date + (fold_duration * (fold_idx + 1))

                fold_period = fold_end - fold_train_start
                train_duration = fold_period * train_ratio

                train_end = fold_train_start + train_duration
                test_start = train_end
                test_end = fold_end

                if (train_end - fold_train_start).days < 7:
                    logger.debug(
                        "WFA Fold %s: トレーニング期間が短すぎるためスキップ", fold_idx
                    )
                    continue

                if (test_end - test_start).days < 1:
                    logger.debug(
                        "WFA Fold %s: テスト期間が短すぎるためスキップ", fold_idx
                    )
                    continue

                train_start_str = fold_train_start.strftime("%Y-%m-%d %H:%M:%S")
                train_end_str = train_end.strftime("%Y-%m-%d %H:%M:%S")
                test_start_str = test_start.strftime("%Y-%m-%d %H:%M:%S")
                test_end_str = test_end.strftime("%Y-%m-%d %H:%M:%S")

                logger.debug(
                    "WFA Fold %s: Train=%s to %s, Test=%s to %s",
                    fold_idx,
                    train_start_str,
                    train_end_str,
                    test_start_str,
                    test_end_str,
                )

                test_config = base_backtest_config.copy()
                test_config["start_date"] = test_start_str
                test_config["end_date"] = test_end_str

                try:
                    oos_fitness = self._evaluator._perform_single_evaluation(
                        gene, test_config, config
                    )
                    oos_fitness_values.append(oos_fitness)
                except Exception as fold_error:
                    logger.warning(f"WFA Fold {fold_idx} 評価エラー: {fold_error}")
                    continue

            if not oos_fitness_values:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluator._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            num_objectives = len(oos_fitness_values[0])
            averaged_fitness = []

            for obj_idx in range(num_objectives):
                obj_values = [f[obj_idx] for f in oos_fitness_values]
                avg_value = sum(obj_values) / len(obj_values)
                averaged_fitness.append(max(0.0, avg_value))

            logger.info(
                f"WFA評価完了: {len(oos_fitness_values)}フォールド, "
                f"平均OOS={tuple(round(v, 4) for v in averaged_fitness)}"
            )

            return tuple(averaged_fitness)

        except Exception as e:
            logger.error(f"WFA評価中エラー: {e}")
            return self._evaluator._perform_single_evaluation(
                gene, base_backtest_config, config
            )
