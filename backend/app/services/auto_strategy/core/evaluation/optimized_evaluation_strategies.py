"""
最適化された評価戦略モジュール

OOS検証、Walk-Forward分析の並列化を提供します。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig

if TYPE_CHECKING:
    from .individual_evaluator import IndividualEvaluator

logger = logging.getLogger(__name__)


class OptimizedEvaluationStrategy:
    """
    最適化された評価戦略ルーティングクラス

    主な最適化ポイント:
    1. OOS検証の並列化（IS/OOS同時実行）
    2. Walk-Forward分析の並列化（n_folds同時実行）
    3. 日付計算のキャッシュ
    4. 効率的なフォールド計算
    """

    def __init__(
        self,
        evaluator: "IndividualEvaluator",
        max_workers: int = 2,
    ) -> None:
        self._evaluator = evaluator
        self._max_workers = max_workers

        # 日付計算キャッシュ
        self._date_cache: Dict[str, Tuple[str, str, str]] = {}

    def execute(
        self, gene: Any, base_backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """
        設定に応じた評価戦略を実行します（最適化版）。
        """
        if getattr(config, "enable_walk_forward", False):
            return self._evaluate_with_walk_forward_parallel(
                gene, base_backtest_config, config
            )

        oos_ratio = getattr(config, "oos_split_ratio", 0.0)
        oos_weight = getattr(config, "oos_fitness_weight", 0.5)

        if oos_ratio > 0.0:
            return self._evaluate_with_oos_parallel(
                gene, base_backtest_config, config, oos_ratio, oos_weight
            )
        else:
            return self._evaluator._perform_single_evaluation(
                gene, base_backtest_config, config
            )

    def _evaluate_with_oos_parallel(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """
        Out-of-Sample (OOS) 検証を並列実行します（最適化版）。

        最適化:
        - IS/OOSのバックテストを並列実行
        - 日付計算のキャッシュ
        """
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                return self._evaluator._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            # 日付計算のキャッシュ
            cache_key = f"{start_date}_{end_date}_{oos_ratio}"
            if cache_key in self._date_cache:
                start_str, split_str, end_str = self._date_cache[cache_key]
            else:
                total_duration = end_date - start_date
                train_duration = total_duration * (1.0 - oos_ratio)
                split_date = start_date + train_duration

                start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
                split_str = split_date.strftime("%Y-%m-%d %H:%M:%S")
                end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

                self._date_cache[cache_key] = (start_str, split_str, end_str)

            # IS/OOS設定を構築
            is_config = base_backtest_config.copy()
            is_config["start_date"] = start_str
            is_config["end_date"] = split_str

            oos_config = base_backtest_config.copy()
            oos_config["start_date"] = split_str
            oos_config["end_date"] = end_str

            # 並列実行
            with ThreadPoolExecutor(max_workers=2) as executor:
                is_future = executor.submit(
                    self._evaluator._perform_single_evaluation,
                    gene, is_config, config
                )
                oos_future = executor.submit(
                    self._evaluator._perform_single_evaluation,
                    gene, oos_config, config
                )

                is_fitness = is_future.result()
                oos_fitness = oos_future.result()

            # フィットネスの結合
            combined_fitness = []
            for f_is, f_oos in zip(is_fitness, oos_fitness):
                combined = f_is * (1.0 - oos_weight) + f_oos * oos_weight
                combined_fitness.append(max(0.0, combined))

            logger.info(
                f"OOS評価完了（並列）: IS={is_fitness}, OOS={oos_fitness}, Combined={combined_fitness}"
            )
            return tuple(combined_fitness)

        except Exception as e:
            logger.error(f"OOS評価中エラー: {e}")
            return self._evaluator._perform_single_evaluation(
                gene, base_backtest_config, config
            )

    def _evaluate_with_walk_forward_parallel(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """
        Walk-Forward Analysis を並列実行します（最適化版）。

        最適化:
        - n_foldsのバックテストを並列実行
        - フォールド計算の効率化
        - 日付計算のキャッシュ
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

            # フォールド設定を事前計算
            fold_configs = self._precompute_fold_configs(
                start_date, end_date, n_folds, train_ratio, anchored, base_backtest_config
            )

            if not fold_configs:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluator._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            # 並列実行
            oos_fitness_values = []
            with ThreadPoolExecutor(max_workers=min(self._max_workers, len(fold_configs))) as executor:
                future_to_fold = {}
                for fold_idx, test_config in fold_configs:
                    future = executor.submit(
                        self._evaluator._perform_single_evaluation,
                        gene, test_config, config
                    )
                    future_to_fold[future] = fold_idx

                for future in as_completed(future_to_fold):
                    fold_idx = future_to_fold[future]
                    try:
                        oos_fitness = future.result()
                        oos_fitness_values.append(oos_fitness)
                    except Exception as fold_error:
                        logger.warning(f"WFA Fold {fold_idx} 評価エラー: {fold_error}")

            if not oos_fitness_values:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluator._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            # 平均フィットネス計算
            num_objectives = len(oos_fitness_values[0])
            averaged_fitness = []

            for obj_idx in range(num_objectives):
                obj_values = [f[obj_idx] for f in oos_fitness_values]
                avg_value = sum(obj_values) / len(obj_values)
                averaged_fitness.append(max(0.0, avg_value))

            logger.info(
                f"WFA評価完了（並列）: {len(oos_fitness_values)}フォールド, "
                f"平均OOS={tuple(round(v, 4) for v in averaged_fitness)}"
            )

            return tuple(averaged_fitness)

        except Exception as e:
            logger.error(f"WFA評価中エラー: {e}")
            return self._evaluator._perform_single_evaluation(
                gene, base_backtest_config, config
            )

    def _precompute_fold_configs(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_folds: int,
        train_ratio: float,
        anchored: bool,
        base_backtest_config: Dict[str, Any],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """
        フォールド設定を事前計算します（最適化版）。

        最適化:
        - 日付計算の効率化
        - 不要なフォールドの早期フィルタリング
        """
        fold_configs = []

        total_duration = end_date - start_date
        fold_duration = total_duration / n_folds

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

            # 早期フィルタリング
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

            # 日付文字列の効率的な生成
            test_start_str = test_start.strftime("%Y-%m-%d %H:%M:%S")
            test_end_str = test_end.strftime("%Y-%m-%d %H:%M:%S")

            test_config = base_backtest_config.copy()
            test_config["start_date"] = test_start_str
            test_config["end_date"] = test_end_str

            fold_configs.append((fold_idx, test_config))

        return fold_configs

    def _evaluate_with_oos(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """OOS検証を含む評価（並列版）"""
        return self._evaluate_with_oos_parallel(
            gene, base_backtest_config, config, oos_ratio, oos_weight
        )

    def _evaluate_with_walk_forward(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """Walk-Forward Analysis による評価（並列版）"""
        return self._evaluate_with_walk_forward_parallel(
            gene, base_backtest_config, config
        )

    def clear_cache(self):
        """キャッシュをクリア"""
        self._date_cache.clear()
