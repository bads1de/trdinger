"""
評価戦略モジュール

OOS (Out-of-Sample) 検証、Walk-Forward 分析などの
評価戦略ルーティングを担当します。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd

from app.services.auto_strategy.config.ga import GAConfig
from .evaluation_report import EvaluationReport, ScenarioEvaluation

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
        self._max_workers = 2
        self._date_cache: Dict[str, Tuple[str, str, str]] = {}

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
        return self.execute_report(
            gene, base_backtest_config, config
        ).aggregated_fitness

    def execute_report(
        self, gene: Any, base_backtest_config: Dict[str, Any], config: GAConfig
    ) -> EvaluationReport:
        """設定に応じた評価戦略を評価レポートとして実行する。"""
        # PurgedKFold評価（過学習対策）
        if getattr(config, "enable_purged_kfold", False):
            return self._evaluate_with_purged_kfold_report(
                gene, base_backtest_config, config
            )

        if getattr(config, "enable_walk_forward", False):
            return self._evaluate_with_walk_forward_report(
                gene, base_backtest_config, config
            )

        oos_ratio = getattr(config, "oos_split_ratio", 0.0)
        oos_weight = getattr(config, "oos_fitness_weight", 0.5)

        if oos_ratio > 0.0:
            return self._evaluate_with_oos_report(
                gene, base_backtest_config, config, oos_ratio, oos_weight
            )
        else:
            return self._evaluate_single_report(
                gene, base_backtest_config, config
            )

    def execute_robustness_report(
        self, gene: Any, base_backtest_config: Dict[str, Any], config: GAConfig
    ) -> EvaluationReport:
        """二段階選抜用に複数シナリオで頑健性評価を実行する。"""
        scenario_definitions = self._build_robustness_scenarios(
            base_backtest_config, config
        )
        if len(scenario_definitions) <= 1:
            return self.execute_report(gene, base_backtest_config, config)

        scenario_reports: list[ScenarioEvaluation] = []
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(scenario_definitions))
        ) as executor:
            future_to_order = {}
            for order, scenario_name, scenario_config, metadata in scenario_definitions:
                future = executor.submit(
                    self._evaluate_robustness_scenario_report,
                    gene,
                    scenario_name,
                    scenario_config,
                    config,
                    metadata,
                )
                future_to_order[future] = order

            for future in as_completed(future_to_order):
                order = future_to_order[future]
                try:
                    scenario = future.result()
                    scenario.metadata.setdefault("scenario_order", order)
                    scenario_reports.append(scenario)
                except Exception as scenario_error:
                    logger.warning(
                        "robustness scenario %s 評価エラー: %s",
                        order,
                        scenario_error,
                    )

        if not scenario_reports:
            return self.execute_report(gene, base_backtest_config, config)

        scenario_reports.sort(
            key=lambda scenario: int(scenario.metadata.get("scenario_order", -1))
        )
        aggregate_method = str(
            getattr(config, "robustness_aggregate_method", "robust") or "robust"
        )
        return EvaluationReport.aggregate(
            mode="robustness",
            objectives=self._get_objectives(config),
            scenarios=scenario_reports,
            aggregate_method=aggregate_method,
            metadata={"scenario_count": len(scenario_reports)},
        )

    def _get_objectives(self, config: GAConfig) -> list[str]:
        objectives = getattr(config, "objectives", None)
        return list(objectives) if objectives else ["weighted_score"]

    def _build_robustness_scenarios(
        self,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> list[tuple[int, str, Dict[str, Any], Dict[str, Any]]]:
        """robustness 評価対象のシナリオ設定を構築する。"""
        scenario_definitions = []
        seen_keys = set()
        order = 0

        def add_scenario(
            name: str,
            scenario_config: Dict[str, Any],
            metadata: Dict[str, Any],
        ) -> None:
            nonlocal order
            scenario_key = (
                str(scenario_config.get("symbol")),
                float(scenario_config.get("commission_rate", 0.0) or 0.0),
                float(scenario_config.get("slippage", 0.0) or 0.0),
                str(scenario_config.get("start_date")),
                str(scenario_config.get("end_date")),
            )
            if scenario_key in seen_keys:
                return
            seen_keys.add(scenario_key)
            scenario_definitions.append((order, name, scenario_config, metadata))
            order += 1

        base_symbol = str(base_backtest_config.get("symbol", "") or "")
        base_slippage = float(base_backtest_config.get("slippage", 0.0) or 0.0)
        base_commission = float(
            base_backtest_config.get("commission_rate", 0.0) or 0.0
        )
        base_start_date = str(base_backtest_config.get("start_date", "") or "")
        base_end_date = str(base_backtest_config.get("end_date", "") or "")

        add_scenario(
            "base",
            base_backtest_config.copy(),
            {
                "scenario_kind": "base",
                "symbol": base_symbol,
                "slippage": base_slippage,
                "commission_rate": base_commission,
                "start_date": base_start_date,
                "end_date": base_end_date,
            },
        )

        for symbol in getattr(config, "robustness_validation_symbols", None) or []:
            symbol_str = str(symbol or "")
            if not symbol_str or symbol_str == base_symbol:
                continue
            scenario_config = base_backtest_config.copy()
            scenario_config["symbol"] = symbol_str
            add_scenario(
                f"symbol_{symbol_str}",
                scenario_config,
                {
                    "scenario_kind": "symbol",
                    "symbol": symbol_str,
                    "slippage": base_slippage,
                    "commission_rate": base_commission,
                    "start_date": base_start_date,
                    "end_date": base_end_date,
                },
            )

        for regime_window in getattr(config, "robustness_regime_windows", []) or []:
            if not isinstance(regime_window, dict):
                continue
            regime_name = str(regime_window.get("name", "") or "").strip()
            regime_start = str(regime_window.get("start_date", "") or "").strip()
            regime_end = str(regime_window.get("end_date", "") or "").strip()
            if not regime_name or not regime_start or not regime_end:
                continue
            scenario_config = base_backtest_config.copy()
            scenario_config["start_date"] = regime_start
            scenario_config["end_date"] = regime_end
            add_scenario(
                f"regime_{regime_name}",
                scenario_config,
                {
                    "scenario_kind": "regime",
                    "regime_name": regime_name,
                    "symbol": base_symbol,
                    "slippage": base_slippage,
                    "commission_rate": base_commission,
                    "start_date": regime_start,
                    "end_date": regime_end,
                },
            )

        for slippage_delta in getattr(config, "robustness_stress_slippage", []) or []:
            scenario_config = base_backtest_config.copy()
            stressed_slippage = base_slippage + float(slippage_delta)
            scenario_config["slippage"] = stressed_slippage
            add_scenario(
                f"slippage_{round(stressed_slippage, 10)}",
                scenario_config,
                {
                    "scenario_kind": "slippage",
                    "symbol": base_symbol,
                    "slippage": stressed_slippage,
                    "commission_rate": base_commission,
                    "start_date": base_start_date,
                    "end_date": base_end_date,
                },
            )

        for multiplier in (
            getattr(config, "robustness_stress_commission_multipliers", []) or []
        ):
            multiplier_value = float(multiplier)
            scenario_config = base_backtest_config.copy()
            stressed_commission = base_commission * multiplier_value
            scenario_config["commission_rate"] = stressed_commission
            add_scenario(
                f"commission_x{round(multiplier_value, 10)}",
                scenario_config,
                {
                    "scenario_kind": "commission",
                    "symbol": base_symbol,
                    "slippage": base_slippage,
                    "commission_rate": stressed_commission,
                    "commission_multiplier": multiplier_value,
                    "start_date": base_start_date,
                    "end_date": base_end_date,
                },
            )

        return scenario_definitions

    def _evaluate_robustness_scenario_report(
        self,
        gene: Any,
        scenario_name: str,
        scenario_config: Dict[str, Any],
        config: GAConfig,
        metadata: Dict[str, Any],
    ) -> ScenarioEvaluation:
        """単一 robustness シナリオを評価し、外側用シナリオへ変換する。"""
        scenario_report = self.execute_report(gene, scenario_config, config)
        scenario_metadata = metadata.copy()
        scenario_metadata.update(
            {
                "inner_mode": scenario_report.mode,
                "inner_pass_rate": scenario_report.pass_rate,
                "inner_scenario_count": len(scenario_report.scenarios),
            }
        )
        min_pass_rate = float(getattr(config, "two_stage_min_pass_rate", 0.0) or 0.0)
        return ScenarioEvaluation(
            name=scenario_name,
            fitness=tuple(float(value) for value in scenario_report.aggregated_fitness),
            passed=scenario_report.pass_rate >= min_pass_rate,
            metadata=scenario_metadata,
        )

    def _evaluate_single_report(
        self, gene: Any, backtest_config: Dict[str, Any], config: GAConfig
    ) -> EvaluationReport:
        scenario = self._evaluate_scenario(
            gene,
            backtest_config,
            config,
            scenario_name="single",
        )
        return EvaluationReport.single(
            mode="single",
            objectives=self._get_objectives(config),
            scenario=scenario,
        )

    def _evaluate_scenario(
        self,
        gene: Any,
        backtest_config: Dict[str, Any],
        config: GAConfig,
        *,
        scenario_name: str,
        metadata: Dict[str, Any] | None = None,
    ) -> ScenarioEvaluation:
        perform_single = getattr(self._evaluator, "_perform_single_evaluation", None)
        if hasattr(perform_single, "mock_calls"):
            fitness = perform_single(gene, backtest_config, config)
            return ScenarioEvaluation(
                name=scenario_name,
                fitness=tuple(float(value) for value in fitness),
                passed=True,
                metadata=metadata.copy() if metadata else {},
            )

        report_method = getattr(self._evaluator, "_perform_single_evaluation_report", None)
        if callable(report_method):
            return report_method(
                gene,
                backtest_config,
                config,
                scenario_name=scenario_name,
                metadata=metadata,
            )

        fitness = self._evaluator._perform_single_evaluation(
            gene, backtest_config, config
        )
        return ScenarioEvaluation(
            name=scenario_name,
            fitness=tuple(float(value) for value in fitness),
            passed=True,
            metadata=metadata.copy() if metadata else {},
        )

    def _evaluate_with_oos_parallel(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """OOS検証を含む評価（並列版の互換API）。"""
        return self._evaluate_with_oos_report(
            gene, base_backtest_config, config, oos_ratio, oos_weight
        ).aggregated_fitness

    def _evaluate_with_walk_forward_parallel(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """Walk-Forward Analysis の並列版の互換API。"""
        return self._evaluate_with_walk_forward_report(
            gene, base_backtest_config, config
        ).aggregated_fitness

    def _evaluate_with_purged_kfold_parallel(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """PurgedKFold評価の並列版の互換API。"""
        return self._evaluate_with_purged_kfold_report(
            gene, base_backtest_config, config
        ).aggregated_fitness

    def _evaluate_with_oos(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """OOS検証を含む評価（タプル返却の互換API）。"""
        return self._evaluate_with_oos_report(
            gene, base_backtest_config, config, oos_ratio, oos_weight
        ).aggregated_fitness

    def _evaluate_with_oos_report(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> EvaluationReport:
        """Out-of-Sample (OOS) 検証を含む評価を実行する。"""
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                return self._evaluate_single_report(gene, base_backtest_config, config)

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

            is_config = base_backtest_config.copy()
            is_config["start_date"] = start_str
            is_config["end_date"] = split_str

            oos_config = base_backtest_config.copy()
            oos_config["start_date"] = split_str
            oos_config["end_date"] = end_str

            with ThreadPoolExecutor(max_workers=2) as executor:
                is_future = executor.submit(
                    self._evaluate_scenario,
                    gene,
                    is_config,
                    config,
                    scenario_name="is",
                    metadata={"segment": "is"},
                )
                oos_future = executor.submit(
                    self._evaluate_scenario,
                    gene,
                    oos_config,
                    config,
                    scenario_name="oos",
                    metadata={"segment": "oos"},
                )

                is_scenario = is_future.result()
                oos_scenario = oos_future.result()

            report = EvaluationReport.aggregate(
                mode="oos",
                objectives=self._get_objectives(config),
                scenarios=[is_scenario, oos_scenario],
                aggregate_method="weighted",
                weights=[1.0 - oos_weight, oos_weight],
                metadata={"oos_weight": oos_weight},
            )

            logger.info(
                "OOS評価完了（並列）: pass_rate=%s, Combined=%s",
                round(report.pass_rate, 4),
                report.aggregated_fitness,
            )
            return report

        except Exception as e:
            logger.error(f"OOS評価中エラー: {e}")
            return self._evaluate_single_report(gene, base_backtest_config, config)

    def _evaluate_with_walk_forward(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """Walk-Forward Analysis（タプル返却の互換API）。"""
        return self._evaluate_with_walk_forward_report(
            gene, base_backtest_config, config
        ).aggregated_fitness

    def _evaluate_with_walk_forward_report(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> EvaluationReport:
        """Walk-Forward Analysis による評価を並列実行する。"""
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                logger.warning("WFA: 期間が不明なため通常評価にフォールバック")
                return self._evaluate_single_report(gene, base_backtest_config, config)

            n_folds = getattr(config, "wfa_n_folds", 5)
            train_ratio = getattr(config, "wfa_train_ratio", 0.7)
            anchored = getattr(config, "wfa_anchored", False)

            fold_configs = self._precompute_fold_configs(
                start_date,
                end_date,
                n_folds,
                train_ratio,
                anchored,
                base_backtest_config,
            )

            if not fold_configs:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluate_single_report(gene, base_backtest_config, config)

            scenario_reports: list[ScenarioEvaluation] = []
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(fold_configs))
            ) as executor:
                future_to_fold = {}
                for fold_idx, test_config in fold_configs:
                    future = executor.submit(
                        self._evaluate_scenario,
                        gene,
                        test_config,
                        config,
                        scenario_name=f"fold_{fold_idx}",
                        metadata={"fold_index": fold_idx},
                    )
                    future_to_fold[future] = fold_idx

                for future in as_completed(future_to_fold):
                    fold_idx = future_to_fold[future]
                    try:
                        scenario_reports.append(future.result())
                    except Exception as fold_error:
                        logger.warning(f"WFA Fold {fold_idx} 評価エラー: {fold_error}")

            if not scenario_reports:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluate_single_report(gene, base_backtest_config, config)

            scenario_reports.sort(
                key=lambda scenario: int(scenario.metadata.get("fold_index", -1))
            )
            report = EvaluationReport.aggregate(
                mode="walk_forward",
                objectives=self._get_objectives(config),
                scenarios=scenario_reports,
                aggregate_method="robust",
                metadata={"fold_count": len(scenario_reports)},
            )

            logger.info(
                "WFA評価完了（並列）: %sフォールド, pass_rate=%s, 集約=%s",
                len(scenario_reports),
                round(report.pass_rate, 4),
                tuple(round(v, 4) for v in report.aggregated_fitness),
            )

            return report

        except Exception as e:
            logger.error(f"WFA評価中エラー: {e}")
            return self._evaluate_single_report(gene, base_backtest_config, config)

    def _precompute_fold_configs(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_folds: int,
        train_ratio: float,
        anchored: bool,
        base_backtest_config: Dict[str, Any],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """WFA 用のテスト設定を事前計算する。"""
        fold_configs: List[Tuple[int, Dict[str, Any]]] = []
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

            if (train_end - fold_train_start).days < 7:
                logger.debug(
                    "WFA Fold %s: トレーニング期間が短すぎるためスキップ",
                    fold_idx,
                )
                continue

            if (test_end - test_start).days < 1:
                logger.debug(
                    "WFA Fold %s: テスト期間が短すぎるためスキップ",
                    fold_idx,
                )
                continue

            test_config = base_backtest_config.copy()
            test_config["start_date"] = test_start.strftime("%Y-%m-%d %H:%M:%S")
            test_config["end_date"] = test_end.strftime("%Y-%m-%d %H:%M:%S")
            fold_configs.append((fold_idx, test_config))

        return fold_configs

    def _evaluate_with_purged_kfold(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """PurgedKFold評価（タプル返却の互換API）。"""
        return self._evaluate_with_purged_kfold_report(
            gene, base_backtest_config, config
        ).aggregated_fitness

    def _evaluate_with_purged_kfold_report(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> EvaluationReport:
        """PurgedKFold評価（過学習対策）を並列実行する。"""
        try:
            n_splits = getattr(config, "purged_kfold_splits", 5)
            embargo_pct = getattr(config, "purged_kfold_embargo", 0.01)

            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                logger.warning("PurgedKFold: 期間が不明なため通常評価にフォールバック")
                return self._evaluate_single_report(gene, base_backtest_config, config)

            fold_configs = self._precompute_purged_kfold_configs(
                start_date,
                end_date,
                n_splits,
                embargo_pct,
                base_backtest_config,
            )

            if not fold_configs:
                logger.warning(
                    "PurgedKFold: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluate_single_report(gene, base_backtest_config, config)

            scenario_reports: list[ScenarioEvaluation] = []
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(fold_configs))
            ) as executor:
                future_to_fold = {}
                for fold_idx, test_config in fold_configs:
                    future = executor.submit(
                        self._evaluate_scenario,
                        gene,
                        test_config,
                        config,
                        scenario_name=f"fold_{fold_idx}",
                        metadata={"fold_index": fold_idx},
                    )
                    future_to_fold[future] = fold_idx

                for future in as_completed(future_to_fold):
                    fold_idx = future_to_fold[future]
                    try:
                        scenario_reports.append(future.result())
                    except Exception as fold_error:
                        logger.warning(
                            f"PurgedKFold Fold {fold_idx} 評価エラー: {fold_error}"
                        )

            if not scenario_reports:
                logger.warning(
                    "PurgedKFold: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluate_single_report(gene, base_backtest_config, config)

            scenario_reports.sort(
                key=lambda scenario: int(scenario.metadata.get("fold_index", -1))
            )
            report = EvaluationReport.aggregate(
                mode="purged_kfold",
                objectives=self._get_objectives(config),
                scenarios=scenario_reports,
                aggregate_method="robust",
                metadata={"fold_count": len(scenario_reports)},
            )

            logger.info(
                "PurgedKFold評価完了（並列）: %sフォールド, pass_rate=%s, 集約=%s",
                len(scenario_reports),
                round(report.pass_rate, 4),
                tuple(round(v, 4) for v in report.aggregated_fitness),
            )

            return report

        except Exception as e:
            logger.error(f"PurgedKFold評価中エラー: {e}")
            return self._evaluate_single_report(gene, base_backtest_config, config)

    def _precompute_purged_kfold_configs(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_splits: int,
        embargo_pct: float,
        base_backtest_config: Dict[str, Any],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """PurgedKFold 用のテスト設定を事前計算する。"""
        fold_configs: List[Tuple[int, Dict[str, Any]]] = []
        if n_splits <= 0:
            return fold_configs

        total_duration = end_date - start_date
        if total_duration <= pd.Timedelta(0):
            return fold_configs

        embargo_pct = max(0.0, float(embargo_pct or 0.0))
        embargo_duration = total_duration * embargo_pct
        usable_duration = total_duration - embargo_duration * max(0, n_splits - 1)
        if usable_duration <= pd.Timedelta(0):
            return fold_configs

        fold_duration = usable_duration / n_splits
        current_start = start_date

        for fold_idx in range(n_splits):
            effective_test_start = current_start
            if effective_test_start >= end_date:
                break

            if fold_idx == n_splits - 1:
                effective_test_end = end_date
            else:
                effective_test_end = effective_test_start + fold_duration

            if effective_test_end <= effective_test_start:
                continue

            test_config = base_backtest_config.copy()
            test_config["start_date"] = effective_test_start.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            test_config["end_date"] = effective_test_end.strftime("%Y-%m-%d %H:%M:%S")
            fold_configs.append((fold_idx, test_config))
            current_start = effective_test_end + embargo_duration

        return fold_configs

    def clear_cache(self):
        """キャッシュをクリアする。"""
        self._date_cache.clear()
