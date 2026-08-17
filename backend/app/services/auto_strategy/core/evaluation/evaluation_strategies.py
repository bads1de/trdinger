"""
評価戦略モジュール

OOS (Out-of-Sample) 検証、Walk-Forward 分析などの
評価戦略ルーティングを担当します。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.config.helpers import (
    normalize_robustness_regime_windows,
)
from app.utils.datetime_utils import parse_datetime_range_optional

from .evaluation_report import (
    _DATETIME_FORMAT,
    EvaluationReport,
    ScenarioEvaluation,
)

if TYPE_CHECKING:
    from .individual_evaluator import IndividualEvaluator

logger = logging.getLogger(__name__)

# 評価モードの取りうる値（レポートの mode / 自動判定にも使用）
EVALUATION_MODES = frozenset({"single", "oos", "walk_forward", "purged_kfold"})
# evaluation_mode 未指定（自動判定）を表す値
AUTO_EVALUATION_MODE = "auto"


def resolve_evaluation_mode(config: GAConfig) -> str:
    """GAConfig から有効な評価モードを決定する。

    明示的な `evaluation_config.evaluation_mode` が設定されていればそれを優先し、
    "auto"（未指定）の場合は従来どおりフラグ群から自動判定する（後方互換）。

    Args:
        config: GA設定。

    Returns:
        str: "single" | "oos" | "walk_forward" | "purged_kfold" のいずれか。
    """
    evaluation_config = getattr(config, "evaluation_config", None)

    explicit = getattr(evaluation_config, "evaluation_mode", AUTO_EVALUATION_MODE)
    if explicit and explicit != AUTO_EVALUATION_MODE:
        if explicit in EVALUATION_MODES:
            return explicit
        logger.warning(
            "未知の evaluation_mode=%r を無視し、フラグから自動判定します", explicit
        )

    if getattr(config, "enable_purged_kfold", False):
        return "purged_kfold"
    if evaluation_config is not None and bool(
        getattr(evaluation_config, "enable_walk_forward", False)
    ):
        return "walk_forward"
    oos_ratio = float(getattr(evaluation_config, "oos_split_ratio", 0.0) or 0.0)
    if oos_ratio > 0.0:
        return "oos"
    return "single"


class EvaluationStrategy:
    """
    評価戦略ルーティングクラス

    IndividualEvaluator から委譲を受け、設定に応じて
    通常評価、OOS検証、Walk-Forward 分析のいずれかに振り分けます。
    """

    def __init__(self, evaluator: "IndividualEvaluator", max_workers: int = 2) -> None:
        self._evaluator = evaluator
        self._max_workers = max_workers
        self._date_cache: dict[str, tuple[str, str, str]] = {}
        self._last_scenario_task_failures = 0

    def execute_report(
        self, gene: Any, base_backtest_config: dict[str, Any], config: GAConfig
    ) -> EvaluationReport:
        """
        GA設定で定義された評価戦略（PurgedKFold、WFA、OOS、単一評価等）を選択・実行し、
        集約された評価レポートを生成します。

        このメソッドは、個体評価の「ルーティング」を担当します。
        モードは `evaluation_config.evaluation_mode` で明示的に指定され、
        未指定 ("auto") の場合は従来のフラグ群から自動判定されます
        （`resolve_evaluation_mode` を参照）。

        Args:
            gene (StrategyGene): 評価対象の戦略遺伝子。
            base_backtest_config (Dict[str, Any]): 基本となるバックテスト条件（銘柄、全期間等）。
            config (GAConfig): 評価モードや期間分割の詳細を含むGA設定。

        Returns:
            EvaluationReport: 実行された全シナリオの結果と、それらを集約した適応度を保持するオブジェクト。
        """
        mode = resolve_evaluation_mode(config)

        # PurgedKFold評価（過学習対策）
        if mode == "purged_kfold":
            return self._evaluate_with_purged_kfold_report(
                gene, base_backtest_config, config
            )

        # Walk-Forward 分析（最終検証・品質ゲート）
        if mode == "walk_forward":
            return self._evaluate_with_walk_forward_report(
                gene, base_backtest_config, config
            )

        # OOS 検証（IS/OOS 分割の重み付き集約）
        if mode == "oos":
            evaluation_config = getattr(config, "evaluation_config", None)
            oos_ratio = float(getattr(evaluation_config, "oos_split_ratio", 0.0) or 0.0)
            oos_weight = float(
                getattr(evaluation_config, "oos_fitness_weight", 0.5) or 0.5
            )
            return self._evaluate_with_oos_report(
                gene, base_backtest_config, config, oos_ratio, oos_weight
            )

        # 単一評価（既定）
        return self._evaluate_single_report(gene, base_backtest_config, config)

    def execute_ga_report(
        self, gene: Any, base_backtest_config: dict[str, Any], config: GAConfig
    ) -> EvaluationReport:
        """GA の選択に使う IS 専用評価を実行する。"""
        evaluation_config = getattr(config, "evaluation_config", None)
        if evaluation_config is not None and bool(
            getattr(evaluation_config, "enable_walk_forward_for_ga", False)
        ):
            ga_backtest_config: dict[str, Any] = base_backtest_config
            oos_ratio_ga = float(
                getattr(evaluation_config, "oos_split_ratio", 0.0) or 0.0
            )
            if oos_ratio_ga > 0.0:
                ga_backtest_config, _, _ = self._build_oos_split_configs(
                    base_backtest_config, oos_ratio_ga
                )
            return self._evaluate_ga_with_walk_forward_report(
                gene, ga_backtest_config, config
            )

        oos_ratio = float(getattr(evaluation_config, "oos_split_ratio", 0.0) or 0.0)
        metadata = {
            "evaluation_layer": "is",
            "method": "is",
            "evaluation_scope": "ga",
            "holdout_reserved": oos_ratio > 0.0,
        }

        if oos_ratio > 0.0:
            is_config, _, split_metadata = self._build_oos_split_configs(
                base_backtest_config,
                oos_ratio,
            )
            scenario = self._evaluate_scenario(
                gene,
                is_config,
                config,
                scenario_name="is",
                metadata={"segment": "is", "evaluation_scope": "ga"},
            )
            metadata.update(split_metadata)
            metadata["oos_fitness_weight_ignored"] = float(
                getattr(evaluation_config, "oos_fitness_weight", 0.5) or 0.5
            )
            return EvaluationReport.single(
                mode="in_sample",
                objectives=self._get_objectives(config),
                scenario=scenario,
                metadata=metadata,
            )

        scenario = self._evaluate_scenario(
            gene,
            base_backtest_config,
            config,
            scenario_name="single",
            metadata={"evaluation_scope": "ga"},
        )
        return EvaluationReport.single(
            mode="in_sample",
            objectives=self._get_objectives(config),
            scenario=scenario,
            metadata=metadata,
        )

    def execute_robustness_report(
        self, gene: Any, base_backtest_config: dict[str, Any], config: GAConfig
    ) -> EvaluationReport:
        """二段階選抜用に複数シナリオで頑健性評価を実行する。"""
        scenario_definitions = self._build_robustness_scenarios(
            base_backtest_config, config
        )
        if not scenario_definitions:
            return EvaluationReport.aggregate(
                mode="robustness",
                objectives=self._get_objectives(config),
                scenarios=[],
                metadata={
                    "evaluation_failed": True,
                    "failure_reason": "robustness シナリオがありません",
                },
            )

        scenario_reports = self._run_parallel_scenario_tasks(
            [
                (
                    order,
                    lambda scenario_name=scenario_name, scenario_config=scenario_config, metadata=metadata: (
                        self._evaluate_robustness_scenario_report(
                            gene,
                            scenario_name,
                            scenario_config,
                            config,
                            metadata,
                        )
                    ),
                )
                for order, scenario_name, scenario_config, metadata in scenario_definitions
            ],
            error_context="robustness scenario",
            order_metadata_key="scenario_order",
        )

        if not scenario_reports:
            return EvaluationReport.aggregate(
                mode="robustness",
                objectives=self._get_objectives(config),
                scenarios=[],
                metadata={
                    "evaluation_failed": True,
                    "failure_reason": "robustness シナリオ評価がすべて失敗しました",
                    "failed_scenario_count": self._last_scenario_task_failures,
                },
            )

        scenario_reports.sort(
            key=lambda scenario: int(scenario.metadata.get("scenario_order", -1))
        )
        robustness_config = getattr(config, "robustness_config", None)
        aggregate_method = str(
            getattr(robustness_config, "aggregate_method", "robust") or "robust"
        )
        failed_scenario_count = self._last_scenario_task_failures
        return EvaluationReport.aggregate(
            mode="robustness",
            objectives=self._get_objectives(config),
            scenarios=scenario_reports,
            aggregate_method=aggregate_method,
            metadata={
                "evaluation_layer": "robustness",
                "scenario_count": len(scenario_reports),
                "failed_scenario_count": failed_scenario_count,
                "evaluation_incomplete": failed_scenario_count > 0,
                "evaluation_failed": failed_scenario_count > 0,
            },
        )

    def _get_objectives(self, config: GAConfig) -> list[str]:
        objectives = getattr(config, "objectives", None)
        return list(objectives) if objectives else ["weighted_score"]

    @staticmethod
    def _resolve_backtest_date_range(
        base_backtest_config: dict[str, Any],
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """バックテスト期間を検証して Timestamp のペアを返す。"""
        start_val = base_backtest_config.get("start_date")
        end_val = base_backtest_config.get("end_date")
        parsed_range = parse_datetime_range_optional(start_val, end_val)
        if parsed_range is None:
            return None

        start_date = pd.Timestamp(parsed_range[0])
        end_date = pd.Timestamp(parsed_range[1])
        return cast(pd.Timestamp, start_date), cast(pd.Timestamp, end_date)

    @staticmethod
    def _get_execution_spread(backtest_config: dict[str, Any]) -> float:
        """spread/slippage 互換キーから execution spread を取得する。"""
        if "spread" in backtest_config and backtest_config.get("spread") is not None:
            return float(backtest_config.get("spread", 0.0) or 0.0)
        return float(backtest_config.get("slippage", 0.0) or 0.0)

    @classmethod
    def _set_execution_spread(
        cls,
        backtest_config: dict[str, Any],
        spread: float,
    ) -> None:
        """spread/slippage の両キーへ同じ値を書き戻す。"""
        normalized_spread = float(spread)
        backtest_config["spread"] = normalized_spread
        backtest_config["slippage"] = normalized_spread

    def _build_robustness_scenarios(
        self,
        base_backtest_config: dict[str, Any],
        config: GAConfig,
    ) -> list[tuple[int, str, dict[str, Any], dict[str, Any]]]:
        """robustness 評価対象のシナリオ設定を構築する。"""
        scenario_definitions = []
        seen_keys = set()
        order = 0

        def add_scenario(
            name: str,
            scenario_config: dict[str, Any],
            metadata: dict[str, Any],
        ) -> None:
            nonlocal order
            scenario_key = (
                str(scenario_config.get("symbol")),
                float(scenario_config.get("commission_rate", 0.0) or 0.0),
                self._get_execution_spread(scenario_config),
                str(scenario_config.get("start_date")),
                str(scenario_config.get("end_date")),
            )
            if scenario_key in seen_keys:
                return
            seen_keys.add(scenario_key)
            scenario_definitions.append((order, name, scenario_config, metadata))
            order += 1

        base_symbol = str(base_backtest_config.get("symbol", "") or "")
        base_spread = self._get_execution_spread(base_backtest_config)
        base_commission = float(base_backtest_config.get("commission_rate", 0.0) or 0.0)
        base_start_date = str(base_backtest_config.get("start_date", "") or "")
        base_end_date = str(base_backtest_config.get("end_date", "") or "")
        self._set_execution_spread(base_backtest_config, base_spread)

        add_scenario(
            "base",
            base_backtest_config.copy(),
            {
                "scenario_kind": "base",
                "symbol": base_symbol,
                "spread": base_spread,
                "slippage": base_spread,
                "commission_rate": base_commission,
                "start_date": base_start_date,
                "end_date": base_end_date,
            },
        )

        for symbol in (
            getattr(config.robustness_config, "validation_symbols", None) or []
        ):
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
                    "spread": base_spread,
                    "slippage": base_spread,
                    "commission_rate": base_commission,
                    "start_date": base_start_date,
                    "end_date": base_end_date,
                },
            )

        for regime_window in normalize_robustness_regime_windows(
            getattr(config.robustness_config, "regime_windows", [])
        ):
            scenario_config = base_backtest_config.copy()
            scenario_config["start_date"] = regime_window.start_date
            scenario_config["end_date"] = regime_window.end_date
            add_scenario(
                f"regime_{regime_window.name}",
                scenario_config,
                {
                    "scenario_kind": "regime",
                    "regime_name": regime_window.name,
                    "symbol": base_symbol,
                    "spread": base_spread,
                    "slippage": base_spread,
                    "commission_rate": base_commission,
                    "start_date": regime_window.start_date,
                    "end_date": regime_window.end_date,
                },
            )

        for slippage_delta in (
            getattr(config.robustness_config, "stress_slippage", []) or []
        ):
            scenario_config = base_backtest_config.copy()
            stressed_spread = base_spread + float(slippage_delta)
            self._set_execution_spread(scenario_config, stressed_spread)
            add_scenario(
                f"slippage_{round(stressed_spread, 10)}",
                scenario_config,
                {
                    "scenario_kind": "slippage",
                    "symbol": base_symbol,
                    "spread": stressed_spread,
                    "slippage": stressed_spread,
                    "commission_rate": base_commission,
                    "start_date": base_start_date,
                    "end_date": base_end_date,
                },
            )

        for multiplier in (
            getattr(config.robustness_config, "stress_commission_multipliers", []) or []
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
                    "spread": base_spread,
                    "slippage": base_spread,
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
        scenario_config: dict[str, Any],
        config: GAConfig,
        metadata: dict[str, Any],
    ) -> ScenarioEvaluation:
        """単一 robustness シナリオを評価し、外側用シナリオへ変換する。"""
        scenario = self._evaluate_scenario(
            gene,
            scenario_config,
            config,
            scenario_name=scenario_name,
            metadata=metadata,
        )
        scenario_metadata = scenario.metadata.copy()
        scenario_metadata["evaluation_layer"] = "robustness"
        return ScenarioEvaluation(
            name=scenario_name,
            fitness=tuple(float(value) for value in scenario.fitness),
            passed=scenario.passed,
            metadata=scenario_metadata,
            performance_metrics=scenario.performance_metrics,
        )

    def _run_parallel_scenario_tasks(
        self,
        tasks: list[tuple[int, Any]],
        *,
        error_context: str,
        order_metadata_key: str | None = None,
    ) -> list[ScenarioEvaluation]:
        """ScenarioEvaluation を返すタスク群を並列実行し、完了順に依らず回収する。"""
        self._last_scenario_task_failures = 0
        if not tasks:
            return []

        scenario_reports: list[ScenarioEvaluation] = []
        failed_count = 0
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(tasks))
        ) as executor:
            future_to_order = {executor.submit(task): order for order, task in tasks}

            for future in as_completed(future_to_order):
                order = future_to_order[future]
                try:
                    scenario = future.result()
                    if order_metadata_key is not None:
                        scenario.metadata.setdefault(order_metadata_key, order)
                    scenario_reports.append(scenario)
                except Exception as scenario_error:
                    failed_count += 1
                    logger.warning(
                        "%s %s 評価エラー: %s",
                        error_context,
                        order,
                        scenario_error,
                    )

        self._last_scenario_task_failures = failed_count
        return scenario_reports

    def _evaluate_single_report(
        self,
        gene: Any,
        backtest_config: dict[str, Any],
        config: GAConfig,
        *,
        metadata: dict[str, Any] | None = None,
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
            metadata=metadata,
        )

    def _evaluate_scenario(
        self,
        gene: Any,
        backtest_config: dict[str, Any],
        config: GAConfig,
        *,
        scenario_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ScenarioEvaluation:
        return self._evaluator._perform_single_evaluation_report(
            gene,
            backtest_config,
            config,
            scenario_name=scenario_name,
            metadata=metadata,
        )

    def _build_oos_split_configs(
        self,
        base_backtest_config: dict[str, Any],
        oos_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """IS/OOS のバックテスト設定と境界メタデータを構築する。"""
        if not 0.0 < oos_ratio < 1.0:
            raise ValueError(
                f"oos_split_ratio は0より大きく1未満である必要があります: {oos_ratio}"
            )

        date_range = self._resolve_backtest_date_range(base_backtest_config)
        if date_range is None:
            raise ValueError("OOS 分割に必要なバックテスト期間がありません")
        start_date, end_date = date_range

        cache_key = f"{start_date}_{end_date}_{oos_ratio}"
        if cache_key in self._date_cache:
            start_str, split_str, end_str = self._date_cache[cache_key]
        else:
            total_duration = end_date - start_date
            train_duration = total_duration * (1.0 - oos_ratio)
            split_date = start_date + train_duration

            start_str = start_date.strftime(_DATETIME_FORMAT)
            split_str = split_date.strftime(_DATETIME_FORMAT)
            end_str = end_date.strftime(_DATETIME_FORMAT)
            self._date_cache[cache_key] = (start_str, split_str, end_str)

        is_config = base_backtest_config.copy()
        is_config["start_date"] = start_str
        is_config["end_date"] = split_str

        timeframe = base_backtest_config.get("timeframe", "1h")
        try:
            bar_offset = pd.Timedelta(timeframe)
        except Exception as e:
            logger.debug(
                "タイムフレームのパーズに失敗しました、デフォルト値を使用します: %s",
                e,
            )
            bar_offset = pd.Timedelta(hours=1)
        oos_start = cast(pd.Timestamp, pd.Timestamp(split_str) + bar_offset)
        oos_config = base_backtest_config.copy()
        oos_config["start_date"] = oos_start.strftime(_DATETIME_FORMAT)
        oos_config["end_date"] = end_str

        metadata = {
            "oos_split_ratio": float(oos_ratio),
            "is_start_date": start_str,
            "is_end_date": split_str,
            "oos_start_date": oos_config["start_date"],
            "oos_end_date": end_str,
        }
        return is_config, oos_config, metadata

    def _evaluate_with_oos_report(
        self,
        gene: Any,
        base_backtest_config: dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> EvaluationReport:
        """Out-of-Sample (OOS) 検証を含む評価を実行する。"""
        try:
            is_config, oos_config, split_metadata = self._build_oos_split_configs(
                base_backtest_config,
                oos_ratio,
            )

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
                metadata={
                    "evaluation_layer": "validation",
                    "method": "oos",
                    "oos_weight": oos_weight,
                    **split_metadata,
                },
            )

            logger.info(
                "OOS評価完了（並列）: pass_rate=%s, Combined=%s",
                round(report.pass_rate, 4),
                report.aggregated_fitness,
            )
            return report

        except Exception as e:
            logger.error(f"OOS評価中エラー: {e}")
            return self._evaluate_single_report(
                gene,
                base_backtest_config,
                config,
                metadata={
                    "evaluation_fallback": True,
                    "fallback_reason": f"OOS評価エラー: {e}",
                },
            )

    def _evaluate_with_walk_forward_report(
        self,
        gene: Any,
        base_backtest_config: dict[str, Any],
        config: GAConfig,
    ) -> EvaluationReport:
        """Walk-Forward Analysis による評価を並列実行する。"""
        try:
            date_range = self._resolve_backtest_date_range(base_backtest_config)
            if date_range is None:
                logger.warning(
                    "WFA: 期間が不明または無効なため通常評価にフォールバック"
                )
                return self._evaluate_single_report(
                    gene,
                    base_backtest_config,
                    config,
                    metadata={
                        "evaluation_fallback": True,
                        "fallback_reason": "WFAのバックテスト期間が不正です",
                    },
                )
            start_date, end_date = date_range

            evaluation_config = getattr(config, "evaluation_config", None)
            n_folds = getattr(evaluation_config, "wfa_n_folds", 5)
            train_ratio = getattr(evaluation_config, "wfa_train_ratio", 0.7)
            anchored = getattr(evaluation_config, "wfa_anchored", False)

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
                return self._evaluate_single_report(
                    gene,
                    base_backtest_config,
                    config,
                    metadata={
                        "evaluation_fallback": True,
                        "fallback_reason": "WFAの有効なフォールドがありません",
                        "expected_fold_count": int(n_folds),
                        "completed_fold_count": 0,
                        "evaluation_incomplete": True,
                    },
                )

            scenario_reports = self._run_parallel_scenario_tasks(
                [
                    (
                        fold_idx,
                        lambda fold_idx=fold_idx, test_config=test_config: (
                            self._evaluate_scenario(
                                gene,
                                test_config,
                                config,
                                scenario_name=f"fold_{fold_idx}",
                                metadata={"fold_index": fold_idx},
                            )
                        ),
                    )
                    for fold_idx, test_config in fold_configs
                ],
                error_context="WFA Fold",
            )

            completed_fold_count = len(scenario_reports)
            expected_fold_count = int(n_folds)
            if not scenario_reports:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluate_single_report(
                    gene,
                    base_backtest_config,
                    config,
                    metadata={
                        "evaluation_fallback": True,
                        "fallback_reason": "WFAのフォールド評価がすべて失敗しました",
                        "expected_fold_count": expected_fold_count,
                        "completed_fold_count": 0,
                        "failed_fold_count": self._last_scenario_task_failures,
                        "evaluation_incomplete": True,
                    },
                )

            scenario_reports.sort(
                key=lambda scenario: int(scenario.metadata.get("fold_index", -1))
            )
            report = EvaluationReport.aggregate(
                mode="walk_forward",
                objectives=self._get_objectives(config),
                scenarios=scenario_reports,
                aggregate_method="robust",
                metadata={
                    "evaluation_layer": "validation",
                    "method": "rolling_holdout",
                    "fold_count": completed_fold_count,
                    "expected_fold_count": expected_fold_count,
                    "completed_fold_count": completed_fold_count,
                    "failed_fold_count": self._last_scenario_task_failures,
                    "evaluation_incomplete": completed_fold_count
                    != expected_fold_count,
                },
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
            return self._evaluate_single_report(
                gene,
                base_backtest_config,
                config,
                metadata={
                    "evaluation_fallback": True,
                    "fallback_reason": f"WFA評価エラー: {e}",
                    "evaluation_incomplete": True,
                },
            )

    def _evaluate_ga_with_walk_forward_report(
        self,
        gene: Any,
        base_backtest_config: dict[str, Any],
        config: GAConfig,
    ) -> EvaluationReport:
        """GA選抜用の IS 内 WFA 評価 (robust 集約で選択圧を汎化側へ寄せる)."""
        evaluation_config = getattr(config, "evaluation_config", None)
        if evaluation_config is not None and bool(
            getattr(evaluation_config, "enable_multi_fidelity_evaluation", False)
        ):
            try:
                from .evaluation_fidelity import is_coarse_fidelity

                if is_coarse_fidelity(config):
                    scenario = self._evaluate_scenario(
                        gene,
                        base_backtest_config,
                        config,
                        scenario_name="single",
                        metadata={
                            "evaluation_scope": "ga",
                            "wfa_ga_coarse_fallback": True,
                        },
                    )
                    return EvaluationReport.single(
                        mode="in_sample",
                        objectives=self._get_objectives(config),
                        scenario=scenario,
                        metadata={
                            "evaluation_layer": "is",
                            "method": "is",
                            "evaluation_scope": "ga",
                            "wfa_ga_coarse_fallback": True,
                        },
                    )
            except Exception:
                pass

        report = self._evaluate_with_walk_forward_report(
            gene, base_backtest_config, config
        )
        if report.metadata.get("evaluation_fallback"):
            return report
        report.metadata["evaluation_layer"] = "is"
        report.metadata["evaluation_scope"] = "ga"
        report.metadata["wfa_ga"] = True
        return report

    def _precompute_fold_configs(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_folds: int,
        train_ratio: float,
        anchored: bool,
        base_backtest_config: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        """WFA 用のテスト設定を事前計算する。"""
        fold_configs: list[tuple[int, dict[str, Any]]] = []
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
            # トレーニング期間の直後のバーからテスト期間を開始し、境界でのデータリークを防ぐ
            timeframe = base_backtest_config.get("timeframe", "1h")
            try:
                bar_offset = pd.Timedelta(timeframe)
            except Exception as e:
                logger.debug(
                    "タイムフレームのパーズに失敗しました、デフォルト値を使用します: %s",
                    e,
                )
                bar_offset = pd.Timedelta(hours=1)
            test_start = train_end + bar_offset
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
            test_config["start_date"] = cast(pd.Timestamp, test_start).strftime(
                _DATETIME_FORMAT
            )
            test_config["end_date"] = cast(pd.Timestamp, test_end).strftime(
                _DATETIME_FORMAT
            )
            fold_configs.append((fold_idx, test_config))

        return fold_configs

    def _evaluate_with_purged_kfold_report(
        self,
        gene: Any,
        base_backtest_config: dict[str, Any],
        config: GAConfig,
    ) -> EvaluationReport:
        """PurgedKFold評価（過学習対策）を並列実行する。"""
        try:
            n_splits = getattr(config, "purged_kfold_splits", 5)
            embargo_pct = getattr(config, "purged_kfold_embargo", 0.01)

            date_range = self._resolve_backtest_date_range(base_backtest_config)
            if date_range is None:
                logger.warning(
                    "PurgedKFold: 期間が不明または無効なため通常評価にフォールバック"
                )
                return self._evaluate_single_report(
                    gene,
                    base_backtest_config,
                    config,
                    metadata={
                        "evaluation_fallback": True,
                        "fallback_reason": "PurgedKFoldのバックテスト期間が不正です",
                        "expected_fold_count": int(n_splits),
                        "completed_fold_count": 0,
                        "evaluation_incomplete": True,
                    },
                )
            start_date, end_date = date_range

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
                return self._evaluate_single_report(
                    gene,
                    base_backtest_config,
                    config,
                    metadata={
                        "evaluation_fallback": True,
                        "fallback_reason": "PurgedKFoldの有効なフォールドがありません",
                        "expected_fold_count": int(n_splits),
                        "completed_fold_count": 0,
                        "evaluation_incomplete": True,
                    },
                )

            scenario_reports = self._run_parallel_scenario_tasks(
                [
                    (
                        fold_idx,
                        lambda fold_idx=fold_idx, test_config=test_config: (
                            self._evaluate_scenario(
                                gene,
                                test_config,
                                config,
                                scenario_name=f"fold_{fold_idx}",
                                metadata={"fold_index": fold_idx},
                            )
                        ),
                    )
                    for fold_idx, test_config in fold_configs
                ],
                error_context="PurgedKFold Fold",
            )

            completed_fold_count = len(scenario_reports)
            expected_fold_count = int(n_splits)
            if not scenario_reports:
                logger.warning(
                    "PurgedKFold: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._evaluate_single_report(
                    gene,
                    base_backtest_config,
                    config,
                    metadata={
                        "evaluation_fallback": True,
                        "fallback_reason": "PurgedKFoldのフォールド評価がすべて失敗しました",
                        "expected_fold_count": expected_fold_count,
                        "completed_fold_count": 0,
                        "failed_fold_count": self._last_scenario_task_failures,
                        "evaluation_incomplete": True,
                    },
                )

            scenario_reports.sort(
                key=lambda scenario: int(scenario.metadata.get("fold_index", -1))
            )
            report = EvaluationReport.aggregate(
                mode="purged_kfold",
                objectives=self._get_objectives(config),
                scenarios=scenario_reports,
                aggregate_method="robust",
                metadata={
                    "fold_count": completed_fold_count,
                    "expected_fold_count": expected_fold_count,
                    "completed_fold_count": completed_fold_count,
                    "failed_fold_count": self._last_scenario_task_failures,
                    "evaluation_incomplete": completed_fold_count
                    != expected_fold_count,
                },
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
            return self._evaluate_single_report(
                gene,
                base_backtest_config,
                config,
                metadata={
                    "evaluation_fallback": True,
                    "fallback_reason": f"PurgedKFold評価エラー: {e}",
                    "evaluation_incomplete": True,
                },
            )

    def _precompute_purged_kfold_configs(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        n_splits: int,
        embargo_pct: float,
        base_backtest_config: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        """PurgedKFold 用のテスト設定を事前計算する。"""
        fold_configs: list[tuple[int, dict[str, Any]]] = []
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
            test_config["start_date"] = effective_test_start.strftime(_DATETIME_FORMAT)
            test_config["end_date"] = effective_test_end.strftime(_DATETIME_FORMAT)
            fold_configs.append((fold_idx, test_config))
            current_start = effective_test_end + embargo_duration

        return fold_configs
