"""
自動検証パイプラインサービス

GA 実行後に Walk-Forward Analysis (WFA) で最終候補戦略を検証し、
合格（Pass）/不合格（Fail）を判定します。
合格した戦略だけが DB へ保存されるように、結果をフィルタリングします。
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

from ..config import objective_registry
from ..config.ga import GAConfig
from ..config.ga.nested_configs import ValidationConfig
from ..core.evaluation.evaluation_report import EvaluationReport
from ..core.evaluation.evaluation_strategies import EvaluationStrategy
from ..core.evaluation.individual_evaluator import IndividualEvaluator
from ..genes.genetic_utils import GeneticUtils

logger = logging.getLogger(__name__)


class StrategyValidationService:
    """
    GA 生成戦略の自動検証サービス

    IndividualEvaluator を再利用して WFA 評価を実行し、
    ValidationConfig の合格基準に基づいて戦略をフィルタリングします。
    """

    def __init__(
        self,
        evaluator: IndividualEvaluator,
        max_workers: int = 2,
    ) -> None:
        """
        初期化

        Args:
            evaluator: GA 実行で使用された個体評価器（データキャッシュ再利用のため）
            max_workers: WFA フォールドの並列実行数
        """
        self._evaluator = evaluator
        self._evaluation_strategy = EvaluationStrategy(
            evaluator, max_workers=max_workers
        )

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def validate_and_filter_result(
        self,
        result: dict[str, Any],
        ga_config: GAConfig,
        backtest_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        GA 実行結果全体を検証し、合格した戦略のみを含む結果に絞り込みます。

        Args:
            result: GA エンジンが返した結果辞書
            ga_config: GA 実行設定（validation_config を含む）
            backtest_config: バックテスト実行設定

        Returns:
            検証結果でフィルタリングされた結果辞書。
            検証が無効な場合は元の結果をそのまま返します。
            検証結果の詳細は result["validation_results"] に格納されます。
        """
        validation_config = ga_config.validation_config
        if not validation_config.enabled:
            return result

        logger.info("自動検証パイプラインを開始します（WFA）")

        validation_ga_config = self._build_validation_ga_config(ga_config)
        validation_results: dict[str, dict[str, Any]] = {}

        best_strategy = result.get("best_strategy")
        best_key = self._get_strategy_key(best_strategy) if best_strategy else None

        # 1. 検証対象の収集（最良戦略 + 上位候補 + パレートフロント）
        candidates_to_validate: list[Any] = []
        if best_strategy is not None:
            candidates_to_validate.append(best_strategy)

        if validation_config.validate_candidates:
            for strategy in result.get("all_strategies", [])[
                : validation_config.max_candidates
            ]:
                key = self._get_strategy_key(strategy)
                if key != best_key:
                    candidates_to_validate.append(strategy)
            # パレートフロントの戦略も検証対象に含める
            for solution in result.get("pareto_front", []):
                strategy = (
                    solution.get("strategy")
                    if isinstance(solution, Mapping)
                    else solution
                )
                if strategy is not None:
                    key = self._get_strategy_key(strategy)
                    if key != best_key:
                        candidates_to_validate.append(strategy)

        # 2. 各候補の WFA 検証
        for strategy in candidates_to_validate:
            key = self._get_strategy_key(strategy)
            if key in validation_results:
                continue
            validation_results[key] = self._validate_strategy(
                strategy,
                backtest_config,
                validation_ga_config,
                validation_config,
            )

        # 3. フィルタリング
        filtered = self._filter_result(result, validation_results, validation_config)
        filtered["validation_results"] = validation_results

        passed_count = sum(
            1 for v in validation_results.values() if v.get("passed", False)
        )
        logger.info(
            "自動検証完了: 検証 %d 件 / 合格 %d 件",
            len(validation_results),
            passed_count,
        )
        return filtered

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------

    def _validate_strategy(
        self,
        strategy: Any,
        backtest_config: dict[str, Any],
        validation_ga_config: GAConfig,
        validation_config: ValidationConfig,
    ) -> dict[str, Any]:
        """単一戦略の WFA 評価を実行し、合格判定を返す。"""
        try:
            report = self._evaluation_strategy.execute_report(
                strategy,
                backtest_config,
                validation_ga_config,
            )
            return self._judge(report, validation_config)
        except Exception as e:
            logger.warning("戦略の自動検証に失敗しました: %s", e)
            return {
                "passed": False,
                "pass_rate": 0.0,
                "primary_fitness": None,
                "worst_case_fitness": None,
                "scenario_count": 0,
                "mode": "error",
                "reasons": [f"検証実行エラー: {e}"],
            }

    def _judge(
        self,
        report: EvaluationReport,
        validation_config: ValidationConfig,
    ) -> dict[str, Any]:
        """評価レポートを合格基準に照らして判定する。"""
        reasons: list[str] = []

        pass_rate = report.pass_rate
        if pass_rate < validation_config.min_pass_rate:
            reasons.append(
                f"WFA合格率 {pass_rate:.2f} が閾値 "
                f"{validation_config.min_pass_rate} 未満"
            )

        primary_fitness = report.primary_aggregated_fitness
        if (
            validation_config.min_primary_fitness is not None
            and primary_fitness is not None
        ):
            # 最小化目的（例: max_drawdown）の場合は方向が逆になる
            primary_objective = report.primary_objective or ""
            is_minimize = objective_registry.is_minimize_objective(primary_objective)
            if is_minimize:
                below_threshold = (
                    primary_fitness > validation_config.min_primary_fitness
                )
            else:
                below_threshold = (
                    primary_fitness < validation_config.min_primary_fitness
                )
            if below_threshold:
                reasons.append(
                    f"集約フィットネス {primary_fitness:.4f} が閾値 "
                    f"{validation_config.min_primary_fitness} を満たしません"
                )

        # 全フォールドの最悪値で取引数 / ドローダウンをチェック
        trades_list = self._collect_scenario_metric(report, "total_trades")
        if validation_config.min_trades is not None and trades_list:
            min_trades = min(trades_list)
            if min_trades < validation_config.min_trades:
                reasons.append(
                    f"最少取引数 {min_trades} が閾値 "
                    f"{validation_config.min_trades} 未満"
                )

        dd_list = self._collect_scenario_metric(report, "max_drawdown")
        if validation_config.max_drawdown is not None and dd_list:
            max_dd = max(abs(v) for v in dd_list)
            if max_dd > validation_config.max_drawdown:
                reasons.append(
                    f"最大ドローダウン {max_dd:.2%} が上限 "
                    f"{validation_config.max_drawdown:.2%} 超過"
                )

        return {
            "passed": len(reasons) == 0,
            "pass_rate": pass_rate,
            "primary_fitness": primary_fitness,
            "worst_case_fitness": report.primary_worst_case_fitness,
            "scenario_count": len(report.scenarios),
            "mode": report.mode,
            "reasons": reasons,
        }

    @staticmethod
    def _collect_scenario_metric(
        report: EvaluationReport,
        metric_name: str,
    ) -> list[float]:
        """全シナリオから指定メトリクスの数値リストを収集する。"""
        values: list[float] = []
        for scenario in report.scenarios:
            metrics = scenario.performance_metrics or {}
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _filter_result(
        self,
        result: dict[str, Any],
        validation_results: dict[str, dict[str, Any]],
        validation_config: ValidationConfig,
    ) -> dict[str, Any]:
        """検証結果に基づいて結果辞書をフィルタリングする。"""
        all_strategies = result.get("all_strategies", [])
        fitness_scores = result.get("fitness_scores", [])

        # 合格した戦略のみを残す
        passing_all: list[Any] = []
        passing_fitness: list[float] = []
        for i, strategy in enumerate(all_strategies):
            key = self._get_strategy_key(strategy)
            validation = validation_results.get(key)
            if validation is not None and validation.get("passed", False):
                passing_all.append(strategy)
                passing_fitness.append(
                    fitness_scores[i] if i < len(fitness_scores) else 0.0
                )

        # パレートフロントも合格のみ
        passing_pareto: list[Any] = []
        for solution in result.get("pareto_front", []):
            strategy = (
                solution.get("strategy") if isinstance(solution, Mapping) else solution
            )
            key = self._get_strategy_key(strategy)
            validation = validation_results.get(key)
            if validation is not None and validation.get("passed", False):
                passing_pareto.append(solution)

        filtered = dict(result)
        filtered["all_strategies"] = passing_all
        filtered["fitness_scores"] = passing_fitness
        filtered["pareto_front"] = passing_pareto

        # 最良戦略の決定（不合格なら上位の合格候補へ昇格）
        best_strategy = result.get("best_strategy")
        best_key = self._get_strategy_key(best_strategy) if best_strategy else None
        best_validation = (
            validation_results.get(best_key) if best_key is not None else None
        )

        if (
            best_strategy is not None
            and best_validation is not None
            and best_validation.get("passed", False)
        ):
            filtered["best_strategy"] = best_strategy
        elif passing_all:
            promoted = passing_all[0]
            filtered["best_strategy"] = promoted
            promoted_key = self._get_strategy_key(promoted)
            for i, strategy in enumerate(all_strategies):
                if self._get_strategy_key(strategy) == promoted_key:
                    filtered["best_fitness"] = (
                        fitness_scores[i] if i < len(fitness_scores) else None
                    )
                    break
            summaries = result.get("evaluation_summaries", {})
            # スタール値回避: 見つからない場合は明示的に None にする
            filtered["best_evaluation_summary"] = summaries.get(promoted_key)
            logger.info("最良戦略が不合格のため、合格した上位候補を昇格させました")
        elif passing_pareto:
            # all_strategies が空でも、合格したパレート戦略があれば昇格
            promoted = (
                passing_pareto[0].get("strategy")
                if isinstance(passing_pareto[0], Mapping)
                else passing_pareto[0]
            )
            filtered["best_strategy"] = promoted
            promoted_key = self._get_strategy_key(promoted)
            filtered["best_fitness"] = None
            summaries = result.get("evaluation_summaries", {})
            filtered["best_evaluation_summary"] = summaries.get(promoted_key)
            logger.info("合格したパレート戦略を最良戦略として昇格させました")
        else:
            filtered["best_strategy"] = None
            filtered["best_fitness"] = None
            filtered["best_evaluation_summary"] = None
            logger.warning("合格した戦略がありませんでした")

        return filtered

    def _build_validation_ga_config(self, ga_config: GAConfig) -> GAConfig:
        """WFA 検証用の GA 設定を構築する。"""
        validation_config = ga_config.validation_config
        validation_ga_config = copy.deepcopy(ga_config)

        evaluation_config = validation_ga_config.evaluation_config
        evaluation_config.enable_walk_forward = True
        evaluation_config.wfa_n_folds = validation_config.wfa_n_folds
        evaluation_config.wfa_train_ratio = validation_config.wfa_train_ratio
        evaluation_config.wfa_anchored = validation_config.wfa_anchored
        # OOS / PurgedKFold が WFA と競合しないように無効化
        evaluation_config.oos_split_ratio = 0.0
        validation_ga_config.enable_purged_kfold = False

        return validation_ga_config

    @staticmethod
    def _get_strategy_key(strategy: Any) -> str:
        """戦略を識別するキーを返す（永続化層と同じルール）。"""
        return GeneticUtils.get_strategy_result_key(strategy)
