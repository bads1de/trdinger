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
from statistics import median
from typing import Any

from ..config import objective_registry
from ..config.ga_config import (
    EarlyTerminationSettings,
    GAConfig,
    RobustnessConfig,
    ValidationConfig,
)
from ..core.evaluation.evaluation_report import EvaluationReport
from ..core.evaluation.evaluation_strategies import EvaluationStrategy
from ..core.evaluation.individual_evaluator import IndividualEvaluator
from ..genes.genetic_utils import GeneticUtils
from .overfitting_metrics import (
    deflated_sharpe_ratio,
    estimate_test_window_years,
    probability_of_backtest_overfitting,
)

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

        validation_ga_config = self._build_validation_ga_config(
            ga_config, backtest_config
        )
        validation_results: dict[str, dict[str, Any]] = {}

        best_strategy = result.get("best_strategy")
        best_key = self._get_strategy_key(best_strategy) if best_strategy else None

        # 1. 検証対象の収集（最良戦略 + 指標構成が多様な上位候補 + パレートフロント）
        # fitness上位をそのまま取ると同一構成（例: MACD+BBANDS+ATR のみ）に偏るため、
        # 指標構成シグネチャが重複しないよう選ぶ（多様性優先の検証）。
        candidates_to_validate: list[Any] = []
        seen_keys: set[str] = set()
        seen_signatures: set[Any] = set()

        def _add_candidate(strategy: Any) -> None:
            if strategy is None:
                return
            key = self._get_strategy_key(strategy)
            if key in seen_keys:
                return
            seen_keys.add(key)
            candidates_to_validate.append(strategy)

        if best_strategy is not None:
            _add_candidate(best_strategy)
            seen_signatures.add(self._indicator_composition_signature(best_strategy))

        if validation_config.validate_candidates:
            # 候補プール: フィットネス順の全戦略 + パレートフロント
            pool: list[Any] = []
            for strategy in result.get("all_strategies", []):
                key = self._get_strategy_key(strategy)
                if key != best_key:
                    pool.append(strategy)
            for solution in result.get("pareto_front", []):
                strategy = (
                    solution.get("strategy")
                    if isinstance(solution, Mapping)
                    else solution
                )
                if strategy is not None:
                    pool.append(strategy)

            # 第1パス: 未見の指標構成を優先して選ぶ（多様性確保）
            target_limit = max(
                0, validation_config.max_candidates - len(candidates_to_validate)
            )
            diverse: list[Any] = []
            for strategy in pool:
                key = self._get_strategy_key(strategy)
                if key in seen_keys:
                    continue
                signature = self._indicator_composition_signature(strategy)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                diverse.append(strategy)
                if len(diverse) >= target_limit:
                    break

            # 第2パス: 多様性で埋め切れない場合は fitness 上位から補充
            if len(diverse) < target_limit:
                for strategy in pool:
                    key = self._get_strategy_key(strategy)
                    if key in seen_keys or strategy in diverse:
                        continue
                    diverse.append(strategy)
                    if len(diverse) >= target_limit:
                        break

            for strategy in diverse:
                _add_candidate(strategy)

        if validation_config.validate_pareto_front:
            # パレートフロントの全メンバーを枠外で検証する。
            # 枠外の無検証パレート解は _filter_result で破棄されるため、
            # 検証しないまま出力すると非劣解セットが欠落する。
            pareto_members: list[Any] = []
            for solution in result.get("pareto_front", []):
                strategy = (
                    solution.get("strategy")
                    if isinstance(solution, Mapping)
                    else solution
                )
                if strategy is not None:
                    pareto_members.append(strategy)
            for strategy in pareto_members:
                _add_candidate(strategy)

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
                ga_config.robustness_config,
            )

        # 3. フィルタリング
        filtered = self._filter_result(result, validation_results, validation_config)
        filtered["validation_results"] = validation_results
        filtered["validation_report_summaries"] = {
            key: value.get("report_summary")
            for key, value in validation_results.items()
            if value.get("report_summary") is not None
        }

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
        robustness_config: RobustnessConfig,
    ) -> dict[str, Any]:
        """単一戦略の WFA 評価を実行し、合格判定を返す。"""
        try:
            report = self._evaluation_strategy.execute_report(
                strategy,
                backtest_config,
                validation_ga_config,
            )
            validation = self._judge(
                report,
                validation_config,
                validation_ga_config,
                backtest_config,
            )
            if validation.get("passed") and robustness_config.enabled:
                validation = self._apply_robustness_gate(
                    validation,
                    strategy,
                    backtest_config,
                    validation_ga_config,
                    robustness_config,
                )
            return validation
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

    def _apply_robustness_gate(
        self,
        validation: dict[str, Any],
        strategy: Any,
        backtest_config: dict[str, Any],
        validation_ga_config: GAConfig,
        robustness_config: RobustnessConfig,
    ) -> dict[str, Any]:
        """WFA合格候補に robustness の fail-closed gate を適用する。"""
        reasons = list(validation.get("reasons", []))
        robustness_result: dict[str, Any]
        try:
            report = self._evaluator.evaluate_robustness_report(
                strategy,
                validation_ga_config,
            )
            if not isinstance(report, EvaluationReport):
                raise TypeError("robustness評価レポートが不正です")

            metadata = report.metadata if isinstance(report.metadata, dict) else {}
            scenario_count = len(report.scenarios)
            pass_rate = report.pass_rate
            robustness_reasons: list[str] = []
            if scenario_count == 0:
                robustness_reasons.append("robustnessシナリオがありません")
            if metadata.get("evaluation_failed"):
                robustness_reasons.append(
                    f"robustness評価に失敗しました: "
                    f"{metadata.get('failure_reason', '理由不明')}"
                )
            if metadata.get("evaluation_incomplete"):
                robustness_reasons.append("robustness評価が不完全です")
            if pass_rate < robustness_config.min_pass_rate:
                robustness_reasons.append(
                    f"robustness合格率 {pass_rate:.2f} が閾値 "
                    f"{robustness_config.min_pass_rate} 未満"
                )

            summary = report.to_summary_dict()
            robustness_result = {
                "passed": not robustness_reasons,
                "pass_rate": pass_rate,
                "scenario_count": scenario_count,
                "mode": report.mode,
                "reasons": robustness_reasons,
                "report_summary": summary,
            }
            if robustness_reasons:
                reasons.append("Robustness gate: " + "; ".join(robustness_reasons))
        except Exception as exc:
            robustness_result = {
                "passed": False,
                "pass_rate": 0.0,
                "scenario_count": 0,
                "mode": "error",
                "reasons": [f"robustness実行エラー: {exc}"],
                "report_summary": None,
            }
            reasons.append(f"Robustness gate: robustness実行エラー: {exc}")

        updated = dict(validation)
        updated["passed"] = not reasons
        updated["reasons"] = reasons
        updated["robustness"] = robustness_result
        return updated

    def _judge(
        self,
        report: EvaluationReport,
        validation_config: ValidationConfig,
        ga_config: GAConfig,
        backtest_config: dict[str, Any],
    ) -> dict[str, Any]:
        """評価レポートを合格基準に照らして判定する。"""
        reasons: list[str] = []
        scenario_count = len(report.scenarios)
        metadata = report.metadata if isinstance(report.metadata, dict) else {}

        # ペナルティ fitness（制約違反・評価エラー）を含むシナリオは
        # 集約 fitness から除外されるため、そのまま合格させると
        # 実質的にフォールドを無視した判定になる。fail-closed で不合格にする。
        penalized_count = int(metadata.get("penalized_scenario_count", 0) or 0)
        if penalized_count > 0:
            reasons.append(
                f"制約違反・評価エラーのフォールドが {penalized_count} 件あります"
            )

        pass_rate = report.pass_rate
        if scenario_count == 0:
            reasons.append("検証シナリオがありません")
        elif pass_rate < validation_config.min_pass_rate:
            reasons.append(
                f"WFA合格率 {pass_rate:.2f} が閾値 "
                f"{validation_config.min_pass_rate} 未満"
            )

        if metadata.get("evaluation_fallback"):
            reasons.append(
                "検証評価が通常評価へフォールバックしました: "
                f"{metadata.get('fallback_reason', '理由不明')}"
            )
        if metadata.get("evaluation_incomplete"):
            expected = metadata.get("expected_fold_count")
            completed = metadata.get("completed_fold_count")
            reasons.append(
                "検証評価が不完全です"
                + (
                    f"（期待 {expected} / 完了 {completed} フォールド）"
                    if expected is not None
                    else ""
                )
            )

        primary_fitness = report.primary_aggregated_fitness
        if validation_config.min_primary_fitness is not None:
            if primary_fitness is None:
                reasons.append("集約フィットネスを取得できません")
            else:
                primary_objective = report.primary_objective or ""
                is_minimize = objective_registry.is_minimize_objective(
                    primary_objective
                )
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

        # フォールド合格は構造的制約のみで判定するため、収益性はここで
        # 集約フィットネスの符号でゲートする。最大化目的で負値は
        # 「全期間を通して収益を上げられていない」ことを意味する。
        if (
            primary_fitness is not None
            and not objective_registry.is_minimize_objective(
                report.primary_objective or ""
            )
            and primary_fitness < 0.0
        ):
            reasons.append(f"集約フィットネス {primary_fitness:.4f} が負値です")

        trades_list = self._collect_scenario_metric(report, "total_trades")
        if validation_config.min_trades is not None:
            if len(trades_list) != scenario_count:
                reasons.append(
                    f"total_trades メトリクスが全シナリオで取得できません"
                    f"（{len(trades_list)}/{scenario_count}）"
                )
            else:
                min_trades = min(trades_list)
                if min_trades < validation_config.min_trades:
                    reasons.append(
                        f"最少取引数 {min_trades} が閾値 "
                        f"{validation_config.min_trades} 未満"
                    )

        dd_list = self._collect_scenario_metric(report, "max_drawdown")
        if validation_config.max_drawdown is not None:
            if len(dd_list) != scenario_count:
                reasons.append(
                    f"max_drawdown メトリクスが全シナリオで取得できません"
                    f"（{len(dd_list)}/{scenario_count}）"
                )
            else:
                max_dd = max(abs(v) for v in dd_list)
                if max_dd > validation_config.max_drawdown:
                    reasons.append(
                        f"最大ドローダウン {max_dd:.2%} が上限 "
                        f"{validation_config.max_drawdown:.2%} 超過"
                    )

        # PBO ゲート: 負けフォールド（total_return < 0）の比率が閾値を超えると不合格。
        # WFA のフォールド合格は構造的制約のみで判定されるため、
        # 「収益性の観点で過半数のフォールドが負け」の戦略をここで弾く。
        pbo: float | None = None
        n_losing_folds: int | None = None
        if validation_config.enable_pbo_gate:
            fold_returns = self._collect_scenario_metric(report, "total_return")
            if scenario_count > 0 and len(fold_returns) == scenario_count:
                pbo = probability_of_backtest_overfitting(fold_returns)
                n_losing_folds = sum(1 for value in fold_returns if value < 0.0)
                if pbo is not None and pbo > validation_config.pbo_threshold:
                    reasons.append(
                        f"PBO超過: 負けフォールド {n_losing_folds}/{scenario_count} "
                        f"(PBO={pbo:.2f}) が閾値 "
                        f"{validation_config.pbo_threshold:.2f} を超えています"
                    )
            else:
                logger.debug(
                    "PBOゲート: total_return が全シナリオで取得できないためスキップ"
                )

        # DSR ゲート: 多重検定補正後のシャープレシオ有意性（厳格な統計検定）。
        dsr: float | None = None
        if validation_config.enable_dsr_gate:
            fold_sharpes = self._collect_scenario_metric(report, "sharpe_ratio")
            if scenario_count > 0 and len(fold_sharpes) == scenario_count:
                observed_sr = float(median(fold_sharpes))
                n_years = estimate_test_window_years(
                    str(backtest_config.get("start_date", "") or ""),
                    str(backtest_config.get("end_date", "") or ""),
                    int(validation_config.wfa_n_folds),
                    float(validation_config.wfa_train_ratio),
                )
                effective_trials = validation_config.dsr_effective_trials
                if effective_trials is None:
                    effective_trials = max(
                        1,
                        int(ga_config.population_size)
                        * max(1, int(ga_config.generations)),
                    )
                if n_years is None or n_years < 1.0:
                    reasons.append(
                        "DSRゲート: 検証期間が短すぎて統計力を確保できません"
                    )
                else:
                    dsr = deflated_sharpe_ratio(
                        observed_sharpe=observed_sr,
                        n_observations=n_years,
                        n_trials=int(effective_trials),
                        sigma_sharpe=float(validation_config.dsr_sigma_sharpe or 1.0),
                    )
                    if dsr < validation_config.min_dsr:
                        reasons.append(
                            f"DSR {dsr:.3f} が閾値 "
                            f"{validation_config.min_dsr} 未満"
                            f"（SR={observed_sr:.2f}, 試行数={effective_trials}）"
                        )
            else:
                logger.debug(
                    "DSRゲート: sharpe_ratio が全シナリオで取得できないためスキップ"
                )

        report_summary = None
        to_summary_dict = getattr(report, "to_summary_dict", None)
        if callable(to_summary_dict):
            summary = to_summary_dict()
            if isinstance(summary, dict):
                report_summary = summary

        return {
            "passed": len(reasons) == 0,
            "pass_rate": pass_rate,
            "primary_fitness": primary_fitness,
            "worst_case_fitness": report.primary_worst_case_fitness,
            "scenario_count": scenario_count,
            "mode": report.mode,
            "reasons": reasons,
            "pbo": pbo,
            "n_losing_folds": n_losing_folds,
            "dsr": dsr,
            "report_summary": report_summary,
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

    def _build_validation_ga_config(
        self, ga_config: GAConfig, backtest_config: dict[str, Any] | None = None
    ) -> GAConfig:
        """WFA 検証用の GA 設定を構築する。"""
        validation_config = ga_config.validation_config
        validation_ga_config = copy.deepcopy(ga_config)

        evaluation_config = validation_ga_config.evaluation_config
        evaluation_config.enable_walk_forward = True
        evaluation_config.wfa_n_folds = validation_config.wfa_n_folds
        evaluation_config.wfa_train_ratio = validation_config.wfa_train_ratio
        evaluation_config.wfa_anchored = validation_config.wfa_anchored
        # 最終品質ゲートは明示的に WFA モードへ固定する。
        evaluation_config.evaluation_mode = "walk_forward"
        # OOS 分割が WFA と競合しないように無効化（後方互換のため明示）
        evaluation_config.oos_split_ratio = 0.0

        # 検証は最終的な品質ゲートのため、完全評価で行う。
        # 早期終了（trade_pace など）が発動するとフォールドが -Infinity で
        # 不合格になり、本来評価可能な戦略まで弾いてしまうため無効化する。
        evaluation_config.early_termination_settings = EarlyTerminationSettings(
            enabled=False
        )

        # フォールド個別の窓長スケーリングは IndividualEvaluator 側で自動適用される
        # (fidelity_backtest_config の start/end から window_days を導出)。
        # ここでは二重スケールを避けるため事前スケールは行わない。
        # WFA のフォールド合格は構造的制約（取引数・ドローダウン）だけで判定する。
        # 収益性制約（負リターン・最低シャープレシオ）をフォールドごとに適用すると、
        # 短期間のテスト窓では相場環境の影響で正当な戦略まで弾かれる。
        # 収益性は集約フィットネス（_judge の負値ゲート）で評価する。
        validation_ga_config.fitness_constraints["min_total_return"] = None
        validation_ga_config.fitness_constraints["min_sharpe_ratio"] = None

        return validation_ga_config

    @staticmethod
    def _get_strategy_key(strategy: Any) -> str:
        """戦略を識別するキーを返す（永続化層と同じルール）。"""
        return GeneticUtils.get_strategy_result_key(strategy)

    @staticmethod
    def _indicator_composition_signature(strategy: Any) -> tuple[Any, ...]:
        """指標構成（タイプの集合）から多様性判定用のシグネチャを構築する。

        パラメータの違いは無視し、どの指標タイプを組み合わせているかのみで
        判定する（例: MACD+BBANDS+ATR はパラメータが異なっても同一扱い）。
        """
        indicator_types: set[Any] = set()
        for indicator in getattr(strategy, "indicators", None) or []:
            if not getattr(indicator, "enabled", True):
                continue
            indicator_types.add(str(getattr(indicator, "type", "")))
        return tuple(sorted(indicator_types))
