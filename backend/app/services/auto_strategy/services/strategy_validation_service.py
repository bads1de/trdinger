"""
自動検証パイプラインサービス

GA 実行後に Walk-Forward Analysis (WFA) で最終候補戦略を検証し、
合格（Pass）/不合格（Fail）を判定します。
合格した戦略だけが DB へ保存されるように、結果をフィルタリングします。
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from statistics import median
from typing import Any

import pandas as pd

from ..config import objective_registry
from ..config.ga_config import (
    EarlyTerminationSettings,
    GAConfig,
    RobustnessConfig,
    ValidationConfig,
)
from ..core.evaluation.evaluation_report import EvaluationReport
from ..core.evaluation.evaluation_strategies import EvaluationStrategy
from ..core.evaluation.individual_evaluator import (
    IndividualEvaluator,
)
from ..genes.genetic_utils import GeneticUtils
from ..utils.indicator_references import (
    collect_indicator_references,
    indicator_reference_names,
)
from .overfitting_metrics import (
    deflated_sharpe_ratio,
    estimate_aggregate_tested_years,
    probability_of_backtest_overfitting,
)
from .parameter_perturbation import build_perturbed_genes

logger = logging.getLogger(__name__)

# 絶対下限ゲート: 全フォールド複利合算の通期リターンの最低水準。
# 市場比でスキルがあっても、通期で赤字の戦略は単独運用に耐えないため
# この水準を下回れば不合格とする。
MIN_AGGREGATED_TOTAL_RETURN = 0.0


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
        # フルピリオド評価時に得た行動フィンガープリント（keyごと）。
        # validate_and_filter_result の冒頭で毎回クリアされる。
        self._behavior_fingerprints: dict[str, str] = {}

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
        if not validation_config.is_enabled:
            return result

        logger.info("自動検証パイプラインを開始します（WFA）")

        validation_ga_config = self._build_validation_ga_config(
            ga_config, backtest_config
        )
        validation_results: dict[str, dict[str, Any]] = {}
        self._behavior_fingerprints.clear()

        best_strategy = result.get("best_strategy")
        best_key = self._get_strategy_key(best_strategy) if best_strategy else None

        # 1. 検証対象の収集（最良戦略 + 行動シグネチャが多様な上位候補 + パレートフロント）
        # fitness上位をそのまま取ると同一骨格のクローン個体に偏るため、
        # 行動シグネチャ（死んだ遺伝子・ID差分を無視）が重複しないよう選ぶ。
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
            seen_signatures.add(self._behavioral_signature(best_strategy))

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

            # 第1パス: 未見の行動シグネチャを優先して選ぶ（多様性確保）
            target_limit = max(
                0, validation_config.max_candidates - len(candidates_to_validate)
            )
            diverse: list[Any] = []
            if target_limit > 0:
                for strategy in pool:
                    key = self._get_strategy_key(strategy)
                    if key in seen_keys:
                        continue
                    signature = self._behavioral_signature(strategy)
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
        key_to_strategy: dict[str, Any] = {}
        for strategy in candidates_to_validate:
            key = self._get_strategy_key(strategy)
            if key in validation_results:
                continue
            key_to_strategy[key] = strategy
            validation_results[key] = self._validate_strategy(
                strategy,
                backtest_config,
                validation_ga_config,
                validation_config,
                ga_config.robustness_config,
            )

        # 2.5 行動重複の排除（取引列フィンガープリント一致クローンの統合）
        self._attach_behavior_fingerprints(validation_results, key_to_strategy)
        validation_results = self._collapse_behavioral_duplicates(validation_results)

        # 3. 頑健性ベースの並び替え
        # メタ分析（n=16, Spearman +0.62 / fitness は -0.04）に基づき、
        # レジーム一貫性の最小スライスマージンを第一キー、fitness を第二キー
        # （未記録・フォールバック）として出力順序を決める。
        # 合格者は常に不合格者より上位に置く。
        validation_results = self._rank_validation_results(validation_results)

        # 4. フィルタリング
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

    def _rank_validation_results(
        self, validation_results: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """検証結果を頑健性ベースの順序に並び替える。

        ソートキー（降順）:
          1. passed（合格者を常に不合格者より上位へ）
          2. レジーム一貫性の最小スライスマージン（未記録は最下位扱い）
          3. primary_fitness（同点・未記録時のフォールバック）

        根拠: OOS予測力メタ分析において fitness は同期間内で予測力ゼロ
        （Spearman -0.04）だった一方、レジーム最小マージンは正の予測力
        （n=16, Spearman +0.62）を示した。サンプル蓄積後の再検証を前提と
        した仮説ベースの改善である。
        """

        def sort_key(item: tuple[str, dict[str, Any]]) -> tuple:
            result = item[1]
            passed = bool(result.get("passed", False))
            rc = result.get("regime_consistency") or {}
            slices = rc.get("slices") or []
            margins = [
                float(s["excess"])
                for s in slices
                if isinstance(s, dict) and isinstance(s.get("excess"), (int, float))
            ]
            has_margin = bool(margins)
            min_margin = min(margins) if margins else 0.0
            fit = result.get("primary_fitness")
            fit_value = (
                float(fit)
                if isinstance(fit, (int, float)) and math.isfinite(float(fit))
                else float("-inf")
            )
            return (passed, has_margin, min_margin, fit_value)

        return dict(sorted(validation_results.items(), key=sort_key, reverse=True))

    # ------------------------------------------------------------------
    # 行動重複排除（バックテスト出力ベース）
    # ------------------------------------------------------------------

    def _behavior_fingerprint_from_cache(self, strategy: Any) -> str | None:
        """評価器キャッシュのフルピリオドレポートから行動フィンガープリントを取得する。

        キャッシュが無い（並列ワーカーで評価された等）場合は None を返し、
        呼び出し側はその個体を重複判定の対象外とする（fail-open）。
        """
        if strategy is None:
            return None
        try:
            report = self._evaluator.get_cached_evaluation_report(strategy)
        except Exception:  # noqa: BLE001
            return None
        scenarios = getattr(report, "scenarios", None)
        if not isinstance(scenarios, list) or not scenarios:
            return None
        metadata = getattr(scenarios[0], "metadata", None)
        if not isinstance(metadata, dict):
            return None
        fingerprint = metadata.get("behavior_fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            return fingerprint
        return None

    def _attach_behavior_fingerprints(
        self,
        validation_results: dict[str, dict[str, Any]],
        key_to_strategy: dict[str, Any],
    ) -> None:
        """検証結果に取引列フィンガープリントを添付する（取得できた場合のみ）。

        第1経路: _get_full_period_metrics 実行時に記録した値。
        第2経路: 評価器キャッシュのレポート（GA評価が同一プロセスだった場合）。
        """
        for key, result in validation_results.items():
            fingerprint = self._behavior_fingerprints.get(key) or (
                self._behavior_fingerprint_from_cache(key_to_strategy.get(key))
            )
            if fingerprint:
                result["behavior_fingerprint"] = fingerprint

    @staticmethod
    def _collapse_behavioral_duplicates(
        validation_results: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """取引フィンガープリントが一致する候補を1体に統合する。

        同一グループ内では (passed, primary_fitness) が最も良い個体を残し、
        残りは不合格へ変更して duplicate_of / reasons を記録する。
        フィンガープリント未取得の個体は対象外（fail-open）。
        """
        groups: dict[str, list[str]] = {}
        for key, result in validation_results.items():
            fingerprint = result.get("behavior_fingerprint")
            if fingerprint:
                groups.setdefault(str(fingerprint), []).append(key)

        def merit(key: str) -> tuple[bool, float]:
            result = validation_results[key]
            fit = result.get("primary_fitness")
            fit_value = (
                float(fit)
                if isinstance(fit, (int, float)) and math.isfinite(float(fit))
                else float("-inf")
            )
            return (bool(result.get("passed", False)), fit_value)

        for _fingerprint, keys in groups.items():
            if len(keys) < 2:
                continue
            ordered = sorted(keys, key=merit, reverse=True)
            kept = ordered[0]
            for duplicate in ordered[1:]:
                dup_result = validation_results[duplicate]
                dup_result["passed"] = False
                dup_result["duplicate_of"] = kept
                reason = (
                    f"行動的に同一なクローン個体（取引fingerprint一致）: {kept} を優先"
                )
                reasons = dup_result.setdefault("reasons", [])
                if reason not in reasons:
                    reasons.append(reason)

        return validation_results

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
                strategy=strategy,
            )
            if (
                validation.get("passed")
                and validation_config.regime_consistency_enabled
            ):
                validation = self._apply_regime_consistency_gate(
                    validation,
                    strategy,
                    backtest_config,
                    validation_ga_config,
                    validation_config,
                )
            if validation.get("passed") and robustness_config.enabled:
                validation = self._apply_robustness_gate(
                    validation,
                    strategy,
                    backtest_config,
                    validation_ga_config,
                    robustness_config,
                )
            if (
                validation.get("passed")
                and robustness_config.param_perturbation_enabled
            ):
                validation = self._apply_parameter_perturbation_gate(
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

    def _apply_regime_consistency_gate(
        self,
        validation: dict[str, Any],
        strategy: Any,
        backtest_config: dict[str, Any],
        validation_ga_config: GAConfig,
        validation_config: ValidationConfig,
    ) -> dict[str, Any]:
        """レジーム一貫性ゲート（fail-closed）。

        学習期間を regime_splits 個の等間隔スライスに分割し、全スライスで:
          - 市場比 ≥ 0（その区間の Buy&Hold に勝つ）
          - 取引数 ≥ ceil(min_trades / splits)
        を要求する。片レジーム依存の戦略（強気にだけ乗る／ベアにだけ
        沿う）を出力から排除し、レジーム適応範囲を学習期間内で検証する。
        """
        reasons = list(validation.get("reasons", []))
        try:
            if not getattr(validation_config, "regime_consistency_enabled", True):
                return validation
            try:
                start = pd.Timestamp(str(backtest_config.get("start_date", "") or ""))
                end = pd.Timestamp(str(backtest_config.get("end_date", "") or ""))
            except Exception:  # noqa: BLE001
                return validation
            splits = int(validation_config.regime_splits)
            if splits < 2 or pd.isna(start) or pd.isna(end) or end <= start:
                return validation

            span = end - start
            min_trades_total = int(validation_config.min_trades or 0)
            min_trades_per_slice = math.ceil(min_trades_total / splits)

            slice_rows: list[dict[str, Any]] = []
            gate_reasons: list[str] = []
            for index in range(splits):
                slice_start = start + span * index / splits
                slice_end = start + span * (index + 1) / splits
                slice_cfg = dict(backtest_config)
                slice_cfg["start_date"] = slice_start.strftime("%Y-%m-%d")
                slice_cfg["end_date"] = slice_end.strftime("%Y-%m-%d")

                metrics = self._evaluate_variant_metrics(
                    strategy, slice_cfg, validation_ga_config
                )
                label = f"slice{index + 1}({slice_cfg['start_date']}~{slice_cfg['end_date']})"
                if metrics is None:
                    slice_rows.append({"label": label, "passed": False, "error": True})
                    gate_reasons.append(f"{label}: 評価に失敗")
                    continue
                total = float(metrics["total_return"])
                bh = float(metrics["buy_hold_return"])
                excess = total - bh
                trades = float(metrics.get("total_trades") or 0.0)
                passed_slice = excess >= 0.0 and trades >= min_trades_per_slice
                if not passed_slice:
                    detail = []
                    if excess < 0.0:
                        detail.append(f"市場比 {excess:+.2%}")
                    if trades < min_trades_per_slice:
                        detail.append(f"取引数 {trades:.0f} < {min_trades_per_slice}")
                    gate_reasons.append(f"{label}: {'、'.join(detail)}")
                slice_rows.append(
                    {
                        "label": label,
                        "passed": passed_slice,
                        "total_return": total,
                        "buy_hold_return": bh,
                        "excess": excess,
                        "trades": trades,
                    }
                )

            regime_passed = not gate_reasons
            validation["regime_consistency"] = {
                "passed": regime_passed,
                "splits": splits,
                "slices": slice_rows,
                "reasons": gate_reasons,
            }
            if not regime_passed:
                validation["passed"] = False
                reasons.append("レジーム一貫性なし: " + " / ".join(gate_reasons))
                validation["reasons"] = reasons
            return validation

        except Exception as exc:  # noqa: BLE001
            # fail-closed: 想定外エラー時は不合格として記録
            logger.warning("レジーム一貫性ゲートでエラー: %s", exc)
            validation["regime_consistency"] = {
                "passed": False,
                "error": str(exc),
            }
            validation["passed"] = False
            validation["reasons"] = reasons + [f"レジーム一貫性ゲート実行エラー: {exc}"]
            return validation

    def _apply_parameter_perturbation_gate(
        self,
        validation: dict[str, Any],
        strategy: Any,
        backtest_config: dict[str, Any],
        validation_ga_config: GAConfig,
        robustness_config: RobustnessConfig,
    ) -> dict[str, Any]:
        """パラメータ摂動頑健性ゲート（fail-closed）。

        合格候補の指標パラメータを ±param_delta 揺らした変異体を
        フルピリオド再評価し、以下を満たす変異体の割合が閾値未満なら
        不合格にする:
          - 絶対リターンが非負
          - 市場比が元戦略の param_min_retention_ratio 倍以上

        「±10% の変更で崩壊する戦略」＝パラメータの針の山に過適合した
        戦略を排除し、OOS への生存率を構造的に引き上げる。
        """
        reasons = list(validation.get("reasons", []))
        perturbation_result: dict[str, Any]

        try:
            original_metrics = self._get_full_period_metrics(
                strategy, backtest_config, validation_ga_config
            )
            variants = build_perturbed_genes(
                strategy,
                float(robustness_config.param_delta),
                int(robustness_config.param_max_variants),
            )

            if not variants:
                # 数値パラメータが無い等のケースでは判定対象が無いまま通過
                perturbation_result = {
                    "passed": True,
                    "variant_count": 0,
                    "survived": 0,
                    "survival_rate": None,
                    "reasons": [],
                    "labels": [],
                }
            elif original_metrics is None:
                # 元戦略の連続指標が無い場合は判定不能 → 判定スキップ
                logger.debug(
                    "摂動ゲート: 元戦略のフルピリオド指標が取得できないためスキップ"
                )
                perturbation_result = {
                    "passed": True,
                    "variant_count": len(variants),
                    "survived": None,
                    "survival_rate": None,
                    "reasons": [],
                    "labels": [label for label, _ in variants],
                }
            else:
                original_excess = float(original_metrics["total_return"]) - float(
                    original_metrics["buy_hold_return"]
                )
                retention = float(robustness_config.param_min_retention_ratio)
                survived = 0
                variant_rows: list[dict[str, Any]] = []
                gate_reasons: list[str] = []
                for label, variant in variants:
                    try:
                        metrics = self._evaluate_variant_metrics(
                            variant, backtest_config, validation_ga_config
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("変異体評価で例外: %s", exc)
                        metrics = None
                    if metrics is None:
                        # 変異体の評価自体が失敗する場合、頑健でない証拠として
                        # 不耐扱い（fail-closed）
                        variant_rows.append({"label": label, "survived": False})
                        gate_reasons.append(f"変異体の評価に失敗: {label}")
                        continue
                    total = float(metrics["total_return"])
                    excess = total - float(metrics["buy_hold_return"])
                    is_survivor = total >= 0.0 and excess >= retention * max(
                        original_excess, 0.0
                    )
                    if is_survivor:
                        survived += 1
                    else:
                        gate_reasons.append(
                            f"{label}: リターン {total:+.2%} / "
                            f"市場比 {excess:+.2%}（元 {original_excess:+.2%}）"
                        )
                    variant_rows.append({"label": label, "survived": is_survivor})

                survival_rate = survived / len(variants) if variants else None
                min_rate = float(robustness_config.param_min_survival_rate)
                perturbation_reasons: list[str] = []
                if survival_rate is not None and survival_rate < min_rate:
                    perturbation_reasons.append(
                        f"パラメータ摂動への耐性不足: 耐久変異体 "
                        f"{survived}/{len(variants)}（ survival_rate="
                        f"{survival_rate:.2f} < {min_rate:.2f}）"
                    )
                    perturbation_reasons.extend(gate_reasons)

                perturbation_result = {
                    "passed": not perturbation_reasons,
                    "variant_count": len(variants),
                    "survived": survived,
                    "survival_rate": survival_rate,
                    "reasons": perturbation_reasons,
                    "labels": [row["label"] for row in variant_rows],
                }

            validation["param_perturbation"] = perturbation_result
            if not perturbation_result.get("passed", True):
                validation["passed"] = False
                reasons.extend(perturbation_result.get("reasons", []))
                validation["reasons"] = reasons
            return validation

        except Exception as e:  # noqa: BLE001
            # fail-closed: 想定外エラー時は不合格として記録
            logger.warning("パラメータ摂動ゲートでエラー: %s", e)
            validation["param_perturbation"] = {
                "passed": False,
                "error": str(e),
            }
            validation["passed"] = False
            validation["reasons"] = reasons + [f"パラメータ摂動ゲート実行エラー: {e}"]
            return validation

    def _evaluate_variant_metrics(
        self,
        variant_gene: Any,
        backtest_config: dict[str, Any],
        ga_config: GAConfig,
    ) -> dict[str, float] | None:
        """摂動変異体をフルピリオド評価した収益メトリクスを返す。"""
        try:
            single_config = copy.deepcopy(ga_config)
            single_config.evaluation_config.evaluation_mode = "single"
            report = self._evaluation_strategy.execute_single_report(
                variant_gene, backtest_config, single_config
            )
            return self._extract_single_scenario_metrics(report)
        except Exception as exc:  # noqa: BLE001
            logger.debug("変異体評価に失敗: %s", exc)
            return None

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
        strategy: Any = None,
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
        # 3段階でゲートする。
        #
        # ① 市場比ゲート: excess_return（戦略 − Buy&Hold）の中央値が負なら
        #    不合格。絶対収益ではなく市場比で判定することで、下落相場で
        #    Buy&Hold を上回る防御的な戦略を誤って不合格にしない。
        #    単一の爆騰フォールド（例: B&H +45%）による平均の歪みを避ける
        #    ため median ベース（DSR ゲートと同じ）。
        # ② 絶対下限ゲート: 全フォールドを複利合算した通期リターンが赤字
        #    なら、市場比でスキルがあっても単独運用に耐えないため不合格。
        # ③ 通期市場比ゲート: 学習期間全体を連続的に評価した戦略リターンと
        #    Buy&Hold リターンを比較し、通期として市場に勝っていなければ
        #    不合格。フォールド窓の複利比較は WFA のテスト窓が期間を分割
        #    するため連続B&Hを過小評価し（トレンド期の急騰区間を取り逃す）、
        #    「フォールドでは勝つが実質B&Hに大敗」の戦略を通してしまう。
        #    フルピリオド指標が取得できない場合は、従来どおりフォールド
        #    複利同士の比較へフォールバックする。
        #
        # 各メトリクスが取得できないレポートの場合は、従来どおり集約
        # フィットネスの符号へフォールバックする。
        if primary_fitness is not None and not objective_registry.is_minimize_objective(
            report.primary_objective or ""
        ):
            excess_returns = self._collect_scenario_metric(report, "excess_return")
            if scenario_count > 0 and len(excess_returns) == scenario_count:
                aggregate_excess = float(median(excess_returns))
                if aggregate_excess < 0.0:
                    reasons.append(
                        "市場比リターン（excess_return 中央値 "
                        f"{aggregate_excess:.4f}）が負値です"
                    )
            elif primary_fitness < 0.0:
                reasons.append(f"集約フィットネス {primary_fitness:.4f} が負値です")

            fold_total_returns = self._collect_scenario_metric(report, "total_return")
            has_total = scenario_count > 0 and len(fold_total_returns) == scenario_count
            if has_total:
                compounded = 1.0
                for value in fold_total_returns:
                    compounded *= 1.0 + float(value)
                aggregated_return = compounded - 1.0
                if aggregated_return < MIN_AGGREGATED_TOTAL_RETURN:
                    reasons.append(
                        f"全期間合算リターン {aggregated_return:.4f} が閾値 "
                        f"{MIN_AGGREGATED_TOTAL_RETURN:.2f} 未満です"
                    )

            # ③ は追加のフルピリオド評価を伴うため、先行ゲートで不合格が
            # 確定している候補には実行しない（計算コスト節減）。
            if not reasons and strategy is not None:
                full_metrics = self._get_full_period_metrics(
                    strategy, backtest_config, ga_config
                )
                if full_metrics is not None:
                    continuous_total = float(full_metrics["total_return"])
                    continuous_bh = float(full_metrics["buy_hold_return"])
                    period_excess = continuous_total - continuous_bh
                    if period_excess < 0.0:
                        reasons.append(
                            f"通期市場比リターン {period_excess:+.4f} が負値です"
                            f"（戦略 {continuous_total:+.2%} vs 連続Buy&Hold "
                            f"{continuous_bh:+.2%}）"
                        )
                elif (
                    has_total
                    and scenario_count > 0
                    and len(self._collect_scenario_metric(report, "buy_hold_return"))
                    == scenario_count
                ):
                    fold_bh_returns = self._collect_scenario_metric(
                        report, "buy_hold_return"
                    )
                    bh_compound = 1.0
                    for value in fold_bh_returns:
                        bh_compound *= 1.0 + float(value)
                    period_excess = compounded - bh_compound
                    if period_excess < 0.0:
                        reasons.append(
                            f"通期市場比リターン {period_excess:+.4f} が負値です"
                            f"（戦略 {(compounded - 1.0):+.2%} vs "
                            f"Buy&Hold {(bh_compound - 1.0):+.2%}）"
                        )

        trades_list = self._collect_scenario_metric(report, "total_trades")
        if validation_config.min_trades is not None:
            if len(trades_list) != scenario_count:
                reasons.append(
                    f"total_trades メトリクスが全シナリオで取得できません"
                    f"（{len(trades_list)}/{scenario_count}）"
                )
            elif trades_list:
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
            elif dd_list:
                max_dd = max(abs(v) for v in dd_list)
                if max_dd > validation_config.max_drawdown:
                    reasons.append(
                        f"最大ドローダウン {max_dd:.2%} が上限 "
                        f"{validation_config.max_drawdown:.2%} 超過"
                    )

        # PBO ゲート: 負けフォールド（市場比 excess_return < 0、すなわち
        # Buy&Hold に劣ったフォールド）の比率が閾値を超えると不合格。
        # 絶対収益ではなく市場比で判定するため、下落相場で Buy&Hold を
        # 上回る防御的なフォールドを「負け」と誤判定しない。
        pbo: float | None = None
        n_losing_folds: int | None = None
        if validation_config.pbo_gate_active:
            fold_returns = self._collect_scenario_metric(report, "excess_return")
            if scenario_count > 0 and len(fold_returns) == scenario_count:
                pbo = probability_of_backtest_overfitting(fold_returns)
                n_losing_folds = sum(1 for value in fold_returns if value < 0.0)
                if pbo is not None and pbo > validation_config.pbo_threshold:
                    reasons.append(
                        f"PBO超過: 市場比で負けたフォールド {n_losing_folds}/{scenario_count} "
                        f"(PBO={pbo:.2f}) が閾値 "
                        f"{validation_config.pbo_threshold:.2f} を超えています"
                    )
            else:
                logger.debug(
                    "PBOゲート: excess_return が全シナリオで取得できないためスキップ"
                )

        # DSR ゲート: 多重検定補正後のシャープレシオ有意性。
        # GA の大量探索によるメタ過適合対策として strict / full で有効。
        # 統計力の基準は個々のフォールド窓ではなく全フォールドの合算
        # テスト期間（OOS 証拠の総量）。閾値未満の場合は判定をスキップ
        # する（fail-closed ではない）。
        dsr: float | None = None
        if validation_config.dsr_gate_active:
            fold_sharpes = self._collect_scenario_metric(report, "sharpe_ratio")
            if scenario_count > 0 and len(fold_sharpes) == scenario_count:
                observed_sr = float(median(fold_sharpes))
                tested_years = estimate_aggregate_tested_years(
                    str(backtest_config.get("start_date", "") or ""),
                    str(backtest_config.get("end_date", "") or ""),
                    float(validation_config.wfa_train_ratio),
                )
                effective_trials = validation_config.dsr_effective_trials
                if effective_trials is None:
                    effective_trials = max(
                        1,
                        int(ga_config.population_size)
                        * max(1, int(ga_config.generations)),
                    )
                if tested_years is None or (
                    tested_years < float(validation_config.dsr_min_tested_years)
                ):
                    logger.debug(
                        "DSRゲート: 合算テスト期間 %.2f年 が閾値 %.2f年 未満のためスキップ",
                        tested_years if tested_years is not None else -1.0,
                        float(validation_config.dsr_min_tested_years),
                    )
                else:
                    dsr = deflated_sharpe_ratio(
                        observed_sharpe=observed_sr,
                        n_observations=tested_years,
                        n_trials=int(effective_trials),
                        sigma_sharpe=float(validation_config.dsr_sigma_sharpe or 1.0),
                    )
                    if dsr < validation_config.min_dsr:
                        reasons.append(
                            f"DSR {dsr:.3f} が閾値 "
                            f"{validation_config.min_dsr} 未満"
                            f"（SR={observed_sr:.2f}, 試行数={effective_trials}, "
                            f"合算テスト={tested_years:.2f}年）"
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
        evaluation_config.wfa_n_folds = validation_config.wfa_n_folds
        evaluation_config.wfa_train_ratio = validation_config.wfa_train_ratio
        # anchored の可否は evaluation_config.wfa_anchored を唯一の情報源とする。
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

    # ------------------------------------------------------------------
    # フルピリオド（連続）評価
    # ------------------------------------------------------------------

    def _get_full_period_metrics(
        self,
        strategy: Any,
        backtest_config: dict[str, Any],
        validation_ga_config: GAConfig,
    ) -> dict[str, float] | None:
        """学習期間全体を連続評価した戦略/B&Hリターンを取得する。

        優先順位:
        1. 評価器のキャッシュ済みフルピリオドレポート（逐次実行時に存在）
        2. single モードでの再評価（WFA のようにテスト窓へ分割しない）

        取得できない場合は None を返し、呼び出し側はフォールド複利比較へ
        フォールバックする。失敗は想定内の経路として debug ログのみ。
        """
        try:
            cached = self._evaluator.get_cached_evaluation_report(strategy)
            if cached is not None:
                self._remember_behavior_fingerprint(strategy, cached)
                metrics = self._extract_single_scenario_metrics(cached)
                if metrics is not None:
                    return metrics

            single_config = copy.deepcopy(validation_ga_config)
            single_config.evaluation_config.evaluation_mode = "single"
            report = self._evaluation_strategy.execute_single_report(
                strategy, backtest_config, single_config
            )
            self._remember_behavior_fingerprint(strategy, report)
            return self._extract_single_scenario_metrics(report)
        except Exception as exc:  # noqa: BLE001
            logger.debug("フルピリオド指標の取得に失敗: %s", exc)
            return None

    def _remember_behavior_fingerprint(self, strategy: Any, report: Any) -> None:
        """フルピリオドレポートから行動フィンガープリントを記録する（取得できれば）。"""
        scenarios = getattr(report, "scenarios", None)
        if not isinstance(scenarios, list) or not scenarios:
            return
        metadata = getattr(scenarios[0], "metadata", None)
        if not isinstance(metadata, dict):
            return
        fingerprint = metadata.get("behavior_fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            try:
                key = self._get_strategy_key(strategy)
            except Exception:  # noqa: BLE001
                return
            self._behavior_fingerprints[key] = fingerprint

    @staticmethod
    def _extract_single_scenario_metrics(
        report: Any,
    ) -> dict[str, float] | None:
        """単一シナリオ（フルピリオド）レポートから収益メトリクスを抽出する。"""
        try:
            if not isinstance(report, EvaluationReport):
                return None
            totals = StrategyValidationService._collect_scenario_metric(
                report, "total_return"
            )
            bhs = StrategyValidationService._collect_scenario_metric(
                report, "buy_hold_return"
            )
        except Exception:  # noqa: BLE001
            return None
        if len(totals) == 1 and len(bhs) == 1:
            trades = StrategyValidationService._collect_scenario_metric(
                report, "total_trades"
            )
            metrics: dict[str, float] = {
                "total_return": float(totals[0]),
                "buy_hold_return": float(bhs[0]),
            }
            if len(trades) == 1:
                metrics["total_trades"] = float(trades[0])
            return metrics
        return None

    # ------------------------------------------------------------------
    # 行動シグネチャ（クローン検出）
    # ------------------------------------------------------------------

    @staticmethod
    def _behavioral_signature(strategy: Any) -> tuple[Any, ...]:
        """取引ロジック本体から行動シグネチャを構築する。

        - 条件式から参照されない「死んだ」指標遺伝子は無視する
          （突然変異が条件に接続されない指標を乗せても同一扱い）
        - 指標IDはランダム付与されるため正規化する
          （同一構成・同一パラメータなら ID が違っても同一シグネチャ）

        これにより、エリート複製＋デッド遺伝子乗せのクローン個体を
        検証対象の重複として扱える。
        """
        indicators = getattr(strategy, "indicators", None) or []
        condition_groups = (
            getattr(strategy, "long_entry_conditions", None),
            getattr(strategy, "short_entry_conditions", None),
            getattr(strategy, "long_exit_conditions", None),
            getattr(strategy, "short_exit_conditions", None),
        )

        referenced_names: set[str] = set()
        for group in condition_groups:
            StrategyValidationService._collect_indicator_references(
                group, referenced_names
            )

        def _indicator_token(indicator: Any) -> str:
            return "|".join(
                (
                    str(getattr(indicator, "type", "")),
                    json.dumps(
                        getattr(indicator, "parameters", {}) or {},
                        sort_keys=True,
                        default=str,
                    ),
                    str(indicator.timeframe or ""),
                )
            )

        replacements: dict[str, str] = {}
        indicator_tokens: list[str] = []
        # 条件式が指標を一切参照しない場合（特殊な遺伝子・テストダブル）
        # は参照フィルタを無効化し、全ての有効指標を対象にする
        filter_by_reference = bool(referenced_names)
        for indicator in indicators:
            if not getattr(indicator, "enabled", True):
                continue
            names = indicator_reference_names(indicator)
            # 参照されていない指標（死んだ遺伝子）はシグネチャから除外
            if filter_by_reference and not (names & referenced_names):
                continue
            token = _indicator_token(indicator)
            indicator_tokens.append(token)
            for name in names:
                replacements[name] = token

        # 条件式内の参照名（ID込み）を正規化トークンへ置換して直列化
        condition_text = json.dumps(
            [StrategyValidationService._condition_to_json(g) for g in condition_groups],
            sort_keys=True,
            default=str,
        )
        for name in sorted(replacements, key=len, reverse=True):
            condition_text = condition_text.replace(name, replacements[name])

        tool_sig = tuple(
            sorted(
                (
                    str(getattr(tool, "tool_name", "")),
                    json.dumps(
                        getattr(tool, "params", {}) or {},
                        sort_keys=True,
                        default=str,
                    ),
                )
                for tool in getattr(strategy, "tool_genes", None) or []
                if getattr(tool, "enabled", True)
            )
        )

        return (
            tuple(sorted(indicator_tokens)),
            condition_text,
            json.dumps(
                getattr(strategy, "risk_management", None) or {},
                sort_keys=True,
                default=str,
            ),
            StrategyValidationService._gene_to_json(
                getattr(strategy, "tpsl_gene", None)
            ),
            StrategyValidationService._gene_to_json(
                getattr(strategy, "position_sizing_gene", None)
            ),
            StrategyValidationService._gene_to_json(
                getattr(strategy, "entry_gene", None)
            ),
            StrategyValidationService._gene_to_json(
                getattr(strategy, "exit_gene", None)
            ),
            tool_sig,
        )

    @staticmethod
    def _collect_indicator_references(node: Any, out: set[str]) -> None:
        """条件ツリーから指標参照を収集する（共通ユーティリティへ委譲）。"""
        collect_indicator_references(node, out)

    @staticmethod
    def _condition_to_json(node: Any) -> Any:
        """条件ツリーを JSON 直列化可能な形へ変換する（非dataclassはそのまま）。"""
        if is_dataclass(node) and not isinstance(node, type):
            return {
                field_info.name: StrategyValidationService._condition_to_json(
                    getattr(node, field_info.name, None)
                )
                for field_info in dataclass_fields(node)
            }
        if isinstance(node, dict):
            return {
                key: StrategyValidationService._condition_to_json(value)
                for key, value in node.items()
            }
        if isinstance(node, (list, tuple)):
            return [StrategyValidationService._condition_to_json(v) for v in node]
        return node

    @staticmethod
    def _gene_to_json(gene: Any) -> str:
        """補助遺伝子（TPSL/サイジング/注文方式）を正規化文字列化する。"""
        if gene is None:
            return ""
        try:
            if is_dataclass(gene) and not isinstance(gene, type):
                payload = {
                    field_info.name: getattr(gene, field_info.name, None)
                    for field_info in dataclass_fields(gene)
                }
                return json.dumps(payload, sort_keys=True, default=str)
            if hasattr(gene, "__dict__"):
                return json.dumps(vars(gene), sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            return ""
        return str(gene)
