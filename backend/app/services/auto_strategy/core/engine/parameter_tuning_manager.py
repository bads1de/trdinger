"""
パラメータ管理モジュール

個体評価のレポート・サマリ構築と full-fidelity 再評価を担当します。
"""

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ...config.ga_config import GAConfig

from ..evaluation.evaluation_fidelity import is_multi_fidelity_enabled
from .report_selection import is_evaluation_report

logger = logging.getLogger(__name__)


class ParameterTuningManager:
    """
    パラメータ管理クラス

    個体の評価レポート構築と、必要に応じた full fidelity 再評価を担当します。
    """

    def __init__(self, individual_evaluator: Any):
        """
        初期化

        Args:
            individual_evaluator: 個体評価器
        """
        self.individual_evaluator = individual_evaluator

    def evaluate_individual_with_full_fidelity(
        self, individual: object, config: "GAConfig"
    ) -> tuple[float, ...]:
        """
        必要に応じて full fidelity で個体を再評価する。
        """
        if is_multi_fidelity_enabled(config):
            return cast(
                tuple[float, ...],
                self.individual_evaluator.evaluate(
                    individual,
                    config,
                    force_refresh=True,
                ),
            )
        return cast(
            tuple[float, ...],
            self.individual_evaluator.evaluate(individual, config),
        )

    @staticmethod
    def extract_primary_fitness_from_result(result: object) -> float:
        """
        評価結果から主 fitness を取り出す。
        """
        from .fitness_utils import extract_primary_fitness_from_result

        return extract_primary_fitness_from_result(result)

    def build_individual_evaluation_summary(
        self,
        individual: object,
        config: "GAConfig",
        *,
        primary_fitness: float | None = None,
        selection_rank_override: int | None = None,
        selection_score_override: tuple[float, ...] | None = None,
    ) -> dict[str, Any] | None:
        """
        個体の評価 report から保存向け summary を構築する。
        """
        if individual is None:
            return None

        get_cached_evaluation_report = getattr(
            self.individual_evaluator,
            "get_cached_evaluation_report",
            None,
        )

        report: object | None = None
        if callable(get_cached_evaluation_report):
            report = get_cached_evaluation_report(individual)

        if report is not None and is_evaluation_report(report):
            if report.metadata.get("evaluation_fidelity") == "coarse":
                report = None

        if report is None and is_multi_fidelity_enabled(config):
            try:
                self.evaluate_individual_with_full_fidelity(individual, config)
            except Exception as exc:
                logger.debug("summary 用 full 評価に失敗しました: %s", exc)
            if callable(get_cached_evaluation_report):
                report = get_cached_evaluation_report(individual)

        if not is_evaluation_report(report):
            return None

        from math import isfinite

        from ..evaluation.report_persistence import build_report_summary
        from .report_selection import (
            extract_primary_fitness,
            get_two_stage_rank,
            get_two_stage_score,
        )

        if primary_fitness is None:
            fitness_score = extract_primary_fitness(individual)
            numeric_fitness = fitness_score if isfinite(fitness_score) else None
        else:
            numeric_fitness = (
                float(primary_fitness) if isfinite(float(primary_fitness)) else None
            )

        selection_rank = selection_rank_override
        if selection_rank is None:
            selection_rank = get_two_stage_rank(individual)

        selection_score: object = selection_score_override
        if selection_score is None:
            selection_score = get_two_stage_score(individual)
        if not isinstance(selection_score, (tuple, list)):
            selection_score = None

        return build_report_summary(
            report,
            selection_rank=(
                selection_rank if isinstance(selection_rank, int) else None
            ),
            selection_score=selection_score,
            fitness_score=numeric_fitness,
        )
