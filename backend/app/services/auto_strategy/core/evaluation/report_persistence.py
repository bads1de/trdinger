"""
評価レポートの永続化ヘルパー

EvaluationReport を保存・表示向けの軽量な summary へ変換する。
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Dict, Mapping, Optional, Sequence, cast

from .evaluation_report import EvaluationReport


def build_report_summary(
    report: EvaluationReport,
    *,
    selection_rank: Optional[int] = None,
    selection_score: Optional[Sequence[float]] = None,
    fitness_score: Optional[float] = None,
    max_scenarios: int = 20,
) -> Dict[str, Any]:
    """EvaluationReport から保存向け summary を構築する。"""
    summary = report.to_summary_dict(max_scenarios=max_scenarios)

    if isinstance(selection_rank, int):
        summary["selection_rank"] = selection_rank

    if selection_score and len(selection_score) >= 4:
        summary["selection_components"] = {
            "pass_gate": float(selection_score[0]),
            "pass_rate": float(selection_score[1]),
            "worst_case": float(selection_score[2]),
            "aggregated": float(selection_score[3]),
        }
        if len(selection_score) > 4:
            extra_components = []
            for index in range(4, len(selection_score), 2):
                if index + 1 >= len(selection_score):
                    break
                extra_components.append(
                    {
                        "worst_case": float(selection_score[index]),
                        "aggregated": float(selection_score[index + 1]),
                    }
                )
            if extra_components:
                cast(Dict[str, Any], summary["selection_components"])[
                    "objective_components"
                ] = extra_components

    if fitness_score is not None:
        numeric_fitness = float(fitness_score)
        if isfinite(numeric_fitness):
            summary["fitness_score"] = numeric_fitness

    return summary


def attach_evaluation_summary(
    gene_data: Dict[str, Any],
    summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """戦略 gene_data の metadata へ評価 summary を埋め込む。"""
    merged = deepcopy(gene_data)
    metadata = merged.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    if isinstance(summary, Mapping):
        metadata["evaluation_summary"] = deepcopy(dict(summary))

    merged["metadata"] = metadata
    return merged


def attach_backtest_evaluation_summary(
    config_json: Dict[str, Any],
    summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """backtest result の config_json へ評価 summary を埋め込む。"""
    merged = deepcopy(config_json)
    if isinstance(summary, Mapping):
        merged["evaluation_summary"] = deepcopy(dict(summary))
    return merged


def extract_evaluation_summary(
    gene_data: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """gene_data.metadata から保存済み評価 summary を取り出す。"""
    metadata = gene_data.get("metadata")
    if not isinstance(metadata, Mapping):
        return None

    summary = metadata.get("evaluation_summary")
    if not isinstance(summary, Mapping):
        return None

    return deepcopy(dict(summary))
