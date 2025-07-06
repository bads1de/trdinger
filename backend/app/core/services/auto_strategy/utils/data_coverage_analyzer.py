"""
データカバレッジ分析システム

OI/FRデータのカバレッジを分析し、戦略の適応度計算にペナルティを反映します。
"""

from typing import Dict, Any
import pandas as pd

# import logging

from ..models.strategy_gene import StrategyGene, Condition

# logger = logging.getLogger(__name__)


class DataCoverageAnalyzer:
    """
    データカバレッジ分析器

    戦略で使用される指標のデータカバレッジを分析し、
    適応度計算に反映するためのペナルティを計算します。
    """

    def __init__(self):
        """データカバレッジ分析器を初期化"""
        self.special_data_sources = {"OpenInterest", "FundingRate"}
        self.coverage_thresholds = {
            "excellent": 0.95,  # 95%以上のカバレッジ
            "good": 0.80,  # 80%以上のカバレッジ
            "fair": 0.60,  # 60%以上のカバレッジ
            "poor": 0.40,  # 40%以上のカバレッジ
            # 40%未満は"very_poor"
        }

    def analyze_strategy_coverage(
        self, strategy_gene: StrategyGene, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """戦略のデータカバレッジを分析

        Args:
            strategy_gene: 戦略遺伝子
            data: バックテストデータ

        Returns:
            カバレッジ分析結果
        """
        try:
            # 戦略で使用される特殊データソースを特定
            used_special_sources = self._extract_special_data_sources(strategy_gene)

            if not used_special_sources:
                # 特殊データソースを使用していない場合はペナルティなし
                return {
                    "uses_special_data": False,
                    "coverage_penalty": 0.0,
                    "coverage_details": {},
                    "overall_coverage_score": 1.0,
                }

            # 各特殊データソースのカバレッジを計算
            coverage_details = {}
            total_penalty = 0.0

            for source in used_special_sources:
                coverage_info = self._calculate_coverage(data, source)
                coverage_details[source] = coverage_info

                # ペナルティを計算
                penalty = self._calculate_coverage_penalty(
                    coverage_info["coverage_ratio"]
                )
                total_penalty += penalty

            # 平均ペナルティを計算
            avg_penalty = (
                total_penalty / len(used_special_sources)
                if used_special_sources
                else 0.0
            )

            # 全体的なカバレッジスコアを計算
            overall_score = max(0.0, 1.0 - avg_penalty)

            return {
                "uses_special_data": True,
                "coverage_penalty": avg_penalty,
                "coverage_details": coverage_details,
                "overall_coverage_score": overall_score,
                "used_special_sources": list(used_special_sources),
            }

        except Exception as e:
            # logger.error(f"データカバレッジ分析エラー: {e}")
            return {
                "uses_special_data": False,
                "coverage_penalty": 0.0,
                "coverage_details": {},
                "overall_coverage_score": 1.0,
                "error": str(e),
            }

    def _extract_special_data_sources(self, strategy_gene: StrategyGene) -> set:
        """戦略で使用される特殊データソースを抽出

        Args:
            strategy_gene: 戦略遺伝子

        Returns:
            使用される特殊データソースのセット
        """
        used_sources = set()

        # エントリー条件をチェック
        for condition in strategy_gene.entry_conditions:
            used_sources.update(self._extract_sources_from_condition(condition))

        # エグジット条件をチェック
        for condition in strategy_gene.exit_conditions:
            used_sources.update(self._extract_sources_from_condition(condition))

        # 特殊データソースのみをフィルタ
        return used_sources.intersection(self.special_data_sources)

    def _extract_sources_from_condition(self, condition: Condition) -> set:
        """条件から使用されるデータソースを抽出

        Args:
            condition: 条件オブジェクト

        Returns:
            使用されるデータソースのセット
        """
        sources = set()

        # 左オペランドをチェック
        if isinstance(condition.left_operand, str):
            if condition.left_operand in self.special_data_sources:
                sources.add(condition.left_operand)

        # 右オペランドをチェック
        if isinstance(condition.right_operand, str):
            if condition.right_operand in self.special_data_sources:
                sources.add(condition.right_operand)

        return sources

    def _calculate_coverage(self, data: pd.DataFrame, source: str) -> Dict[str, Any]:
        """指定されたデータソースのカバレッジを計算

        Args:
            data: バックテストデータ
            source: データソース名

        Returns:
            カバレッジ情報
        """
        if source not in data.columns:
            return {
                "coverage_ratio": 0.0,
                "total_points": len(data),
                "valid_points": 0,
                "missing_points": len(data),
                "quality": "very_poor",
            }

        total_points = len(data)

        # 有効なデータポイント数を計算（0.0でないデータ）
        valid_points = (data[source] != 0.0).sum()
        missing_points = total_points - valid_points

        coverage_ratio = valid_points / total_points if total_points > 0 else 0.0

        # カバレッジの品質を判定
        quality = self._determine_coverage_quality(coverage_ratio)

        return {
            "coverage_ratio": coverage_ratio,
            "total_points": total_points,
            "valid_points": valid_points,
            "missing_points": missing_points,
            "quality": quality,
        }

    def _determine_coverage_quality(self, coverage_ratio: float) -> str:
        """カバレッジ比率から品質を判定

        Args:
            coverage_ratio: カバレッジ比率（0.0-1.0）

        Returns:
            品質レベル
        """
        if coverage_ratio >= self.coverage_thresholds["excellent"]:
            return "excellent"
        elif coverage_ratio >= self.coverage_thresholds["good"]:
            return "good"
        elif coverage_ratio >= self.coverage_thresholds["fair"]:
            return "fair"
        elif coverage_ratio >= self.coverage_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"

    def _calculate_coverage_penalty(self, coverage_ratio: float) -> float:
        """カバレッジ比率からペナルティを計算

        Args:
            coverage_ratio: カバレッジ比率（0.0-1.0）

        Returns:
            ペナルティ値（0.0-1.0）
        """
        if coverage_ratio >= self.coverage_thresholds["excellent"]:
            return 0.0  # ペナルティなし
        elif coverage_ratio >= self.coverage_thresholds["good"]:
            return 0.1  # 軽微なペナルティ
        elif coverage_ratio >= self.coverage_thresholds["fair"]:
            return 0.25  # 中程度のペナルティ
        elif coverage_ratio >= self.coverage_thresholds["poor"]:
            return 0.5  # 重いペナルティ
        else:
            return 0.8  # 非常に重いペナルティ

    def get_coverage_summary(self, analysis_result: Dict[str, Any]) -> str:
        """カバレッジ分析結果のサマリーを生成

        Args:
            analysis_result: 分析結果

        Returns:
            サマリー文字列
        """
        if not analysis_result.get("uses_special_data", False):
            return "特殊データソース（OI/FR）を使用していません"

        details = analysis_result.get("coverage_details", {})
        penalty = analysis_result.get("coverage_penalty", 0.0)

        summary_parts = []
        for source, info in details.items():
            coverage_pct = info["coverage_ratio"] * 100
            quality = info["quality"]
            summary_parts.append(f"{source}: {coverage_pct:.1f}% ({quality})")

        summary = ", ".join(summary_parts)
        summary += f" | 総合ペナルティ: {penalty:.3f}"

        return summary


data_coverage_analyzer = DataCoverageAnalyzer()
