"""
AutoML特徴量分析サービス

AutoML生成特徴量の重要度分析と可視化機能を提供します。
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import numpy as np


class FeaturePatternConfig(TypedDict):
    """特徴量パターン設定の型定義"""

    prefix: str
    patterns: List[str]


logger = logging.getLogger(__name__)


@dataclass
class FeatureAnalysis:
    """特徴量分析結果"""

    feature_name: str
    importance: float
    feature_type: str  # 'manual', 'autofeat'
    category: str  # 特徴量のカテゴリ
    description: str  # 特徴量の説明


class AutoMLFeatureAnalyzer:
    """
    AutoML特徴量分析クラス

    AutoML生成特徴量を分類・分析し、重要度を可視化します。
    """

    def __init__(self):
        """初期化"""
        self.feature_patterns: Dict[str, FeaturePatternConfig] = (
            self._initialize_feature_patterns()
        )
        self.category_descriptions = self._initialize_category_descriptions()

    def _initialize_feature_patterns(self) -> Dict[str, FeaturePatternConfig]:
        """特徴量パターンを初期化"""
        return {}  # autofeat機能は削除されました

    def _initialize_category_descriptions(self) -> Dict[str, str]:
        """カテゴリ説明を初期化"""
        return {
            "statistical": "統計的特徴量（平均、分散、歪度など）",
            "frequency": "周波数領域特徴量（FFT、スペクトラムなど）",
            "temporal": "時間的特徴量（トレンド、自己相関など）",
            "aggregation": "集約特徴量（合計、カウントなど）",
            "interaction": "相互作用特徴量（特徴量間の関係）",
            "genetic": "遺伝的アルゴリズム生成特徴量",
            "manual": "手動作成特徴量",
            "unknown": "分類不明特徴量",
        }

    def analyze_feature_importance(
        self, feature_importance: Dict[str, float], top_n: int = 20
    ) -> Dict[str, Any]:
        """
        特徴量重要度を分析

        Args:
            feature_importance: 特徴量重要度辞書
            top_n: 分析する上位特徴量数

        Returns:
            分析結果辞書
        """
        try:
            if not feature_importance:
                return {"error": "特徴量重要度データがありません"}

            # 特徴量を分析
            analyzed_features = []
            for feature_name, importance in feature_importance.items():
                analysis = self._analyze_single_feature(feature_name, importance)
                analyzed_features.append(analysis)

            # 重要度順にソート
            analyzed_features.sort(key=lambda x: x.importance, reverse=True)
            top_features = analyzed_features[:top_n]

            # タイプ別統計を計算
            type_stats = self._calculate_type_statistics(analyzed_features)

            # カテゴリ別統計を計算
            category_stats = self._calculate_category_statistics(analyzed_features)

            # AutoML効果を分析
            automl_impact = self._analyze_automl_impact(analyzed_features)

            return {
                "top_features": [
                    {
                        "feature_name": f.feature_name,
                        "importance": f.importance,
                        "feature_type": f.feature_type,
                        "category": f.category,
                        "description": f.description,
                    }
                    for f in top_features
                ],
                "type_statistics": type_stats,
                "category_statistics": category_stats,
                "automl_impact": automl_impact,
                "total_features": len(analyzed_features),
                "analysis_summary": self._generate_analysis_summary(
                    analyzed_features, type_stats, automl_impact
                ),
            }

        except Exception as e:
            logger.error(f"特徴量重要度分析エラー: {e}")
            return {"error": f"分析エラー: {str(e)}"}

    def _analyze_single_feature(
        self, feature_name: str, importance: float
    ) -> FeatureAnalysis:
        """単一特徴量を分析"""
        feature_type = self._identify_feature_type(feature_name)
        category = self._categorize_feature(feature_name, feature_type)
        description = self._generate_feature_description(
            feature_name, feature_type, category
        )

        return FeatureAnalysis(
            feature_name=feature_name,
            importance=importance,
            feature_type=feature_type,
            category=category,
            description=description,
        )

    def _identify_feature_type(self, feature_name: str) -> str:
        """特徴量タイプを識別"""
        # AutoML特徴量のパターンマッチング
        for feature_type, config in self.feature_patterns.items():
            # プレフィックスチェック
            if feature_name.startswith(config["prefix"]):
                return feature_type

            # パターンマッチング
            for pattern in config["patterns"]:
                if re.match(pattern, feature_name):
                    return feature_type

        # 手動特徴量の判定
        manual_patterns = [
            r".*_sma_.*",
            r".*_ema_.*",
            r".*_rsi.*",
            r".*_macd.*",
            r".*_bb_.*",
            r".*_volume_.*",
            r".*_price_.*",
        ]

        for pattern in manual_patterns:
            if re.match(pattern, feature_name, re.IGNORECASE):
                return "manual"

        return "unknown"

    def _categorize_feature(self, feature_name: str, feature_type: str) -> str:
        """特徴量をカテゴリ分類"""
        name_lower = feature_name.lower()

        # 統計的特徴量
        if any(stat in name_lower for stat in ["mean", "std", "var", "skew", "kurt"]):
            return "statistical"

        # 周波数領域特徴量
        if any(
            freq in name_lower for freq in ["fft", "frequency", "spectral", "energy"]
        ):
            return "frequency"

        # 時間的特徴量
        if any(temp in name_lower for temp in ["trend", "autocorr", "lag", "shift"]):
            return "temporal"

        # 集約特徴量
        if any(agg in name_lower for agg in ["sum", "count", "max", "min"]):
            return "aggregation"

        # 相互作用特徴量
        if any(inter in name_lower for inter in ["*", "+", "/", "interaction"]):
            return "interaction"

        # autofeat機能は削除されました

        # 手動特徴量
        if feature_type == "manual":
            return "manual"

        return "unknown"

    def _generate_feature_description(
        self, feature_name: str, feature_type: str, category: str
    ) -> str:
        """特徴量の説明を生成"""
        base_desc = self.category_descriptions.get(category, "特徴量")

        if feature_type == "manual":
            return f"手動作成: {base_desc}"
        else:
            return base_desc

    def _calculate_type_statistics(
        self, features: List[FeatureAnalysis]
    ) -> Dict[str, Any]:
        """タイプ別統計を計算"""
        type_counts = {}
        type_importance = {}

        for feature in features:
            feature_type = feature.feature_type
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1

            if feature_type not in type_importance:
                type_importance[feature_type] = []
            type_importance[feature_type].append(feature.importance)

        # 統計計算
        type_stats = {}
        for feature_type in type_counts:
            importances = type_importance[feature_type]
            type_stats[feature_type] = {
                "count": type_counts[feature_type],
                "total_importance": sum(importances),
                "mean_importance": np.mean(importances),
                "max_importance": max(importances),
                "percentage": (type_counts[feature_type] / len(features)) * 100,
            }

        return type_stats

    def _calculate_category_statistics(
        self, features: List[FeatureAnalysis]
    ) -> Dict[str, Any]:
        """カテゴリ別統計を計算"""
        category_counts = {}
        category_importance = {}

        for feature in features:
            category = feature.category
            category_counts[category] = category_counts.get(category, 0) + 1

            if category not in category_importance:
                category_importance[category] = []
            category_importance[category].append(feature.importance)

        # 統計計算
        category_stats = {}
        for category in category_counts:
            importances = category_importance[category]
            category_stats[category] = {
                "count": category_counts[category],
                "total_importance": sum(importances),
                "mean_importance": np.mean(importances),
                "max_importance": max(importances),
                "percentage": (category_counts[category] / len(features)) * 100,
                "description": self.category_descriptions.get(category, ""),
            }

        return category_stats

    def _analyze_automl_impact(self, features: List[FeatureAnalysis]) -> Dict[str, Any]:
        """AutoML効果を分析"""
        automl_types = []  # autofeat機能は削除されました
        manual_features = [f for f in features if f.feature_type == "manual"]
        automl_features = [f for f in features if f.feature_type in automl_types]

        if not features:
            return {"error": "分析対象の特徴量がありません"}

        total_importance = sum(f.importance for f in features)
        manual_importance = sum(f.importance for f in manual_features)
        automl_importance = sum(f.importance for f in automl_features)

        return {
            "total_features": len(features),
            "manual_features": len(manual_features),
            "automl_features": len(automl_features),
            "manual_importance_ratio": (
                (manual_importance / total_importance) * 100
                if total_importance > 0
                else 0
            ),
            "automl_importance_ratio": (
                (automl_importance / total_importance) * 100
                if total_importance > 0
                else 0
            ),
            "automl_feature_ratio": (len(automl_features) / len(features)) * 100,
            "top_automl_features": [
                {
                    "name": f.feature_name,
                    "type": f.feature_type,
                    "importance": f.importance,
                }
                for f in sorted(
                    automl_features, key=lambda x: x.importance, reverse=True
                )[:5]
            ],
        }

    def _generate_analysis_summary(
        self,
        features: List[FeatureAnalysis],
        type_stats: Dict[str, Any],
        automl_impact: Dict[str, Any],
    ) -> str:
        """分析サマリーを生成"""
        total_features = len(features)
        automl_ratio = automl_impact.get("automl_importance_ratio", 0)

        if automl_ratio > 60:
            impact_level = "高"
        elif automl_ratio > 30:
            impact_level = "中"
        else:
            impact_level = "低"

        return (
            f"総特徴量数: {total_features}個\n"
            f"AutoML特徴量の重要度貢献: {automl_ratio:.1f}%\n"
            f"AutoML効果レベル: {impact_level}\n"
            f"最も効果的なAutoMLタイプ: {max(type_stats.keys(), key=lambda k: type_stats[k]['total_importance']) if type_stats else 'なし'}"
        )
