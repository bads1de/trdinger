"""
インデックス整合性管理ユーティリティ

MLワークフローにおける特徴量とラベルのインデックス整合性を保証するためのユーティリティ。
pandasの強力なインデックス操作機能（align, reindex, intersection）を直接活用し、
シンプルで効率的なインデックス管理を提供します。
"""

import logging
import pandas as pd
from typing import Dict, Tuple, Any, Union, Literal

logger = logging.getLogger(__name__)


def align_data(
    features: pd.DataFrame,
    labels: pd.Series,
    method: Literal[
        "intersection", "features_priority", "labels_priority", "outer"
    ] = "intersection",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    特徴量とラベルのインデックスを整合

    pandasのalign機能を活用してシンプルで効率的なインデックス整合を実現。

    Args:
        features: 特徴量DataFrame
        labels: ラベルSeries
        method: 整合方法
            - "intersection": 共通インデックスのみ使用（inner join）
            - "features_priority": 特徴量のインデックスを優先（left join）
            - "labels_priority": ラベルのインデックスを優先（right join）
            - "outer": 全インデックスを使用（outer join）

    Returns:
        整合された特徴量とラベルのタプル
    """
    logger.info(f"インデックス整合開始: 特徴量{len(features)}行, ラベル{len(labels)}行")

    # pandasのjoin方法にマッピング
    join_mapping = {
        "intersection": "inner",
        "features_priority": "left",
        "labels_priority": "right",
        "outer": "outer",
    }

    if method not in join_mapping:
        raise ValueError(
            f"不正な整合方法: {method}. 利用可能: {list(join_mapping.keys())}"
        )

    join_method = join_mapping[method]

    # pandasのalignを使用してインデックス整合
    aligned_features, aligned_labels = features.align(labels, join=join_method, axis=0)

    # NaN値を含む行を除去（全てのjoin方法で実行）
    # 特徴量にNaNがある行を特定
    features_na_mask = aligned_features.isna().any(axis=1)
    # ラベルにNaNがある行を特定
    labels_na_mask = aligned_labels.isna()
    # どちらかにNaNがある行を除去
    valid_mask = ~(features_na_mask | labels_na_mask)

    aligned_features = aligned_features[valid_mask]
    aligned_labels = aligned_labels[valid_mask]

    alignment_ratio = (
        len(aligned_features) / max(len(features), len(labels))
        if max(len(features), len(labels)) > 0
        else 0
    )

    logger.info(
        f"インデックス整合完了: {len(aligned_features)}行 "
        f"(整合率: {alignment_ratio*100:.1f}%)"
    )

    return aligned_features, aligned_labels


def validate_alignment(
    features: pd.DataFrame,
    labels: pd.Series,
    min_alignment_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    インデックス整合性を検証

    pandasのインデックス操作を使用してシンプルで効率的な検証を実現。

    Args:
        features: 特徴量DataFrame
        labels: ラベルSeries
        min_alignment_ratio: 最小整合率

    Returns:
        検証結果の辞書
    """
    # pandasのintersectionとdifferenceを使用
    common_index = features.index.intersection(labels.index)
    max_possible = max(len(features), len(labels))
    alignment_ratio = len(common_index) / max_possible if max_possible > 0 else 0

    validation_result = {
        "is_valid": alignment_ratio >= min_alignment_ratio,
        "alignment_ratio": alignment_ratio,
        "common_rows": len(common_index),
        "features_rows": len(features),
        "labels_rows": len(labels),
        "missing_in_features": len(labels.index.difference(features.index)),
        "missing_in_labels": len(features.index.difference(labels.index)),
        "issues": [],
    }

    # 問題の特定
    if alignment_ratio < min_alignment_ratio:
        validation_result["issues"].append(
            f"整合率が低すぎます: {alignment_ratio*100:.1f}% < {min_alignment_ratio*100:.1f}%"
        )

    if validation_result["missing_in_features"] > 0:
        validation_result["issues"].append(
            f"特徴量に不足しているインデックス: {validation_result['missing_in_features']}個"
        )

    if validation_result["missing_in_labels"] > 0:
        validation_result["issues"].append(
            f"ラベルに不足しているインデックス: {validation_result['missing_in_labels']}個"
        )

    return validation_result


def preserve_index_during_processing(
    data: pd.DataFrame, processing_func: callable, *args, **kwargs
) -> pd.DataFrame:
    """
    処理中にインデックスを保持

    シンプルなユーティリティ関数として実装し、pandasの機能を活用。

    Args:
        data: 処理対象データ
        processing_func: 処理関数
        *args, **kwargs: 処理関数の引数

    Returns:
        インデックスが保持された処理結果
    """
    original_index = data.index.copy()

    # 処理実行
    result = processing_func(data, *args, **kwargs)

    # インデックスの整合性確認と復元
    if len(result) == len(original_index):
        # 行数が同じ場合はインデックスを復元
        result.index = original_index
    elif len(result) < len(original_index):
        # 行数が減った場合は対応するインデックスを特定
        logger.warning(f"処理により行数が変化: {len(original_index)} → {len(result)}")
        # 結果のインデックスが元のインデックスのサブセットであることを確認
        if not result.index.isin(original_index).all():
            logger.warning("処理結果のインデックスが元のインデックスと整合していません")

    return result


def reindex_with_intersection(
    data: Union[pd.DataFrame, pd.Series], target_index: pd.Index
) -> Union[pd.DataFrame, pd.Series]:
    """
    インデックスの共通部分でデータを再インデックス

    pandasのintersectionとreindexを組み合わせた効率的な実装。

    Args:
        data: 再インデックス対象のデータ
        target_index: 目標インデックス

    Returns:
        再インデックスされたデータ
    """
    # 共通インデックスを取得
    common_index = data.index.intersection(target_index)

    # 共通インデックスで再インデックス
    return data.reindex(common_index)


def get_index_statistics(features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
    """
    インデックス統計情報を取得

    pandasのインデックス操作を使用した統計情報の計算。

    Args:
        features: 特徴量DataFrame
        labels: ラベルSeries

    Returns:
        インデックス統計情報の辞書
    """
    common_index = features.index.intersection(labels.index)
    features_only = features.index.difference(labels.index)
    labels_only = labels.index.difference(features.index)

    return {
        "features_count": len(features),
        "labels_count": len(labels),
        "common_count": len(common_index),
        "features_only_count": len(features_only),
        "labels_only_count": len(labels_only),
        "alignment_ratio": (
            len(common_index) / max(len(features), len(labels))
            if max(len(features), len(labels)) > 0
            else 0
        ),
        "coverage_in_features": (
            len(common_index) / len(features) if len(features) > 0 else 0
        ),
        "coverage_in_labels": len(common_index) / len(labels) if len(labels) > 0 else 0,
    }


# 後方互換性のためのレガシークラス（非推奨）
# 新しい関数ベースのアプローチを使用することを推奨
class MLWorkflowIndexManager:
    """
    MLワークフロー専用のインデックス管理クラス（非推奨）

    注意: このクラスは後方互換性のために残されています。
    新しいコードでは上記の関数ベースのアプローチを使用してください。
    """

    def __init__(self):
        """初期化"""
        self.workflow_state = {
            "original_data": None,
            "preprocessed_data": None,
            "features": None,
            "labels": None,
        }
        logger.warning(
            "MLWorkflowIndexManagerは非推奨です。関数ベースのアプローチを使用してください。"
        )

    def initialize_workflow(self, original_data: pd.DataFrame) -> None:
        """ワークフローを初期化"""
        self.workflow_state["original_data"] = original_data.copy()
        logger.info(f"MLワークフロー初期化: {len(original_data)}行")

    def process_with_index_tracking(
        self,
        stage_name: str,
        data: pd.DataFrame,
        processing_func: callable,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """インデックス追跡付きで処理を実行"""
        logger.info(f"{stage_name}処理開始: {len(data)}行")

        result = preserve_index_during_processing(
            data, processing_func, *args, **kwargs
        )

        logger.info(f"{stage_name}処理完了: {len(result)}行")

        # ワークフロー状態を更新
        if stage_name == "前処理":
            self.workflow_state["preprocessed_data"] = result
        elif stage_name == "特徴量エンジニアリング":
            self.workflow_state["features"] = result

        return result

    def finalize_workflow(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        alignment_method: str = "intersection",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """ワークフローを完了し、最終的な整合を実行"""
        logger.info("MLワークフロー最終整合開始")

        # 整合性検証
        validation_result = validate_alignment(features, labels)

        if not validation_result["is_valid"]:
            logger.warning("インデックス整合性に問題があります:")
            for issue in validation_result["issues"]:
                logger.warning(f"  - {issue}")

        # 最終整合
        aligned_features, aligned_labels = align_data(
            features, labels, method=alignment_method
        )

        # ワークフロー状態を更新
        self.workflow_state["features"] = aligned_features
        self.workflow_state["labels"] = aligned_labels

        # 統計情報を取得
        stats = get_index_statistics(aligned_features, aligned_labels)
        logger.info(
            f"MLワークフロー完了: 最終データ{len(aligned_features)}行 "
            f"(整合率: {stats['alignment_ratio']*100:.1f}%)"
        )

        return aligned_features, aligned_labels

    def get_workflow_summary(self) -> Dict[str, Any]:
        """ワークフロー全体のサマリーを取得"""
        summary = {
            "original_rows": (
                len(self.workflow_state["original_data"])
                if self.workflow_state["original_data"] is not None
                else 0
            ),
            "final_rows": (
                len(self.workflow_state["features"])
                if self.workflow_state["features"] is not None
                else 0
            ),
            "data_retention_rate": 0,
        }

        if summary["original_rows"] > 0:
            summary["data_retention_rate"] = (
                summary["final_rows"] / summary["original_rows"]
            )

        return summary
