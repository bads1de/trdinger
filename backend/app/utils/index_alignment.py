"""
インデックス整合性管理ユーティリティ

MLワークフローにおける特徴量とラベルのインデックス整合性を保証するためのユーティリティ。
前処理、特徴量エンジニアリング、ラベル生成の各段階でのインデックス管理を一貫化します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class IndexAlignmentManager:
    """
    インデックス整合性管理クラス
    
    MLワークフローの各段階でインデックスの整合性を保証し、
    特徴量とラベルの対応関係を維持します。
    """
    
    def __init__(self):
        """初期化"""
        self.original_index = None
        self.alignment_history = []
        
    def set_reference_index(self, reference_data: pd.DataFrame) -> None:
        """
        基準インデックスを設定
        
        Args:
            reference_data: 基準となるデータ（通常は元のOHLCVデータ）
        """
        self.original_index = reference_data.index.copy()
        logger.info(f"基準インデックスを設定: {len(self.original_index)}行")
        
    def align_data(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series,
        method: str = "intersection"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特徴量とラベルのインデックスを整合
        
        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            method: 整合方法 ("intersection", "features_priority", "labels_priority")
            
        Returns:
            整合された特徴量とラベルのタプル
        """
        logger.info(f"インデックス整合開始: 特徴量{len(features)}行, ラベル{len(labels)}行")
        
        if method == "intersection":
            # 共通インデックスを使用
            common_index = features.index.intersection(labels.index)
            aligned_features = features.loc[common_index]
            aligned_labels = labels.loc[common_index]
            
        elif method == "features_priority":
            # 特徴量のインデックスを優先
            aligned_features = features
            aligned_labels = labels.reindex(features.index)
            # NaNが発生した場合は該当行を削除
            valid_mask = ~aligned_labels.isna()
            aligned_features = aligned_features[valid_mask]
            aligned_labels = aligned_labels[valid_mask]
            
        elif method == "labels_priority":
            # ラベルのインデックスを優先
            aligned_labels = labels
            aligned_features = features.reindex(labels.index)
            # NaNが発生した場合は該当行を削除
            valid_mask = ~aligned_features.isna().any(axis=1)
            aligned_features = aligned_features[valid_mask]
            aligned_labels = aligned_labels[valid_mask]
            
        else:
            raise ValueError(f"不正な整合方法: {method}")
        
        # 整合結果をログ記録
        alignment_info = {
            "method": method,
            "original_features": len(features),
            "original_labels": len(labels),
            "aligned_features": len(aligned_features),
            "aligned_labels": len(aligned_labels),
            "alignment_ratio": len(aligned_features) / max(len(features), len(labels))
        }
        
        self.alignment_history.append(alignment_info)
        
        logger.info(f"インデックス整合完了: {len(aligned_features)}行 "
                   f"(整合率: {alignment_info['alignment_ratio']*100:.1f}%)")
        
        return aligned_features, aligned_labels
    
    def validate_alignment(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series,
        min_alignment_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        インデックス整合性を検証
        
        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            min_alignment_ratio: 最小整合率
            
        Returns:
            検証結果の辞書
        """
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
            "issues": []
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
        self, 
        data: pd.DataFrame,
        processing_func: callable,
        *args, **kwargs
    ) -> pd.DataFrame:
        """
        処理中にインデックスを保持
        
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
        
        # インデックスの整合性確認
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
    
    def get_alignment_summary(self) -> Dict[str, Any]:
        """
        整合履歴のサマリーを取得
        
        Returns:
            整合履歴のサマリー
        """
        if not self.alignment_history:
            return {"message": "整合履歴がありません"}
        
        latest = self.alignment_history[-1]
        
        summary = {
            "total_alignments": len(self.alignment_history),
            "latest_alignment": latest,
            "average_alignment_ratio": np.mean([h["alignment_ratio"] for h in self.alignment_history]),
            "min_alignment_ratio": min([h["alignment_ratio"] for h in self.alignment_history]),
            "max_alignment_ratio": max([h["alignment_ratio"] for h in self.alignment_history])
        }
        
        return summary


class MLWorkflowIndexManager:
    """
    MLワークフロー専用のインデックス管理クラス
    
    前処理→特徴量エンジニアリング→ラベル生成の流れでインデックス整合性を保証
    """
    
    def __init__(self):
        """初期化"""
        self.alignment_manager = IndexAlignmentManager()
        self.workflow_state = {
            "original_data": None,
            "preprocessed_data": None,
            "features": None,
            "labels": None
        }
    
    def initialize_workflow(self, original_data: pd.DataFrame) -> None:
        """
        ワークフローを初期化
        
        Args:
            original_data: 元のOHLCVデータ
        """
        self.workflow_state["original_data"] = original_data.copy()
        self.alignment_manager.set_reference_index(original_data)
        logger.info(f"MLワークフロー初期化: {len(original_data)}行")
    
    def process_with_index_tracking(
        self,
        stage_name: str,
        data: pd.DataFrame,
        processing_func: callable,
        *args, **kwargs
    ) -> pd.DataFrame:
        """
        インデックス追跡付きで処理を実行
        
        Args:
            stage_name: 処理段階名
            data: 処理対象データ
            processing_func: 処理関数
            *args, **kwargs: 処理関数の引数
            
        Returns:
            処理結果
        """
        logger.info(f"{stage_name}処理開始: {len(data)}行")
        
        result = self.alignment_manager.preserve_index_during_processing(
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
        alignment_method: str = "intersection"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        ワークフローを完了し、最終的な整合を実行
        
        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            alignment_method: 整合方法
            
        Returns:
            整合された特徴量とラベル
        """
        logger.info("MLワークフロー最終整合開始")
        
        # 整合性検証
        validation_result = self.alignment_manager.validate_alignment(features, labels)
        
        if not validation_result["is_valid"]:
            logger.warning("インデックス整合性に問題があります:")
            for issue in validation_result["issues"]:
                logger.warning(f"  - {issue}")
        
        # 最終整合
        aligned_features, aligned_labels = self.alignment_manager.align_data(
            features, labels, method=alignment_method
        )
        
        # ワークフロー状態を更新
        self.workflow_state["features"] = aligned_features
        self.workflow_state["labels"] = aligned_labels
        
        # サマリーログ
        summary = self.alignment_manager.get_alignment_summary()
        logger.info(f"MLワークフロー完了: 最終データ{len(aligned_features)}行 "
                   f"(平均整合率: {summary.get('average_alignment_ratio', 0)*100:.1f}%)")
        
        return aligned_features, aligned_labels
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        ワークフロー全体のサマリーを取得
        
        Returns:
            ワークフローサマリー
        """
        summary = {
            "original_rows": len(self.workflow_state["original_data"]) if self.workflow_state["original_data"] is not None else 0,
            "final_rows": len(self.workflow_state["features"]) if self.workflow_state["features"] is not None else 0,
            "data_retention_rate": 0,
            "alignment_summary": self.alignment_manager.get_alignment_summary()
        }
        
        if summary["original_rows"] > 0:
            summary["data_retention_rate"] = summary["final_rows"] / summary["original_rows"]
        
        return summary
