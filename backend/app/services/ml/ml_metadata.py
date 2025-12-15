"""
MLモデルのメタデータ管理

型安全なメタデータ管理のためのdataclassを提供します。
BaseMLTrainerの責務分割の一環として、冗長なメタデータ構築ロジックを簡潔にします。
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    MLモデルのメタデータを管理するdataclass

    BaseMLTrainerで使用される冗長なメタデータ構築ロジックを
    型安全で簡潔な形に置き換えます。
    """

    # 基本性能指標
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # AUC指標
    auc_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0

    # 高度な指標
    balanced_accuracy: float = 0.0
    matthews_corrcoef: float = 0.0
    cohen_kappa: float = 0.0

    # 専門指標
    specificity: float = 0.0
    sensitivity: float = 0.0
    npv: float = 0.0
    ppv: float = 0.0

    # 確率指標
    log_loss: float = 0.0
    brier_score: float = 0.0

    # モデル情報
    feature_count: int = 0
    training_samples: int = 0
    test_samples: int = 0
    best_iteration: int = 0
    num_classes: int = 2

    # 学習パラメータ
    train_test_split: float = 0.8
    random_state: int = 42

    # システム情報
    model_type: str = ""
    created_at: Optional[str] = None

    def __post_init__(self):
        """初期化後の処理"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)

    @classmethod
    def from_training_result(
        cls,
        training_result: Dict[str, Any],
        training_params: Dict[str, Any],
        model_type: str = "",
        feature_count: int = 0,
    ) -> "ModelMetadata":
        """
        学習結果からModelMetadataを構築するファクトリメソッド

        Args:
            training_result: 学習結果の辞書
            training_params: 学習パラメータの辞書
            model_type: モデルタイプ
            feature_count: 特徴量数

        Returns:
            ModelMetadata インスタンス
        """
        return cls(
            # 基本性能指標
            accuracy=training_result.get("accuracy", 0.0),
            precision=training_result.get("precision", 0.0),
            recall=training_result.get("recall", 0.0),
            f1_score=training_result.get("f1_score", 0.0),
            # AUC指標
            auc_score=training_result.get("auc_score", 0.0),
            auc_roc=training_result.get("roc_auc", training_result.get("auc_roc", 0.0)),
            auc_pr=training_result.get("pr_auc", training_result.get("auc_pr", 0.0)),
            # 高度な指標
            balanced_accuracy=training_result.get("balanced_accuracy", 0.0),
            matthews_corrcoef=training_result.get("matthews_corrcoef", 0.0),
            cohen_kappa=training_result.get("cohen_kappa", 0.0),
            # 専門指標
            specificity=training_result.get("specificity", 0.0),
            sensitivity=training_result.get("sensitivity", 0.0),
            npv=training_result.get("npv", 0.0),
            ppv=training_result.get("ppv", 0.0),
            # 確率指標
            log_loss=training_result.get("log_loss", 0.0),
            brier_score=training_result.get("brier_score", 0.0),
            # モデル情報
            feature_count=feature_count,
            training_samples=training_result.get("training_samples", 0),
            test_samples=training_result.get("test_samples", 0),
            best_iteration=training_result.get("best_iteration", 0),
            num_classes=training_result.get("num_classes", 2),
            # 学習パラメータ
            train_test_split=training_params.get("train_test_split", 0.8),
            random_state=training_params.get("random_state", 42),
            # システム情報
            model_type=model_type,
        )

    def log_summary(self) -> None:
        """メタデータのサマリーをログ出力"""
        logger.info(
            f"モデルメタデータ: 精度={self.accuracy:.4f}, "
            f"F1={self.f1_score:.4f}, 特徴量数={self.feature_count}, "
            f"学習サンプル数={self.training_samples}"
        )

    def validate(self) -> Dict[str, Any]:
        """
        メタデータの妥当性を検証

        Returns:
            検証結果の辞書
        """
        errors = []
        warnings = []

        # 基本的な妥当性チェック
        if not 0.0 <= self.accuracy <= 1.0:
            errors.append(f"精度が範囲外です: {self.accuracy}")

        if not 0.0 <= self.f1_score <= 1.0:
            errors.append(f"F1スコアが範囲外です: {self.f1_score}")

        if self.feature_count <= 0:
            warnings.append(f"特徴量数が0以下です: {self.feature_count}")

        if self.training_samples <= 0:
            warnings.append(f"学習サンプル数が0以下です: {self.training_samples}")

        # パフォーマンス警告
        if self.accuracy < 0.5:
            warnings.append(f"精度が低いです: {self.accuracy:.4f}")

        if self.f1_score < 0.3:
            warnings.append(f"F1スコアが低いです: {self.f1_score:.4f}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


