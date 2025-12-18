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
        学習結果とパラメータを統合してメタデータオブジェクトを生成

        Args:
            training_result: 学習器が返した生のメトリクス辞書
            training_params: 学習時に使用された設定パラメータ
            model_type: モデルのクラス名等
            feature_count: 入力特徴量の数

        Returns:
            正規化されたメタデータインスタンス
        """
        r, p = training_result, training_params
        return cls(
            # 指標
            accuracy=r.get("accuracy", 0.0),
            precision=r.get("precision", 0.0),
            recall=r.get("recall", 0.0),
            f1_score=r.get("f1_score", 0.0),
            auc_score=r.get("auc_score", 0.0),
            auc_roc=r.get("roc_auc", r.get("auc_roc", 0.0)),
            auc_pr=r.get("pr_auc", r.get("auc_pr", 0.0)),
            balanced_accuracy=r.get("balanced_accuracy", 0.0),
            matthews_corrcoef=r.get("matthews_corrcoef", 0.0),
            cohen_kappa=r.get("cohen_kappa", 0.0),
            specificity=r.get("specificity", 0.0),
            sensitivity=r.get("sensitivity", 0.0),
            npv=r.get("npv", 0.0),
            ppv=r.get("ppv", 0.0),
            log_loss=r.get("log_loss", 0.0),
            brier_score=r.get("brier_score", 0.0),
            # 情報
            feature_count=feature_count,
            training_samples=r.get("training_samples", 0),
            test_samples=r.get("test_samples", 0),
            best_iteration=r.get("best_iteration", 0),
            num_classes=r.get("num_classes", 2),
            train_test_split=p.get("train_test_split", 0.8),
            random_state=p.get("random_state", 42),
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
        保持しているメタデータが物理的・統計的に妥当であるか検証

        精度が0〜1の範囲内であるか、特徴量数やサンプル数が正の値であるか
        などをチェックし、異常があればエラーまたは警告を返します。

        Returns:
            {'is_valid': bool, 'errors': List[str], 'warnings': List[str]}
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
