"""
MLモデルレジストリ・メタデータ

アルゴリズム名の標準化マッピングと、型安全なモデルメタデータ管理を提供します。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from app.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """
    アルゴリズム名レジストリクラス

    モデルクラス名と標準化されたアルゴリズム名のマッピングを一元管理します。
    異なるライブラリ（LightGBM、XGBoost、CatBoost等）のクラス名を
    標準化されたアルゴリズム名に変換します。

    サポートされるアルゴリズム:
        - lightgbm: LightGBM関連クラス
        - xgboost: XGBoost関連クラス
        - catboost: CatBoost関連クラス
        - ensemble: アンサンブル関連クラス
        - stacking: スタッキング関連クラス
    """

    _CLASS_TO_ALGORITHM_MAPPING = {
        "lightgbmmodel": "lightgbm",
        "lgbmclassifier": "lightgbm",
        "lgbmregressor": "lightgbm",
        "xgbclassifier": "xgboost",
        "xgbregressor": "xgboost",
        "ensembletrainer": "ensemble",
        "stackingensemble": "stacking",
        "catboostmodel": "catboost",
        "catboostclassifier": "catboost",
        "catboostregressor": "catboost",
    }

    _SUPPORTED_ALGORITHMS = set(_CLASS_TO_ALGORITHM_MAPPING.values())

    @classmethod
    def get_algorithm_name(cls, model_class_name: str) -> str:
        """
        モデルクラス名から標準化されたアルゴリズム名を取得

        モデルクラス名（例: 'LGBMClassifier'、'XGBRegressor'）を
        標準化されたアルゴリズム名（例: 'lightgbm'、'xgboost'）に変換します。

        Args:
            model_class_name: モデルクラス名

        Returns:
            str: 標準化されたアルゴリズム名（未知の場合は'unknown'）

        変換ロジック:
            1. 定義済みマッピングから完全一致を検索
            2. 部分一致（クラス名にキーが含まれる）を検索
            3. サフィックス（trainer、model、classifier等）を除去して検索
            4. いずれにも一致しない場合は'unknown'を返す
        """
        if not model_class_name:
            return "unknown"
        name = model_class_name.lower()

        for key, val in cls._CLASS_TO_ALGORITHM_MAPPING.items():
            if key in name:
                return val

        base = name
        for s in ["trainer", "model", "classifier", "regressor", "wrapper"]:
            if base.endswith(s):
                base = base[: -len(s)]
                break

        return base if base in cls._SUPPORTED_ALGORITHMS else "unknown"


# グローバルインスタンス
algorithm_registry = AlgorithmRegistry()


@dataclass
class ModelMetadata:
    """
    MLモデルのメタデータを管理するdataclass

    モデルの性能指標、学習パラメータ、作成日時などの
    メタデータを一元管理します。

    Attributes:
        task_type: タスクタイプ（例: 'volatility_regression'、'classification'）
        target_kind: 目的変数の種類（例: 'log_realized_vol'、'classification_label'）
        accuracy: 精度
        precision: 適合率
        recall: 再現率
        f1_score: F1スコア
        auc_score: AUCスコア
        auc_roc: ROC-AUCスコア
        auc_pr: PR-AUCスコア
        balanced_accuracy: バランス精度
        matthews_corrcoef: マシューズ相関係数
        cohen_kappa: コーエンのカッパ係数
        specificity: 特異度
        sensitivity: 感度
        npv: 陰性的中率
        ppv: 陽性的中率
        log_loss: 対数損失
        brier_score: ブライアスコア
        qlike: QLIKE指標
        rmse_log_rv: 対数実現ボラティリティのRMSE
        mae_log_rv: 対数実現ボラティリティのMAE
        feature_count: 特徴量数
        training_samples: 学習サンプル数
        test_samples: テストサンプル数
        best_iteration: 最適イテレーション
        num_classes: クラス数
        train_test_split: 学習/テスト分割比率
        random_state: ランダムシード
        model_type: モデルタイプ
        symbol: シンボル
        timeframe: 時間足
        prediction_horizon: 予測ホライズン
        gate_quantile: ゲート分位点
        gate_cutoff_log_rv: ゲートカットオフ（対数実現ボラティリティ）
        gate_cutoff_vol: ゲートカットオフ（ボラティリティ）
        created_at: 作成日時（ISO形式）
    """

    # タスク情報
    task_type: str = "volatility_regression"
    target_kind: str = "log_realized_vol"

    # 基本性能指標
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0

    # 高度な指標
    balanced_accuracy: float = 0.0
    matthews_corrcoef: float = 0.0
    cohen_kappa: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    npv: float = 0.0
    ppv: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0

    # 回帰指標
    qlike: float = 0.0
    rmse_log_rv: float = 0.0
    mae_log_rv: float = 0.0

    # モデル情報
    feature_count: int = 0
    training_samples: int = 0
    test_samples: int = 0
    best_iteration: int = 0
    num_classes: int = 1
    train_test_split: float = 0.8
    random_state: int = 42
    model_type: str = ""
    symbol: str = ""
    timeframe: str = ""
    prediction_horizon: int = 1
    gate_quantile: float = 0.67
    gate_cutoff_log_rv: float = 0.0
    gate_cutoff_vol: float = 1.0
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        メタデータを辞書形式に変換

        dataclassを辞書に変換してJSONシリアライズ可能な形式にします。

        Returns:
            Dict[str, Any]: メタデータの辞書表現
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_training_result(
        cls,
        training_result: Dict[str, Any],
        training_params: Dict[str, Any],
        model_type: str = "",
        feature_count: int = 0,
    ) -> "ModelMetadata":
        """
        学習結果からメタデータを生成

        学習結果とパラメータからModelMetadataインスタンスを作成します。

        Args:
            training_result: 学習結果の辞書（精度、損失等の指標を含む）
            training_params: 学習パラメータの辞書（タスクタイプ、シンボル等を含む）
            model_type: モデルタイプ（オプション）
            feature_count: 特徴量数（オプション）

        Returns:
            ModelMetadata: 生成されたメタデータインスタンス
        """
        r, p = training_result, training_params
        return cls(
            task_type=p.get("task_type", "volatility_regression"),
            target_kind=p.get("target_kind", "log_realized_vol"),
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
            qlike=r.get("qlike", 0.0),
            rmse_log_rv=r.get("rmse_log_rv", 0.0),
            mae_log_rv=r.get("mae_log_rv", 0.0),
            feature_count=feature_count,
            training_samples=r.get("training_samples", r.get("train_samples", 0)),
            test_samples=r.get("test_samples", 0),
            best_iteration=r.get("best_iteration", 0),
            num_classes=r.get("num_classes", 1),
            train_test_split=p.get("train_test_split", 0.8),
            random_state=p.get("random_state", 42),
            model_type=model_type,
            symbol=p.get("symbol", ""),
            timeframe=p.get("timeframe", ""),
            prediction_horizon=int(p.get("prediction_horizon", p.get("horizon_n", 1))),
            gate_quantile=float(p.get("gate_quantile", 0.67)),
            gate_cutoff_log_rv=float(r.get("gate_cutoff_log_rv", 0.0)),
            gate_cutoff_vol=float(r.get("gate_cutoff_vol", 1.0)),
        )

    def log_summary(self) -> None:
        """
        メタデータの要約をログ出力

        主要なメトリクス（精度、F1スコア、特徴量数、学習サンプル数）を
        ログに出力します。
        """
        logger.info(
            f"モデルメタデータ: 精度={self.accuracy:.4f}, "
            f"F1={self.f1_score:.4f}, 特徴量数={self.feature_count}, "
            f"学習サンプル数={self.training_samples}"
        )

    def validate(self) -> Dict[str, Any]:
        """メタデータの妥当性を検証

        メタデータの値が合理的な範囲内にあるかを検証します。

        Returns:
            Dict[str, Any]: 検証結果を含む辞書。
                - is_valid (bool): 全ての検証をパスした場合はTrue。
                - errors (List[str]): エラーメッセージのリスト。1つ以上ある場合はinvalid。
                - warnings (List[str]): 警告メッセージのリスト。検証には影響しない参考情報。

        検証ルール:
            - volatility_regressionタスク:
              - RMSE(log_rv)、MAE(log_rv)が負でないこと
              - QLIKEが負の場合は警告
            - classificationタスク:
              - 精度、F1スコアが0-1の範囲内であること
            - 共通:
              - 特徴量数、学習サンプル数が正であること
        """
        errors, warnings = [], []
        if self.task_type == "volatility_regression":
            if self.rmse_log_rv < 0.0:
                errors.append(f"RMSE(log_rv) が範囲外です: {self.rmse_log_rv}")
            if self.mae_log_rv < 0.0:
                errors.append(f"MAE(log_rv) が範囲外です: {self.mae_log_rv}")
            if self.qlike < 0.0:
                warnings.append(f"QLIKE が負値です: {self.qlike}")
        else:
            if not 0.0 <= self.accuracy <= 1.0:
                errors.append(f"精度が範囲外です: {self.accuracy}")
            if not 0.0 <= self.f1_score <= 1.0:
                errors.append(f"F1スコアが範囲外です: {self.f1_score}")
        if self.feature_count <= 0:
            warnings.append(f"特徴量数が0以下です: {self.feature_count}")
        if self.training_samples <= 0:
            warnings.append(f"学習サンプル数が0以下です: {self.training_samples}")
        return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings}
