"""
拡張モデル管理システム

分析報告書で提案されたモデル管理システムの改善を実装。
バージョン管理、パフォーマンス監視、自動デプロイメントを提供します。
"""

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """モデルステータス"""

    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class PerformanceMetric(Enum):
    """パフォーマンス指標"""

    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    PRECISION = "precision"
    RECALL = "recall"


@dataclass
class ModelMetadata:
    """モデルメタデータ"""

    model_id: str
    version: str
    name: str
    algorithm: str
    created_at: datetime
    status: ModelStatus

    # 学習情報
    training_data_hash: str
    feature_count: int
    sample_count: int
    training_duration: float

    # パフォーマンス指標
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]

    # 設定情報
    hyperparameters: Dict[str, Any]
    feature_selection_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]

    # ファイルパス
    model_path: str
    metadata_path: str

    # その他
    tags: List[str]
    description: str
    author: str


@dataclass
class PerformanceMonitoringConfig:
    """パフォーマンス監視設定"""

    enable_monitoring: bool = True
    alert_threshold: float = 0.05  # パフォーマンス低下の閾値
    monitoring_window: int = 100  # 監視ウィンドウサイズ
    auto_retrain_threshold: float = 0.10  # 自動再学習の閾値
    max_performance_history: int = 1000


class EnhancedModelManager:
    """
    拡張モデル管理システム

    モデルのライフサイクル全体を管理し、
    バージョン管理とパフォーマンス監視を提供します。
    """

    def __init__(
        self,
        base_path: str = "models",
        monitoring_config: PerformanceMonitoringConfig = None,
    ):
        """
        初期化

        Args:
            base_path: モデル保存ベースパス
            monitoring_config: パフォーマンス監視設定
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.monitoring_config = monitoring_config or PerformanceMonitoringConfig()

        # メタデータストレージ
        self.metadata_file = self.base_path / "model_registry.json"
        self.performance_history_file = self.base_path / "performance_history.json"

        # モデルレジストリの初期化
        self.model_registry = self._load_model_registry()
        self.performance_history = self._load_performance_history()

    def register_model(
        self,
        model: Any,
        model_name: str,
        algorithm: str,
        training_data: pd.DataFrame,
        performance_metrics: Dict[str, float],
        validation_metrics: Dict[str, float] = None,
        hyperparameters: Dict[str, Any] = None,
        feature_selection_config: Dict[str, Any] = None,
        preprocessing_config: Dict[str, Any] = None,
        tags: List[str] = None,
        description: str = "",
        author: str = "system",
    ) -> str:
        """
        モデルを登録

        Args:
            model: 学習済みモデル
            model_name: モデル名
            algorithm: アルゴリズム名
            training_data: 学習データ
            performance_metrics: パフォーマンス指標
            validation_metrics: 検証指標
            hyperparameters: ハイパーパラメータ
            feature_selection_config: 特徴量選択設定
            preprocessing_config: 前処理設定
            tags: タグ
            description: 説明
            author: 作成者

        Returns:
            モデルID
        """
        logger.info(f"🔄 モデル登録開始: {model_name}")

        # モデルIDとバージョンを生成
        model_id = self._generate_model_id(model_name, algorithm)
        version = self._generate_version(model_id)

        # データハッシュを計算
        training_data_hash = self._calculate_data_hash(training_data)

        # モデルファイルパスを生成
        model_dir = self.base_path / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        metadata_path = model_dir / "metadata.json"

        # モデルを保存
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"モデルファイル保存: {model_path}")
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise

        # メタデータを作成
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=model_name,
            algorithm=algorithm,
            created_at=datetime.now(timezone.utc),
            status=ModelStatus.TRAINED,
            training_data_hash=training_data_hash,
            feature_count=training_data.shape[1],
            sample_count=training_data.shape[0],
            training_duration=0.0,  # 実際の学習時間は外部から設定
            performance_metrics=performance_metrics,
            validation_metrics=validation_metrics or {},
            hyperparameters=hyperparameters or {},
            feature_selection_config=feature_selection_config or {},
            preprocessing_config=preprocessing_config or {},
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            tags=tags or [],
            description=description,
            author=author,
        )

        # メタデータを保存
        self._save_metadata(metadata)

        # レジストリに追加
        self.model_registry[f"{model_id}:{version}"] = asdict(metadata)
        self._save_model_registry()

        logger.info(f"✅ モデル登録完了: {model_id}:{version}")
        return f"{model_id}:{version}"

    def load_model(self, model_key: str) -> Tuple[Any, ModelMetadata]:
        """
        モデルをロード

        Args:
            model_key: モデルキー (model_id:version)

        Returns:
            モデルとメタデータのタプル
        """
        if model_key not in self.model_registry:
            raise ValueError(f"モデルが見つかりません: {model_key}")

        metadata_dict = self.model_registry[model_key]
        metadata = ModelMetadata(**metadata_dict)

        try:
            with open(metadata.model_path, "rb") as f:
                model = pickle.load(f)

            logger.info(f"モデルロード完了: {model_key}")
            return model, metadata

        except Exception as e:
            logger.error(f"モデルロードエラー: {e}")
            raise

    def get_best_model(
        self,
        metric: PerformanceMetric = PerformanceMetric.BALANCED_ACCURACY,
        algorithm: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, ModelMetadata]]:
        """
        最高性能のモデルを取得

        Args:
            metric: 評価指標
            algorithm: アルゴリズム名でフィルタ
            tags: タグでフィルタ

        Returns:
            モデルキーとメタデータのタプル
        """
        candidates = []

        for model_key, metadata_dict in self.model_registry.items():
            metadata = ModelMetadata(**metadata_dict)

            # フィルタ条件をチェック
            if algorithm and metadata.algorithm != algorithm:
                continue

            if tags and not any(tag in metadata.tags for tag in tags):
                continue

            if metadata.status not in [
                ModelStatus.TRAINED,
                ModelStatus.VALIDATED,
                ModelStatus.DEPLOYED,
            ]:
                continue

            # 指標値を取得
            metric_value = metadata.performance_metrics.get(metric.value, 0.0)
            candidates.append((model_key, metadata, metric_value))

        if not candidates:
            return None

        # 最高スコアのモデルを選択
        best_model_key, best_metadata, _ = max(candidates, key=lambda x: x[2])
        return best_model_key, best_metadata

    def update_model_status(self, model_key: str, status: ModelStatus):
        """モデルステータスを更新"""
        if model_key not in self.model_registry:
            raise ValueError(f"モデルが見つかりません: {model_key}")

        self.model_registry[model_key]["status"] = status.value
        self._save_model_registry()

        logger.info(f"モデルステータス更新: {model_key} -> {status.value}")

    def record_performance(
        self,
        model_key: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ):
        """
        パフォーマンス履歴を記録

        Args:
            model_key: モデルキー
            metrics: パフォーマンス指標
            timestamp: タイムスタンプ
        """
        if not self.monitoring_config.enable_monitoring:
            return

        timestamp = timestamp or datetime.now(timezone.utc)

        if model_key not in self.performance_history:
            self.performance_history[model_key] = []

        # 履歴に追加
        record = {"timestamp": timestamp.isoformat(), "metrics": metrics}

        self.performance_history[model_key].append(record)

        # 履歴サイズ制限
        max_history = self.monitoring_config.max_performance_history
        if len(self.performance_history[model_key]) > max_history:
            self.performance_history[model_key] = self.performance_history[model_key][
                -max_history:
            ]

        self._save_performance_history()

        # パフォーマンス低下の検出
        self._check_performance_degradation(model_key, metrics)

    def _check_performance_degradation(
        self, model_key: str, current_metrics: Dict[str, float]
    ):
        """パフォーマンス低下を検出"""
        if model_key not in self.model_registry:
            return

        baseline_metrics = self.model_registry[model_key]["performance_metrics"]

        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                degradation = baseline_value - current_value

                if degradation > self.monitoring_config.alert_threshold:
                    logger.warning(
                        f"⚠️ パフォーマンス低下検出: {model_key} "
                        f"{metric_name}: {baseline_value:.4f} -> {current_value:.4f} "
                        f"(低下: {degradation:.4f})"
                    )

                    # 自動再学習の閾値をチェック
                    if degradation > self.monitoring_config.auto_retrain_threshold:
                        logger.warning(f"🔄 自動再学習が推奨されます: {model_key}")

    def get_model_list(
        self, status: Optional[ModelStatus] = None, algorithm: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """モデル一覧を取得"""
        models = []

        for model_key, metadata_dict in self.model_registry.items():
            if status and metadata_dict["status"] != status.value:
                continue

            if algorithm and metadata_dict["algorithm"] != algorithm:
                continue

            models.append({"model_key": model_key, **metadata_dict})

        # 作成日時でソート
        models.sort(key=lambda x: x["created_at"], reverse=True)
        return models

    def delete_model(self, model_key: str):
        """モデルを削除"""
        if model_key not in self.model_registry:
            raise ValueError(f"モデルが見つかりません: {model_key}")

        metadata_dict = self.model_registry[model_key]

        # ファイルを削除
        try:
            model_path = Path(metadata_dict["model_path"])
            if model_path.exists():
                model_path.unlink()

            metadata_path = Path(metadata_dict["metadata_path"])
            if metadata_path.exists():
                metadata_path.unlink()

            # ディレクトリが空なら削除
            model_dir = model_path.parent
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()

        except Exception as e:
            logger.warning(f"ファイル削除エラー: {e}")

        # レジストリから削除
        del self.model_registry[model_key]

        # パフォーマンス履歴も削除
        if model_key in self.performance_history:
            del self.performance_history[model_key]

        self._save_model_registry()
        self._save_performance_history()

        logger.info(f"モデル削除完了: {model_key}")

    def _generate_model_id(self, model_name: str, algorithm: str) -> str:
        """モデルIDを生成"""
        base_id = f"{model_name}_{algorithm}"
        # 特殊文字を除去
        model_id = "".join(c for c in base_id if c.isalnum() or c in "_-")
        return model_id

    def _generate_version(self, model_id: str) -> str:
        """バージョンを生成"""
        existing_versions = [
            key.split(":")[1]
            for key in self.model_registry.keys()
            if key.startswith(f"{model_id}:")
        ]

        if not existing_versions:
            return "v1.0.0"

        # 最新バージョンを取得してインクリメント
        latest_version = max(existing_versions)
        try:
            major, minor, patch = map(int, latest_version[1:].split("."))
            return f"v{major}.{minor}.{patch + 1}"
        except ValueError:
            # 期待するフォーマット 'vX.Y.Z' を満たさない場合は後方互換のフォールバック
            return f"v1.0.{len(existing_versions)}"

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """データハッシュを計算"""
        try:
            data_str = data.to_string()
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            # DataFrame でない/シリアライズ失敗などの場合のフォールバック
            return "unknown"

    def _save_metadata(self, metadata: ModelMetadata):
        """メタデータを保存"""
        try:
            with open(metadata.metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    asdict(metadata), f, indent=2, default=str, ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")

    def _load_model_registry(self) -> Dict[str, Any]:
        """モデルレジストリをロード"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"レジストリロードエラー: {e}")

        return {}

    def _save_model_registry(self):
        """モデルレジストリを保存"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.model_registry, f, indent=2, default=str, ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"レジストリ保存エラー: {e}")

    def _load_performance_history(self) -> Dict[str, List[Dict]]:
        """パフォーマンス履歴をロード"""
        try:
            if self.performance_history_file.exists():
                with open(self.performance_history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"履歴ロードエラー: {e}")

        return {}

    def _save_performance_history(self):
        """パフォーマンス履歴を保存"""
        try:
            with open(self.performance_history_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.performance_history,
                    f,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"履歴保存エラー: {e}")


# グローバルインスタンス
enhanced_model_manager = EnhancedModelManager()
