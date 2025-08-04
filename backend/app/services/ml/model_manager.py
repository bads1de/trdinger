"""
統合MLモデル管理サービス

モデルの保存・読み込み・一覧・クリーンアップを一元管理するサービスです。
バージョン管理、パフォーマンス監視、メタデータ管理などの高機能も提供します。
既存のAPIとの互換性を保持しながら、拡張機能も利用できます。
"""

import glob
import hashlib
import logging
import os
import warnings
 
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from .config import ml_config

from ...utils.unified_error_handler import UnifiedModelError, safe_ml_operation

logger = logging.getLogger(__name__)


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
class PerformanceMonitoringConfig:
    """パフォーマンス監視設定"""

    enable_monitoring: bool = True
    alert_threshold: float = 0.05  # パフォーマンス低下の閾値
    monitoring_window: int = 100  # 監視ウィンドウサイズ
    auto_retrain_threshold: float = 0.10  # 自動再学習の閾値
    max_performance_history: int = 1000


class ModelManager:
    """
    統合MLモデル管理クラス

    モデルの保存、読み込み、バージョン管理、クリーンアップなど、
    モデル管理に関する全ての機能を提供します。
    既存のAPIとの互換性を保持しながら、バージョン管理、パフォーマンス監視、
    メタデータ管理などの高機能も提供します。
    """

    def __init__(self, monitoring_config: PerformanceMonitoringConfig = None):
        """
        初期化

        Args:
            monitoring_config: パフォーマンス監視設定
        """
        # 既存設定の初期化
        self.config = ml_config.model
        self._ensure_directories()

        # 拡張機能の初期化（ファイルシステムベース）
        self.base_path = Path(self.config.MODEL_SAVE_PATH)
        self.monitoring_config = monitoring_config or PerformanceMonitoringConfig()

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

    # ========================================
    # 既存API（互換性維持）
    # ========================================

    def _extract_algorithm_name(
        self, model: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        モデルからアルゴリズム名を抽出

        Args:
            model: モデルオブジェクト
            metadata: メタデータ

        Returns:
            アルゴリズム名
        """
        try:
            # メタデータから best_algorithm を取得（アンサンブルの場合）
            if metadata and "best_algorithm" in metadata:
                algorithm = metadata["best_algorithm"]
                if algorithm:
                    return algorithm.lower()

            # メタデータから model_type を取得
            if metadata and "model_type" in metadata:
                model_type = metadata["model_type"]
                if model_type and model_type != "unknown":
                    # EnsembleTrainer などのクラス名から推定
                    if "ensemble" in model_type.lower():
                        return "ensemble"
                    elif "single" in model_type.lower():
                        return "single"
                    else:
                        return model_type.lower()

            # モデルオブジェクトのクラス名から推定
            model_class_name = type(model).__name__.lower()

            # AlgorithmRegistry を使用してアルゴリズム名を取得
            from .common.algorithm_registry import algorithm_registry

            algorithm_name = algorithm_registry.get_algorithm_name(model_class_name)

            if algorithm_name != "unknown":
                logger.debug(
                    f"AlgorithmRegistryからアルゴリズム名を取得: {model_class_name} -> {algorithm_name}"
                )
                return algorithm_name

            # デフォルト値
            return "unknown"

        except Exception as e:
            logger.warning(f"アルゴリズム名の抽出に失敗: {e}")
            return "unknown"

    @safe_ml_operation(default_return=None, context="モデル保存でエラーが発生しました")
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        scaler: Optional[Any] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        モデルを保存（既存API）

        Args:
            model: 保存するモデル
            model_name: モデル名
            metadata: メタデータ
            scaler: スケーラー（オプション）
            feature_columns: 特徴量カラム（オプション）

        Returns:
            保存されたモデルのパス

        Raises:
            UnifiedModelError: モデル保存に失敗した場合
        """
        try:
            if model is None:
                raise UnifiedModelError("保存するモデルがNullです")

            # アルゴリズム名を取得
            algorithm_name = self._extract_algorithm_name(model, metadata)

            # 日付のみのタイムスタンプを生成（アルゴリズム名_日付形式）
            date_stamp = datetime.now().strftime("%Y%m%d")
            base_filename = f"{algorithm_name}_{date_stamp}"

            # 同じファイル名が存在する場合は連番を追加
            counter = 1
            filename = f"{base_filename}{self.config.MODEL_FILE_EXTENSION}"
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, filename)

            while os.path.exists(model_path):
                filename = (
                    f"{base_filename}_{counter:02d}{self.config.MODEL_FILE_EXTENSION}"
                )
                model_path = os.path.join(self.config.MODEL_SAVE_PATH, filename)
                counter += 1

            # モデルデータを構築
            model_data = {
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "timestamp": date_stamp,
                "model_name": algorithm_name,  # アルゴリズム名を保存
                "original_model_name": model_name,  # 元のモデル名も保持
                "metadata": metadata or {},
            }

            # メタデータにシステム情報を追加
            model_data["metadata"].update(
                {
                    "created_at": datetime.now().isoformat(),
                    "file_size_bytes": 0,  # 後で更新
                    "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                    "model_type": type(model).__name__,
                }
            )

            # モデルを保存
            joblib.dump(model_data, model_path)

            # ファイルサイズを更新
            file_size = os.path.getsize(model_path)
            model_data["metadata"]["file_size_bytes"] = file_size
            joblib.dump(model_data, model_path)

            logger.info(
                f"モデル保存完了: {filename} (アルゴリズム: {algorithm_name}, サイズ: {file_size / 1024 / 1024:.2f}MB)"
            )

            # 古いモデルのクリーンアップ
            self._cleanup_old_models(model_name)

            return model_path

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise UnifiedModelError(f"モデル保存に失敗しました: {e}")

    @safe_ml_operation(
        default_return=None, context="モデル読み込みでエラーが発生しました"
    )
    def load_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        モデルを読み込み（既存API）

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込まれたモデルデータ

        Raises:
            UnifiedModelError: モデル読み込みに失敗した場合
        """
        try:
            if not os.path.exists(model_path):
                raise UnifiedModelError(f"モデルファイルが見つかりません: {model_path}")

            # モデルデータを読み込み
            from sklearn.exceptions import InconsistentVersionWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                model_data = joblib.load(model_path)
 
            # 古い形式との互換性を保つ
            if not isinstance(model_data, dict):
                # 古い形式（直接モデルオブジェクト）の場合
                model_data = {
                    "model": model_data,
                    "scaler": None,
                    "feature_columns": None,
                    "timestamp": None,
                    "model_name": os.path.basename(model_path),
                    "metadata": {},
                }

            # 必要なキーが存在しない場合のデフォルト値設定
            if "model" not in model_data:
                raise UnifiedModelError("モデルデータに'model'キーが見つかりません")

            model_data.setdefault("scaler", None)
            model_data.setdefault("feature_columns", None)
            model_data.setdefault("metadata", {})

            logger.info(f"モデル読み込み完了: {os.path.basename(model_path)}")
            return model_data

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise UnifiedModelError(f"モデル読み込みに失敗しました: {e}")

    def get_latest_model(self, model_name_pattern: str = "*") -> Optional[str]:
        """
        最新のモデルファイルパスを取得

        Args:
            model_name_pattern: モデル名のパターン（ワイルドカード使用可能）

        Returns:
            最新モデルのファイルパス
        """
        try:
            # 複数の検索パスから最新モデルを検索
            all_model_files = []

            for search_path in ml_config.get_model_search_paths():
                if os.path.exists(search_path):
                    # .pkl と .joblib ファイルを検索
                    pattern_pkl = os.path.join(
                        search_path, f"{model_name_pattern}*.pkl"
                    )
                    pattern_joblib = os.path.join(
                        search_path, f"{model_name_pattern}*.joblib"
                    )

                    pkl_files = glob.glob(pattern_pkl)
                    joblib_files = glob.glob(pattern_joblib)
                    all_model_files.extend(pkl_files + joblib_files)

            if not all_model_files:
                logger.info(f"モデルファイルが見つかりません: {model_name_pattern}")
                return None

            # 最新のモデルファイルを取得（更新時刻でソート）
            latest_model = max(all_model_files, key=os.path.getmtime)
            logger.info(f"最新モデルを発見: {os.path.basename(latest_model)}")

            return latest_model

        except Exception as e:
            logger.error(f"最新モデル検索エラー: {e}")
            return None

    def list_models(self, model_name_pattern: str = "*") -> List[Dict[str, Any]]:
        """
        モデルファイルの一覧を取得

        Args:
            model_name_pattern: モデル名のパターン

        Returns:
            モデル情報のリスト
        """
        try:
            models = []
            seen_files = set()  # 重複を防ぐためのセット

            for search_path in ml_config.get_model_search_paths():
                if not os.path.exists(search_path):
                    continue

                pattern_pkl = os.path.join(search_path, f"{model_name_pattern}*.pkl")
                pattern_joblib = os.path.join(
                    search_path, f"{model_name_pattern}*.joblib"
                )

                for pattern in [pattern_pkl, pattern_joblib]:
                    for model_path in glob.glob(pattern):
                        try:
                            # 絶対パスで正規化して重複チェック
                            normalized_path = os.path.abspath(model_path)
                            if normalized_path in seen_files:
                                continue
                            seen_files.add(normalized_path)

                            stat = os.stat(model_path)
                            models.append(
                                {
                                    "path": model_path,
                                    "name": os.path.basename(model_path),
                                    "size_mb": stat.st_size / 1024 / 1024,
                                    "modified_at": datetime.fromtimestamp(
                                        stat.st_mtime
                                    ),
                                    "directory": os.path.dirname(model_path),
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"モデルファイル情報取得エラー {model_path}: {e}"
                            )

            # 更新時刻でソート（新しい順）
            models.sort(key=lambda x: x["modified_at"], reverse=True)

            return models

        except Exception as e:
            logger.error(f"モデル一覧取得エラー: {e}")
            return []

    def cleanup_expired_models(self):
        """期限切れのモデルファイルをクリーンアップ"""
        try:
            cutoff_date = datetime.now() - timedelta(
                days=self.config.MODEL_RETENTION_DAYS
            )

            for search_path in ml_config.get_model_search_paths():
                if not os.path.exists(search_path):
                    continue

                for model_file in glob.glob(
                    os.path.join(search_path, f"*{self.config.MODEL_FILE_EXTENSION}")
                ):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                        if file_time < cutoff_date:
                            os.remove(model_file)
                            logger.info(
                                f"期限切れモデルを削除: {os.path.basename(model_file)}"
                            )
                    except Exception as e:
                        logger.warning(f"期限切れモデル削除エラー {model_file}: {e}")

        except Exception as e:
            logger.error(f"期限切れモデルクリーンアップエラー: {e}")

    def _cleanup_old_models(self, model_name: str):
        """古いモデルファイルをクリーンアップ"""
        try:
            # 同じモデル名のファイルを検索
            pattern = os.path.join(
                self.config.MODEL_SAVE_PATH,
                f"{model_name}_*{self.config.MODEL_FILE_EXTENSION}",
            )
            model_files = glob.glob(pattern)

            if len(model_files) <= self.config.MAX_MODEL_VERSIONS:
                return

            # 更新時刻でソート（古い順）
            model_files.sort(key=os.path.getmtime)

            # 古いファイルを削除（最新のN個を残す）
            files_to_delete = model_files[: -self.config.MAX_MODEL_VERSIONS]

            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.info(
                        f"古いモデルファイルを削除: {os.path.basename(file_path)}"
                    )
                except Exception as e:
                    logger.warning(f"モデルファイル削除エラー {file_path}: {e}")

        except Exception as e:
            logger.error(f"モデルクリーンアップエラー: {e}")

    # ========================================
    # 拡張API（新機能）
    # ========================================

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
        モデルを登録（拡張機能 - ファイルシステムベース）

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
            モデルファイルパス
        """
        logger.info(f"🔄 モデル登録開始: {model_name}")

        # 拡張メタデータを含むモデルデータを構築
        extended_metadata = {
            "algorithm": algorithm,
            "performance_metrics": performance_metrics,
            "validation_metrics": validation_metrics or {},
            "hyperparameters": hyperparameters or {},
            "feature_selection_config": feature_selection_config or {},
            "preprocessing_config": preprocessing_config or {},
            "tags": tags or [],
            "description": description,
            "author": author,
            "feature_count": training_data.shape[1],
            "sample_count": training_data.shape[0],
            "training_data_hash": self._calculate_data_hash(training_data),
        }

        # 既存のsave_modelメソッドを使用（拡張メタデータ付き）
        model_path = self.save_model(
            model=model,
            model_name=model_name,
            metadata=extended_metadata,
            scaler=None,
            feature_columns=(
                list(training_data.columns)
                if hasattr(training_data, "columns")
                else None
            ),
        )

        logger.info(f"✅ モデル登録完了: {model_path}")
        return model_path

    def load_model_enhanced(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        モデルをロード（拡張機能 - ファイルシステムベース）

        Args:
            model_path: モデルファイルパス

        Returns:
            モデルとメタデータのタプル
        """
        # 既存のload_modelメソッドを使用
        model_data = self.load_model(model_path)

        if model_data is None:
            raise ValueError(f"モデルが見つかりません: {model_path}")

        # モデルとメタデータを分離
        model = model_data["model"]
        metadata = model_data.get("metadata", {})

        logger.info(f"拡張モデルロード完了: {os.path.basename(model_path)}")
        return model, metadata

    def get_best_model(
        self,
        metric: PerformanceMetric = PerformanceMetric.BALANCED_ACCURACY,
        algorithm: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        最高性能のモデルを取得（ファイルシステムベース）

        Args:
            metric: 評価指標
            algorithm: アルゴリズム名でフィルタ
            tags: タグでフィルタ

        Returns:
            最高性能モデルのファイルパス
        """
        models = self.list_models("*")
        candidates = []

        for model_info in models:
            try:
                # モデルファイルからメタデータを読み込み
                model_data = self.load_model(model_info["path"])
                if not model_data:
                    continue

                metadata = model_data.get("metadata", {})

                # フィルタ条件をチェック
                if algorithm and metadata.get("algorithm") != algorithm:
                    continue

                if tags and not any(tag in metadata.get("tags", []) for tag in tags):
                    continue

                # パフォーマンス指標を取得
                performance_metrics = metadata.get("performance_metrics", {})
                metric_value = performance_metrics.get(metric.value, 0.0)

                candidates.append((model_info["path"], metric_value))

            except Exception as e:
                logger.warning(
                    f"モデルメタデータ読み込みエラー {model_info['path']}: {e}"
                )
                continue

        if not candidates:
            return None

        # 最高スコアのモデルを選択
        best_model_path, _ = max(candidates, key=lambda x: x[1])
        return best_model_path

    def get_model_list_enhanced(
        self, algorithm: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """モデル一覧を取得（拡張機能 - ファイルシステムベース）"""
        models = self.list_models("*")
        enhanced_models = []

        for model_info in models:
            try:
                # モデルファイルからメタデータを読み込み
                model_data = self.load_model(model_info["path"])
                if not model_data:
                    continue

                metadata = model_data.get("metadata", {})

                # フィルタ条件をチェック
                if algorithm and metadata.get("algorithm") != algorithm:
                    continue

                # 拡張情報を追加
                enhanced_info = {
                    **model_info,
                    "algorithm": metadata.get("algorithm", "unknown"),
                    "performance_metrics": metadata.get("performance_metrics", {}),
                    "validation_metrics": metadata.get("validation_metrics", {}),
                    "feature_count": metadata.get("feature_count", 0),
                    "sample_count": metadata.get("sample_count", 0),
                    "tags": metadata.get("tags", []),
                    "description": metadata.get("description", ""),
                    "author": metadata.get("author", "unknown"),
                }

                enhanced_models.append(enhanced_info)

            except Exception as e:
                logger.warning(
                    f"モデルメタデータ読み込みエラー {model_info['path']}: {e}"
                )
                # メタデータ読み込みに失敗してもモデル情報は追加
                enhanced_models.append(
                    {
                        **model_info,
                        "algorithm": "unknown",
                        "performance_metrics": {},
                        "validation_metrics": {},
                        "feature_count": 0,
                        "sample_count": 0,
                        "tags": [],
                        "description": "",
                        "author": "unknown",
                    }
                )

        # 更新時刻でソート（新しい順）
        enhanced_models.sort(key=lambda x: x["modified_at"], reverse=True)
        return enhanced_models

    # ========================================
    # ヘルパーメソッド
    # ========================================

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """データハッシュを計算"""
        try:
            data_str = data.to_string()
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            # DataFrame でない/シリアライズ失敗などの場合のフォールバック
            return "unknown"


# グローバルインスタンス
model_manager = ModelManager()
