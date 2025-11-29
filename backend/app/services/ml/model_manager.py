"""
統合MLモデル管理サービス

モデルの保存・読み込み・一覧・クリーンアップを一元管理するサービスです。
既存のAPIとの互換性を保持し、効率的なモデル管理を提供します。
"""

import glob
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib

from ...utils.error_handler import safe_ml_operation
from ...config.unified_config import unified_config
from .exceptions import MLModelError

logger = logging.getLogger(__name__)


class ModelManager:
    """
    統合MLモデル管理クラス

    モデルの保存、読み込み、バージョン管理、クリーンアップなど、
    モデル管理に関する全ての機能を提供します。
    既存のAPIとの互換性を保持し、効率的なモデル管理を提供します。
    """

    def __init__(self):
        """
        初期化
        """
        # 既存設定の初期化
        self.config = unified_config.ml.model
        self._ensure_directories()

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.config.model_save_path, exist_ok=True)

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
            ModelError: モデル保存に失敗した場合
        """
        try:
            if model is None:
                raise MLModelError("保存するモデルがNullです")

            # アルゴリズム名を取得
            algorithm_name = self._extract_algorithm_name(model, metadata)

            # 日付のみのタイムスタンプを生成（アルゴリズム名_日付形式）
            date_stamp = datetime.now().strftime("%Y%m%d")
            base_filename = f"{algorithm_name}_{date_stamp}"

            # 同じファイル名が存在する場合は連番を追加
            counter = 1
            filename = f"{base_filename}{self.config.model_file_extension}"
            model_path = os.path.join(self.config.model_save_path, filename)

            while os.path.exists(model_path):
                filename = (
                    f"{base_filename}_{counter:02d}{self.config.model_file_extension}"
                )
                model_path = os.path.join(self.config.model_save_path, filename)
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
            raise MLModelError(f"モデル保存に失敗しました: {e}")

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
            ModelError: モデル読み込みに失敗した場合
        """
        try:
            if not os.path.exists(model_path):
                raise MLModelError(f"モデルファイルが見つかりません: {model_path}")

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
                raise MLModelError("モデルデータに'model'キーが見つかりません")

            model_data.setdefault("scaler", None)
            model_data.setdefault("feature_columns", None)
            model_data.setdefault("metadata", {})

            return model_data

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise MLModelError(f"モデル読み込みに失敗しました: {e}")

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

            for search_path in unified_config.ml.get_model_search_paths():
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
                return None

            # 最新のモデルファイルを取得（更新時刻でソート）
            latest_model = max(all_model_files, key=os.path.getmtime)

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

            for search_path in unified_config.ml.get_model_search_paths():
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
                days=self.config.model_retention_days
            )

            for search_path in unified_config.ml.get_model_search_paths():
                if not os.path.exists(search_path):
                    continue

                for model_file in glob.glob(
                    os.path.join(search_path, f"*{self.config.model_file_extension}")
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
                self.config.model_save_path,
                f"{model_name}_*{self.config.model_file_extension}",
            )
            model_files = glob.glob(pattern)

            if len(model_files) <= self.config.max_model_versions:
                return

            # 更新時刻でソート（古い順）
            model_files.sort(key=os.path.getmtime)

            # 古いファイルを削除（最新のN個を残す）
            files_to_delete = model_files[: -self.config.max_model_versions]

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


    def extract_model_performance_metrics(self, model_path: str) -> Dict[str, float]:
        """
        モデルから性能メトリクスを抽出（classification_reportからのフォールバック含む）
        """
        from .common.evaluation_utils import get_default_metrics

        try:
            model_data = self.load_model(model_path)
            if not model_data or "metadata" not in model_data:
                return get_default_metrics()

            metadata = model_data["metadata"]
            metrics = get_default_metrics()

            # メタデータからメトリクスを更新
            for key in metrics:
                if key in metadata:
                    metrics[key] = metadata[key]

            # classification_report から macro avg をフォールバック
            if "classification_report" in metadata:
                report = metadata["classification_report"]
                if isinstance(report, dict) and "macro avg" in report:
                    macro_avg = report["macro avg"]
                    if metrics["precision"] == 0.0:
                        metrics["precision"] = macro_avg.get("precision", 0.0)
                    if metrics["recall"] == 0.0:
                        metrics["recall"] = macro_avg.get("recall", 0.0)
                    if metrics["f1_score"] == 0.0:
                        metrics["f1_score"] = macro_avg.get("f1-score", 0.0)

            return metrics

        except Exception as e:
            logger.warning(f"メトリクス抽出エラー {model_path}: {e}")
            return get_default_metrics()


# グローバルインスタンス
model_manager = ModelManager()