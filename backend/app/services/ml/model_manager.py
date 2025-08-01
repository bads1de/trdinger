"""
ML モデル管理サービス

分散していたモデルの保存・読み込み・管理機能を統一し、
一貫性のあるモデル管理を提供します。
"""

import os
import glob
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


from app.services.ml.config import ml_config
from ...utils.unified_error_handler import UnifiedModelError, safe_ml_operation

logger = logging.getLogger(__name__)


class ModelManager:
    """
    MLモデルの統一管理クラス

    モデルの保存、読み込み、バージョン管理、クリーンアップなど、
    モデル管理に関する全ての機能を提供します。
    """

    def __init__(self):
        """初期化"""
        self.config = ml_config.model
        self._ensure_directories()

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

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

            # 一般的なアルゴリズム名にマッピング
            algorithm_mapping = {
                "lightgbmmodel": "lightgbm",
                "lgbmclassifier": "lightgbm",
                "lgbmregressor": "lightgbm",
                "xgbclassifier": "xgboost",
                "xgbregressor": "xgboost",
                "catboostclassifier": "catboost",
                "catboostregressor": "catboost",
                "randomforestclassifier": "randomforest",
                "randomforestregressor": "randomforest",
                "tabnetclassifier": "tabnet",
                "tabnetregressor": "tabnet",
                "ensembletrainer": "ensemble",
                "singlemodeltrainer": "single",
                "baggingensemble": "bagging",
                "stackingensemble": "stacking",
            }

            # マッピングから検索
            for key, value in algorithm_mapping.items():
                if key in model_class_name:
                    return value

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
        モデルを保存

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
        default_return=None, context="アンサンブルモデル保存でエラーが発生しました"
    )
    def save_ensemble_model(
        self,
        ensemble_trainer: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        アンサンブルモデルを保存

        Args:
            ensemble_trainer: EnsembleTrainerインスタンス
            model_name: モデル名
            metadata: メタデータ

        Returns:
            保存されたモデルのベースパス

        Raises:
            UnifiedModelError: モデル保存に失敗した場合
        """
        try:
            if ensemble_trainer is None:
                raise UnifiedModelError("保存するアンサンブルトレーナーがNullです")

            # アルゴリズム名と日付でディレクトリ名を生成
            algorithm_name = self._extract_algorithm_name(ensemble_trainer, metadata)
            date_stamp = datetime.now().strftime("%Y%m%d")
            base_ensemble_dir = f"{algorithm_name}_{date_stamp}"

            # 同じディレクトリ名が存在する場合は連番を追加
            counter = 1
            ensemble_dir = base_ensemble_dir
            ensemble_path = os.path.join(self.config.MODEL_SAVE_PATH, ensemble_dir)

            while os.path.exists(ensemble_path):
                ensemble_dir = f"{base_ensemble_dir}_{counter:02d}"
                ensemble_path = os.path.join(self.config.MODEL_SAVE_PATH, ensemble_dir)
                counter += 1

            # ディレクトリを作成
            os.makedirs(ensemble_path, exist_ok=True)

            # ベースパスを設定
            base_path = os.path.join(ensemble_path, "ensemble_model")

            # アンサンブルモデルを保存
            success = ensemble_trainer.save_model(base_path)

            if not success:
                raise UnifiedModelError("アンサンブルモデルの保存に失敗しました")

            # アンサンブル全体のメタデータを保存
            ensemble_metadata = {
                "model_name": algorithm_name,
                "original_model_name": model_name,
                "timestamp": date_stamp,
                "ensemble_method": getattr(
                    ensemble_trainer, "ensemble_method", "unknown"
                ),
                "model_type": "EnsembleModel",
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            # メタデータファイルを保存
            metadata_path = os.path.join(ensemble_path, "ensemble_metadata.json")
            import json

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(ensemble_metadata, f, ensure_ascii=False, indent=2)

            # ディレクトリサイズを計算
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(ensemble_path)
                for filename in filenames
            )

            logger.info(
                f"アンサンブルモデル保存完了: {ensemble_dir} (アルゴリズム: {algorithm_name}, サイズ: {total_size / 1024 / 1024:.2f}MB)"
            )

            # 古いモデルのクリーンアップ
            self._cleanup_old_ensemble_models(model_name)

            return ensemble_path

        except Exception as e:
            logger.error(f"アンサンブルモデル保存エラー: {e}")
            raise UnifiedModelError(f"アンサンブルモデル保存に失敗しました: {e}")

    @safe_ml_operation(
        default_return=None, context="モデル読み込みでエラーが発生しました"
    )
    def load_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        モデルを読み込み

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

    def load_ensemble_model(self, ensemble_path: str) -> Optional[Any]:
        """
        アンサンブルモデルを読み込み

        Args:
            ensemble_path: アンサンブルモデルのディレクトリパス

        Returns:
            読み込まれたEnsembleTrainerインスタンス

        Raises:
            UnifiedModelError: モデル読み込みに失敗した場合
        """
        try:
            from .ensemble.ensemble_trainer import EnsembleTrainer
            import json

            # メタデータを読み込み
            metadata_path = os.path.join(ensemble_path, "ensemble_metadata.json")
            if not os.path.exists(metadata_path):
                raise UnifiedModelError(
                    "アンサンブルメタデータファイルが見つかりません"
                )

            with open(metadata_path, "r", encoding="utf-8") as f:
                ensemble_metadata = json.load(f)

            # ベースパスを設定
            base_path = os.path.join(ensemble_path, "ensemble_model")

            # EnsembleTrainerを作成（仮の設定で初期化）
            dummy_config = {
                "method": ensemble_metadata.get("ensemble_method", "bagging")
            }
            ensemble_trainer = EnsembleTrainer(ensemble_config=dummy_config)

            # モデルを読み込み
            success = ensemble_trainer.load_model(base_path)

            if not success:
                raise UnifiedModelError("アンサンブルモデルの読み込みに失敗しました")

            logger.info(
                f"アンサンブルモデル読み込み完了: {os.path.basename(ensemble_path)}"
            )
            return ensemble_trainer

        except Exception as e:
            logger.error(f"アンサンブルモデル読み込みエラー: {e}")
            raise UnifiedModelError(f"アンサンブルモデル読み込みに失敗しました: {e}")

    def _cleanup_old_ensemble_models(self, model_name: str):
        """
        古いアンサンブルモデルをクリーンアップ

        Args:
            model_name: モデル名
        """
        try:
            # 保持する最大バージョン数を超えた古いモデルを削除
            ensemble_dirs = []

            for search_path in ml_config.get_model_search_paths():
                if not os.path.exists(search_path):
                    continue

                pattern = os.path.join(search_path, f"{model_name}_ensemble_*")
                for ensemble_dir in glob.glob(pattern):
                    if os.path.isdir(ensemble_dir):
                        ensemble_dirs.append(ensemble_dir)

            # 作成日時でソート（新しい順）
            ensemble_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            # 保持数を超えた古いモデルを削除
            for old_dir in ensemble_dirs[self.config.MAX_MODEL_VERSIONS :]:
                try:
                    import shutil

                    shutil.rmtree(old_dir)
                    logger.info(
                        f"古いアンサンブルモデルを削除: {os.path.basename(old_dir)}"
                    )
                except Exception as e:
                    logger.warning(f"古いアンサンブルモデル削除エラー {old_dir}: {e}")

        except Exception as e:
            logger.error(f"古いアンサンブルモデルクリーンアップエラー: {e}")


# グローバルインスタンス
model_manager = ModelManager()
