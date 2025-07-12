"""
ML モデル管理サービス

分散していたモデルの保存・読み込み・管理機能を統一し、
一貫性のあるモデル管理を提供します。
"""

import os
import glob
import joblib
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ...config.ml_config import ml_config
from ...utils.ml_error_handler import MLErrorHandler, MLModelError, safe_ml_operation

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
        os.makedirs(self.config.MODEL_BACKUP_PATH, exist_ok=True)
    
    @safe_ml_operation(default_value=None, error_message="モデル保存でエラーが発生しました")
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        scaler: Optional[Any] = None,
        feature_columns: Optional[List[str]] = None
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
            MLModelError: モデル保存に失敗した場合
        """
        try:
            if model is None:
                raise MLModelError("保存するモデルがNullです")
            
            # タイムスタンプ付きファイル名を生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}{self.config.MODEL_FILE_EXTENSION}"
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, filename)
            
            # モデルデータを構築
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'timestamp': timestamp,
                'model_name': model_name,
                'metadata': metadata or {}
            }
            
            # メタデータにシステム情報を追加
            model_data['metadata'].update({
                'created_at': datetime.now().isoformat(),
                'file_size_bytes': 0,  # 後で更新
                'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                'model_type': type(model).__name__
            })
            
            # モデルを保存
            joblib.dump(model_data, model_path)
            
            # ファイルサイズを更新
            file_size = os.path.getsize(model_path)
            model_data['metadata']['file_size_bytes'] = file_size
            joblib.dump(model_data, model_path)
            
            logger.info(f"モデル保存完了: {filename} ({file_size / 1024 / 1024:.2f}MB)")
            
            # 古いモデルのクリーンアップ
            self._cleanup_old_models(model_name)
            
            return model_path
            
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise MLModelError(f"モデル保存に失敗しました: {e}")
    
    @safe_ml_operation(default_value=None, error_message="モデル読み込みでエラーが発生しました")
    def load_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        モデルを読み込み
        
        Args:
            model_path: モデルファイルパス
        
        Returns:
            読み込まれたモデルデータ
        
        Raises:
            MLModelError: モデル読み込みに失敗した場合
        """
        try:
            if not os.path.exists(model_path):
                raise MLModelError(f"モデルファイルが見つかりません: {model_path}")
            
            # モデルデータを読み込み
            model_data = joblib.load(model_path)
            
            # 古い形式との互換性を保つ
            if not isinstance(model_data, dict):
                # 古い形式（直接モデルオブジェクト）の場合
                model_data = {
                    'model': model_data,
                    'scaler': None,
                    'feature_columns': None,
                    'timestamp': None,
                    'model_name': os.path.basename(model_path),
                    'metadata': {}
                }
            
            # 必要なキーが存在しない場合のデフォルト値設定
            if 'model' not in model_data:
                raise MLModelError("モデルデータに'model'キーが見つかりません")
            
            model_data.setdefault('scaler', None)
            model_data.setdefault('feature_columns', None)
            model_data.setdefault('metadata', {})
            
            logger.info(f"モデル読み込み完了: {os.path.basename(model_path)}")
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
            
            for search_path in ml_config.get_model_search_paths():
                if os.path.exists(search_path):
                    # .pkl と .joblib ファイルを検索
                    pattern_pkl = os.path.join(search_path, f"{model_name_pattern}*.pkl")
                    pattern_joblib = os.path.join(search_path, f"{model_name_pattern}*.joblib")
                    
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
                pattern_joblib = os.path.join(search_path, f"{model_name_pattern}*.joblib")

                for pattern in [pattern_pkl, pattern_joblib]:
                    for model_path in glob.glob(pattern):
                        try:
                            # 絶対パスで正規化して重複チェック
                            normalized_path = os.path.abspath(model_path)
                            if normalized_path in seen_files:
                                continue
                            seen_files.add(normalized_path)

                            stat = os.stat(model_path)
                            models.append({
                                'path': model_path,
                                'name': os.path.basename(model_path),
                                'size_mb': stat.st_size / 1024 / 1024,
                                'modified_at': datetime.fromtimestamp(stat.st_mtime),
                                'directory': os.path.dirname(model_path)
                            })
                        except Exception as e:
                            logger.warning(f"モデルファイル情報取得エラー {model_path}: {e}")

            # 更新時刻でソート（新しい順）
            models.sort(key=lambda x: x['modified_at'], reverse=True)

            return models
            
        except Exception as e:
            logger.error(f"モデル一覧取得エラー: {e}")
            return []
    
    def backup_model(self, model_path: str) -> Optional[str]:
        """
        モデルをバックアップ

        Args:
            model_path: バックアップするモデルのパス

        Returns:
            バックアップファイルのパス
        """
        try:
            logger.info(f"モデルバックアップ開始: {model_path}")

            if not os.path.exists(model_path):
                logger.error(f"バックアップ対象のモデルが見つかりません: {model_path}")
                raise MLModelError(f"バックアップ対象のモデルが見つかりません: {model_path}")

            # バックアップディレクトリの確認・作成
            os.makedirs(self.config.MODEL_BACKUP_PATH, exist_ok=True)

            # バックアップファイル名を生成
            model_name = os.path.basename(model_path)
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
            backup_path = os.path.join(self.config.MODEL_BACKUP_PATH, backup_name)

            # ファイルをコピー
            shutil.copy2(model_path, backup_path)

            logger.info(f"モデルバックアップ完了: {backup_name} -> {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"モデルバックアップエラー: {e}")
            return None
    
    def _cleanup_old_models(self, model_name: str):
        """古いモデルファイルをクリーンアップ"""
        try:
            # 同じモデル名のファイルを検索
            pattern = os.path.join(self.config.MODEL_SAVE_PATH, f"{model_name}_*{self.config.MODEL_FILE_EXTENSION}")
            model_files = glob.glob(pattern)
            
            if len(model_files) <= self.config.MAX_MODEL_VERSIONS:
                return
            
            # 更新時刻でソート（古い順）
            model_files.sort(key=os.path.getmtime)
            
            # 古いファイルを削除（最新のN個を残す）
            files_to_delete = model_files[:-self.config.MAX_MODEL_VERSIONS]
            
            for file_path in files_to_delete:
                try:
                    # バックアップしてから削除
                    self.backup_model(file_path)
                    os.remove(file_path)
                    logger.info(f"古いモデルファイルを削除: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"モデルファイル削除エラー {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"モデルクリーンアップエラー: {e}")
    
    def cleanup_expired_models(self):
        """期限切れのモデルファイルをクリーンアップ"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.MODEL_RETENTION_DAYS)
            
            for search_path in ml_config.get_model_search_paths():
                if not os.path.exists(search_path):
                    continue
                
                for model_file in glob.glob(os.path.join(search_path, f"*{self.config.MODEL_FILE_EXTENSION}")):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                        if file_time < cutoff_date:
                            # バックアップしてから削除
                            self.backup_model(model_file)
                            os.remove(model_file)
                            logger.info(f"期限切れモデルを削除: {os.path.basename(model_file)}")
                    except Exception as e:
                        logger.warning(f"期限切れモデル削除エラー {model_file}: {e}")
                        
        except Exception as e:
            logger.error(f"期限切れモデルクリーンアップエラー: {e}")


# グローバルインスタンス
model_manager = ModelManager()
