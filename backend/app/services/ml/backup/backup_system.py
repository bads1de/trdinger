"""
バックアップ・災害復旧システム

データ、モデル、設定の自動バックアップと復旧機能を提供します。
"""

import logging
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """バックアップタイプ"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """バックアップステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackupConfig:
    """バックアップ設定"""
    # バックアップ対象
    model_directory: str = "backend/models"
    data_directory: str = "backend/data"
    config_directory: str = "backend/config"
    
    # バックアップ先
    backup_root: str = "backups"
    
    # スケジュール
    auto_backup_enabled: bool = True
    full_backup_interval_hours: int = 24
    incremental_backup_interval_hours: int = 6
    
    # 保持ポリシー
    max_full_backups: int = 7
    max_incremental_backups: int = 30
    
    # 圧縮設定
    compression_enabled: bool = True
    compression_level: int = 6
    
    # 検証設定
    verify_backups: bool = True
    checksum_algorithm: str = "sha256"


@dataclass
class BackupRecord:
    """バックアップ記録"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime]
    file_path: str
    file_size_bytes: int
    checksum: str
    error_message: Optional[str] = None
    files_count: int = 0
    compressed_size_bytes: int = 0


class BackupSystem:
    """
    バックアップ・災害復旧システム
    
    データ、モデル、設定の自動バックアップと復旧機能を提供します。
    """
    
    def __init__(self, config: Optional[BackupConfig] = None):
        """
        初期化
        
        Args:
            config: バックアップ設定
        """
        self.config = config or BackupConfig()
        self.backup_records: List[BackupRecord] = []
        self.is_running = False
        self.backup_thread: Optional[threading.Thread] = None
        
        # バックアップディレクトリを作成
        self._ensure_backup_directories()
        
        # 既存のバックアップ記録を読み込み
        self._load_backup_records()
    
    def _ensure_backup_directories(self):
        """バックアップディレクトリを確保"""
        backup_root = Path(self.config.backup_root)
        backup_root.mkdir(parents=True, exist_ok=True)
        
        # サブディレクトリを作成
        (backup_root / "full").mkdir(exist_ok=True)
        (backup_root / "incremental").mkdir(exist_ok=True)
        (backup_root / "metadata").mkdir(exist_ok=True)
    
    def _load_backup_records(self):
        """バックアップ記録を読み込み"""
        try:
            records_file = Path(self.config.backup_root) / "metadata" / "backup_records.json"
            if records_file.exists():
                with open(records_file, 'r', encoding='utf-8') as f:
                    records_data = json.load(f)
                
                self.backup_records = []
                for record_data in records_data:
                    # 日時文字列をdatetimeオブジェクトに変換
                    record_data['start_time'] = datetime.fromisoformat(record_data['start_time'])
                    if record_data.get('end_time'):
                        record_data['end_time'] = datetime.fromisoformat(record_data['end_time'])
                    
                    # EnumをPythonオブジェクトに変換
                    record_data['backup_type'] = BackupType(record_data['backup_type'])
                    record_data['status'] = BackupStatus(record_data['status'])
                    
                    self.backup_records.append(BackupRecord(**record_data))
                
                logger.info(f"バックアップ記録を読み込みました: {len(self.backup_records)}件")
                
        except Exception as e:
            logger.error(f"バックアップ記録読み込みエラー: {e}")
            self.backup_records = []
    
    def _save_backup_records(self):
        """バックアップ記録を保存"""
        try:
            records_file = Path(self.config.backup_root) / "metadata" / "backup_records.json"
            
            # シリアライズ可能な形式に変換
            records_data = []
            for record in self.backup_records:
                record_dict = {
                    'backup_id': record.backup_id,
                    'backup_type': record.backup_type.value,
                    'status': record.status.value,
                    'start_time': record.start_time.isoformat(),
                    'end_time': record.end_time.isoformat() if record.end_time else None,
                    'file_path': record.file_path,
                    'file_size_bytes': record.file_size_bytes,
                    'checksum': record.checksum,
                    'error_message': record.error_message,
                    'files_count': record.files_count,
                    'compressed_size_bytes': record.compressed_size_bytes
                }
                records_data.append(record_dict)
            
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(records_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"バックアップ記録保存エラー: {e}")
    
    def start_auto_backup(self):
        """自動バックアップを開始"""
        if not self.config.auto_backup_enabled:
            logger.info("自動バックアップは無効です")
            return
        
        if self.is_running:
            logger.warning("自動バックアップは既に実行中です")
            return
        
        self.is_running = True
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.backup_thread.start()
        logger.info("自動バックアップを開始しました")
    
    def stop_auto_backup(self):
        """自動バックアップを停止"""
        self.is_running = False
        if self.backup_thread:
            self.backup_thread.join(timeout=10)
        logger.info("自動バックアップを停止しました")
    
    def _backup_loop(self):
        """バックアップループ"""
        last_full_backup = self._get_last_backup_time(BackupType.FULL)
        last_incremental_backup = self._get_last_backup_time(BackupType.INCREMENTAL)
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # フルバックアップのチェック
                if (not last_full_backup or 
                    current_time - last_full_backup >= timedelta(hours=self.config.full_backup_interval_hours)):
                    
                    logger.info("フルバックアップを実行します")
                    self.create_backup(BackupType.FULL)
                    last_full_backup = current_time
                
                # インクリメンタルバックアップのチェック
                elif (not last_incremental_backup or 
                      current_time - last_incremental_backup >= timedelta(hours=self.config.incremental_backup_interval_hours)):
                    
                    logger.info("インクリメンタルバックアップを実行します")
                    self.create_backup(BackupType.INCREMENTAL)
                    last_incremental_backup = current_time
                
                # 古いバックアップのクリーンアップ
                self._cleanup_old_backups()
                
                # 1時間待機
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"バックアップループエラー: {e}")
                time.sleep(300)  # エラー時は5分待機
    
    def create_backup(self, backup_type: BackupType) -> Optional[BackupRecord]:
        """バックアップを作成"""
        backup_id = f"{backup_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = BackupRecord(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            file_path="",
            file_size_bytes=0,
            checksum="",
            files_count=0,
            compressed_size_bytes=0
        )
        
        try:
            logger.info(f"バックアップ開始: {backup_id}")
            record.status = BackupStatus.RUNNING
            
            # バックアップファイルパスを決定
            backup_dir = Path(self.config.backup_root) / backup_type.value
            backup_file = backup_dir / f"{backup_id}.tar.gz"
            record.file_path = str(backup_file)
            
            # バックアップを実行
            if backup_type == BackupType.FULL:
                files_count = self._create_full_backup(backup_file)
            else:
                files_count = self._create_incremental_backup(backup_file)
            
            record.files_count = files_count
            
            # ファイルサイズとチェックサムを計算
            if backup_file.exists():
                record.file_size_bytes = backup_file.stat().st_size
                record.compressed_size_bytes = record.file_size_bytes
                record.checksum = self._calculate_checksum(backup_file)
                
                # 検証
                if self.config.verify_backups:
                    if self._verify_backup(backup_file):
                        record.status = BackupStatus.COMPLETED
                        logger.info(f"バックアップ完了: {backup_id}")
                    else:
                        record.status = BackupStatus.FAILED
                        record.error_message = "バックアップ検証に失敗しました"
                        logger.error(f"バックアップ検証失敗: {backup_id}")
                else:
                    record.status = BackupStatus.COMPLETED
                    logger.info(f"バックアップ完了: {backup_id}")
            else:
                record.status = BackupStatus.FAILED
                record.error_message = "バックアップファイルが作成されませんでした"
                logger.error(f"バックアップファイル作成失敗: {backup_id}")
            
        except Exception as e:
            record.status = BackupStatus.FAILED
            record.error_message = str(e)
            logger.error(f"バックアップエラー ({backup_id}): {e}")
        
        finally:
            record.end_time = datetime.now()
            self.backup_records.append(record)
            self._save_backup_records()
        
        return record
    
    def _create_full_backup(self, backup_file: Path) -> int:
        """フルバックアップを作成"""
        import tarfile
        
        files_count = 0
        
        with tarfile.open(backup_file, 'w:gz', compresslevel=self.config.compression_level) as tar:
            # モデルディレクトリ
            if os.path.exists(self.config.model_directory):
                tar.add(self.config.model_directory, arcname="models")
                files_count += self._count_files(self.config.model_directory)
            
            # データディレクトリ
            if os.path.exists(self.config.data_directory):
                tar.add(self.config.data_directory, arcname="data")
                files_count += self._count_files(self.config.data_directory)
            
            # 設定ディレクトリ
            if os.path.exists(self.config.config_directory):
                tar.add(self.config.config_directory, arcname="config")
                files_count += self._count_files(self.config.config_directory)
        
        return files_count
    
    def _create_incremental_backup(self, backup_file: Path) -> int:
        """インクリメンタルバックアップを作成"""
        # 最後のフルバックアップ時刻を取得
        last_full_backup_time = self._get_last_backup_time(BackupType.FULL)
        if not last_full_backup_time:
            # フルバックアップがない場合はフルバックアップを作成
            return self._create_full_backup(backup_file)
        
        import tarfile
        
        files_count = 0
        
        with tarfile.open(backup_file, 'w:gz', compresslevel=self.config.compression_level) as tar:
            # 変更されたファイルのみを追加
            for directory in [self.config.model_directory, self.config.data_directory, self.config.config_directory]:
                if os.path.exists(directory):
                    files_count += self._add_modified_files(tar, directory, last_full_backup_time)
        
        return files_count
    
    def _add_modified_files(self, tar, directory: str, since_time: datetime) -> int:
        """変更されたファイルをtarに追加"""
        files_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mtime > since_time:
                        arcname = os.path.relpath(file_path, os.path.dirname(directory))
                        tar.add(file_path, arcname=arcname)
                        files_count += 1
                except Exception as e:
                    logger.warning(f"ファイル追加エラー ({file_path}): {e}")
        
        return files_count
    
    def _count_files(self, directory: str) -> int:
        """ディレクトリ内のファイル数をカウント"""
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """ファイルのチェックサムを計算"""
        hash_algo = hashlib.new(self.config.checksum_algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        
        return hash_algo.hexdigest()
    
    def _verify_backup(self, backup_file: Path) -> bool:
        """バックアップファイルを検証"""
        try:
            import tarfile
            
            # tarファイルの整合性をチェック
            with tarfile.open(backup_file, 'r:gz') as tar:
                # 全メンバーをリストして整合性を確認
                members = tar.getmembers()
                return len(members) > 0
                
        except Exception as e:
            logger.error(f"バックアップ検証エラー: {e}")
            return False
    
    def _get_last_backup_time(self, backup_type: BackupType) -> Optional[datetime]:
        """最後のバックアップ時刻を取得"""
        completed_backups = [
            record for record in self.backup_records
            if record.backup_type == backup_type and record.status == BackupStatus.COMPLETED
        ]
        
        if completed_backups:
            return max(record.start_time for record in completed_backups)
        return None
    
    def _cleanup_old_backups(self):
        """古いバックアップをクリーンアップ"""
        try:
            # フルバックアップのクリーンアップ
            self._cleanup_backups_by_type(BackupType.FULL, self.config.max_full_backups)
            
            # インクリメンタルバックアップのクリーンアップ
            self._cleanup_backups_by_type(BackupType.INCREMENTAL, self.config.max_incremental_backups)
            
        except Exception as e:
            logger.error(f"バックアップクリーンアップエラー: {e}")
    
    def _cleanup_backups_by_type(self, backup_type: BackupType, max_count: int):
        """指定タイプのバックアップをクリーンアップ"""
        completed_backups = [
            record for record in self.backup_records
            if record.backup_type == backup_type and record.status == BackupStatus.COMPLETED
        ]
        
        # 古い順にソート
        completed_backups.sort(key=lambda x: x.start_time)
        
        # 制限を超えた分を削除
        if len(completed_backups) > max_count:
            to_delete = completed_backups[:-max_count]
            
            for record in to_delete:
                try:
                    # ファイルを削除
                    if os.path.exists(record.file_path):
                        os.remove(record.file_path)
                    
                    # 記録から削除
                    self.backup_records.remove(record)
                    logger.info(f"古いバックアップを削除: {record.backup_id}")
                    
                except Exception as e:
                    logger.error(f"バックアップ削除エラー ({record.backup_id}): {e}")
    
    def restore_backup(self, backup_id: str, restore_path: Optional[str] = None) -> bool:
        """バックアップから復元"""
        # バックアップ記録を検索
        backup_record = next(
            (record for record in self.backup_records if record.backup_id == backup_id),
            None
        )
        
        if not backup_record:
            logger.error(f"バックアップが見つかりません: {backup_id}")
            return False
        
        if backup_record.status != BackupStatus.COMPLETED:
            logger.error(f"バックアップが完了していません: {backup_id}")
            return False
        
        try:
            import tarfile
            
            restore_root = restore_path or "restored"
            os.makedirs(restore_root, exist_ok=True)
            
            with tarfile.open(backup_record.file_path, 'r:gz') as tar:
                tar.extractall(path=restore_root)
            
            logger.info(f"バックアップから復元完了: {backup_id} -> {restore_root}")
            return True
            
        except Exception as e:
            logger.error(f"復元エラー ({backup_id}): {e}")
            return False
    
    def get_backup_status(self) -> Dict[str, Any]:
        """バックアップ状態を取得"""
        recent_backups = [
            record for record in self.backup_records
            if record.start_time >= datetime.now() - timedelta(days=7)
        ]
        
        return {
            "auto_backup_enabled": self.config.auto_backup_enabled,
            "is_running": self.is_running,
            "total_backups": len(self.backup_records),
            "recent_backups": len(recent_backups),
            "last_full_backup": self._get_last_backup_time(BackupType.FULL).isoformat() if self._get_last_backup_time(BackupType.FULL) else None,
            "last_incremental_backup": self._get_last_backup_time(BackupType.INCREMENTAL).isoformat() if self._get_last_backup_time(BackupType.INCREMENTAL) else None,
            "backup_size_total_mb": sum(record.file_size_bytes for record in self.backup_records) / (1024 * 1024),
            "failed_backups": len([r for r in recent_backups if r.status == BackupStatus.FAILED])
        }
