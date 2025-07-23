"""
性能最適化システム

並列処理、キャッシュ、メモリ最適化を実装します。
"""

import logging
import hashlib
import pickle
import time
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import gc

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能最適化クラス"""

    def __init__(self, cache_dir: str = "cache/tsfresh", max_cache_size_mb: int = 1000):
        """
        初期化

        Args:
            cache_dir: キャッシュディレクトリ
            max_cache_size_mb: 最大キャッシュサイズ（MB）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_index = {}
        self.performance_stats = {}

        # キャッシュインデックスを読み込み
        self._load_cache_index()

    def _load_cache_index(self):
        """キャッシュインデックスを読み込み"""
        try:
            index_file = self.cache_dir / "cache_index.pkl"
            if index_file.exists():
                with open(index_file, "rb") as f:
                    self.cache_index = pickle.load(f)
                logger.debug(
                    f"キャッシュインデックス読み込み完了: {len(self.cache_index)}件"
                )
        except Exception as e:
            logger.warning(f"キャッシュインデックス読み込みエラー: {e}")
            self.cache_index = {}

    def _save_cache_index(self):
        """キャッシュインデックスを保存"""
        try:
            index_file = self.cache_dir / "cache_index.pkl"
            with open(index_file, "wb") as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"キャッシュインデックス保存エラー: {e}")

    def _generate_cache_key(self, data: pd.DataFrame, settings: Dict[str, Any]) -> str:
        """キャッシュキーを生成"""
        try:
            # データのハッシュを計算
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(data, index=True).values
            ).hexdigest()

            # 設定のハッシュを計算
            settings_str = str(sorted(settings.items()))
            settings_hash = hashlib.md5(settings_str.encode()).hexdigest()

            return f"{data_hash}_{settings_hash}"
        except Exception as e:
            logger.error(f"キャッシュキー生成エラー: {e}")
            return f"fallback_{int(time.time())}"

    def get_cached_features(
        self, data: pd.DataFrame, settings: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """キャッシュから特徴量を取得（無効化）"""
        # キャッシュ機能を無効化して毎回新鮮な特徴量を生成
        logger.debug("キャッシュ機能が無効化されています")
        return None

    def cache_features(
        self, data: pd.DataFrame, settings: Dict[str, Any], features: pd.DataFrame
    ):
        """特徴量をキャッシュに保存（無効化）"""
        # キャッシュ保存を無効化して毎回新鮮な特徴量を生成
        logger.debug("キャッシュ保存機能が無効化されています")

    def _cleanup_cache_if_needed(self):
        """必要に応じてキャッシュをクリーンアップ"""
        try:
            # 現在のキャッシュサイズを計算
            total_size = sum(info["file_size"] for info in self.cache_index.values())

            max_size_bytes = self.max_cache_size_mb * 1024 * 1024

            if total_size > max_size_bytes:
                logger.info(
                    f"キャッシュサイズ制限超過: {total_size / 1024 / 1024:.1f}MB"
                )

                # 最後にアクセスされた時間でソート
                sorted_items = sorted(
                    self.cache_index.items(), key=lambda x: x[1]["last_accessed"]
                )

                # 古いキャッシュから削除
                removed_size = 0
                for cache_key, info in sorted_items:
                    if total_size - removed_size <= max_size_bytes * 0.8:  # 80%まで削減
                        break

                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    if cache_file.exists():
                        cache_file.unlink()

                    removed_size += info["file_size"]
                    del self.cache_index[cache_key]

                logger.info(
                    f"キャッシュクリーンアップ完了: {removed_size / 1024 / 1024:.1f}MB削除"
                )
                self._save_cache_index()

        except Exception as e:
            logger.error(f"キャッシュクリーンアップエラー: {e}")

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameのメモリ使用量を最適化"""
        try:
            original_memory = df.memory_usage(deep=True).sum()
            optimized_df = df.copy()

            for col in optimized_df.columns:
                col_type = optimized_df[col].dtype

                if col_type != "object":
                    c_min = optimized_df[col].min()
                    c_max = optimized_df[col].max()

                    if str(col_type)[:3] == "int":
                        if (
                            c_min > np.iinfo(np.int8).min
                            and c_max < np.iinfo(np.int8).max
                        ):
                            optimized_df[col] = optimized_df[col].astype(np.int8)
                        elif (
                            c_min > np.iinfo(np.int16).min
                            and c_max < np.iinfo(np.int16).max
                        ):
                            optimized_df[col] = optimized_df[col].astype(np.int16)
                        elif (
                            c_min > np.iinfo(np.int32).min
                            and c_max < np.iinfo(np.int32).max
                        ):
                            optimized_df[col] = optimized_df[col].astype(np.int32)

                    elif str(col_type)[:5] == "float":
                        if (
                            c_min > np.finfo(np.float32).min
                            and c_max < np.finfo(np.float32).max
                        ):
                            optimized_df[col] = optimized_df[col].astype(np.float32)

            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            reduction = (original_memory - optimized_memory) / original_memory * 100

            if reduction > 5:  # 5%以上の削減があった場合のみログ出力
                logger.info(f"メモリ最適化: {reduction:.1f}%削減")

            return optimized_df

        except Exception as e:
            logger.error(f"メモリ最適化エラー: {e}")
            return df

    def monitor_system_resources(self) -> Dict[str, Any]:
        """システムリソースを監視"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # プロセスのメモリ使用量
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)

            resource_info = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory_available_gb,
                "process_memory_mb": process_memory_mb,
                "timestamp": time.time(),
            }

            # 警告レベルのチェック
            warnings = []
            if cpu_percent > 80:
                warnings.append("CPU使用率が高い")
            if memory_percent > 85:
                warnings.append("メモリ使用率が高い")
            if process_memory_mb > 2000:  # 2GB
                warnings.append("プロセスメモリ使用量が多い")

            resource_info["warnings"] = warnings

            if warnings:
                logger.warning(f"リソース警告: {', '.join(warnings)}")

            return resource_info

        except Exception as e:
            logger.error(f"リソース監視エラー: {e}")
            return {}

    def suggest_optimization(
        self, data_size: int, feature_count: int
    ) -> Dict[str, Any]:
        """最適化提案を生成"""
        suggestions = {
            "parallel_jobs": 1,
            "batch_processing": False,
            "memory_optimization": False,
            "cache_enabled": True,
            "performance_mode": "balanced",
        }

        # データサイズに基づく提案
        if data_size > 10000:
            suggestions["parallel_jobs"] = min(4, psutil.cpu_count())
            suggestions["batch_processing"] = True
            suggestions["memory_optimization"] = True
        elif data_size > 5000:
            suggestions["parallel_jobs"] = 2
            suggestions["memory_optimization"] = True

        # 特徴量数に基づく提案
        if feature_count > 200:
            suggestions["performance_mode"] = "fast"
        elif feature_count > 500:
            suggestions["performance_mode"] = "lightweight"

        # システムリソースに基づく提案
        resource_info = self.monitor_system_resources()
        if resource_info.get("memory_percent", 0) > 70:
            suggestions["memory_optimization"] = True
            suggestions["batch_processing"] = True

        return suggestions

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        try:
            total_files = len(self.cache_index)
            total_size = sum(info["file_size"] for info in self.cache_index.values())
            total_hits = sum(info["hit_count"] for info in self.cache_index.values())

            return {
                "total_files": total_files,
                "total_size_mb": total_size / (1024**2),
                "total_hits": total_hits,
                "hit_rate": total_hits / max(total_files, 1),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"キャッシュ統計取得エラー: {e}")
            return {}

    def clear_cache(self):
        """キャッシュをクリア"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

            self.cache_index.clear()
            self._save_cache_index()

            logger.info("キャッシュクリア完了")

        except Exception as e:
            logger.error(f"キャッシュクリアエラー: {e}")

    def force_garbage_collection(self):
        """強制ガベージコレクション"""
        try:
            collected = gc.collect()
            logger.debug(f"ガベージコレクション実行: {collected}オブジェクト回収")
        except Exception as e:
            logger.error(f"ガベージコレクションエラー: {e}")
