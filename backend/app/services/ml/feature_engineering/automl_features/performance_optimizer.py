"""
性能最適化システム

並列処理、キャッシュ、メモリ最適化を実装します。
"""

import logging
import hashlib
import pickle
import time
from typing import Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import gc
import functools
import tracemalloc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# memory-profilerのインポート（オプション）
try:
    from memory_profiler import memory_usage

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning(
        "memory-profilerが利用できません。詳細なメモリ分析機能は無効化されます。"
    )

# line-profilerのインポート（オプション）
try:
    import importlib.util

    LINE_PROFILER_AVAILABLE = importlib.util.find_spec("line_profiler") is not None
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    logger.warning(
        "line-profilerが利用できません。ライン単位の分析機能は無効化されます。"
    )


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

    def cleanup_autofeat_memory(self, autofeat_model=None):
        """AutoFeat特有のメモリクリーンアップ"""
        try:
            memory_before = self._get_memory_usage()

            # AutoFeatモデルの詳細クリーンアップ
            if autofeat_model is not None:
                self._cleanup_autofeat_model(autofeat_model)

            # NumPyキャッシュのクリア
            self._clear_numpy_cache()

            # Scikit-learnキャッシュのクリア
            self._clear_sklearn_cache()

            # 強制ガベージコレクション
            self.force_garbage_collection()

            memory_after = self._get_memory_usage()
            memory_freed = memory_before - memory_after

            if memory_freed > 1.0:  # 1MB以上解放された場合のみログ出力
                logger.info(f"AutoFeatメモリクリーンアップ: {memory_freed:.2f}MB解放")

        except Exception as e:
            logger.error(f"AutoFeatメモリクリーンアップエラー: {e}")

    def _cleanup_autofeat_model(self, autofeat_model):
        """AutoFeatモデルの詳細クリーンアップ"""
        try:
            # AutoFeatモデル内部の属性をクリア
            if hasattr(autofeat_model, "feateng_cols_"):
                autofeat_model.feateng_cols_ = None
            if hasattr(autofeat_model, "featsel_"):
                autofeat_model.featsel_ = None
            if hasattr(autofeat_model, "model_"):
                autofeat_model.model_ = None
            if hasattr(autofeat_model, "scaler_"):
                autofeat_model.scaler_ = None
            if hasattr(autofeat_model, "feature_importances_"):
                autofeat_model.feature_importances_ = None

        except Exception as e:
            logger.warning(f"AutoFeatモデルクリーンアップエラー: {e}")

    def _clear_numpy_cache(self):
        """NumPyキャッシュのクリア"""
        try:
            import numpy as np

            # NumPyの内部キャッシュをクリア（可能な場合）
            if hasattr(np, "_NoValue"):
                # NumPyの内部状態をリセット
                pass
        except Exception as e:
            logger.debug(f"NumPyキャッシュクリアエラー: {e}")

    def _clear_sklearn_cache(self):
        """Scikit-learnキャッシュのクリア"""
        try:
            # Scikit-learnのキャッシュをクリア

            # 可能であればScikit-learnの内部キャッシュをクリア
            pass
        except Exception as e:
            logger.debug(f"Scikit-learnキャッシュクリアエラー: {e}")

    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB単位）"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception as e:
            logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0

    def monitor_memory_usage(self, operation_name: str = "操作"):
        """メモリ使用量を監視するコンテキストマネージャー"""

        @contextmanager
        def memory_monitor():
            start_memory = self._get_memory_usage()
            logger.debug(f"{operation_name}開始時メモリ: {start_memory:.2f}MB")

            try:
                yield
            finally:
                end_memory = self._get_memory_usage()
                memory_diff = end_memory - start_memory
                if abs(memory_diff) > 5.0:  # 5MB以上の変化があった場合のみログ出力
                    logger.info(
                        f"{operation_name}完了時メモリ変化: {memory_diff:+.2f}MB"
                    )

        return memory_monitor()

    def get_memory_recommendations(
        self, data_size_mb: float, feature_count: int
    ) -> Dict[str, Any]:
        """メモリ使用量に基づく推奨設定を取得"""
        recommendations = {
            "use_batch_processing": False,
            "batch_size": 5000,
            "max_memory_gb": 4,
            "enable_memory_monitoring": False,
            "cleanup_frequency": "after_each_operation",
        }

        # データサイズに基づく推奨設定
        if data_size_mb > 500:  # 500MB以上
            recommendations.update(
                {
                    "use_batch_processing": True,
                    "batch_size": 2000,
                    "max_memory_gb": 2,
                    "enable_memory_monitoring": True,
                    "cleanup_frequency": "after_each_batch",
                }
            )
        elif data_size_mb > 100:  # 100MB以上
            recommendations.update(
                {
                    "use_batch_processing": True,
                    "batch_size": 3000,
                    "max_memory_gb": 3,
                    "enable_memory_monitoring": True,
                }
            )

        # 特徴量数に基づく調整
        if feature_count > 1000:
            recommendations["max_memory_gb"] = min(recommendations["max_memory_gb"], 2)
            recommendations["enable_memory_monitoring"] = True

        return recommendations

    def detailed_memory_profile(
        self, func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        詳細なメモリプロファイリングを実行

        Args:
            func: プロファイリング対象の関数
            *args: 関数の引数
            **kwargs: 関数のキーワード引数

        Returns:
            メモリプロファイリング結果
        """
        if not MEMORY_PROFILER_AVAILABLE:
            logger.warning(
                "memory-profilerが利用できないため、基本的なメモリ監視のみ実行します"
            )
            return self._basic_memory_profile(func, *args, **kwargs)

        try:
            # tracemalloc開始
            tracemalloc.start()

            # メモリ使用量の詳細測定
            start_memory = self._get_memory_usage()
            start_time = time.time()

            # memory-profilerを使用した詳細分析
            mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=None)

            # 実行時間とメモリ使用量の計算
            end_time = time.time()
            end_memory = self._get_memory_usage()

            # tracemalloc統計取得
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 結果の整理
            result = {
                "execution_time": end_time - start_time,
                "memory_start_mb": start_memory,
                "memory_end_mb": end_memory,
                "memory_diff_mb": end_memory - start_memory,
                "memory_usage_timeline": mem_usage,
                "peak_memory_mb": max(mem_usage) if mem_usage else end_memory,
                "tracemalloc_current_mb": current / 1024 / 1024,
                "tracemalloc_peak_mb": peak / 1024 / 1024,
                "memory_efficiency": self._calculate_memory_efficiency(mem_usage),
            }

            logger.info(
                f"詳細メモリプロファイリング完了: ピーク使用量 {result['peak_memory_mb']:.2f}MB"
            )
            return result

        except Exception as e:
            logger.error(f"詳細メモリプロファイリングエラー: {e}")
            return self._basic_memory_profile(func, *args, **kwargs)

    def _basic_memory_profile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """基本的なメモリプロファイリング（memory-profiler不使用）"""
        start_memory = self._get_memory_usage()
        start_time = time.time()

        # 関数実行
        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = self._get_memory_usage()

        return {
            "execution_time": end_time - start_time,
            "memory_start_mb": start_memory,
            "memory_end_mb": end_memory,
            "memory_diff_mb": end_memory - start_memory,
            "result": result,
        }

    def _calculate_memory_efficiency(self, mem_usage: list) -> Dict[str, float]:
        """メモリ効率性を計算"""
        if not mem_usage or len(mem_usage) < 2:
            return {"efficiency_score": 0.0, "stability_score": 0.0}

        # メモリ使用量の変動を分析
        mem_array = np.array(mem_usage)
        mean_usage = np.mean(mem_array)
        std_usage = np.std(mem_array)
        max_usage = np.max(mem_array)
        min_usage = np.min(mem_array)

        # 効率性スコア（低いほど良い）
        efficiency_score = (
            (max_usage - min_usage) / mean_usage if mean_usage > 0 else 1.0
        )

        # 安定性スコア（低いほど良い）
        stability_score = std_usage / mean_usage if mean_usage > 0 else 1.0

        return {
            "efficiency_score": efficiency_score,
            "stability_score": stability_score,
            "mean_usage_mb": mean_usage,
            "std_usage_mb": std_usage,
            "max_usage_mb": max_usage,
            "min_usage_mb": min_usage,
        }

    def memory_profiling_decorator(self, enable_detailed: bool = True):
        """
        メモリプロファイリングデコレータ

        Args:
            enable_detailed: 詳細プロファイリングを有効にするか
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if enable_detailed and MEMORY_PROFILER_AVAILABLE:
                    profile_result = self.detailed_memory_profile(func, *args, **kwargs)
                    logger.info(f"{func.__name__} メモリプロファイル: {profile_result}")
                    return profile_result.get("result")
                else:
                    with self.monitor_memory_usage(func.__name__):
                        return func(*args, **kwargs)

            return wrapper

        return decorator

    def optimize_pandas_memory_usage(
        self, df: pd.DataFrame, aggressive: bool = False
    ) -> pd.DataFrame:
        """
        pandasのメモリ使用量を最適化（公式推奨手法）

        Args:
            df: 最適化するDataFrame
            aggressive: より積極的な最適化を行うか

        Returns:
            最適化されたDataFrame
        """
        try:
            original_memory = df.memory_usage(deep=True).sum()
            optimized_df = df.copy()

            for col in optimized_df.columns:
                col_type = optimized_df[col].dtype

                # 数値型の最適化
                if col_type in ["int64", "int32"]:
                    # より小さい整数型に変換可能かチェック
                    c_min = optimized_df[col].min()
                    c_max = optimized_df[col].max()

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
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

                elif col_type in ["float64", "float32"]:
                    # float32で十分な精度かチェック
                    if aggressive or col_type == "float64":
                        optimized_df[col] = pd.to_numeric(
                            optimized_df[col], downcast="float"
                        )

                # オブジェクト型の最適化
                elif col_type == "object":
                    # カテゴリカルデータに変換可能かチェック
                    num_unique_values = len(optimized_df[col].unique())
                    num_total_values = len(optimized_df[col])

                    if (
                        num_unique_values / num_total_values < 0.5
                    ):  # ユニーク値が50%未満
                        optimized_df[col] = optimized_df[col].astype("category")

            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            reduction = (original_memory - optimized_memory) / original_memory * 100

            logger.info(
                f"pandasメモリ最適化: {reduction:.1f}%削減 "
                f"({original_memory/1024/1024:.1f}MB → {optimized_memory/1024/1024:.1f}MB)"
            )

            return optimized_df

        except Exception as e:
            logger.error(f"pandasメモリ最適化エラー: {e}")
            return df

    def cleanup(self):
        """
        PerformanceOptimizerのリソースクリーンアップ
        EnhancedFeatureEngineeringServiceから呼び出される統一インターフェース
        """
        try:
            logger.debug("PerformanceOptimizerのクリーンアップを開始")

            # キャッシュインデックスをクリア
            if hasattr(self, 'cache_index'):
                self.cache_index.clear()

            # NumPyキャッシュのクリア
            self._clear_numpy_cache()

            # Scikit-learnキャッシュのクリア
            self._clear_sklearn_cache()

            # 強制ガベージコレクション
            self.force_garbage_collection()

            logger.debug("PerformanceOptimizerのクリーンアップ完了")

        except Exception as e:
            logger.error(f"PerformanceOptimizerクリーンアップエラー: {e}")
