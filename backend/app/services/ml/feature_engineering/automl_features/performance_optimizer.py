"""
性能最適化システム

並列処理、キャッシュ、メモリ最適化を実装します。
"""

import gc
import hashlib
import logging
import pickle
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage

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
            # データのハッシュを計算（pandas.utilの代わりに独自実装）
            data_str = f"{data.shape}_{data.dtypes.to_dict()}_{data.head(10).to_dict()}"
            data_hash = hashlib.md5(data_str.encode()).hexdigest()

            # 設定のハッシュを計算
            settings_str = str(sorted(settings.items()))
            settings_hash = hashlib.md5(settings_str.encode()).hexdigest()

            return f"{data_hash}_{settings_hash}"
        except Exception as e:
            logger.error(f"キャッシュキー生成エラー: {e}")
            return f"fallback_{int(time.time())}"

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

                    # より正確な型チェック
                    if pd.api.types.is_integer_dtype(col_type):
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

                    elif pd.api.types.is_float_dtype(col_type):
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
        cpu_count = psutil.cpu_count()
        if cpu_count is None:
            cpu_count = 1

        if data_size > 10000:
            suggestions["parallel_jobs"] = min(4, cpu_count)
            suggestions["batch_processing"] = True
            suggestions["memory_optimization"] = True
        elif data_size > 5000:
            suggestions["parallel_jobs"] = min(2, cpu_count)
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
            # NumPyのキャッシュは自動管理されるため、
            # 明示的なクリアは必要ないが、メモリ使用量をログ出力
            logger.debug("NumPyキャッシュクリアは自動管理のためスキップ")
        except Exception as e:
            logger.debug(f"NumPyキャッシュクリアエラー: {e}")

    def _clear_sklearn_cache(self):
        """Scikit-learnキャッシュのクリア"""
        try:
            # Scikit-learnはメモリキャッシュを自動管理するため、
            # 明示的なクリアは必要ないが、メモリ使用量をログ出力
            logger.debug("Scikit-learnキャッシュクリアは自動管理のためスキップ")
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

        try:
            # tracemalloc開始
            tracemalloc.start()

            # メモリ使用量の詳細測定
            start_memory = self._get_memory_usage()
            start_time = time.time()

            # memory-profilerを使用した詳細分析
            mem_usage = memory_usage(func, interval=0.1, timeout=None)  # type: ignore

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
        func_result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = self._get_memory_usage()

        return {
            "execution_time": end_time - start_time,
            "memory_start_mb": start_memory,
            "memory_end_mb": end_memory,
            "memory_diff_mb": end_memory - start_memory,
            "result": func_result,
        }

    def _calculate_memory_efficiency(self, mem_usage: list) -> Dict[str, Any]:
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
            if hasattr(self, "cache_index"):
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
