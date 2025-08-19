"""
AutoFeat特徴量選択クラス

遺伝的アルゴリズムによる最適特徴量組み合わせの自動発見を実装します。
"""

import gc
import logging
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autofeat import AutoFeatClassifier, AutoFeatRegressor

from .....utils.data_processing import data_processor as data_preprocessor
from .....utils.error_handler import safe_ml_operation
from .automl_config import AutoFeatConfig
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class AutoFeatCalculator:
    """
    AutoFeat特徴量選択クラス

    遺伝的アルゴリズムによる最適特徴量組み合わせを発見します。
    """

    def __init__(self, config: Optional[AutoFeatConfig] = None):
        """
        初期化

        Args:
            config: AutoFeat設定
        """
        self.config = config or AutoFeatConfig()
        self.autofeat_model = None
        self.selected_features = None
        self.feature_scores = {}
        self.last_selection_info = {}
        self._memory_usage_before = 0
        self._memory_usage_after = 0

        # パフォーマンス最適化ツールを初期化
        self.performance_optimizer = PerformanceOptimizer()

        # メモリプロファイリングを有効にするかどうか
        self.enable_memory_profiling = True

    def __enter__(self):
        """コンテキストマネージャーの開始"""
        self._memory_usage_before = self._get_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了時にリソースをクリーンアップ"""
        _ = exc_type, exc_val, exc_tb  # 未使用パラメータ
        try:
            self.clear_model()
            self._force_garbage_collection()
            self._memory_usage_after = self._get_memory_usage()
            memory_freed = self._memory_usage_before - self._memory_usage_after
            if memory_freed > 0:
                logger.info(f"AutoFeatCalculator終了時メモリ解放: {memory_freed:.2f}MB")
        except Exception as e:
            logger.error(f"AutoFeatCalculatorクリーンアップエラー: {e}")

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

    def _force_garbage_collection(self):
        """強制ガベージコレクション"""
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"ガベージコレクションエラー: {e}")

    @contextmanager
    def _memory_managed_operation(self, operation_name: str):
        """メモリ管理付きの操作を実行するコンテキストマネージャー"""
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            # 操作完了後にガベージコレクションを実行
            self._force_garbage_collection()
            end_memory = self._get_memory_usage()
            memory_diff = end_memory - start_memory
            if abs(memory_diff) > 1.0:  # 1MB以上の変化があった場合のみログ出力
                logger.info(f"{operation_name}完了時メモリ変化: {memory_diff:+.2f}MB")

    def _should_use_batch_processing(self, df: pd.DataFrame) -> bool:
        """バッチ処理を使用すべきかどうかを判定"""
        # データサイズが大きい場合はバッチ処理を使用
        data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        row_count = len(df)

        # 100MB以上または10,000行以上の場合はバッチ処理
        return data_size_mb > 100 or row_count > 10000

    def _process_in_batches(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        optimized_config: AutoFeatConfig,
        batch_size: int = 5000,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """大量データをバッチ処理で特徴量生成"""
        logger.info(
            f"バッチ処理モードで特徴量生成開始: {len(df)}行を{batch_size}行ずつ処理"
        )

        all_results = []
        batch_info = []

        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end].copy()
            batch_target = target.iloc[i:batch_end].copy()

            logger.debug(f"バッチ {i//batch_size + 1}: {len(batch_df)}行処理中")

            with self._memory_managed_operation(f"バッチ{i//batch_size + 1}処理"):
                # 小さなバッチで特徴量生成（最適化設定を使用）
                batch_result, batch_meta = self._generate_features_single_batch(
                    batch_df, batch_target, optimized_config
                )

                if "error" not in batch_meta:
                    all_results.append(batch_result)
                    batch_info.append(batch_meta)

        if not all_results:
            return df, {"error": "All batches failed"}

        # バッチ結果を結合
        final_result = pd.concat(all_results, ignore_index=True)

        # メタデータを統合
        combined_info = {
            "batch_count": len(batch_info),
            "total_rows": len(final_result),
            "batch_processing": True,
            "generation_time": sum(
                info.get("generation_time", 0) for info in batch_info
            ),
        }

        return final_result, combined_info

    @safe_ml_operation(
        default_return=None, context="AutoFeat特徴量生成でエラーが発生しました"
    )
    def generate_features(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        task_type: str = "regression",
        max_features: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        AutoFeatによる自動特徴量生成

        Args:
            df: 入力DataFrame
            target: ターゲット変数
            task_type: タスクタイプ（regression/classification）
            max_features: 最大特徴量数

        Returns:
            生成された特徴量とメタデータ
        """

        if df is None or df.empty or target is None:
            logger.warning("空のデータまたはターゲットが提供されました")
            return df, {"error": "Empty data"}

        try:
            with self._memory_managed_operation("AutoFeat特徴量生成"):
                start_time = time.time()

                # データサイズを計算してメモリ最適化設定を取得
                data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                optimized_config = self.config.get_memory_optimized_config(data_size_mb)

                logger.info(f"データサイズ: {data_size_mb:.2f}MB, 最適化設定適用")

                # 設定の決定
                max_feat = (
                    max_features
                    if max_features is not None
                    else optimized_config.max_features
                )

                logger.info(
                    f"AutoFeat特徴量生成開始: 最大特徴量={max_feat}, メモリ制限={optimized_config.max_gb}GB"
                )

                # データの前処理とメモリ最適化
                processed_df, processed_target = self._preprocess_data(df, target)

                # pandasメモリ最適化を適用
                processed_df = self.performance_optimizer.optimize_pandas_memory_usage(
                    processed_df, aggressive=data_size_mb > 100
                )

                if processed_df.empty:
                    logger.warning("前処理後のデータが空です")
                    return df, {"error": "Empty processed data"}

                # 大量データの場合はバッチ処理を使用
                if self._should_use_batch_processing(processed_df):
                    logger.info("大量データのためバッチ処理モードを使用します")
                    return self._process_in_batches(
                        processed_df, processed_target, optimized_config
                    )

                # NumPyのランダムシードを現在時刻で設定（AutoFeat内部で使用される）
                np.random.seed(int(time.time()) % 2147483647)

                # AutoFeatモデルを初期化（最適化された設定を使用）
                with self._memory_managed_operation("AutoFeatモデル初期化"):
                    # メモリ制限をさらに厳しく設定（データサイズに基づく）
                    if data_size_mb < 1:
                        actual_max_gb = 0.05  # 50MB
                    elif data_size_mb < 10:
                        actual_max_gb = 0.1  # 100MB
                    else:
                        actual_max_gb = min(
                            optimized_config.max_gb, 0.2
                        )  # 最大でも200MB

                    if task_type.lower() == "classification":
                        self.autofeat_model = AutoFeatClassifier(
                            feateng_steps=optimized_config.feateng_steps,
                            max_gb=actual_max_gb,
                            verbose=optimized_config.verbose,
                            featsel_runs=optimized_config.featsel_runs,
                            n_jobs=optimized_config.n_jobs,  # 並列処理数を制御
                            units=None,  # 単位を指定しない（メモリ節約）
                        )
                    else:
                        self.autofeat_model = AutoFeatRegressor(
                            feateng_steps=optimized_config.feateng_steps,
                            max_gb=actual_max_gb,
                            verbose=optimized_config.verbose,
                            featsel_runs=optimized_config.featsel_runs,
                            n_jobs=optimized_config.n_jobs,  # 並列処理数を制御
                            units=None,  # 単位を指定しない（メモリ節約）
                        )

                    logger.info(
                        f"AutoFeatモデル初期化完了: max_gb={actual_max_gb}, "
                        f"feateng_steps={optimized_config.feateng_steps}, "
                        f"featsel_runs={optimized_config.featsel_runs}"
                    )

                # 特徴量生成を実行
                with self._memory_managed_operation("AutoFeat学習・変換"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        logger.info("AutoFeat特徴量生成実行中...")
                        logger.info(f"入力データ形状: {processed_df.shape}")
                        logger.info(f"入力列: {list(processed_df.columns)}")
                        logger.info(
                            f"AutoFeat設定詳細: feateng_steps={optimized_config.feateng_steps}, "
                            f"max_gb={actual_max_gb}, max_features={optimized_config.max_features}"
                        )

                        # AutoFeatの推奨方法：fit_transformを使用（メモリ効率が良い）
                        transformed_df = self.autofeat_model.fit_transform(
                            processed_df, processed_target
                        )

                        # AutoFeatの内部状態をログ出力
                        if hasattr(self.autofeat_model, "feateng_cols_"):
                            feateng_count = (
                                len(self.autofeat_model.feateng_cols_)
                                if self.autofeat_model.feateng_cols_
                                else 0
                            )
                            logger.info(f"AutoFeat生成特徴量列: {feateng_count}個")
                        if hasattr(self.autofeat_model, "featsel_"):
                            featsel_count = (
                                len(self.autofeat_model.featsel_)
                                if self.autofeat_model.featsel_
                                else 0
                            )
                            logger.info(f"AutoFeat選択特徴量: {featsel_count}個")

                logger.info(f"変換後データ形状: {transformed_df.shape}")
                logger.info(f"変換後列: {list(transformed_df.columns)}")
                logger.info(f"元データ形状: {df.shape}")

                # AutoFeatプレフィックスを追加して元のDataFrameと結合
                result_df = df.copy()

                # 元データのNaN値を統計的手法で処理
                if result_df.isnull().any().any():
                    logger.info("元データのNaN値を統計的手法で補完します")
                    result_df = data_preprocessor.transform_missing_values(
                        result_df, strategy="median"
                    )

                # インデックスを統一（長さ不一致問題を防ぐ）
                if hasattr(transformed_df, "index"):
                    transformed_df = transformed_df.reset_index(drop=True)
                if hasattr(result_df, "index"):
                    result_df = result_df.reset_index(drop=True)

                # 新しい特徴量にプレフィックスを追加
                for col in transformed_df.columns:
                    if col not in processed_df.columns:  # 新しく生成された特徴量のみ
                        new_col_name = f"AF_{col}"

                        # 長さを合わせる処理
                        if len(transformed_df) == len(result_df):
                            result_df[new_col_name] = transformed_df[col].values
                        elif len(transformed_df) < len(result_df):
                            # 変換データが短い場合、最後の値で埋める
                            full_data = np.full(
                                len(result_df),
                                transformed_df[col].iloc[-1],
                                dtype=float,
                            )
                            full_data[: len(transformed_df)] = transformed_df[
                                col
                            ].values
                            result_df[new_col_name] = full_data
                        else:
                            # 変換データが長い場合、切り詰める
                            result_df[new_col_name] = (
                                transformed_df[col].iloc[: len(result_df)].values
                            )

            # 生成された特徴量の情報を取得
            new_features = [col for col in result_df.columns if col.startswith("AF_")]

            # 生成情報を保存
            generation_time = time.time() - start_time
            self.last_selection_info = {
                "original_features": len(df.columns),
                "generated_features": len(new_features),
                "total_features": len(result_df.columns),
                "generation_time": generation_time,
                "task_type": task_type,
                "feateng_steps": self.config.feateng_steps,
                "max_gb": self.config.max_gb,
            }

            self.selected_features = new_features

            logger.info(
                f"AutoFeat特徴量生成完了: {len(df.columns)}個 → {len(result_df.columns)}個 "
                f"(新規: {len(new_features)}個, {generation_time:.2f}秒)"
            )

            return result_df, self.last_selection_info

        except Exception as e:
            logger.error(f"AutoFeat特徴量選択エラー: {e}")
            return df, {"error": str(e)}

    def _preprocess_data(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """データの前処理"""
        try:
            # 数値列のみを選択
            numeric_df = df.select_dtypes(include=[np.number]).copy()

            # 無限値とNaNを処理
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

            # インデックスをリセット（タイムゾーン問題を回避）
            numeric_df = numeric_df.reset_index(drop=True)
            target_reset = target.reset_index(drop=True)

            # NaNを含む行を除去（より厳密なチェック）
            valid_mask = numeric_df.notna().all(axis=1) & target_reset.notna()

            # 追加の検証：無限値や異常値のチェック
            for col in numeric_df.columns:
                col_valid = (
                    np.isfinite(numeric_df[col])
                    & (numeric_df[col] != np.inf)
                    & (numeric_df[col] != -np.inf)
                )
                valid_mask = valid_mask & col_valid

            processed_df = numeric_df[valid_mask].copy()
            processed_target = target_reset[valid_mask].copy()

            # 最終的なNaNチェックと統計的補完
            if processed_df.isnull().any().any():
                logger.warning(
                    "前処理後にもNaN値が残っています。統計的手法で補完します。"
                )
                processed_df = data_preprocessor.transform_missing_values(
                    processed_df, strategy="median"
                )

            if processed_target.isnull().any():
                logger.warning(
                    "ターゲット変数にNaN値があります。統計的手法で補完します。"
                )
                target_df = pd.DataFrame({"target": processed_target})
                target_df = data_preprocessor.transform_missing_values(
                    target_df, strategy="median"
                )
                processed_target = target_df["target"]

            # 定数列を除去
            constant_columns = []
            for col in processed_df.columns:
                if processed_df[col].nunique() <= 1:
                    constant_columns.append(col)

            if constant_columns:
                processed_df = processed_df.drop(columns=constant_columns)
                logger.info(f"定数列を除去: {len(constant_columns)}個")

            # 特徴量数を制限（AutoFeatの制限対応）
            if len(processed_df.columns) > 100:
                # 分散の大きい特徴量を優先的に選択
                feature_variances = processed_df.var().sort_values(ascending=False)
                selected_cols = feature_variances.head(100).index
                processed_df = processed_df[selected_cols]
                logger.info("特徴量数を100個に制限しました")

            logger.info(
                f"前処理完了: {len(processed_df)}行, {len(processed_df.columns)}列"
            )
            return processed_df, processed_target

        except Exception as e:
            logger.error(f"データ前処理エラー: {e}")
            return pd.DataFrame(), pd.Series()

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        if self.selected_features:
            return [name for name in self.selected_features if name.startswith("AF_")]
        else:
            return []

    def clear_model(self):
        """モデルをクリア（メモリリーク防止のため徹底的にクリーンアップ）"""
        try:
            # AutoFeatモデルの詳細なクリーンアップ
            if self.autofeat_model is not None:
                # AutoFeatモデル内部の属性を徹底的にクリア
                autofeat_attrs = [
                    "feateng_cols_",
                    "featsel_",
                    "model_",
                    "scaler_",
                    "X_",
                    "y_",
                    "feature_names_in_",
                    "n_features_in_",
                    "feature_importances_",
                    "coef_",
                    "intercept_",
                    "_feature_names",
                    "_feature_types",
                    "_transformations",
                ]

                for attr in autofeat_attrs:
                    if hasattr(self.autofeat_model, attr):
                        try:
                            setattr(self.autofeat_model, attr, None)
                        except Exception as attr_error:
                            logger.debug(
                                f"属性 '{attr}' のクリア中にエラー: {attr_error}"
                            )

                # モデル自体をクリア
                self.autofeat_model = None

            # その他の属性をクリア
            self.selected_features = None
            self.feature_scores = {}
            self.last_selection_info = {}

            # パフォーマンス最適化ツールもクリア
            if hasattr(self, "performance_optimizer") and self.performance_optimizer:
                try:
                    if hasattr(self.performance_optimizer, "clear"):
                        self.performance_optimizer.clear()
                except Exception as perf_error:
                    logger.error(
                        f"パフォーマンス最適化ツールクリア中にエラー: {perf_error}"
                    )

            # 強制ガベージコレクション
            self._force_garbage_collection()

        except Exception as e:
            logger.error(f"モデルクリア中にエラー: {e}")
            # エラーが発生してもクリーンアップは続行
            self.autofeat_model = None
            self.selected_features = None
            self.feature_scores = {}
            self.last_selection_info = {}

    def cleanup(self):
        """
        AutoFeatCalculatorのリソースクリーンアップ
        EnhancedFeatureEngineeringServiceから呼び出される統一インターフェース
        """
        try:
            self.clear_model()
        except Exception as e:
            logger.error(f"AutoFeatCalculatorクリーンアップエラー: {e}")

    def _generate_features_single_batch(
        self, df: pd.DataFrame, target: pd.Series, optimized_config: AutoFeatConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """単一バッチでの特徴量生成（バッチ処理用）"""
        try:
            start_time = time.time()

            # ランダム性を追加
            np.random.seed(int(time.time()) % 2147483647)

            # AutoFeatモデルを初期化（バッチ処理用の厳しい制限）
            batch_max_gb = min(optimized_config.max_gb, 0.2)  # バッチ処理では0.2GB以下
            autofeat_model = AutoFeatRegressor(
                feateng_steps=1,  # バッチ処理では常に1ステップ
                max_gb=batch_max_gb,
                verbose=0,  # バッチ処理では詳細ログを抑制
                featsel_runs=1,  # バッチ処理では1回のみ
                n_jobs=1,  # バッチ処理では並列処理を無効化
                units=None,  # 単位を指定しない（メモリ節約）
            )

            # 特徴量生成を実行
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # AutoFeatの推奨方法：fit_transformを使用（メモリ効率が良い）
                transformed_df = autofeat_model.fit_transform(df, target)

                # 結果をDataFrameに変換
                result_df = df.copy()

                # インデックスを統一
                if hasattr(transformed_df, "index"):
                    transformed_df = transformed_df.reset_index(drop=True)
                if hasattr(result_df, "index"):
                    result_df = result_df.reset_index(drop=True)

                # 新しい特徴量を追加
                for i, col_name in enumerate(transformed_df.columns):
                    new_col_name = (
                        f"AF_{col_name}" if not col_name.startswith("AF_") else col_name
                    )
                    if len(transformed_df) == len(result_df):
                        result_df[new_col_name] = transformed_df.iloc[:, i].values

                # 生成情報
                generation_time = time.time() - start_time
                new_features = [
                    col for col in result_df.columns if col.startswith("AF_")
                ]

                batch_info = {
                    "original_features": len(df.columns),
                    "generated_features": len(new_features),
                    "total_features": len(result_df.columns),
                    "generation_time": generation_time,
                }

                # バッチ処理用にモデルを即座にクリア
                del autofeat_model
                self._force_garbage_collection()

                return result_df, batch_info

        except Exception as e:
            logger.error(f"バッチ特徴量生成エラー: {e}")
            return df, {"error": str(e)}
