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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from .....utils.unified_error_handler import safe_ml_operation
from .automl_config import AutoFeatConfig
from autofeat import AutoFeatRegressor, AutoFeatClassifier

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

    def __enter__(self):
        """コンテキストマネージャーの開始"""
        self._memory_usage_before = self._get_memory_usage()
        logger.debug(
            f"AutoFeatCalculator開始時メモリ使用量: {self._memory_usage_before:.2f}MB"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了時にリソースをクリーンアップ"""
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
            collected = gc.collect()
            logger.debug(f"ガベージコレクション実行: {collected}オブジェクト回収")
        except Exception as e:
            logger.error(f"ガベージコレクションエラー: {e}")

    @contextmanager
    def _memory_managed_operation(self, operation_name: str):
        """メモリ管理付きの操作を実行するコンテキストマネージャー"""
        start_memory = self._get_memory_usage()
        logger.debug(f"{operation_name}開始時メモリ: {start_memory:.2f}MB")

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
        self, df: pd.DataFrame, target: pd.Series, batch_size: int = 5000
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
                # 小さなバッチで特徴量生成
                batch_result, batch_meta = self._generate_features_single_batch(
                    batch_df, batch_target
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

                # 設定の決定
                max_feat = (
                    max_features
                    if max_features is not None
                    else self.config.max_features
                )

                logger.info(f"AutoFeat特徴量生成開始: 最大特徴量={max_feat}")

                # データの前処理
                processed_df, processed_target = self._preprocess_data(df, target)

                if processed_df.empty:
                    logger.warning("前処理後のデータが空です")
                    return df, {"error": "Empty processed data"}

                # 大量データの場合はバッチ処理を使用
                if self._should_use_batch_processing(processed_df):
                    logger.info("大量データのためバッチ処理モードを使用します")
                    return self._process_in_batches(processed_df, processed_target)

                # NumPyのランダムシードを現在時刻で設定（AutoFeat内部で使用される）
                np.random.seed(int(time.time()) % 2147483647)

                # AutoFeatモデルを初期化（正しいAPI）
                with self._memory_managed_operation("AutoFeatモデル初期化"):
                    if task_type.lower() == "classification":
                        self.autofeat_model = AutoFeatClassifier(
                            feateng_steps=self.config.feateng_steps,
                            max_gb=self.config.max_gb,
                            verbose=1,
                            featsel_runs=1,  # 特徴量選択の実行回数を減らす
                        )
                    else:
                        self.autofeat_model = AutoFeatRegressor(
                            feateng_steps=self.config.feateng_steps,
                            max_gb=self.config.max_gb,
                            verbose=1,
                            featsel_runs=1,  # 特徴量選択の実行回数を減らす
                        )

                # 特徴量生成を実行
                with self._memory_managed_operation("AutoFeat学習・変換"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        logger.info("AutoFeat特徴量生成実行中...")
                        logger.info(f"入力データ形状: {processed_df.shape}")
                        logger.info(f"入力列: {list(processed_df.columns)}")

                        # AutoFeatは自動的に特徴量を生成し、最適なものを選択
                        self.autofeat_model.fit(processed_df, processed_target)

                        # 変換されたデータを取得
                        transformed_df = self.autofeat_model.transform(processed_df)

                logger.info(f"変換後データ形状: {transformed_df.shape}")
                logger.info(f"変換後列: {list(transformed_df.columns)}")
                logger.info(f"元データ形状: {df.shape}")

                # AutoFeatプレフィックスを追加して元のDataFrameと結合
                result_df = df.copy()

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

            # NaNを含む行を除去
            valid_mask = numeric_df.notna().all(axis=1) & target_reset.notna()
            processed_df = numeric_df[valid_mask]
            processed_target = target_reset[valid_mask]

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

    def _extract_selected_features(
        self, original_df: pd.DataFrame, processed_df: pd.DataFrame
    ) -> pd.DataFrame:
        """選択された特徴量を抽出"""
        try:
            if self.autofeat_model is None:
                return original_df

            # AutoFeatで生成された特徴量を取得
            if hasattr(self.autofeat_model, "transform"):
                # 変換された特徴量を取得
                transformed_features = self.autofeat_model.transform(processed_df)

                # 特徴量名を生成
                feature_names = [
                    f"AF_{i}" for i in range(transformed_features.shape[1])
                ]

                # DataFrameに変換
                selected_df = pd.DataFrame(
                    transformed_features,
                    columns=feature_names,
                    index=processed_df.index,
                )

                # 元のDataFrameのインデックスに合わせる
                result_df = original_df.copy()
                for col in selected_df.columns:
                    if len(selected_df) == len(result_df):
                        result_df[col] = selected_df[col].values
                    else:
                        # 長さが異なる場合は短い方に合わせる
                        min_length = min(len(selected_df), len(result_df))
                        result_df.loc[: min_length - 1, col] = (
                            selected_df[col].iloc[:min_length].values
                        )

                return result_df
            else:
                logger.warning("AutoFeatモデルにtransformメソッドがありません")
                return original_df

        except Exception as e:
            logger.error(f"特徴量抽出エラー: {e}")
            return original_df

    def _calculate_feature_scores(
        self, features_df: pd.DataFrame, target: pd.Series, task_type: str
    ) -> Dict[str, float]:
        """特徴量スコアを計算"""
        try:
            scores = {}

            # AutoFeat特徴量のみを対象とする
            autofeat_columns = [
                col for col in features_df.columns if col.startswith("AF_")
            ]

            if not autofeat_columns:
                return scores

            autofeat_features = features_df[autofeat_columns]

            # 有効なデータのマスクを作成
            valid_mask = target.notna() & autofeat_features.notna().all(axis=1)
            valid_features = autofeat_features[valid_mask]
            valid_target = target[valid_mask]

            if len(valid_features) == 0:
                return scores

            # RandomForestで重要度を計算
            if task_type.lower() == "classification":
                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            else:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf.fit(valid_features, valid_target)

            # 重要度をスコアとして保存
            for i, col in enumerate(autofeat_columns):
                if i < len(rf.feature_importances_):
                    scores[col] = float(rf.feature_importances_[i])

            return scores

        except Exception as e:
            logger.error(f"特徴量スコア計算エラー: {e}")
            return {}

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        if self.selected_features:
            return [name for name in self.selected_features if name.startswith("AF_")]
        else:
            return []

    def get_generation_info(self) -> Dict[str, Any]:
        """最後の生成情報を取得"""
        return self.last_selection_info.copy()

    def get_feature_scores(self) -> Dict[str, float]:
        """特徴量スコアを取得"""
        return self.feature_scores.copy()

    def evaluate_selected_features(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        task_type: str = "regression",
    ) -> Dict[str, float]:
        """選択された特徴量の性能を評価"""
        try:
            # AutoFeat特徴量のみを使用
            autofeat_columns = [
                col for col in features_df.columns if col.startswith("AF_")
            ]

            if not autofeat_columns:
                return {"error": "AutoFeat特徴量が見つかりません"}

            autofeat_features = features_df[autofeat_columns]

            # 有効なデータのマスクを作成
            valid_mask = target.notna() & autofeat_features.notna().all(axis=1)
            valid_features = autofeat_features[valid_mask]
            valid_target = target[valid_mask]

            if len(valid_features) < 10:
                return {"error": "評価に十分なデータがありません"}

            # クロスバリデーションで性能を評価
            if task_type.lower() == "classification":
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=50, random_state=42, n_jobs=1
                )
                cv_scores = cross_val_score(
                    model, valid_features, valid_target, cv=3, scoring="accuracy"
                )

                return {
                    "cv_accuracy_mean": float(cv_scores.mean()),
                    "cv_accuracy_std": float(cv_scores.std()),
                    "feature_count": len(autofeat_columns),
                }
            else:
                model = RandomForestRegressor(
                    n_estimators=50, random_state=42, n_jobs=1
                )
                cv_scores = cross_val_score(
                    model, valid_features, valid_target, cv=3, scoring="r2"
                )

                return {
                    "cv_r2_mean": float(cv_scores.mean()),
                    "cv_r2_std": float(cv_scores.std()),
                    "feature_count": len(autofeat_columns),
                }

        except Exception as e:
            logger.error(f"特徴量評価エラー: {e}")
            return {"error": str(e)}

    def clear_model(self):
        """モデルをクリア（メモリリーク防止のため徹底的にクリーンアップ）"""
        try:
            # AutoFeatモデルの詳細なクリーンアップ
            if self.autofeat_model is not None:
                # AutoFeatモデル内部の属性をクリア
                if hasattr(self.autofeat_model, "feateng_cols_"):
                    self.autofeat_model.feateng_cols_ = None
                if hasattr(self.autofeat_model, "featsel_"):
                    self.autofeat_model.featsel_ = None
                if hasattr(self.autofeat_model, "model_"):
                    self.autofeat_model.model_ = None
                if hasattr(self.autofeat_model, "scaler_"):
                    self.autofeat_model.scaler_ = None

                # モデル自体をクリア
                self.autofeat_model = None
                logger.debug("AutoFeatモデルの詳細クリーンアップ完了")

            # その他の属性をクリア
            self.selected_features = None
            self.feature_scores = {}
            self.last_selection_info = {}

            # 強制ガベージコレクション
            self._force_garbage_collection()

            logger.debug("AutoFeatモデルをクリアしました")

        except Exception as e:
            logger.error(f"モデルクリア中にエラー: {e}")
            # エラーが発生してもクリーンアップは続行
            self.autofeat_model = None
            self.selected_features = None
            self.feature_scores = {}
            self.last_selection_info = {}

    def _generate_features_single_batch(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """単一バッチでの特徴量生成（バッチ処理用）"""
        try:
            start_time = time.time()

            # ランダム性を追加
            np.random.seed(int(time.time()) % 2147483647)

            # AutoFeatモデルを初期化
            autofeat_model = AutoFeatRegressor(
                feateng_steps=self.config.feateng_steps,
                max_gb=min(self.config.max_gb, 2),  # バッチ処理では少ないメモリを使用
                verbose=0,  # バッチ処理では詳細ログを抑制
                featsel_runs=1,
            )

            # 特徴量生成を実行
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 学習と変換
                autofeat_model.fit(df, target)
                transformed_df = autofeat_model.transform(df)

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
