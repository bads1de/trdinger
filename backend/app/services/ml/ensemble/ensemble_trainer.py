"""
アンサンブル学習トレーナー

BaseMLTrainerを継承し、スタッキングアンサンブル学習のオーケストレーションを行います。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....utils.error_handler import ModelError
from ..base_ml_trainer import BaseMLTrainer
from .stacking import StackingEnsemble
from ..meta_labeling_service import MetaLabelingService # Added

logger = logging.getLogger(__name__)


class EnsembleTrainer(BaseMLTrainer):
    """
    アンサンブル学習トレーナー

    BaseMLTrainerを継承し、スタッキングアンサンブル学習機能を提供します。
    """

    def __init__(
        self,
        ensemble_config: Dict[str, Any],
    ):
        """
        初期化

        Args:
            ensemble_config: アンサンブル設定
        """
        super().__init__()

        self.ensemble_config = ensemble_config
        self.model_type = "EnsembleModel"
        self.ensemble_method = ensemble_config.get("method", "stacking")
        self.ensemble_model = None
        self.meta_labeling_service = None # メタラベリングサービスを追加
        self.meta_model_threshold = 0.5 # メタモデルの予測閾値

        logger.info(f"EnsembleTrainer初期化: method={self.ensemble_method}")

    def _extract_optimized_parameters(
        self, training_params: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        最適化されたパラメータを各モデル・手法別に分離

        Args:
            training_params: 最適化されたパラメータを含む学習パラメータ

        Returns:
            分離されたパラメータ辞書
        """
        optimized_params = {
            "base_models": {
                "lightgbm": {},
                "xgboost": {},
            },
            "stacking": {},
        }

        for param_name, param_value in training_params.items():
            # LightGBMパラメータ
            if param_name.startswith("lgb_"):
                clean_name = param_name.replace("lgb_", "")
                optimized_params["base_models"]["lightgbm"][clean_name] = param_value

            # XGBoostパラメータ
            elif param_name.startswith("xgb_"):
                clean_name = param_name.replace("xgb_", "")
                optimized_params["base_models"]["xgboost"][clean_name] = param_value

            # スタッキングパラメータ
            elif param_name.startswith("stacking_"):
                clean_name = param_name.replace("stacking_", "")
                if clean_name.startswith("meta_"):
                    # メタモデルパラメータ
                    meta_param = clean_name.replace("meta_", "")
                    if "meta_model_params" not in optimized_params["stacking"]:
                        optimized_params["stacking"]["meta_model_params"] = {}
                    optimized_params["stacking"]["meta_model_params"][meta_param] = (
                        param_value
                    )
                else:
                    optimized_params["stacking"][clean_name] = param_value

        return optimized_params

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        アンサンブルモデルで予測確率を取得（フィルタリングなし）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の配列 (2クラス分類)
        """
        if self.ensemble_model is None or not self.ensemble_model.is_fitted:
            raise ModelError("学習済みアンサンブルモデルがありません")

        try:
            features_scaled = features_df
            predictions = self.ensemble_model.predict_proba(features_scaled)

            # StackingEnsembleのpredict_probaは2クラスの確率を返す想定
            if predictions.ndim == 2 and predictions.shape[1] == 2:
                return predictions
            else:
                raise ModelError(
                    f"予期しない予測確率の形状: {predictions.shape}. "
                    f"2クラス分類の確率が期待されます。"
                )

        except Exception as e:
            logger.error(f"アンサンブル予測確率取得エラー: {e}")
            raise ModelError(f"アンサンブル予測確率の取得に失敗しました: {e}")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        アンサンブルモデルで予測を実行（メタラベリング適用）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測クラスの配列 (0=Range, 1=Trend)
        """
        if self.ensemble_model is None or not self.ensemble_model.is_fitted:
            raise ModelError("学習済みアンサンブルモデルがありません")

        try:
            features_scaled = features_df # アンサンブルモデルは主にLightGBMベースなのでスケーリング不要

            # StackingEnsembleから予測確率を取得
            predictions_proba = self.ensemble_model.predict_proba(features_scaled)

            # ポジティブクラス（Trend）の確率を取得
            primary_proba = predictions_proba[:, 1]
            primary_proba_series = pd.Series(primary_proba, index=features_df.index) # ここで定義

            # メタラベリング適用
            if self.meta_labeling_service and self.meta_labeling_service.is_trained:
                logger.debug("メタラベリングによる予測フィルタリングを適用中...")
                
                # 各ベースモデルの予測確率を取得

                try:
                    base_model_probs_df = self.ensemble_model.predict_base_models_proba(features_scaled)
                except Exception as e:
                    logger.warning(f"ベースモデル予測確率の取得に失敗したため、空のDataFrameを使用します: {e}")
                    base_model_probs_df = pd.DataFrame(index=features_scaled.index) # indexをfeatures_scaledに合わせる

                # メタモデルは0/1を返す
                filtered_predictions = self.meta_labeling_service.predict(
                    X=features_scaled, # Xをfeatures_scaledに修正
                    primary_proba=primary_proba_series,
                    base_model_probs_df=base_model_probs_df
                )
                return filtered_predictions.values # Seriesをnp.ndarrayに変換
            else:
                logger.debug("メタラベリングは有効化されていません。StackingEnsembleの直接予測を使用。")
                # メタラベリングが有効でない場合は、StackingEnsembleの直接予測を2値に変換
                return (primary_proba >= self.meta_model_threshold).astype(int)

        except Exception as e:
            logger.error(f"アンサンブル予測エラー: {e}")
            raise ModelError(f"アンサンブル予測に失敗しました: {e}")

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        アンサンブルモデルを学習（BaseMLTrainerの抽象メソッド実装）

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果
        """
        try:
            logger.info(f"アンサンブル学習開始: method={self.ensemble_method}")

            # 入力データの検証
            if X_train is None or X_train.empty:
                raise ValueError("学習用特徴量データが空です")
            if y_train is None or len(y_train) == 0:
                raise ValueError("学習用ターゲットデータが空です")
            if len(X_train) != len(y_train):
                raise ValueError("特徴量とターゲットの長さが一致しません")

            logger.info(
                f"学習データサイズ: {len(X_train)}行, {len(X_train.columns)}特徴量"
            )
            logger.info(f"テストデータサイズ: {len(X_test)}行")
            logger.info(f"ターゲット分布: {y_train.value_counts().to_dict()}")

            # ハイパーパラメータ最適化からのパラメータを分離
            optimized_params = self._extract_optimized_parameters(training_params)

            # スタッキングアンサンブルモデルを作成
            if self.ensemble_method.lower() == "stacking":
                # スタッキング設定を準備
                stacking_config = self.ensemble_config.get("stacking_params", {})

                # 最適化されたスタッキングパラメータを適用
                if "stacking" in optimized_params:
                    stacking_config.update(optimized_params["stacking"])

                stacking_config.update(
                    {
                        "random_state": training_params.get("random_state", 42),
                        "n_jobs": training_params.get("n_jobs", -1),  # 並列処理を有効化
                    }
                )

                self.ensemble_model = StackingEnsemble(config=stacking_config)

            else:
                raise ModelError(
                    f"サポートされていないアンサンブル手法: {self.ensemble_method}"
                )

            # 最適化されたベースモデルパラメータをアンサンブルモデルに渡す
            base_model_params = optimized_params.get("base_models", {})

            # アンサンブルモデルを学習
            try:
                logger.info("アンサンブルモデルの学習を開始")
                training_result = self.ensemble_model.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    base_model_params=base_model_params,
                )
                logger.info("アンサンブルモデルの学習が完了")
            except Exception as e:
                logger.error(f"アンサンブルモデル学習エラー: {e}")
                raise ModelError(f"アンサンブルモデルの学習に失敗しました: {e}")

            # 予測と評価
            y_pred_proba = self.ensemble_model.predict_proba(X_test)
            y_pred = self.ensemble_model.predict(X_test)

            # 予測確率が3クラス分類であることを確認
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 3:
                logger.warning(
                    f"予測確率の形状が期待と異なります: {y_pred_proba.shape}"
                )
                # 3クラス分類でない場合は評価をスキップ

            # 統一された評価指標計算器を使用
            from ..evaluation.metrics import (
                MetricsCalculator,
                MetricsConfig,
            )

            config = MetricsConfig(
                include_balanced_accuracy=True,
                include_pr_auc=True,
                include_roc_auc=True,
                include_confusion_matrix=True,
                include_classification_report=True,
                average_method="weighted",
                zero_division="0",
            )

            metrics_calculator = MetricsCalculator(config)
            detailed_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )

            # 分類レポート
            from sklearn.metrics import classification_report

            class_report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0.0
            )

            # 特徴量重要度
            feature_importance = self.ensemble_model.get_feature_importance()
            if not feature_importance:
                logger.warning(
                    "アンサンブルモデルから特徴量重要度を取得できませんでした"
                )
                feature_importance = {}
            else:
                logger.info(f"特徴量重要度を取得: {len(feature_importance)}個")

            # 結果をまとめ
            result = {
                **detailed_metrics,
                **training_result,
                "classification_report": class_report,
                "feature_importance": feature_importance,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "model_type": self.model_type,
                "ensemble_method": self.ensemble_method,
            }

            # 学習完了フラグを設定
            self.is_trained = True

            # 精度を取得
            accuracy = detailed_metrics.get("accuracy", 0.0)
            logger.info(
                f"アンサンブルモデル学習完了: method={self.ensemble_method}, 精度={accuracy:.4f}"
            )

            # --- メタラベリングモデルの学習 ---
            try:
                logger.info("メタラベリングモデルの学習を開始...")
                # アンサンブルモデルのOOF予測値を取得 (これが一次モデルの確信度となる)
                # StackingServiceにOOF予測の公開メソッドがあればそれを使う
                # 現状はrun_ml_pipeline.pyで計算しているものを渡す必要あり
                # ここでは簡易的にX_trainで予測したものをOOFとみなす（厳密にはリークするが、ここでは動作確認）
                # TODO: StackingServiceからOOF予測値を公開するように修正
                primary_oof_proba_train = self.ensemble_model.get_oof_predictions()
                primary_oof_series_train = pd.Series(primary_oof_proba_train, index=X_train.index) # X_trainのindexを使う

                oof_base_model_probs_df = self.ensemble_model.get_oof_base_model_predictions()
                X_train_original_for_meta = self.ensemble_model.get_X_train_original()
                y_train_original_for_meta = self.ensemble_model.get_y_train_original()

                if oof_base_model_probs_df is None or X_train_original_for_meta is None or y_train_original_for_meta is None:
                    logger.warning("StackingEnsembleからOOF予測値またはオリジナルデータが取得できませんでした。メタラベリング学習をスキップします。")
                    return result

                self.meta_labeling_service = MetaLabelingService()
                meta_result = self.meta_labeling_service.train(
                    X_train=X_train_original_for_meta, # オリジナルのX_trainを使用
                    y_train=y_train_original_for_meta, # オリジナルのy_trainを使用
                    primary_proba_train=primary_oof_series_train,
                    base_model_probs_df=oof_base_model_probs_df # 各ベースモデルのOOF予測確率DataFrameを渡す
                )
                if meta_result["status"] == "success":
                    logger.info("メタラベリングモデルの学習が完了しました。")
                else:
                    logger.warning(f"メタラベリングモデルの学習がスキップされました: {meta_result.get('reason')}")
            except Exception as e:
                logger.error(f"メタラベリングモデル学習エラー: {e}")
                logger.warning("メタラベリングモデルの学習をスキップしました。")


            return result

        except Exception as e:
            logger.error(f"アンサンブルモデル学習エラー: {e}")
            raise ModelError(f"アンサンブルモデル学習に失敗しました: {e}")

    def save_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        アンサンブルモデルを統一されたモデル管理システムで保存

        Args:
            model_name: モデル名
            metadata: モデルメタデータ（オプション）

        Returns:
            保存されたモデルのパス
        """
        if self.ensemble_model is None:
            raise ModelError("アンサンブルモデルが初期化されていません")
        if not self.is_trained:
            raise ModelError("アンサンブルモデルが学習されていません")
        if (
            not hasattr(self.ensemble_model, "is_fitted")
            or not self.ensemble_model.is_fitted
        ):
            raise ModelError("アンサンブルモデルの学習が完了していません")

        try:
            from ..model_manager import model_manager

            # アンサンブル用メタデータを準備
            algorithm_name = getattr(self.ensemble_model, "best_algorithm", "unknown")

            final_metadata = metadata or {}
            final_metadata.update(
                {
                    "model_type": algorithm_name,  # 最高性能アルゴリズム名を使用
                    "trainer_type": "ensemble",
                    "ensemble_method": self.ensemble_method,
                    "feature_count": (
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                    "best_algorithm": algorithm_name,
                    "best_model_score": getattr(
                        self.ensemble_model, "best_model_score", None
                    ),
                    "selected_model_only": True,
                    "ensemble_config": self.ensemble_config,
                }
            )

            # --- メタラベリングモデルの保存 ---
            meta_model_path = None
            if self.meta_labeling_service and self.meta_labeling_service.is_trained:
                meta_model_path = model_manager.save_model(
                    model=self.meta_labeling_service,
                    model_name=f"{model_name}_meta", # メタモデル用の名前
                    metadata={"primary_model_name": model_name},
                    scaler=None, # メタモデルはスケーラーを使わない
                    feature_columns=self.feature_columns # メタモデルも元の特徴量を使う
                )
                final_metadata["meta_model_path"] = meta_model_path
                logger.info(f"メタラベリングモデル保存完了: {meta_model_path}")
            # ----------------------------------

            # 特徴量重要度をメタデータに追加
            try:
                feature_importance = self.get_feature_importance(top_n=100)
                if feature_importance:
                    final_metadata["feature_importance"] = feature_importance
                    logger.info(
                        f"特徴量重要度をメタデータに追加: {len(feature_importance)}個"
                    )
            except Exception as e:
                logger.warning(f"特徴量重要度の取得に失敗: {e}")

            # アンサンブルモデルの保存
            # StackingEnsembleクラスのsave_modelsを使う
            if self.ensemble_model:
                model_paths = self.ensemble_model.save_models(base_path=model_manager.model_save_path / model_name)
                # StackingEnsemble.save_modelsが複数パスを返すので、最初のものを代表パスとする
                model_path = model_paths[0] if model_paths else None
            else:
                model_path = None

            if model_path is None:
                raise ModelError("モデル保存に失敗しました")
            
            # 統一されたモデル保存のメタデータ更新
            model_manager.update_model_metadata(model_path, final_metadata)

            logger.info(
                f"アンサンブル最高性能モデル保存完了: {model_path} (アルゴリズム: {algorithm_name})"
            )
            return model_path

        except Exception as e:
            logger.error(f"アンサンブルモデル保存エラー: {e}")
            raise ModelError(f"アンサンブルモデルの保存に失敗しました: {e}")

    def load_model(self, model_path: str) -> bool:
        """
        アンサンブルモデルを読み込み

        Args:
            model_path: 読み込み元パス

        Returns:
            読み込み成功フラグ
        """
        try:
            # メタデータを読み込み（タイムスタンプ付きファイルに対応）
            import glob
            import os
            import warnings

            import joblib
            from sklearn.exceptions import InconsistentVersionWarning

            metadata_patterns = [
                f"{model_path}_ensemble_metadata_*.pkl",  # 新形式
                f"{model_path}_ensemble_metadata.pkl",  # 旧形式
            ]

            metadata_path = None
            for pattern in metadata_patterns:
                files = glob.glob(pattern)
                if files:
                    metadata_path = sorted(files)[-1]  # 最新のファイルを選択
                    break

            if metadata_path and os.path.exists(metadata_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    metadata = joblib.load(metadata_path)
                self.ensemble_config = metadata["ensemble_config"]
                self.model_type = metadata["model_type"]
                self.ensemble_method = metadata["ensemble_method"]
                self.feature_columns = metadata["feature_columns"]
                self.scaler = metadata["scaler"]
                self.is_trained = metadata["is_trained"]

                # --- メタラベリングモデルのロード ---
                if "meta_model_path" in metadata and metadata["meta_model_path"]:
                    try:
                        from ..model_manager import model_manager
                        self.meta_labeling_service = model_manager.load_model(metadata["meta_model_path"])
                        if self.meta_labeling_service:
                            logger.info(f"メタラベリングモデルをロードしました: {metadata['meta_model_path']}")
                        else:
                            logger.warning(f"メタラベリングモデルのロードに失敗しました: {metadata['meta_model_path']}")
                    except Exception as e:
                        logger.error(f"メタラベリングモデルのロードエラー: {e}")
                # ------------------------------------

            # スタッキングアンサンブルモデルを作成
            if self.ensemble_method.lower() == "stacking":
                stacking_config = self.ensemble_config.get("stacking_params", {})
                self.ensemble_model = StackingEnsemble(config=stacking_config)
            else:
                raise ModelError(
                    f"サポートされていないアンサンブル手法: {self.ensemble_method}"
                )

            # アンサンブルモデルを読み込み
            success = self.ensemble_model.load_models(model_path)

            if success:
                logger.info(
                    f"アンサンブルモデル読み込み完了: method={self.ensemble_method}"
                )
            else:
                logger.error("アンサンブルモデルの読み込みに失敗")

            return success

        except Exception as e:
            logger.error(f"アンサンブルモデル読み込みエラー: {e}")
            return False

    def _cleanup_models(self, level):
        """
        EnsembleTrainer固有のモデルクリーンアップ

        Args:
            level: クリーンアップレベル
        """
        try:
            # アンサンブルモデルのクリーンアップ
            if self.ensemble_model is not None:
                try:
                    # アンサンブルモデル自体をクリア
                    self.ensemble_model = None
                    logger.debug("アンサンブルモデルをクリアしました")

                except Exception as ensemble_error:
                    logger.warning(
                        f"アンサンブルモデルクリーンアップ警告: {ensemble_error}"
                    )

            # 親クラスのモデルクリーンアップを呼び出し
            super()._cleanup_models(level)

        except Exception as e:
            logger.warning(f"EnsembleTrainerモデルクリーンアップエラー: {e}")
            # エラーが発生してもクリーンアップは続行
            self.ensemble_model = None

    def get_feature_importance(self, top_n: int = 100) -> Dict[str, float]:
        """
        アンサンブルモデルから特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or not self.ensemble_model:
            logger.warning("学習済みアンサンブルモデルがありません")
            return {}

        try:
            # アンサンブルモデルから特徴量重要度を取得
            feature_importance = self.ensemble_model.get_feature_importance()
            if not feature_importance:
                logger.warning(
                    "アンサンブルモデルから特徴量重要度を取得できませんでした"
                )
                return {}

            # 上位N個を取得
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            logger.info(
                f"アンサンブルから特徴量重要度を取得: {len(sorted_importance)}個"
            )
            return dict(sorted_importance)

        except Exception as e:
            logger.error(f"アンサンブル特徴量重要度取得エラー: {e}")
            return {}
