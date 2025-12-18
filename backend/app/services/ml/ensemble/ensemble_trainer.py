"""
アンサンブル学習トレーナー

BaseMLTrainerを継承し、スタッキングアンサンブル学習のオーケストレーションを行います。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....utils.error_handler import ModelError
from ..base_ml_trainer import BaseMLTrainer
from ..common.evaluation_utils import evaluate_model_predictions
from ..common.ml_utils import predict_class_from_proba, validate_training_inputs
from .meta_labeling import MetaLabelingService
from .stacking import StackingEnsemble

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
        初期化（統一トレーナー: 単一モデル・アンサンブル両対応）

        Args:
            ensemble_config: アンサンブル設定
                - models: モデルのリスト (例: ["lightgbm"] or ["lightgbm", "xgboost"])
                - model_type: 単一モデルタイプ (例: "lightgbm") ※後方互換
                - method: アンサンブル手法 (デフォルト: "stacking")
        """
        super().__init__()

        self.ensemble_config = ensemble_config
        self.ensemble_method = ensemble_config.get("method", "stacking")
        self.ensemble_model = None
        self.meta_labeling_service = None  # メタラベリング サービスを追加
        self.meta_model_threshold = 0.5  # メタモデルの予測閾値
        self.strict_error_mode = ensemble_config.get(
            "strict_error_mode", True
        )  # エラー時に例外を発生させるか

        # 単一モデルモードかアンサンブルモードかを判定
        models = ensemble_config.get("models", [])
        model_type = ensemble_config.get("model_type")

        if model_type or len(models) == 1:
            self.is_single_model = True
            self.model_type = model_type or models[0]
            if not models:
                self.ensemble_config["models"] = [self.model_type]
        else:
            self.is_single_model = False
            self.model_type = "EnsembleModel"

        mode = "単一モデル" if self.is_single_model else "アンサンブル"
        logger.info(
            f"EnsembleTrainer初期化: mode={mode}, "
            f"method={self.ensemble_method}, model_type={self.model_type}, "
            f"strict_error_mode={self.strict_error_mode}"
        )

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
            "base_models": {"lightgbm": {}, "xgboost": {}},
            "stacking": {},
        }

        prefix_map = {
            "lgb_": ("base_models", "lightgbm"),
            "xgb_": ("base_models", "xgboost"),
        }

        for param_name, param_value in training_params.items():
            # ベースモデルパラメータ
            for prefix, (cat, subcat) in prefix_map.items():
                if param_name.startswith(prefix):
                    clean_name = param_name[len(prefix) :]
                    optimized_params[cat][subcat][clean_name] = param_value
                    break
            else:
                # スタッキングパラメータ
                if param_name.startswith("stacking_"):
                    clean_name = param_name[len("stacking_") :]
                    if clean_name.startswith("meta_"):
                        meta_param = clean_name[len("meta_") :]
                        meta_params = optimized_params["stacking"].setdefault(
                            "meta_model_params", {}
                        )
                        meta_params[meta_param] = param_value
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

            return predictions

        except Exception as e:
            logger.error(f"アンサンブル予測確率取得エラー: {e}")
            raise ModelError(f"アンサンブル予測確率の取得に失敗しました: {e}")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        アンサンブルモデルで予測を実行（メタラベリング適用）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の配列（2次元配列）
            メタラベリングが有効な場合は、フィルタリングされた結果（0/1の1次元配列）を返す。
        """
        if self.ensemble_model is None or not self.ensemble_model.is_fitted:
            raise ModelError("学習済みアンサンブルモデルがありません")

        try:
            features_scaled = features_df  # アンサンブルモデルは主にLightGBMベースなのでスケーリング不要

            # StackingEnsembleから予測確率を取得
            predictions_proba = self.ensemble_model.predict_proba(features_scaled)

            # ポジティブクラス（Trend）の確率を取得
            primary_proba = predictions_proba[:, 1]
            primary_proba_series = pd.Series(
                primary_proba, index=features_df.index
            )  # ここで定義

            # メタラベリング適用
            if self.meta_labeling_service and self.meta_labeling_service.is_trained:
                logger.debug("メタラベリングによる予測フィルタリングを適用中...")

                # 各ベースモデルの予測確率を取得

                try:
                    base_model_probs_df = self.ensemble_model.predict_base_models_proba(
                        features_scaled
                    )
                except Exception as e:
                    if self.strict_error_mode:
                        # 厳格モード: 明示的にエラーを発生させる
                        logger.error(f"ベースモデル予測確率の取得に失敗しました: {e}")
                        raise ModelError(
                            f"ベースモデル予測確率の取得に失敗しました。"
                            f"メタラベリングには完全な入力データが必要です: {e}"
                        )
                    else:
                        # 寛容モード: 全てNo Trade（0）を返す
                        logger.warning(
                            f"ベースモデル予測確率の取得に失敗しました。"
                            f"strict_error_mode=Falseのため、全てNo Trade（0）を返します: {e}"
                        )
                        return np.zeros(len(features_df), dtype=int)

                # メタモデルは0/1を返す
                filtered_predictions = self.meta_labeling_service.predict(
                    X=features_scaled,  # Xをfeatures_scaledに修正
                    primary_proba=primary_proba_series,
                    base_model_probs_df=base_model_probs_df,
                )
                return filtered_predictions.values  # Seriesをnp.ndarrayに変換
            else:
                # メタラベリングが有効でない場合は、予測確率をそのまま返す
                return predictions_proba

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

            # 入力データの検証（共通関数を使用）
            validate_training_inputs(X_train, y_train, X_test, y_test, log_info=True)

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
            y_pred = predict_class_from_proba(y_pred_proba)

            # 統一された評価システムを使用
            detailed_metrics = evaluate_model_predictions(
                y_true=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
            )

            # 分類レポート（詳細メトリクスに含まれているが、互換性のために明示的に取得）
            class_report = detailed_metrics.get("classification_report", {})

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

            # BaseMLTrainer用のモデル参照を設定
            self._model = self.ensemble_model

            # 精度を取得
            accuracy = detailed_metrics.get("accuracy", 0.0)
            logger.info(
                f"アンサンブルモデル学習完了: method={self.ensemble_method}, 精度={accuracy:.4f}"
            )

            # --- メタラベリングモデルの学習 ---
            try:
                logger.info("メタラベリングモデルの学習を開始...")
                # アンサンブルモデルのOOF予測値を取得 (これが一次モデルの確信度となる)
                primary_oof_proba_train = self.ensemble_model.get_oof_predictions()
                primary_oof_series_train = pd.Series(
                    primary_oof_proba_train, index=X_train.index
                )  # X_trainのindexを使う

                oof_base_model_probs_df = (
                    self.ensemble_model.get_oof_base_model_predictions()
                )
                X_train_original_for_meta = self.ensemble_model.get_X_train_original()
                y_train_original_for_meta = self.ensemble_model.get_y_train_original()

                if (
                    oof_base_model_probs_df is None
                    or X_train_original_for_meta is None
                    or y_train_original_for_meta is None
                ):
                    logger.warning(
                        "StackingEnsembleからOOF予測値またはオリジナルデータが取得できませんでした。メタラベリング学習をスキップします。"
                    )
                    return result

                # メタラベリングの設定
                meta_config = self.ensemble_config.get("meta_labeling_params", {})
                meta_model_type = meta_config.get("model_type", "lightgbm")
                meta_params = meta_config.get("model_params", {}).copy()

                # 最適化されたパラメータがあれば適用
                if (
                    "stacking" in optimized_params
                    and "meta_model_params" in optimized_params["stacking"]
                ):
                    meta_params.update(
                        optimized_params["stacking"]["meta_model_params"]
                    )

                meta_service = MetaLabelingService(
                    model_type=meta_model_type, model_params=meta_params
                )
                meta_result = meta_service.train(
                    X_train=X_train_original_for_meta,  # オリジナルのX_trainを使用
                    y_train=y_train_original_for_meta,  # オリジナルのy_trainを使用
                    primary_proba_train=primary_oof_series_train,
                    base_model_probs_df=oof_base_model_probs_df,  # 各ベースモデルのOOF予測確率DataFrameを渡す
                )
                if meta_result["status"] == "success":
                    self.meta_labeling_service = meta_service
                    logger.info("メタラベリングモデルの学習が完了しました。")
                else:
                    logger.warning(
                        f"メタラベリングモデルの学習がスキップされました: {meta_result.get('reason')}"
                    )
            except Exception as e:
                logger.error(f"メタラベリングモデル学習エラー: {e}")
                logger.warning("メタラベリングモデルの学習をスキップしました。")

            # メタラベリング学習完了後、StackingEnsemble内の学習データを解放
            # これによりメモリリークを防ぎ、保存されるモデルファイルのサイズを削減
            if hasattr(self.ensemble_model, "clear_training_data"):
                self.ensemble_model.clear_training_data()

            return result

        except Exception as e:
            logger.error(f"アンサンブルモデル学習エラー: {e}")
            raise ModelError(f"アンサンブルモデル学習に失敗しました: {e}")

    def _get_model_to_save(self) -> Any:
        """保存対象のモデルオブジェクトを取得"""
        return self.ensemble_model

    def _get_model_specific_metadata(self, model_name: str) -> Dict[str, Any]:
        """モデル固有のメタデータを取得"""
        if self.ensemble_model is None:
            return {}

        algorithm_name = getattr(self.ensemble_model, "best_algorithm", "unknown")

        metadata = {
            "model_type": algorithm_name,
            "trainer_type": "ensemble",
            "ensemble_method": self.ensemble_method,
            "best_algorithm": algorithm_name,
            "best_model_score": getattr(self.ensemble_model, "best_model_score", None),
            "selected_model_only": True,
            "ensemble_config": self.ensemble_config,
        }

        # メタラベリングモデルの保存
        if self.meta_labeling_service and self.meta_labeling_service.is_trained:
            try:
                from ..model_manager import model_manager

                meta_model_path = model_manager.save_model(
                    model=self.meta_labeling_service,
                    model_name=f"{model_name}_meta",
                    metadata={"primary_model_name": model_name},
                    scaler=None,
                    feature_columns=self.feature_columns,
                )
                metadata["meta_model_path"] = meta_model_path
                logger.info(f"メタラベリングモデル保存完了: {meta_model_path}")
            except Exception as e:
                logger.error(f"メタラベリングモデル保存エラー: {e}")

        return metadata

    def load_model(self, model_path: str) -> bool:
        """
        アンサンブルモデルを読み込み

        Args:
            model_path: 読み込み元パス

        Returns:
            読み込み成功フラグ
        """
        try:
            # BaseMLTrainerのload_modelを使用
            if not super().load_model(model_path):
                return False

            # self._modelにロードされたモデルをensemble_modelに設定
            self.ensemble_model = self._model

            # メタデータから設定を復元
            if hasattr(self, "metadata"):
                self.ensemble_config = self.metadata.get("ensemble_config", {})
                self.ensemble_method = self.metadata.get("ensemble_method", "stacking")

                # メタラベリングモデルのロード
                meta_model_path = self.metadata.get("meta_model_path")
                if meta_model_path:
                    try:
                        from ..model_manager import model_manager

                        # model_manager.load_model returns dict with 'model' key
                        meta_data = model_manager.load_model(meta_model_path)
                        if meta_data and meta_data.get("model"):
                            self.meta_labeling_service = meta_data.get("model")
                            logger.info(
                                f"メタラベリングモデルをロードしました: {meta_model_path}"
                            )
                        else:
                            logger.warning(
                                f"メタラベリングモデルのロードに失敗しました: {meta_model_path}"
                            )
                    except Exception as e:
                        logger.error(f"メタラベリングモデルのロードエラー: {e}")

            logger.info(
                f"アンサンブルモデル読み込み完了: method={self.ensemble_method}"
            )
            return True

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
