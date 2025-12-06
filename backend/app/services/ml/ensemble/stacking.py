"""
スタッキング（Stacking）アンサンブル手法の実装

自前実装によるスタッキング。OOF予測を生成し、メタモデルを直接学習することで
計算コストを最小化しつつ、各モデルの強みを活かした高精度な予測を実現します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from ....utils.error_handler import ModelError
from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEnsemble):
    """
    自前実装によるスタッキングアンサンブル

    sklearnのStackingClassifierを使用せず、以下の3ステップで実装:
    1. OOF予測生成: 各ベースモデルでcross_val_predictを実行
    2. メタモデル学習: OOF予測をメタ特徴量としてメタモデルを直接学習
    3. ベースモデル最終fit: 全データで各ベースモデルをフィット（推論用）

    これにより、sklearn.StackingClassifierを使用した場合と比較して
    ベースモデルの学習回数が約半分になり、計算コストを削減できます。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: スタッキング設定
        """
        super().__init__(config)

        # スタッキング固有の設定
        self._base_model_types: List[str] = config.get(
            "base_models", ["lightgbm", "xgboost", "catboost"]
        )
        self._meta_model_type: str = (
            config.get("meta_model", "logistic_regression") or "logistic_regression"
        )
        self.cv_folds = config.get("cv_folds", 5)
        self.stack_method = config.get("stack_method", "predict_proba")
        self.random_state = config.get("random_state", 42)
        self.n_jobs = config.get("n_jobs", -1)  # 並列処理
        self.passthrough = config.get(
            "passthrough", False
        )  # 元特徴量をメタモデルに渡すか

        # 学習済みモデルを保持
        self._fitted_base_models: Dict[str, Any] = {}  # {name: fitted_model}
        self._fitted_meta_model: Optional[Any] = None
        self.oof_predictions: Optional[np.ndarray] = None  # OOF予測値を保持
        self.oof_base_model_predictions: Optional[pd.DataFrame] = (
            None  # 各ベースモデルのOOF予測確率を保持
        )
        self.X_train_original: Optional[pd.DataFrame] = (
            None  # メタモデルの特徴量として使うため
        )
        self.y_train_original: Optional[pd.Series] = None  # メタラベル生成のため

        # 後方互換性のためのエイリアス
        self.stacking_classifier = None  # 互換性のために残す（ただし使用しない）

        logger.info(
            f"StackingEnsemble初期化（自前実装）: base_models={self._base_model_types}, "
            f"meta_model={self._meta_model_type}, cv_folds={self.cv_folds}, "
            f"stack_method={self.stack_method}, n_jobs={self.n_jobs}, "
            f"passthrough={self.passthrough}"
        )

    @property
    def base_models(self) -> List[str]:
        """後方互換性のためのプロパティ"""
        return self._base_model_types

    @base_models.setter
    def base_models(self, value: List[str]) -> None:
        """後方互換性のためのセッター"""
        self._base_model_types = value

    @property
    def meta_model(self) -> str:
        """後方互換性のためのプロパティ"""
        return self._meta_model_type

    @meta_model.setter
    def meta_model(self, value: str) -> None:
        """後方互換性のためのセッター"""
        self._meta_model_type = value

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        base_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        自前実装のスタッキングアンサンブルモデルを学習

        処理フロー:
        1. 各ベースモデルでcross_val_predictを実行してOOF予測を生成
        2. OOF予測（+オプションで元特徴量）をメタ特徴量としてメタモデルを学習
        3. 全データで各ベースモデルを最終フィット（推論時に使用）

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_test: テスト用特徴量（オプション）
            y_test: テスト用ターゲット（オプション）
            base_model_params: 各ベースモデルの最適化されたパラメータ（オプション）

        Returns:
            学習結果の辞書
        """
        try:
            logger.info(
                "スタッキングアンサンブル学習を開始（自前実装・計算コスト最適化版）"
            )

            # 入力データの検証（共通関数を使用）
            from ..common.ml_utils import validate_training_inputs

            validate_training_inputs(X_train, y_train, X_test, y_test, log_info=True)

            self.feature_columns = X_train.columns.tolist()

            # ベースモデルのリストを作成
            try:
                estimators = self._create_base_estimators()
                logger.info(f"ベースモデル作成完了: {[name for name, _ in estimators]}")
            except Exception as e:
                logger.error(f"ベースモデル作成エラー: {e}")
                raise ModelError(f"ベースモデルの作成に失敗しました: {e}")

            # クロスバリデーション設定
            cv = self._create_cv_splitter(X_train)

            # ============================================================
            # ステップ1: 各ベースモデルのOOF予測を計算
            # ============================================================
            logger.info(
                f"[Step 1/3] 各ベースモデルのOOF予測を計算中（{self.cv_folds}フォールドCV）..."
            )

            oof_base_model_preds = pd.DataFrame(index=X_train.index)

            for name, estimator in estimators:
                logger.debug(f"  {name}のOOF予測を計算中...")
                try:
                    # cross_val_predictで各サンプルをOOF方式で予測
                    oof_pred = cross_val_predict(
                        estimator,
                        X_train,
                        y_train,
                        cv=cv,
                        method="predict_proba",
                        n_jobs=self.n_jobs,
                        verbose=0,
                    )
                    # ポジティブクラス（通常はクラス1）の確率を取得
                    oof_base_model_preds[name] = oof_pred[:, 1]
                except Exception as e:
                    logger.warning(
                        f"  {name}のOOF予測計算でエラー: {e}。スキップします。"
                    )
                    continue

            if oof_base_model_preds.empty:
                raise ModelError("有効なベースモデルのOOF予測が生成できませんでした")

            logger.info(
                f"ベースモデルのOOF予測計算完了: {len(oof_base_model_preds.columns)}モデル"
            )

            # ============================================================
            # ステップ2: OOF予測を使ってメタモデルを直接学習
            # ============================================================
            logger.info("[Step 2/3] メタモデルを学習中...")

            # メタ特徴量を構築
            if self.passthrough:
                # 元特徴量も含める
                meta_features = pd.concat([oof_base_model_preds, X_train], axis=1)
                logger.info(
                    f"メタ特徴量: OOF予測({len(oof_base_model_preds.columns)}) + "
                    f"元特徴量({len(X_train.columns)}) = {len(meta_features.columns)}"
                )
            else:
                meta_features = oof_base_model_preds
                logger.info(f"メタ特徴量: OOF予測のみ ({len(meta_features.columns)})")

            # メタモデルを作成して学習
            try:
                self._fitted_meta_model = self._create_base_model(self._meta_model_type)
                self._fitted_meta_model.fit(meta_features, y_train)
                logger.info(f"メタモデル({self._meta_model_type})の学習完了")
            except Exception as e:
                logger.error(f"メタモデル学習エラー: {e}")
                raise ModelError(f"メタモデルの学習に失敗しました: {e}")

            # メタモデルを使ってOOF予測を生成（メタラベリング用）
            meta_oof_pred = cross_val_predict(
                clone(self._create_base_model(self._meta_model_type)),
                meta_features,
                y_train,
                cv=self.cv_folds,  # メタモデルには単純なKFoldでOK
                method="predict_proba",
                n_jobs=self.n_jobs,
                verbose=0,
            )

            # ============================================================
            # ステップ3: 全データでベースモデルを最終フィット（推論用）
            # ============================================================
            logger.info("[Step 3/3] ベースモデルを全データでフィット中（推論用）...")

            self._fitted_base_models.clear()
            for name, estimator in estimators:
                if name not in oof_base_model_preds.columns:
                    continue  # OOF予測生成でスキップされたモデル

                try:
                    # cloneして新しいインスタンスで学習
                    fitted_model = clone(estimator)
                    fitted_model.fit(X_train, y_train)
                    self._fitted_base_models[name] = fitted_model
                    logger.debug(f"  {name}のフィット完了")
                except Exception as e:
                    logger.warning(
                        f"  {name}の最終フィットでエラー: {e}。スキップします。"
                    )
                    continue

            logger.info(
                f"ベースモデル最終フィット完了: {len(self._fitted_base_models)}モデル"
            )

            # ============================================================
            # 学習完了・結果保存
            # ============================================================
            self.is_fitted = True

            # OOF予測を保存（メタラベリング用）
            self.oof_predictions = meta_oof_pred[:, 1]  # ポジティブクラスの確率
            self.oof_base_model_predictions = oof_base_model_preds
            self.X_train_original = X_train.copy()
            self.y_train_original = y_train.copy()

            logger.info("スタッキングアンサンブル学習完了（計算コスト最適化版）")

            # アンサンブル全体の評価
            ensemble_result = self._evaluate_ensemble(X_test, y_test)

            # 学習結果情報を追加
            ensemble_result.update(
                {
                    "model_type": "StackingEnsemble",
                    "base_models": self._base_model_types,
                    "meta_model": self._meta_model_type,
                    "cv_folds": self.cv_folds,
                    "stack_method": self.stack_method,
                    "n_jobs": self.n_jobs,
                    "passthrough": self.passthrough,
                    "sklearn_implementation": False,  # 自前実装
                    "training_samples": len(X_train),
                    "test_samples": len(X_test) if X_test is not None else 0,
                    "fitted_base_models": list(self._fitted_base_models.keys()),
                }
            )

            return ensemble_result

        except Exception as e:
            logger.error(f"スタッキングアンサンブル学習エラー: {e}")
            raise ModelError(f"スタッキングアンサンブル学習に失敗しました: {e}")

    def _create_cv_splitter(self, X_train: pd.DataFrame) -> Any:
        """
        クロスバリデーション分割器を作成

        Args:
            X_train: 学習データ（インデックスから時系列情報を取得）

        Returns:
            CVスプリッター
        """
        from ..cross_validation.purged_kfold import PurgedKFold
        from app.config.unified_config import unified_config
        from ..common.time_series_utils import get_t1_series
        from sklearn.model_selection import StratifiedKFold, KFold

        cv_strategy = self.config.get("cv_strategy", "purged_kfold")

        if cv_strategy == "kfold":
            return KFold(n_splits=self.cv_folds, shuffle=False)
        elif cv_strategy == "stratified_kfold":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=False)
        else:
            # デフォルト: PurgedKFold
            t1_horizon_n = unified_config.ml.training.prediction_horizon
            t1 = get_t1_series(X_train.index, t1_horizon_n)
            pct_embargo = getattr(unified_config.ml.training, "pct_embargo", 0.01)
            return PurgedKFold(n_splits=self.cv_folds, t1=t1, pct_embargo=pct_embargo)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングアンサンブルで予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果 (predict_probaの結果を返す、後方互換性のため)
        """
        return self.predict_proba(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングアンサンブルで予測確率を取得

        処理フロー:
        1. 各ベースモデルで予測確率を取得
        2. 予測確率をメタ特徴量としてメタモデルで最終予測

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列 (shape: [n_samples, 2])
        """
        if not self.is_fitted:
            raise ModelError("モデルが学習されていません")

        if not self._fitted_base_models or self._fitted_meta_model is None:
            raise ModelError("ベースモデルまたはメタモデルが学習されていません")

        # ベースモデルの予測確率を取得
        base_preds = pd.DataFrame(index=X.index)
        for name, model in self._fitted_base_models.items():
            try:
                proba = model.predict_proba(X)
                base_preds[name] = proba[:, 1]  # ポジティブクラスの確率
            except Exception as e:
                logger.warning(f"ベースモデル({name})の予測でエラー: {e}")
                raise ModelError(f"ベースモデル({name})の予測に失敗しました: {e}")

        # メタ特徴量を構築
        if self.passthrough:
            meta_features = pd.concat([base_preds, X], axis=1)
        else:
            meta_features = base_preds

        # メタモデルで最終予測
        return self._fitted_meta_model.predict_proba(meta_features)

    def _create_base_estimators(self) -> List[Tuple[str, Any]]:
        """
        ベースモデルのリストを作成（StackingClassifier用）

        Returns:
            (name, estimator)のタプルのリスト
        """
        estimators = []

        for model_type in self.base_models:
            try:
                model = self._create_base_model(model_type)
                estimators.append((model_type, model))
                logger.info(f"ベースモデル追加: {model_type}")
            except Exception as e:
                logger.warning(f"ベースモデル({model_type})の作成をスキップ: {e}")
                continue

        if not estimators:
            raise ModelError("有効なベースモデルが作成できませんでした")

        return estimators

    def _evaluate_ensemble(
        self,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """
        スタッキングアンサンブル全体の評価を実行

        Args:
            X_test: テスト用特徴量
            y_test: テスト用ターゲット

        Returns:
            アンサンブル評価結果
        """
        result = {
            "model_type": "StackingEnsemble",
            "base_models": self._base_model_types,
            "meta_model": self._meta_model_type,
            "cv_folds": self.cv_folds,
            "stack_method": self.stack_method,
            "fitted_base_models": list(self._fitted_base_models.keys()),
        }

        # テストデータがある場合はアンサンブル予測を評価
        if X_test is not None and y_test is not None:
            try:
                y_pred = self.predict(X_test)
                y_pred_proba = self.predict_proba(X_test)

                ensemble_metrics = self._evaluate_predictions(
                    y_test, y_pred, y_pred_proba
                )
                result.update(ensemble_metrics)

            except Exception as e:
                logger.warning(f"アンサンブル評価でエラー: {e}")
                result["evaluation_error"] = str(e)

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """
        スタッキングアンサンブルの特徴量重要度を取得

        メタモデルの特徴量重要度を返します。
        ベースモデルの重要度は個別に取得する必要があります。

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_fitted or self._fitted_meta_model is None:
            return {}

        try:
            # メタモデルの特徴量重要度を取得
            meta_model = self._fitted_meta_model

            # ベースモデル名を取得して特徴量名として使用
            estimator_names = list(self._fitted_base_models.keys())

            if hasattr(meta_model, "feature_importances_"):
                # Tree-based models
                importances = meta_model.feature_importances_

                if len(importances) == len(estimator_names):
                    feature_names = estimator_names
                else:
                    feature_names = [
                        f"meta_feature_{i}" for i in range(len(importances))
                    ]

                return dict(zip(feature_names, importances))
            elif hasattr(meta_model, "coef_"):
                # Linear models
                coef = meta_model.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                else:
                    coef = np.abs(coef)

                if len(coef) == len(estimator_names):
                    feature_names = estimator_names
                else:
                    feature_names = [f"meta_feature_{i}" for i in range(len(coef))]

                return dict(zip(feature_names, coef))
            else:
                logger.warning(
                    f"メタモデル({self._meta_model_type})は特徴量重要度をサポートしていません"
                )
                return {}

        except Exception as e:
            logger.warning(f"特徴量重要度の取得でエラー: {e}")
            return {}

    def load_models(self, base_path: str) -> bool:
        """
        スタッキングアンサンブルモデルを読み込み

        Args:
            base_path: モデル読み込みパス

        Returns:
            読み込み成功フラグ
        """
        try:
            import glob
            import json
            import os
            import joblib

            # 新形式: 自前実装のモデルファイルを探す
            pattern = f"{base_path}_stacking_ensemble_*.pkl"
            model_files = glob.glob(pattern)

            # 旧形式へのフォールバック
            if not model_files:
                pattern = f"{base_path}_stacking_classifier_*.pkl"
                model_files = glob.glob(pattern)

            if not model_files:
                logger.warning(f"モデルファイルが見つかりません: {pattern}")
                return False

            model_path = sorted(model_files)[-1]  # 最新のファイルを選択

            # モデルを読み込み
            model_data = joblib.load(model_path)

            if isinstance(model_data, dict):
                # 新形式: 自前実装
                if "fitted_base_models" in model_data:
                    self._fitted_base_models = model_data["fitted_base_models"]
                    self._fitted_meta_model = model_data["fitted_meta_model"]
                    self._base_model_types = model_data.get(
                        "base_model_types", list(self._fitted_base_models.keys())
                    )
                    self._meta_model_type = model_data.get(
                        "meta_model_type", "logistic_regression"
                    )
                    self.feature_columns = model_data.get("feature_columns", None)
                    self.config = model_data.get("config", self.config)
                    self.passthrough = model_data.get("passthrough", False)
                    logger.info("自前実装のスタッキングモデルを読み込み完了")
                # 旧形式: StackingClassifier
                elif "ensemble_classifier" in model_data:
                    # 旧形式は後方互換性のために読み込むが警告を出す
                    logger.warning(
                        "旧形式（StackingClassifier）のモデルを読み込みます。"
                        "このモデルは新しい自前実装とは互換性がありません。再学習を推奨します。"
                    )
                    self.stacking_classifier = model_data["ensemble_classifier"]
                    self.feature_columns = model_data.get("feature_columns", None)
                    self._base_model_types = model_data.get(
                        "base_models", self._base_model_types
                    )
                else:
                    logger.warning("不明なモデル形式です")
                    return False
            else:
                logger.warning("不明なモデル形式です")
                return False

            # メタデータを読み込み（存在する場合）
            metadata_path = model_path.replace(".pkl", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                self._base_model_types = metadata.get(
                    "base_models", self._base_model_types
                )
                self._meta_model_type = metadata.get(
                    "meta_model", self._meta_model_type
                )
                self.cv_folds = metadata.get("cv_folds", self.cv_folds)
                self.stack_method = metadata.get("stack_method", self.stack_method)
                if "feature_columns" in metadata:
                    self.feature_columns = metadata["feature_columns"]

            self.is_fitted = True
            logger.info(f"スタッキングアンサンブルモデルを読み込みました: {model_path}")
            return True

        except Exception as e:
            logger.error(f"アンサンブルモデル読み込みエラー: {e}")
            return False

    def get_oof_predictions(self) -> Optional[np.ndarray]:
        """スタッキングモデルのOOF予測確率を返す"""
        return self.oof_predictions

    def get_oof_base_model_predictions(self) -> Optional[pd.DataFrame]:
        """各ベースモデルのOOF予測確率をDataFrameで返す"""
        return self.oof_base_model_predictions

    def get_X_train_original(self) -> Optional[pd.DataFrame]:
        """学習に使用した元の特徴量Xを返す"""
        return self.X_train_original

    def get_y_train_original(self) -> Optional[pd.Series]:
        """学習に使用した元のラベルyを返す"""
        return self.y_train_original

    def predict_base_models_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        各ベースモデルの予測確率を取得する

        Args:
            X: 特徴量DataFrame

        Returns:
            各ベースモデルの予測確率を含むDataFrame
        """
        if not self.is_fitted or not self._fitted_base_models:
            raise ModelError("モデルが学習されていません")

        try:
            base_preds = pd.DataFrame(index=X.index)
            for name, model in self._fitted_base_models.items():
                proba = model.predict_proba(X)
                base_preds[name] = proba[:, 1]  # ポジティブクラスの確率

            return base_preds

        except Exception as e:
            logger.error(f"ベースモデル予測確率取得エラー: {e}")
            raise ModelError(f"ベースモデル予測確率の取得に失敗しました: {e}")

    def clear_training_data(self) -> None:
        """
        学習用の一時データをクリアしてメモリを解放する

        推論（predict）には不要な、学習時の元データやOOF予測値を削除します。
        モデルのシリアライズ（保存）前に実行することで、ファイルサイズを削減できます。
        """
        self.X_train_original = None
        self.y_train_original = None
        self.oof_predictions = None
        self.oof_base_model_predictions = None
        logger.debug("StackingEnsembleの学習用一時データをクリアしました")

    def cleanup(self) -> None:
        """
        リソースをクリーンアップ

        大量のメモリを消費するモデルオブジェクトを明示的に解放します。
        """
        logger.info("StackingEnsembleのリソースをクリーンアップ中...")

        # ベースモデルのクリーンアップ
        if self._fitted_base_models:
            for name in list(self._fitted_base_models.keys()):
                self._fitted_base_models[name] = None
            self._fitted_base_models.clear()

        # メタモデルのクリーンアップ
        self._fitted_meta_model = None

        # OOF予測データのクリーンアップ
        self.oof_predictions = None
        self.oof_base_model_predictions = None
        self.X_train_original = None
        self.y_train_original = None

        # 後方互換性用
        self.stacking_classifier = None

        self.is_fitted = False
        logger.info("StackingEnsembleのリソースクリーンアップ完了")
