"""
ML モデル最適化サービスモジュール

このモジュールは、機械学習モデル（特にアンサンブルモデル）のハイパーパラメータ最適化を実行するための
高レベルサービスを提供します。Optuna をバックエンドとして使用し、目的関数の定義、パラメータ空間の探索、
学習プロセスの実行を管理します。

主なクラス:
    - OptimizationSettings: 最適化の実行設定（試行回数、パラメータ空間など）を定義するデータクラス。
    - OptimizationService: 最適化プロセス全体を統括するサービスクラス。トレーナーと連携して最適なパラメータを探索します。
"""

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.utils.error_handler import safe_operation

from .optuna_optimizer import OptunaOptimizer, ParameterSpace

logger = logging.getLogger(__name__)


class OptimizationSettings:
    """最適化設定クラス"""

    def __init__(
        self,
        enabled: bool = False,
        n_calls: int = 50,
        parameter_space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.enabled = enabled
        self.n_calls = n_calls
        self.parameter_space = parameter_space or {}


class OptimizationService:
    """
    ML モデルのハイパーパラメータ最適化を統括するサービス

    `OptunaOptimizer` をバックエンドとして使用し、アンサンブルモデルを構成する
    各ベースモデルやメタモデルの最適なパラメータセットを自動探索します。
    目的関数（Objective Function）の生成、探索空間の設定、
    CV（交差検証）回数の調整等を行い、指定された試行回数内で
    モデル性能（主にマクロ F1 スコア）を最大化します。
    """

    def __init__(self):
        self.optimizer = OptunaOptimizer()

    @safe_operation(context="パラメータ最適化", is_api_call=False)
    def optimize_parameters(
        self,
        trainer: Any,
        training_data: pd.DataFrame,
        optimization_settings: OptimizationSettings,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        model_name: Optional[str] = None,
        **training_params,
    ) -> Dict[str, Any]:
        """
        MLモデル（単一またはアンサンブル）のハイパーパラメータ最適化を実行します。

        このメソッドは、Optunaを使用して指定された試行回数内で目的関数を最大化するパラメータを探索します。
        内部プロセス：
        1. パラメータ探索空間の構築（`_prepare_parameter_space`）。
        2. 学習と評価を繰り返す目的関数の生成（`_create_objective_function`）。
        3. 指定された試行回数（`n_calls`）に基づく最適化の実行。

        Args:
            trainer (Any): 最適化対象のトレーナーインスタンス。
            training_data (pd.DataFrame): 学習・検証に使用するOHLCVデータ。
            optimization_settings (OptimizationSettings): 最適化の設定（有効化、試行回数、カスタム探索空間）。
            funding_rate_data (Optional[pd.DataFrame]): 追加の市場データ。
            open_interest_data (Optional[pd.DataFrame]): 追加の市場データ。
            model_name (Optional[str]): 最適化プロセス中には保存されませんが、ログ出力等で使用される識別名。
            **training_params: トレーナーの `train_model` メソッドに渡される追加のパラメータ。

        Returns:
            Dict[str, Any]: 最適化結果のサマリー。
                主なキー：
                - "best_params" (Dict): 発見された最適なパラメータセット。
                - "best_score" (float): 達成された最高評価値（通常はF1スコア）。
                - "optimization_time" (float): 最適化に要した時間（秒）。

        Note:
            最適化プロセスは計算負荷が高いため、APIリクエストとは独立したワーカープロセスで実行されることを想定しています。
        """
        try:
            logger.info("🚀 最適化プロセスを開始")

            # パラメータ空間を準備
            parameter_space = self._prepare_parameter_space(
                trainer, optimization_settings
            )

            # 目的関数を作成
            objective_function = self._create_objective_function(
                trainer=trainer,
                training_data=training_data,
                optimization_settings=optimization_settings,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                **training_params,
            )

            # 最適化を実行
            result = self.optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                n_calls=optimization_settings.n_calls,
            )

            return {
                "method": "optuna",
                "best_params": result.best_params,
                "best_score": result.best_score,
                "total_evaluations": result.total_evaluations,
                "optimization_time": result.optimization_time,
            }

        finally:
            # 確実にリソースを解放
            self.optimizer.cleanup()

    def optimize_full_pipeline(
        self,
        feature_superset: pd.DataFrame,
        labels: pd.Series,
        ohlcv_data: pd.DataFrame,
        n_trials: int = 50,
        test_ratio: float = 0.2,
        frac_diff_d_values: Optional[list] = None,
        fixed_label_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        特徴量エンジニアリング、特徴量選択、およびモデル学習を同時に最適化（CASH）します。

        このメソッドは、単なるモデルのハイパーパラメータ探索を超え、パイプライン全体の最適化を行います：
        1. **特徴量抽出**: 分数次差分（FracDiff）の次数 `d` を探索パラメータに含め、最適な `d` のカラムをスーパーセットから選択。
        2. **特徴量選択**: 生成された数百〜数千の特徴量から、性能に寄与する上位 `k` 個を抽出する手法やパラメータを最適化。
        3. **モデル選択と学習**: ベースモデル（LightGBM, XGBoost, CatBoost等）の選択と、それぞれのハイパーパラメータを探索。
        4. **評価**: ホールドアウト検証または交差検証により、汎化性能を最大化する組み合わせを特定。

        Args:
            feature_superset (pd.DataFrame): create_feature_superset で生成した全パターン特徴量。
            labels (pd.Series): 目的変数（ターゲットラベル）。
            ohlcv_data (pd.DataFrame): 市場データ（期間計算用）。
            n_trials (int): Optunaの試行回数。デフォルトは50。
            test_ratio (float): 最終的なモデル評価に使用するテストデータの割合。
            frac_diff_d_values (Optional[list]): 探索対象とする分数次差分の `d` 値のリスト。
            fixed_label_params (Optional[Dict]): ラベル生成に関連する固定パラメータ（例: {"tbm_horizon": 24}）。

        Returns:
            Dict[str, Any]: 最適化されたパイプライン構成。
                主なキー：
                - "best_pipeline_params" (Dict): 最適な特徴量 `d`、選択手法、モデル設定。
                - "performance_metrics" (Dict): テストデータでの評価メトリクス。
                - "feature_importance" (Dict): 最終的に選択された特徴量の重要度。

        Note:
            この「全方位最適化」は、特徴量とモデルの間の隠れた相互作用（例: 特定のモデルは特定の `d` 値で性能が出る等）を
            捉えることができ、手動での調整に比べて大幅な精度向上が期待できます。
        """
        if frac_diff_d_values is None:
            frac_diff_d_values = [0.3, 0.4, 0.5, 0.6]

        logger.info(f"🚀 パイプライン同時最適化を開始: 試行回数={n_trials}")

        # データ分割基準日を決定 (feature_supersetのインデックスを使用)
        n_samples = len(feature_superset)
        split_idx = int(n_samples * (1 - test_ratio))
        split_date = feature_superset.index[split_idx]

        logger.info(f"データ分割基準日: {split_date}")

        # パラメータ空間を定義
        parameter_space = self._get_pipeline_parameter_space(
            frac_diff_d_values, fixed_label_params
        )

        # 目的関数を作成
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # 0. ラベルの動的生成 (Triple Barrier Method)
                # 固定パラメータがある場合はそれを優先し、なければOptunaの提案値(params)を使用
                label_params = params.copy()
                if fixed_label_params:
                    label_params.update(fixed_label_params)

                df_for_label = (
                    ohlcv_data if ohlcv_data is not None else feature_superset
                )

                current_labels = self._generate_pipeline_labels(
                    df_for_label, label_params
                )

                if current_labels.empty:
                    return 0.0

                # ... (以下同様)

                # 1. 特徴量とラベルのアラインメント
                X_aligned, y_aligned = self._align_feature_superset_and_labels(
                    feature_superset, current_labels
                )
                if len(X_aligned) < 100:
                    return 0.0

                # 2. TrainVal / Test 分割 (split_date基準)
                X_trainval_curr, y_trainval_curr, _, _ = self._split_by_date(
                    X_aligned, y_aligned, split_date
                )

                if len(X_trainval_curr) < 50:
                    return 0.0

                # 3. FracDiff d値でフィルタ
                d_value = params["frac_diff_d"]

                # 4. 内部CV用分割 (時系列ホールドアウト 20%)
                val_split_idx = int(len(X_trainval_curr) * 0.8)

                X_train = X_trainval_curr.iloc[:val_split_idx]
                y_train = y_trainval_curr.iloc[:val_split_idx]
                X_val = X_trainval_curr.iloc[val_split_idx:]
                y_val = y_trainval_curr.iloc[val_split_idx:]

                # 5. 特徴量選択 + モデル学習 + 評価
                score, _ = self._evaluate_selected_model_pipeline(
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_val,
                    y_eval=y_val,
                    d_value=d_value,
                    selection_method=params["selection_method"],
                    correlation_threshold=params["correlation_threshold"],
                    min_features=params["min_features"],
                    learning_rate=params["learning_rate"],
                    num_leaves=params["num_leaves"],
                )
                return score

            except Exception as e:
                logger.warning(f"パイプライン評価エラー: {e}")
                return 0.0

        # 最適化を実行
        result = self.optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=n_trials,
        )

        best_params = result.best_params
        best_score = result.best_score

        # 最終評価: ベストパラメータでテストデータを評価
        logger.info("🔍 ベストパラメータでテストデータを評価中...")

        # 固定パラメータをマージ
        final_params = best_params.copy()
        if fixed_label_params:
            final_params.update(fixed_label_params)

        df_for_label = ohlcv_data if ohlcv_data is not None else feature_superset

        labels_best = self._generate_pipeline_labels(df_for_label, final_params)

        # アラインメント
        X_aligned, y_aligned = self._align_feature_superset_and_labels(
            feature_superset, labels_best
        )

        # 分割
        X_trainval, y_trainval, X_test, y_test = self._split_by_date(
            X_aligned, y_aligned, split_date
        )

        # ベストパラメータでフルTrainValデータで再学習
        test_score, n_selected_features = self._evaluate_selected_model_pipeline(
            X_train=X_trainval,
            y_train=y_trainval,
            X_eval=X_test,
            y_eval=y_test,
            d_value=best_params["frac_diff_d"],
            selection_method=best_params["selection_method"],
            correlation_threshold=best_params["correlation_threshold"],
            min_features=best_params["min_features"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            cv_folds=3,
            cv_strategy="timeseries",
            n_jobs=-1,
        )

        baseline_score = 0.0
        self.optimizer.cleanup()

        # 固定パラメータも結果に含める
        result_params = best_params.copy()
        if fixed_label_params:
            result_params.update(fixed_label_params)

        return {
            "best_params": result_params,
            "best_score": best_score,
            "test_score": test_score,
            "baseline_score": baseline_score,
            "improvement": test_score - baseline_score,
            "total_evaluations": result.total_evaluations,
            "optimization_time": result.optimization_time,
            "n_selected_features": n_selected_features,
        }

    def _generate_pipeline_labels(
        self, df_for_label: pd.DataFrame, label_params: Dict[str, Any]
    ) -> pd.Series:
        """パイプライン最適化向けに triple barrier ラベルを一元生成する"""
        from app.services.ml.label_generation.presets import (
            triple_barrier_method_preset,
        )

        return triple_barrier_method_preset(
            df=df_for_label,
            timeframe="1h",
            horizon_n=label_params.get("tbm_horizon", 24),
            pt=label_params.get("tbm_pt", 1.0),
            sl=label_params.get("tbm_sl", 1.0),
            min_ret=0.001,
            price_column="close",
            use_atr=True,
        )

    @staticmethod
    def _align_feature_superset_and_labels(
        feature_superset: pd.DataFrame, labels: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """特徴量スーパーセットとラベルを共通インデックスで揃える。"""
        common_idx = feature_superset.index.intersection(labels.index)
        return feature_superset.loc[common_idx], labels.loc[common_idx]

    @staticmethod
    def _split_by_date(
        X_aligned: pd.DataFrame,
        y_aligned: pd.Series,
        split_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """時系列順に TrainVal / Test を分割する。"""
        mask_trainval = X_aligned.index < split_date
        X_trainval = X_aligned[mask_trainval]
        y_trainval = y_aligned[mask_trainval]
        X_test = X_aligned[~mask_trainval]
        y_test = y_aligned[~mask_trainval]
        return X_trainval, y_trainval, X_test, y_test

    def _evaluate_selected_model_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        *,
        d_value: float,
        selection_method: str,
        correlation_threshold: float,
        min_features: int,
        learning_rate: float,
        num_leaves: int,
        cv_folds: int = 3,
        cv_strategy: str = "timeseries",
        n_jobs: int = -1,
        n_estimators: int = 100,
        random_state: int = 42,
        verbosity: int = -1,
        force_col_wise: bool = True,
    ) -> tuple[float, int]:
        """FracDiff フィルタ、特徴量選択、学習、評価を 1 箇所にまとめる"""
        from lightgbm import LGBMClassifier
        from sklearn.metrics import balanced_accuracy_score as sklearn_metric

        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )
        from app.services.ml.feature_selection.feature_selector import FeatureSelector

        X_train_filtered = FeatureEngineeringService.filter_superset_for_d(
            X_train, d_value
        )
        X_eval_filtered = FeatureEngineeringService.filter_superset_for_d(
            X_eval, d_value
        )

        exclude_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [c for c in X_train_filtered.columns if c not in exclude_cols]
        X_train_features = X_train_filtered[feature_cols]
        X_eval_features = X_eval_filtered[feature_cols]

        selector = FeatureSelector(
            method=selection_method,
            correlation_threshold=correlation_threshold,
            min_features=min_features,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            n_jobs=n_jobs,
        )

        X_train_selected = selector.fit_transform(X_train_features, y_train)
        X_eval_selected = selector.transform(X_eval_features)

        model = LGBMClassifier(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            random_state=random_state,
            verbosity=verbosity,
            force_col_wise=force_col_wise,
        )

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_eval_selected)
        score = sklearn_metric(y_eval, y_pred)
        return score, int(X_train_selected.shape[1])

    def optimize_meta_model_with_oof(
        self,
        primary_pipeline: Any,
        X_superset: pd.DataFrame,
        y_true: pd.Series,
        n_trials: int = 30,
        cv_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        OOF予測を用いたメタモデルの最適化
        ...
        """
        import numpy as np
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )
        from app.services.ml.feature_selection.feature_selector import FeatureSelector

        logger.info("🚀 メタモデル最適化 (OOF) を開始")

        # 1. 一次モデルの OOF 予測を生成
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        oof_probs = pd.Series(np.nan, index=y_true.index)

        logger.info(
            f"[*] Generating OOF predictions for primary model ({cv_splits} folds)..."
        )
        for train_idx, val_idx in tscv.split(X_superset, y_true):
            X_train, X_val = X_superset.iloc[train_idx], X_superset.iloc[val_idx]
            y_train = y_true.iloc[train_idx]

            primary_pipeline.fit(X_train, y_train)
            oof_probs.iloc[val_idx] = primary_pipeline.predict_proba(X_val)[:, 1]

        # 予測が生成されたサンプルのみを対象にする (最初のFoldはnanになる)
        valid_mask = oof_probs.notna()
        X_meta_full = X_superset[valid_mask]
        y_true_meta = y_true[valid_mask]
        primary_probs = oof_probs[valid_mask]

        # メタラベルの生成: 一次モデルが「1」と予測し、かつ正解も「1」なら 1, 外れたら 0
        # ここでは閾値 0.5 で「エントリー判断」をしたポイントのみを抽出
        entry_mask = primary_probs >= 0.5
        if entry_mask.sum() < 100:
            return {"status": "skipped", "reason": "too few entries for meta-learning"}

        X_meta = X_meta_full[entry_mask]
        # メタラベル: 1 = 成功(TP), 0 = 失敗(FP/ダマシ)
        y_meta = (y_true_meta[entry_mask] == 1).astype(int)

        logger.info(
            f"[*] Meta-dataset prepared: {len(y_meta)} samples (Win Rate: {y_meta.mean():.2%})"
        )

        # 2. メタモデルの最適化
        # メタモデルには「マイクロストラクチャ系」などの特権特徴量を優先的に選ばせる
        # また、一次モデルの予測確率自体も強力な特徴量になる
        micro_keywords = [
            "LS_",
            "OI_",
            "VPIN",
            "Roll_",
            "Amihud_",
            "Kyles_",
            "Spread",
            "Volume_CV",
        ]
        meta_feature_cols = [
            c for c in X_meta.columns if any(k in c for k in micro_keywords)  # type: ignore[reportAttributeAccessIssue]
        ]

        X_meta_specialized = X_meta[meta_feature_cols].copy()
        X_meta_specialized["primary_prob"] = primary_probs

        logger.info(
            f"[*] Specialized meta-features: {len(meta_feature_cols)} microstructure cols + primary_prob"
        )

        frac_diff_d_values = [0.3, 0.4, 0.5]
        parameter_space = self._get_pipeline_parameter_space(frac_diff_d_values)

        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # 特徴量フィルタリング (FracDiff適用済みのカラムから選択)
                d_value = params["frac_diff_d"]
                X_filt = FeatureEngineeringService.filter_superset_for_d(
                    X_meta_specialized, d_value
                )

                # 特徴量選択
                selector = FeatureSelector(
                    method=params["selection_method"],
                    min_features=params["min_features"],
                    cv_strategy="timeseries",
                )
                X_sel = selector.fit_transform(X_filt, y_meta)

                # メタモデル (LightGBM)
                model = LGBMClassifier(
                    learning_rate=params["learning_rate"],
                    num_leaves=params["num_leaves"],
                    n_estimators=50,  # 高速化のため少なめ
                    class_weight="balanced",
                    random_state=42,
                    verbosity=-1,
                )

                # CV評価 (F1スコアを重視)
                scores = cross_val_score(
                    model, X_sel, y_meta, cv=TimeSeriesSplit(n_splits=3), scoring="f1"
                )
                return scores.mean()
            except Exception:
                return 0.0

        result = self.optimizer.optimize(
            objective_function, parameter_space, n_calls=n_trials
        )

        return {
            "best_params": result.best_params,
            "best_f1": result.best_score,
            "n_samples": len(y_meta),
            "base_win_rate": y_meta.mean(),
            "n_meta_features": X_meta_specialized.shape[1],
        }

    def _get_pipeline_parameter_space(
        self,
        frac_diff_d_values: list,
        fixed_label_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ParameterSpace]:
        """パイプライン同時最適化用のパラメータ空間を定義"""

        space = {
            # Feature Engineering
            "frac_diff_d": ParameterSpace(
                type="categorical", categories=frac_diff_d_values
            ),
            # Feature Selection
            "selection_method": ParameterSpace(
                type="categorical", categories=["staged", "rfecv", "mutual_info"]
            ),
            "correlation_threshold": ParameterSpace(type="real", low=0.85, high=0.99),
            "min_features": ParameterSpace(type="integer", low=5, high=30),
            # Model (LightGBM)
            "learning_rate": ParameterSpace(type="real", low=0.005, high=0.1),
            "num_leaves": ParameterSpace(type="integer", low=16, high=128),
        }

        # ラベル生成パラメータ（固定されていない場合のみ追加）
        fixed_keys = fixed_label_params.keys() if fixed_label_params else []

        if "tbm_pt" not in fixed_keys:
            space["tbm_pt"] = ParameterSpace(type="real", low=0.5, high=3.0)
        if "tbm_sl" not in fixed_keys:
            space["tbm_sl"] = ParameterSpace(type="real", low=0.5, high=3.0)
        if "tbm_horizon" not in fixed_keys:
            space["tbm_horizon"] = ParameterSpace(type="integer", low=4, high=48)

        return space

    def _evaluate_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """ベースライン（デフォルトパラメータ）での評価"""
        try:
            score, _ = self._evaluate_selected_model_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_test,
                y_eval=y_test,
                d_value=0.4,
                selection_method="staged",
                correlation_threshold=0.90,
                min_features=10,
                learning_rate=0.05,
                num_leaves=31,
                cv_folds=3,
                cv_strategy="timeseries",
                n_jobs=-1,
            )
            return score

        except Exception as e:
            logger.warning(f"ベースライン評価エラー: {e}")
            return 0.0

    def _prepare_parameter_space(
        self, trainer: Any, optimization_settings: OptimizationSettings
    ) -> Dict[str, ParameterSpace]:
        """
        探索対象となるパラメータ空間を定義

        設定で探索空間が明示されている場合はそれを使用し、
        そうでない場合はトレーナーのアンサンブル設定に基づいたデフォルト空間を生成します。

        Args:
            trainer: 対象のトレーナー
            optimization_settings: 最適化設定

        Returns:
            パラメータ名をキー、探索範囲（ParameterSpace）を値とする辞書
        """
        if optimization_settings.parameter_space:
            return self._convert_parameter_space_config(
                optimization_settings.parameter_space
            )

        # EnsembleTrainerの場合（単一モデルも含む）
        if hasattr(trainer, "ensemble_config"):
            c = trainer.ensemble_config
            return self.optimizer.get_ensemble_parameter_space(
                c.get("method", "stacking"), c.get("models", ["lightgbm", "xgboost"])
            )

        from app.services.ml.optimization.optuna_optimizer import (
            build_lightgbm_parameter_space,
        )

        return build_lightgbm_parameter_space()

    def _convert_parameter_space_config(
        self, parameter_space_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ParameterSpace]:
        """設定辞書をParameterSpaceオブジェクトに変換"""
        return {
            name: ParameterSpace(
                type=cfg["type"],
                low=int(cfg["low"]) if cfg["type"] == "integer" else cfg.get("low"),
                high=int(cfg["high"]) if cfg["type"] == "integer" else cfg.get("high"),
                categories=cfg.get("categories"),
            )
            for name, cfg in parameter_space_config.items()
        }

    def _create_objective_function(
        self,
        trainer: Any,
        training_data: pd.DataFrame,
        optimization_settings: OptimizationSettings,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        **base_training_params,
    ) -> Callable[[Dict[str, Any]], float]:
        """
        Optuna に渡す目的関数（Objective Function）を作成

        各試行で渡されるパラメータを受け取り、モデル学習と評価を行い、
        最大化すべきスコア（F1 スコア）を返します。

        Args:
            trainer: テンプレートとなるトレーナー
            training_data: 学習データ
            optimization_settings: 最適化設定
            **base_training_params: 固定の学習パラメータ

        Returns:
            パラメータ辞書を受け取りスコア（float）を返す関数
        """
        evaluation_count = 0

        def objective_function(params: Dict[str, Any]) -> float:
            nonlocal evaluation_count
            evaluation_count += 1

            try:
                logger.info(
                    f"🔍 試行 {evaluation_count}/{optimization_settings.n_calls}: {params}"
                )

                # パラメータのマージ
                training_params = {**base_training_params, **params}

                # 一時的なトレーナーを作成
                temp_trainer = self._create_temp_trainer(trainer, params)

                # 学習実行（保存なし）
                result = temp_trainer.train_model(
                    training_data=training_data,
                    funding_rate_data=funding_rate_data,
                    open_interest_data=open_interest_data,
                    save_model=False,
                    model_name=None,
                    **training_params,
                )

                # スコア取得
                f1_score = result.get("f1_score", 0.0)
                if "classification_report" in result:
                    f1_score = (
                        result["classification_report"]
                        .get("macro avg", {})
                        .get("f1-score", f1_score)
                    )

                return f1_score

            except Exception as e:
                logger.warning(f"目的関数評価エラー: {e}")
                return 0.0

        return objective_function

    def _create_temp_trainer(
        self, original_trainer: object, params: Dict[str, object]
    ) -> object:
        """一時的なトレーナーを作成（全てEnsembleTrainerで統一）"""
        # オリジナルのトレーナーがEnsembleTrainerであることを前提
        if hasattr(original_trainer, "ensemble_config"):
            temp_config = original_trainer.ensemble_config.copy()

            # 最適化用にCV foldsを減らす（速度向上）
            if "stacking_params" in temp_config:
                stacking_params = temp_config["stacking_params"].copy()
                stacking_params["cv_folds"] = 3
                temp_config["stacking_params"] = stacking_params

            return EnsembleTrainer(ensemble_config=temp_config)
        else:
            # フォールバック: デフォルト設定でEnsembleTrainer作成
            return EnsembleTrainer(ensemble_config={"method": "stacking"})
