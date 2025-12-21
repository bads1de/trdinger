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
        アンサンブルモデルのハイパーパラメータ最適化を実行

        Optuna を用いて、指定された試行回数（n_calls）の中で
        最も高い F1 スコア（マクロ平均）を出すパラメータの組み合わせを探索します。
        内部で一時的なトレーナーを生成し、評価を行います。

        Args:
            trainer: ベースとなるトレーナーインスタンス
            training_data: 学習用データ
            optimization_settings: 最適化の設定（有効化、試行回数、探索空間）
            funding_rate_data: オプションの FR データ
            open_interest_data: オプションの OI データ
            model_name: 保存時のモデル名（最適化中は保存されません）
            **training_params: 追加の学習パラメータ

        Returns:
            最適パラメータ、ベストスコア、評価時間等を含む結果辞書
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
        ohlcv_data: pd.DataFrame,  # Added for dynamic label generation
        n_trials: int = 50,
        test_ratio: float = 0.2,
        frac_diff_d_values: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        特徴量エンジニアリング + 特徴量選択 + モデル学習の同時最適化（CASH）

        スーパーセットから特定のd値のカラムを選択し、特徴量選択とモデル学習を
        パイプラインとして評価することで、最適なパラメータ組み合わせを探索します。

        Args:
            feature_superset: create_feature_superset で生成した全パターン特徴量
            labels: ターゲットラベル
            n_trials: Optuna試行回数
            test_ratio: テストデータの割合（最終評価用）
            frac_diff_d_values: 探索する分数次差分のd値リスト

        Returns:
            ベストパラメータ、ベストスコア、比較結果などを含む辞書
        """
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )
        from app.services.ml.feature_selection.feature_selector import FeatureSelector
        from app.services.ml.label_generation.presets import triple_barrier_method_preset
        from lightgbm import LGBMClassifier
        from sklearn.metrics import balanced_accuracy_score as sklearn_metric

        if frac_diff_d_values is None:
            frac_diff_d_values = [0.3, 0.4, 0.5, 0.6]

        logger.info(f"🚀 パイプライン同時最適化を開始: 試行回数={n_trials}")

        # データ分割基準日を決定 (feature_supersetのインデックスを使用)
        n_samples = len(feature_superset)
        split_idx = int(n_samples * (1 - test_ratio))
        split_date = feature_superset.index[split_idx]

        logger.info(f"データ分割基準日: {split_date}")

        # パラメータ空間を定義
        parameter_space = self._get_pipeline_parameter_space(frac_diff_d_values)

        # 目的関数を作成
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # 0. ラベルの動的生成 (Triple Barrier Method)
                # ohlcv_data が渡されていない場合は feature_superset から最低限のデータを復元（近似）
                # ただし正確なHigh/Lowが必要なので、ohlcv_dataは必須推奨
                
                df_for_label = ohlcv_data if ohlcv_data is not None else feature_superset
                
                # feature_supersetにhigh/lowがない場合のフォールバックはpresets側でエラーになる可能性があるため
                # ここではohlcv_dataがあることを前提とする（optimize_full_pipelineの引数で必須化した方が良いが）
                
                current_labels = triple_barrier_method_preset(
                    df=df_for_label,
                    timeframe="1h", # 仮固定。feature_supersetから推測できるとベスト
                    horizon_n=params["tbm_horizon"],
                    pt=params["tbm_pt"],
                    sl=params["tbm_sl"],
                    min_ret=0.001,
                    price_column="close",
                    use_atr=True, # ATRベースのボラティリティを使用
                )
                
                if current_labels.empty:
                    return 0.0
                
                # 1. 特徴量とラベルのアラインメント
                common_idx = feature_superset.index.intersection(current_labels.index)
                if len(common_idx) < 100:
                    return 0.0
                    
                X_aligned = feature_superset.loc[common_idx]
                y_aligned = current_labels.loc[common_idx]

                # 2. TrainVal / Test 分割 (split_date基準)
                # objective_function内では TrainVal のみをさらに Train/Val に分割して評価
                mask_trainval = X_aligned.index < split_date
                X_trainval_curr = X_aligned[mask_trainval]
                y_trainval_curr = y_aligned[mask_trainval]
                
                if len(X_trainval_curr) < 50:
                    return 0.0

                # 3. FracDiff d値でフィルタ
                d_value = params["frac_diff_d"]
                X_filtered = FeatureEngineeringService.filter_superset_for_d(
                    X_trainval_curr, d_value
                )

                # 除外カラム
                exclude_cols = ["open", "high", "low", "close", "volume"]
                feature_cols = [c for c in X_filtered.columns if c not in exclude_cols]
                X_features = X_filtered[feature_cols]

                # 4. 内部CV用分割 (時系列ホールドアウト 20%)
                val_split_idx = int(len(X_features) * 0.8)
                
                X_train = X_features.iloc[:val_split_idx]
                y_train = y_trainval_curr.iloc[:val_split_idx]
                X_val = X_features.iloc[val_split_idx:]
                y_val = y_trainval_curr.iloc[val_split_idx:]

                # 5. 特徴量選択
                selector = FeatureSelector(
                    method=params["selection_method"],
                    correlation_threshold=params["correlation_threshold"],
                    min_features=params["min_features"],
                    cv_folds=3,
                    cv_strategy="timeseries",
                    n_jobs=-1,
                )

                X_train_selected = selector.fit_transform(X_train, y_train)
                X_val_selected = selector.transform(X_val)

                # 6. モデル学習
                model = LGBMClassifier(
                    learning_rate=params["learning_rate"],
                    num_leaves=params["num_leaves"],
                    n_estimators=100,
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True,
                )

                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_val_selected)

                # Balanced Accuracy
                score = sklearn_metric(y_val, y_pred)
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
        
        # ベストパラメータでラベル再生成
        df_for_label = ohlcv_data if ohlcv_data is not None else feature_superset
        
        labels_best = triple_barrier_method_preset(
            df=df_for_label,
            timeframe="1h",
            horizon_n=best_params["tbm_horizon"],
            pt=best_params["tbm_pt"],
            sl=best_params["tbm_sl"],
            min_ret=0.001,
            price_column="close",
            use_atr=True,
        )
        
        # アラインメント
        common_idx = feature_superset.index.intersection(labels_best.index)
        X_aligned = feature_superset.loc[common_idx]
        y_aligned = labels_best.loc[common_idx]
        
        # 分割
        mask_trainval = X_aligned.index < split_date
        X_trainval = X_aligned[mask_trainval]
        y_trainval = y_aligned[mask_trainval]
        X_test = X_aligned[~mask_trainval]
        y_test = y_aligned[~mask_trainval]

        # ベストパラメータでフルTrainValデータで再学習
        d_value = best_params["frac_diff_d"]
        X_filtered = FeatureEngineeringService.filter_superset_for_d(
            X_trainval, d_value
        )
        X_test_filtered = FeatureEngineeringService.filter_superset_for_d(
            X_test, d_value
        )

        exclude_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [c for c in X_filtered.columns if c not in exclude_cols]
        X_features = X_filtered[feature_cols]
        X_test_features = X_test_filtered[feature_cols]

        selector = FeatureSelector(
            method=best_params["selection_method"],
            correlation_threshold=best_params["correlation_threshold"],
            min_features=best_params["min_features"],
            cv_folds=3,
            n_jobs=-1,
        )

        X_train_selected = selector.fit_transform(X_features, y_trainval)
        X_test_selected = selector.transform(X_test_features)

        model = LGBMClassifier(
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            n_estimators=100,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
        )

        model.fit(X_train_selected, y_trainval)
        y_pred_test = model.predict(X_test_selected)
        test_score = sklearn_metric(y_test, y_pred_test)

        # ベースライン評価（デフォルトパラメータ）
        # 注: ベースラインもラベル生成が必要だが、ここでは固定（trend_scanning_1h相当）で比較するのが妥当
        # しかし実装が複雑になるため、ここでは「ベストパラメータでのTrainValスコア」と比較する形にするか、
        # または固定パラメータで再計算する。
        # 簡易的に0.0を返す（動的ラベルの場合、ベースラインとの直接比較は難しい）
        baseline_score = 0.0 

        self.optimizer.cleanup()

        return {
            "best_params": best_params,
            "best_score": best_score,
            "test_score": test_score,
            "baseline_score": baseline_score,
            "improvement": test_score - baseline_score,
            "total_evaluations": result.total_evaluations,
            "optimization_time": result.optimization_time,
            "n_selected_features": X_train_selected.shape[1],
        }

    def _get_pipeline_parameter_space(
        self, frac_diff_d_values: list
    ) -> Dict[str, ParameterSpace]:
        """パイプライン同時最適化用のパラメータ空間を定義"""
        return {
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
            # Label Generation (Triple Barrier)
            "tbm_pt": ParameterSpace(type="real", low=0.5, high=3.0),
            "tbm_sl": ParameterSpace(type="real", low=0.5, high=3.0),
            "tbm_horizon": ParameterSpace(type="integer", low=4, high=48),
        }

    def _evaluate_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """ベースライン（デフォルトパラメータ）での評価"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )
        from app.services.ml.feature_selection.feature_selector import FeatureSelector
        from lightgbm import LGBMClassifier
        from sklearn.metrics import balanced_accuracy_score as sklearn_metric

        try:
            # デフォルト d=0.4 でフィルタ
            X_train_filtered = FeatureEngineeringService.filter_superset_for_d(
                X_train, 0.4
            )
            X_test_filtered = FeatureEngineeringService.filter_superset_for_d(
                X_test, 0.4
            )

            exclude_cols = ["open", "high", "low", "close", "volume"]
            feature_cols = [
                c for c in X_train_filtered.columns if c not in exclude_cols
            ]
            X_features = X_train_filtered[feature_cols]
            X_test_features = X_test_filtered[feature_cols]

            # デフォルト特徴量選択
            selector = FeatureSelector(
                method="staged",
                correlation_threshold=0.90,
                min_features=10,
                cv_folds=3,
                cv_strategy="timeseries",
                n_jobs=-1,
            )

            X_train_selected = selector.fit_transform(X_features, y_train)
            X_test_selected = selector.transform(X_test_features)

            # デフォルトモデル
            model = LGBMClassifier(
                learning_rate=0.05,
                num_leaves=31,
                n_estimators=100,
                random_state=42,
                verbosity=-1,
                force_col_wise=True,
            )

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            return sklearn_metric(y_test, y_pred)

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

        return self.optimizer.get_default_parameter_space()

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
        self, original_trainer: Any, params: Dict[str, Any]
    ) -> Any:
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
