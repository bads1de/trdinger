import logging
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class MetaLabelingService:
    """
    メタラベリングサービス

    一次モデル（Primary Model）の予測結果を入力として、
    その予測が正解（収益につながる）かどうかを判定する二次モデル（Meta Model）を構築します。
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        base_model_names: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.base_model_names = base_model_names
        self.model_params = model_params or {}

    def _add_base_model_statistics(
        self, X_meta: pd.DataFrame, base_probs_filtered: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ベースモデルの予測確率から統計量を計算してメタ特徴量に追加
        """
        if base_probs_filtered.empty:
            return X_meta

        # Pandasの統計メソッドがNumPy不具合で失敗するため、値を一度リスト化して計算
        probs_values = base_probs_filtered.values
        
        # 行ごとの統計量を計算 (NumPyの不具合を避けるため、可能な限りPythonレベルで)
        means = []
        stds = []
        mins = []
        maxs = []
        
        for i in range(len(probs_values)):
            row = probs_values[i].tolist()
            means.append(sum(row) / len(row) if row else 0.0)
            mins.append(min(row) if row else 0.0)
            maxs.append(max(row) if row else 0.0)
            
            # 標準偏差
            if len(row) > 1:
                m = sum(row) / len(row)
                variance = sum((x - m) ** 2 for x in row) / len(row)
                stds.append(variance ** 0.5)
            else:
                stds.append(0.0)

        X_meta["base_prob_mean"] = means
        X_meta["base_prob_std"] = stds
        X_meta["base_prob_min"] = mins
        X_meta["base_prob_max"] = maxs
        
        return X_meta

    def create_meta_labels(
        self, primary_preds_proba: pd.Series, y_true: pd.Series, threshold: float = 0.5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        メタラベルとフィルタリング用のマスクを作成します。

        Args:
            primary_preds_proba: 一次モデルの予測確率 (Trendである確率)
            y_true: 正解ラベル (1=Trend, 0=Range)
            threshold: 一次モデルがTrendと判定する閾値

        Returns:
            trend_mask: フィルタリング用マスク (一次モデルがTrendと判定した箇所がTrue)
            y_meta: メタラベル (1=一次モデル正解, 0=一次モデル不正解)
        """
        # 一次モデルが「トレンド」と予測したサンプルのみを対象にする
        trend_mask = primary_preds_proba >= threshold

        # フィルタリングされたインデックス
        indices = primary_preds_proba.index[trend_mask]

        if len(indices) == 0:
            logger.warning("一次モデルがトレンドと予測したサンプルがありません。")
            return pd.Series(dtype=bool), pd.Series()

        # メタラベルの生成
        # 一次モデルがTrend(1)と予測し、かつ正解もTrend(1)なら -> 1 (Execute)
        # 一次モデルがTrend(1)と予測したが、正解はRange(0)なら -> 0 (Pass)
        y_meta = y_true.loc[indices]

        return trend_mask, y_meta

    def _init_model(self) -> Any:
        """モデルを初期化"""
        if self.model_type == "random_forest":
            params = {
                "n_estimators": 100, "max_depth": 5, "class_weight": "balanced",
                "random_state": 42, "n_jobs": -1, **self.model_params
            }
            return RandomForestClassifier(**params)
        if self.model_type == "lightgbm":
            params = {
                "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
                "random_state": 42, "n_jobs": -1, "class_weight": "balanced",
                "reg_lambda": 1.0, "verbose": -1, **self.model_params
            }
            return lgb.LGBMClassifier(**params)
        raise ValueError(f"未サポートのモデルタイプ: {self.model_type}")

    def _prepare_meta_features(
        self, X: pd.DataFrame, primary_proba: pd.Series, base_probs: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """メタ特徴量を準備"""
        X_meta = X.loc[mask].copy()
        X_meta["primary_proba"] = primary_proba.loc[mask]
        X_meta = pd.concat([X_meta, base_probs.loc[mask]], axis=1)
        return self._add_base_model_statistics(X_meta, base_probs.loc[mask])

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        primary_proba_train: pd.Series,
        base_model_probs_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """メタモデルを学習"""
        self.base_model_names = base_model_probs_df.columns.tolist()

        trend_mask, y_meta = self.create_meta_labels(primary_proba_train, y_train, threshold)
        if len(y_meta) < 50:
            return {"status": "skipped", "reason": "insufficient_data"}

        X_meta = self._prepare_meta_features(X_train, primary_proba_train, base_model_probs_df, trend_mask)
        self.model = self._init_model()
        self.model.fit(X_meta, y_meta)
        self.is_trained = True
        return {"status": "success", "samples": len(X_meta)}

    def predict(
        self,
        X: pd.DataFrame,
        primary_proba: pd.Series,
        base_model_probs_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.Series:
        """予測を実行"""
        if not self.is_trained:
            raise RuntimeError("メタモデルが学習されていません")

        final_pred = pd.Series(0, index=X.index)
        trend_mask = primary_proba >= threshold
        if not trend_mask.any():
            return final_pred

        X_meta = self._prepare_meta_features(X, primary_proba, base_model_probs_df, trend_mask)
        final_pred.loc[trend_mask] = self.model.predict(X_meta)
        return final_pred

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        primary_proba: pd.Series,
        base_model_probs_df: pd.DataFrame,
        threshold: float = 0.5,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ) -> pd.Series:
        """Cross-Validationを実行"""
        from sklearn.model_selection import KFold
        from ..cross_validation.purged_kfold import PurgedKFold

        trend_mask = primary_proba >= threshold
        oof_preds = pd.Series(0, index=X.index, dtype=int)
        if not trend_mask.any():
            return oof_preds

        target_idx = X.index[trend_mask]
        X_meta_target = self._prepare_meta_features(X, primary_proba, base_model_probs_df, trend_mask)
        y_target = y.loc[target_idx]

        cv = PurgedKFold(n_splits=n_splits, t1=t1.loc[target_idx], pct_embargo=pct_embargo) if t1 is not None else KFold(n_splits=n_splits, shuffle=False)

        for tr_idx, val_idx in cv.split(X_meta_target, y_target):
            tr_indices, val_indices = X_meta_target.index[tr_idx], X_meta_target.index[val_idx]
            model = self._init_model()
            model.fit(X_meta_target.loc[tr_indices], y_target.loc[tr_indices])
            oof_preds.loc[val_indices] = model.predict(X_meta_target.loc[val_indices])

        return oof_preds

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        primary_proba_test: pd.Series,
        base_model_probs_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """メタラベリング適用後のパフォーマンスを評価"""
        from ..common.evaluation import evaluate_model_predictions

        # メタ予測と一次予測（バイナリ）
        final_pred = self.predict(X_test, primary_proba_test, base_model_probs_df, threshold)
        primary_pred = (primary_proba_test >= threshold).astype(int)

        # メトリクス計算
        m_met = evaluate_model_predictions(y_test, final_pred.values)
        p_met = evaluate_model_predictions(y_test, primary_pred.values)

        return {
            "meta_accuracy": m_met["accuracy"], "meta_precision": m_met["precision"],
            "meta_recall": m_met["recall"], "meta_f1": m_met["f1_score"],
            "primary_accuracy": p_met["accuracy"], "primary_precision": p_met["precision"],
            "primary_recall": p_met["recall"], "primary_f1": p_met["f1_score"],
            "improvement_precision": m_met["precision"] - p_met["precision"],
            "improvement_recall": m_met["recall"] - p_met["recall"],
            "improvement_f1": m_met["f1_score"] - p_met["f1_score"],
            "meta_classification_report": m_met.get("classification_report", {}),
            "primary_classification_report": p_met.get("classification_report", {}),
            "meta_balanced_accuracy": m_met.get("balanced_accuracy", 0.0),
            "primary_balanced_accuracy": p_met.get("balanced_accuracy", 0.0),
        }